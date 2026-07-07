"""The transport-agnostic Ziggurat API, in terms of the binary `protocol` structs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
import logging

from zigpy.exceptions import DeliveryError

from zigpy_ziggurat.zigbee import protocol as p
from zigpy_ziggurat.zigbee.transport import Transport, select_transport

_LOGGER = logging.getLogger(__name__)

# The end-to-end APS ack (or local handoff) must arrive within this window.
CONFIRM_TIMEOUT = 30


class _Pending:
    """One in-flight request: its response body, plus an event queue when streaming."""

    def __init__(self, *, streaming: bool) -> None:
        loop = asyncio.get_running_loop()
        self.response: asyncio.Future[bytes] = loop.create_future()
        self.events: asyncio.Queue[bytes | None] | None = (
            asyncio.Queue() if streaming else None
        )


class ZigguratApi:
    """The request surface `ControllerApplication` speaks, over any `Transport`."""

    def __init__(
        self,
        url: str,
        on_notification: Callable[[p.Notification], None],
        on_disconnect: Callable[[BaseException | None], None],
        *,
        baudrate: int = 115200,
        flow_control: str | None = None,
    ) -> None:
        self._on_notification = on_notification
        self._on_disconnect = on_disconnect
        self._closing = False
        self._request_id = 1
        self._pending: dict[int, _Pending] = {}
        self._pending_confirms: dict[
            int, asyncio.Future[p.SendConfirm | p.ApsAckConfirm]
        ] = {}
        self._awaiting_aps_ack: set[int] = set()

        self._transport: Transport = select_transport(url)(
            url,
            self._handle_frame,
            self._on_transport_lost,
            baudrate=baudrate,
            flow_control=flow_control,
        )

    async def connect(self) -> None:
        await self._transport.connect()

    async def disconnect(self) -> None:
        self._closing = True
        await self._transport.disconnect()

    def _on_transport_lost(self, exc: BaseException | None) -> None:
        for pending in self._pending.values():
            if not pending.response.done():
                pending.response.set_exception(ConnectionError("Connection lost"))
        self._pending.clear()
        for confirm in self._pending_confirms.values():
            if not confirm.done():
                confirm.set_exception(ConnectionError("Connection lost"))
        self._pending_confirms.clear()
        self._awaiting_aps_ack.clear()
        if not self._closing:
            self._on_disconnect(exc)

    def _next_id(self) -> int:
        request_id = self._request_id
        self._request_id = (self._request_id % 0xFFFF) + 1
        return request_id

    # -- request surface -----------------------------------------------------------

    async def request(self, request: p.Request) -> p.Response | None:
        """Send a request; return its response, or None if the OK reply is empty."""
        request_id = self._next_id()
        pending = _Pending(streaming=False)
        self._pending[request_id] = pending
        try:
            await self._transport.send_frame(p.encode_request(request, request_id))
            body = await pending.response
        finally:
            self._pending.pop(request_id, None)

        if request.response is None:
            return None
        return request.response.deserialize(body)[0]

    async def request_confirmed(self, send: p.SendAps) -> None:
        """Send and await the terminal confirmation."""

        # The terminal confirmation is the end-to-end APS ack for an ack-requested
        # unicast, otherwise the local handoff. A rejected frame raises `DeliveryError`
        # before any confirm; a failed confirmation raises it too.
        request_id = self._next_id()
        pending = _Pending(streaming=False)
        self._pending[request_id] = pending
        confirm: asyncio.Future[p.SendConfirm | p.ApsAckConfirm] = (
            asyncio.get_running_loop().create_future()
        )
        self._pending_confirms[request_id] = confirm
        if send.aps_ack:
            self._awaiting_aps_ack.add(request_id)

        try:
            async with asyncio.timeout(CONFIRM_TIMEOUT):
                await self._transport.send_frame(p.encode_request(send, request_id))
                await pending.response  # accepted / rejected
                result = await confirm
        finally:
            self._pending.pop(request_id, None)
            self._pending_confirms.pop(request_id, None)
            self._awaiting_aps_ack.discard(request_id)

        if isinstance(result, p.SendConfirm) and not result.confirmed:
            raise DeliveryError(result.reason_text)
        if isinstance(result, p.ApsAckConfirm) and not result.acked:
            raise DeliveryError(result.reason_text)

    async def request_stream(
        self, request: p.Request
    ) -> AsyncGenerator[p.Response, None]:
        """Yield each streamed `request.event` item until the terminal response."""
        # An error response or disconnect is raised once the stream is exhausted.
        assert request.event is not None
        request_id = self._next_id()
        pending = _Pending(streaming=True)
        self._pending[request_id] = pending
        assert pending.events is not None

        await self._transport.send_frame(p.encode_request(request, request_id))
        try:
            while (item := await pending.events.get()) is not None:
                yield request.event.deserialize(item)[0]
            await pending.response  # surface an error
        finally:
            self._pending.pop(request_id, None)

    # -- inbound frame handling ----------------------------------------------------

    def _handle_frame(self, frame: bytes) -> None:
        header, body = p.FrameHeader.deserialize(frame)
        request_id = header.request_id

        if header.frame_type == p.FrameType.RESPONSE:
            pending = self._pending.get(request_id)
            if pending is None or pending.response.done():
                return
            status = p.Status(body[0])
            if status != p.Status.OK:
                err = p.Error.deserialize(body)[0]
                pending.response.set_exception(
                    p.ProtocolError(status, err.message_text)
                )
            else:
                pending.response.set_result(body[1:])
            if pending.events is not None:
                pending.events.put_nowait(None)
        elif header.frame_type == p.FrameType.EVENT:
            pending = self._pending.get(request_id)
            if pending is not None and pending.events is not None:
                pending.events.put_nowait(body)
        elif header.frame_type == p.FrameType.NOTIFICATION:
            try:
                command = p.CommandId(header.command)
            except ValueError:
                _LOGGER.debug("Unknown notification command %#x", header.command)
                return
            self._handle_notification(command, request_id, body)

    def _handle_notification(
        self, command: p.CommandId, request_id: int, body: bytes
    ) -> None:
        if command == p.CommandId.SEND_CONFIRM:
            confirm = self._pending_confirms.get(request_id)
            if confirm is None or confirm.done():
                return
            payload = p.SendConfirm.deserialize(body)[0]
            # A confirmed handoff is not terminal for an ack-requested send.
            if payload.confirmed and request_id in self._awaiting_aps_ack:
                return
            self._awaiting_aps_ack.discard(request_id)
            confirm.set_result(payload)
        elif command == p.CommandId.APS_ACK_CONFIRM:
            self._awaiting_aps_ack.discard(request_id)
            confirm = self._pending_confirms.get(request_id)
            if confirm is not None and not confirm.done():
                confirm.set_result(p.ApsAckConfirm.deserialize(body)[0])
        elif command == p.CommandId.HELLO:
            _LOGGER.debug("Ziggurat stack started")
        elif command == p.CommandId.LAST_RESET:
            reset = p.LastReset.deserialize(body)[0]
            logging.getLogger("ziggurat.fw").warning(
                "The firmware's previous reset was abnormal: %s", reset.message_text
            )
        elif command in p.NOTIFICATIONS:
            self._on_notification(p.NOTIFICATIONS[command].deserialize(body)[0])
        else:
            _LOGGER.debug("Unhandled notification %r", command)
