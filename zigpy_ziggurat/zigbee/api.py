"""The transport-agnostic Ziggurat API, in terms of the binary `protocol` structs."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator, Callable
from datetime import timedelta
import logging

from zigpy.exceptions import DeliveryError
import zigpy.types as t

from zigpy_ziggurat.zigbee import protocol as p
from zigpy_ziggurat.zigbee.transport import Transport, connect_transport

_LOGGER = logging.getLogger(__name__)

# The end-to-end APS ack (or local handoff) must arrive within this window.
CONFIRM_TIMEOUT = 30


class _Pending:
    """One in-flight request: the request itself, its response, and a stream queue."""

    def __init__(self, request: p.Request, *, streaming: bool) -> None:
        loop = asyncio.get_running_loop()
        self.request = request
        self.response: asyncio.Future[p.Response | None] = loop.create_future()
        self.events: asyncio.Queue[p.Response | None] | None = (
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
        self._url = url
        self._baudrate = baudrate
        self._flow_control = flow_control
        self._on_notification = on_notification
        self._on_disconnect = on_disconnect
        self._closing = False
        self._request_id = 1
        self._pending: dict[int, _Pending] = {}
        self._pending_confirms: dict[
            int, asyncio.Future[p.SendConfirm | p.ApsAckConfirm | p.BroadcastConfirm]
        ] = {}
        self._awaiting_aps_ack: set[int] = set()
        self._transport: Transport | None = None

    async def connect(self) -> None:
        self._transport = await connect_transport(
            self._url,
            self._handle_frame,
            self._on_transport_lost,
            baudrate=self._baudrate,
            flow_control=self._flow_control,
        )

    async def disconnect(self) -> None:
        self._closing = True
        if self._transport is not None:
            await self._transport.disconnect()

    def _on_transport_lost(self, exc: BaseException | None) -> None:
        self._connection_lost(exc)

    def _connection_lost(self, exc: BaseException | None) -> None:
        for pending in self._pending.values():
            if not pending.response.done():
                pending.response.set_exception(ConnectionError("Connection lost"))
        self._pending.clear()
        for confirm in self._pending_confirms.values():
            if not confirm.done():
                confirm.set_exception(ConnectionError("Connection lost"))
        self._pending_confirms.clear()
        self._awaiting_aps_ack.clear()
        # Report the loss once. `_closing` also suppresses it during our own teardown.
        if not self._closing:
            self._closing = True
            self._on_disconnect(exc)

    def _next_id(self) -> int:
        # Request ids are 14 bits on the wire; 0 is left to unsolicited notifications.
        request_id = self._request_id
        self._request_id = (self._request_id % 0x3FFF) + 1
        return request_id

    async def _cancel_send(self, request_id: int) -> None:
        """Best-effort cancel of an in-flight send by its id."""
        if self._transport is None or self._closing:
            return
        frame = p.encode_request(
            p.CancelRequest(request_id=t.uint16_t(request_id)),
            self._next_id(),
        )
        await asyncio.shield(self._transport.send_frame(frame))

    # -- request surface -----------------------------------------------------------

    async def request(self, request: p.Request) -> p.Response | None:
        """Send a request; return its response, or None if the OK reply is empty."""
        request_id = self._next_id()
        pending = _Pending(request, streaming=False)
        self._pending[request_id] = pending

        _LOGGER.debug("Sending request (id=%d): %r", request_id, request)

        assert self._transport is not None
        try:
            await self._transport.send_frame(p.encode_request(request, request_id))
            return await pending.response
        finally:
            self._pending.pop(request_id, None)

    async def request_confirmed(
        self, send: p.SendUnicast | p.SendBroadcast | p.SendGroupcast
    ) -> None:
        """Send and await the terminal confirmation."""

        # The terminal confirmation is the end-to-end APS ack for an ack-requested
        # unicast, the passive-ack quorum for a broadcast/groupcast, otherwise the local
        # handoff. A rejected frame raises `DeliveryError` before any confirm; a failed
        # confirmation raises it too.
        request_id = self._next_id()
        pending = _Pending(send, streaming=False)
        self._pending[request_id] = pending
        confirm: asyncio.Future[
            p.SendConfirm | p.ApsAckConfirm | p.BroadcastConfirm
        ] = asyncio.get_running_loop().create_future()
        self._pending_confirms[request_id] = confirm
        if isinstance(send, p.SendUnicast) and send.aps_ack:
            self._awaiting_aps_ack.add(request_id)

        _LOGGER.debug("Sending request with confirmation (id=%d): %r", request_id, send)

        assert self._transport is not None
        try:
            async with asyncio.timeout(CONFIRM_TIMEOUT):
                await self._transport.send_frame(p.encode_request(send, request_id))
                await pending.response  # accepted / rejected
                result = await confirm
        finally:
            self._pending.pop(request_id, None)
            self._pending_confirms.pop(request_id, None)
            self._awaiting_aps_ack.discard(request_id)

            if not confirm.done() or confirm.cancelled():
                await self._cancel_send(request_id)

        if result.status != p.SendStatus.SUCCESS:
            raise DeliveryError(f"Send failed: {result.status.name}")

    async def request_stream(
        self, request: p.Request
    ) -> AsyncGenerator[p.Response, None]:
        """Yield each streamed `request.event` item until the terminal response."""
        # An error response or disconnect is raised once the stream is exhausted.
        assert request.event is not None
        request_id = self._next_id()
        pending = _Pending(request, streaming=True)
        self._pending[request_id] = pending
        assert pending.events is not None

        _LOGGER.debug("Sending stream request (id=%d): %r", request_id, request)

        assert self._transport is not None
        await self._transport.send_frame(p.encode_request(request, request_id))
        try:
            while (item := await pending.events.get()) is not None:
                yield item
            await pending.response  # surface an error
        finally:
            self._pending.pop(request_id, None)

    # -- inbound frame handling ----------------------------------------------------

    def _handle_frame(self, frame: bytes) -> None:
        header, body = p.Header.deserialize(frame)
        request_id = header.request_id

        if header.frame_type == p.FrameType.RESPONSE:
            self._handle_response(request_id, body)
        elif header.frame_type == p.FrameType.EVENT:
            self._handle_event(request_id, body)
        elif header.frame_type == p.FrameType.NOTIFICATION:
            command = p.NotificationCommand(header.command)
            if command not in p.NOTIFICATIONS:
                _LOGGER.debug("Unhandled notification %r", command)
                return
            notification = p.NOTIFICATIONS[command].deserialize(body)[0]
            _LOGGER.debug("Received notification (id=%d): %r", request_id, notification)
            self._handle_notification(request_id, notification)

    def _handle_response(self, request_id: int, body: bytes) -> None:
        pending = self._pending.get(request_id)
        if pending is None or pending.response.done():
            return

        status = p.Status(body[0])
        if status == p.Status.RATE_LIMITED:
            rate_limited = p.RateLimitedPayload.deserialize(body)[0]
            retry_in = timedelta(milliseconds=rate_limited.retry_in_ms)
            _LOGGER.debug(
                "Received rate-limited response (id=%d): retry in %s",
                request_id,
                retry_in,
            )
            pending.response.set_exception(p.RateLimitedError(retry_in))
        elif status != p.Status.OK:
            _LOGGER.debug("Received error response (id=%d): %r", request_id, status)
            pending.response.set_exception(p.ProtocolError(status))
        else:
            response = (
                pending.request.response.deserialize(body[1:])[0]
                if pending.request.response is not None
                else None
            )
            _LOGGER.debug("Received response (id=%d): %r", request_id, response)
            pending.response.set_result(response)

        if pending.events is not None:
            pending.events.put_nowait(None)

    def _handle_event(self, request_id: int, body: bytes) -> None:
        pending = self._pending.get(request_id)
        if pending is None or pending.events is None:
            return
        assert pending.request.event is not None
        event = pending.request.event.deserialize(body)[0]
        _LOGGER.debug("Received event (id=%d): %r", request_id, event)
        pending.events.put_nowait(event)

    def _handle_notification(
        self, request_id: int, notification: p.Notification
    ) -> None:
        if isinstance(notification, p.SendConfirm):
            confirm = self._pending_confirms.get(request_id)
            if confirm is None or confirm.done():
                return
            # A confirmed handoff is not terminal for an ack-requested send.
            if (
                notification.status == p.SendStatus.SUCCESS
                and request_id in self._awaiting_aps_ack
            ):
                return
            self._awaiting_aps_ack.discard(request_id)
            confirm.set_result(notification)
        elif isinstance(notification, (p.ApsAckConfirm, p.BroadcastConfirm)):
            self._awaiting_aps_ack.discard(request_id)
            confirm = self._pending_confirms.get(request_id)
            if confirm is not None and not confirm.done():
                confirm.set_result(notification)
        elif isinstance(notification, p.Hello):
            # The firmware only sends `hello` when it reboots
            self._connection_lost(ConnectionError("Ziggurat firmware reset"))
        elif isinstance(notification, p.LastReset):
            logging.getLogger("ziggurat.fw").warning(
                "The firmware's previous reset was abnormal: %s",
                notification.message,
            )
        else:
            self._on_notification(notification)

    async def set_tunable(self, name: str, value: int | timedelta) -> None:
        """Set a stack tunable by its Rust field name (a debug/experiment surface)."""
        await self.request(p.SetTunable.build(name, value))
