"""Frame transports for the binary protocol: serial (Spinel tunnel) and WebSocket."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import logging
from typing import Any, Protocol, cast

import aiohttp
import aiospinel
import zigpy.serial

_LOGGER = logging.getLogger(__name__)

WEBSOCKET_HEARTBEAT = 15

PROP_VENDOR_ZIGGURAT = aiospinel.PackedUInt21(0x3D5A)

# The callback the API installs to receive a device -> host binary frame.
OnFrame = Callable[[bytes], None]
OnLost = Callable[[BaseException | None], None]


# The server must announce itself with a hello within this window.
HANDSHAKE_TIMEOUT = 5


class Transport(Protocol):
    """Moves binary protocol frames between the API and a device, once connected."""

    async def disconnect(self) -> None: ...

    async def send_frame(self, frame: bytes) -> None: ...


async def connect_transport(
    url: str,
    on_frame: OnFrame,
    on_lost: OnLost,
    *,
    baudrate: int = 115200,
    flow_control: str | None = None,
) -> Transport:
    """Open a connected transport for `url`, probing a WebSocket for its protocol."""
    if not url.startswith(("ws://", "wss://", "ws+unix://")):
        spinel = SpinelTransport(
            url, on_frame, on_lost, baudrate=baudrate, flow_control=flow_control
        )
        await spinel.connect()
        return spinel

    return await _probe_websocket(url, on_frame, on_lost)


# -- serial (Spinel tunnel) ------------------------------------------------------


class _SpinelProtocol(aiospinel.SpinelProtocol):
    """Tunnels binary frames over the vendor Spinel stream property."""

    def __init__(self, on_frame: OnFrame, on_lost: OnLost) -> None:
        super().__init__()
        self._on_frame = on_frame
        self._on_lost = on_lost
        self.add_property_listener(PROP_VENDOR_ZIGGURAT, self._stream_frame_received)

    def connection_lost(self, exc: BaseException | None) -> None:
        super().connection_lost(exc)
        self._on_lost(exc)

    def _stream_frame_received(self, data: bytes) -> None:
        # Responses to our own property SETs also land here, with no payload.
        if len(data) < 2:
            return
        length = int.from_bytes(data[:2], "little")
        try:
            self._on_frame(data[2 : 2 + length])
        except Exception:
            _LOGGER.exception("Failed to handle frame: %r", data)

    async def start_ziggurat(self) -> None:
        rsp = await self.send_command(
            aiospinel.CommandID.PROP_VALUE_GET,
            PROP_VENDOR_ZIGGURAT.serialize(),
        )
        prop_id, _ = aiospinel.PackedUInt21.deserialize(rsp.data)
        if prop_id != PROP_VENDOR_ZIGGURAT:
            raise ConnectionError(
                f"Firmware does not embed the Ziggurat stack: {rsp!r}"
            )
        _LOGGER.debug("Embedded Ziggurat firmware detected")

    async def tunnel_send(self, frame: bytes) -> None:
        # No retries: a timed-out tunnel write must not resend the request (the first
        # copy may already have been processed).
        rsp = await self.send_command(
            aiospinel.CommandID.PROP_VALUE_SET,
            (
                PROP_VENDOR_ZIGGURAT.serialize()
                + len(frame).to_bytes(2, "little")
                + frame
            ),
            retries=0,
        )
        prop_id, _ = aiospinel.PackedUInt21.deserialize(rsp.data)
        if prop_id != PROP_VENDOR_ZIGGURAT:
            raise ConnectionError(f"Tunnel write rejected: {rsp!r}")


class SpinelTransport:
    """The binary protocol tunneled over a serial OpenThread RCP's Spinel stream."""

    def __init__(
        self,
        url: str,
        on_frame: OnFrame,
        on_lost: OnLost,
        *,
        baudrate: int = 115200,
        flow_control: str | None = None,
    ) -> None:
        self._url = url
        self._on_frame = on_frame
        self._on_lost = on_lost
        self._baudrate = baudrate
        self._flow_control = flow_control
        self._protocol: _SpinelProtocol | None = None

    async def connect(self) -> None:
        _, protocol = await zigpy.serial.create_serial_connection(
            loop=asyncio.get_running_loop(),
            protocol_factory=lambda: _SpinelProtocol(self._on_frame, self._on_lost),
            url=self._url,
            baudrate=self._baudrate,
            flow_control=cast(Any, self._flow_control),
        )
        self._protocol = cast(_SpinelProtocol, protocol)
        await self._protocol.wait_until_connected()
        await self._protocol.start_ziggurat()

    async def disconnect(self) -> None:
        if self._protocol is not None:
            self._protocol.close()
            await self._protocol.wait_until_closed()
            self._protocol = None

    async def send_frame(self, frame: bytes) -> None:
        assert self._protocol is not None
        await self._protocol.tunnel_send(frame)


# -- WebSocket -------------------------------------------------------------------


async def _open_websocket(
    url: str,
) -> tuple[aiohttp.ClientSession, aiohttp.ClientWebSocketResponse]:
    if url.startswith("ws+unix://"):
        # The URL's path is the socket path; the HTTP host is a placeholder.
        connector: aiohttp.BaseConnector | None = aiohttp.UnixConnector(
            path=url.removeprefix("ws+unix://")
        )
        ws_url = "ws://localhost/"
    else:
        connector = None
        ws_url = url

    session = aiohttp.ClientSession(connector=connector)
    websocket = await session.ws_connect(ws_url, heartbeat=WEBSOCKET_HEARTBEAT)
    return session, websocket


async def _probe_websocket(url: str, on_frame: OnFrame, on_lost: OnLost) -> Transport:
    """Open the WebSocket and check that the opening hello is a binary frame."""
    session, websocket = await _open_websocket(url)
    async with asyncio.timeout(HANDSHAKE_TIMEOUT):
        hello = await websocket.receive()

    if hello.type != aiohttp.WSMsgType.BINARY:
        await session.close()
        raise ConnectionError(f"Unexpected handshake from ziggurat: {hello!r}")

    transport = WebSocketTransport(on_frame, on_lost)
    transport._adopt(session, websocket)
    return transport


class WebSocketTransport:
    """Shared aiohttp WebSocket plumbing, driven from a socket passed to `_adopt`."""

    def __init__(self, on_frame: OnFrame, on_lost: OnLost) -> None:
        self._on_frame = on_frame
        self._on_lost = on_lost
        self._session: aiohttp.ClientSession | None = None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._receiver_task: asyncio.Task[None] | None = None

    def _adopt(
        self,
        session: aiohttp.ClientSession,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> None:
        self._session = session
        self._websocket = websocket
        self._receiver_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self) -> None:
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            self._receiver_task = None
        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _receive_loop(self) -> None:
        websocket = self._websocket
        assert websocket is not None
        exc: BaseException | None = None
        try:
            async for msg in websocket:
                if msg.type == aiohttp.WSMsgType.BINARY:
                    try:
                        self._on_frame(msg.data)
                    except Exception:
                        _LOGGER.exception("Failed to handle frame: %r", msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    exc = websocket.exception()
                    break
        except asyncio.CancelledError:
            # A deliberate disconnect; the API is already tearing down.
            self._on_lost(None)
            raise
        self._on_lost(exc)

    async def send_frame(self, frame: bytes) -> None:
        if self._websocket is None:
            raise ConnectionError("Not connected")
        await self._websocket.send_bytes(frame)
