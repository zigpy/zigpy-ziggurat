"""Tests for `connect_transport`, which probes a WebSocket for its protocol, and for
the transports it returns."""

import asyncio
from collections.abc import AsyncIterator
import socket
from typing import Any

import aiohttp
from aiohttp import web
from aiohttp.test_utils import TestServer
from aiospinel import PropertyStatusError, Status
import pytest

from tests.common import (
    ClosingZiggurat,
    ProtocolErrorWebSocket,
    SyntheticSpinelRcp,
    SyntheticZiggurat,
    closing_server,
    protocol_error_server,
    server,
    spinel_rcp,
)
from zigpy_ziggurat.zigbee import protocol as p, transport as transport_module
from zigpy_ziggurat.zigbee.transport import (
    SpinelTransport,
    WebSocketTransport,
    connect_transport,
)


@pytest.fixture
def client_sessions(monkeypatch: pytest.MonkeyPatch) -> list[aiohttp.ClientSession]:
    """Record every `ClientSession` the WebSocket transport opens."""
    sessions: list[aiohttp.ClientSession] = []
    client_session = aiohttp.ClientSession

    def recording_session(*args: Any, **kwargs: Any) -> aiohttp.ClientSession:
        session = client_session(*args, **kwargs)
        sessions.append(session)
        return session

    monkeypatch.setattr(aiohttp, "ClientSession", recording_session)
    return sessions


class SilentZiggurat:
    """A WebSocket server that accepts the upgrade but never sends a hello."""

    def __init__(self) -> None:
        self.web_app = web.Application()
        self.web_app.router.add_get("/", self._handle_connection)
        self.url = ""
        self.disconnected = asyncio.Event()

    async def _handle_connection(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        try:
            async for _ in ws:
                pass
        finally:
            # The server may cancel the handler once the client drops the socket.
            self.disconnected.set()
        return ws


@pytest.fixture
async def silent_server() -> AsyncIterator[SilentZiggurat]:
    ziggurat = SilentZiggurat()
    test_server = TestServer(ziggurat.web_app)
    await test_server.start_server()
    ziggurat.url = f"ws://localhost:{test_server.port}/"

    yield ziggurat

    await test_server.close()


async def _wait_for(frames: list[bytes], count: int = 1) -> None:
    async with asyncio.timeout(2):
        while len(frames) < count:
            await asyncio.sleep(0.01)


async def test_probe_selects_binary(server: SyntheticZiggurat) -> None:
    frames: list[bytes] = []
    transport = await connect_transport(server.url, frames.append, lambda exc: None)
    try:
        assert isinstance(transport, WebSocketTransport)
        await transport.send_frame(p.encode_request(p.Shutdown(), 1))
        await _wait_for(frames)
    finally:
        await transport.disconnect()

    # The opening hello is consumed by the probe, so the only frame is the response.
    assert len(frames) == 1
    header, body = p.Header.deserialize(frames[0])
    assert header.frame_type == p.FrameType.RESPONSE
    assert header.command == p.RequestCommand.SHUTDOWN
    assert body == bytes([p.Status.OK])
    assert isinstance(server.requests[0], p.Shutdown)


async def test_probe_rejects_unexpected_handshake(
    closing_server: ClosingZiggurat,
    client_sessions: list[aiohttp.ClientSession],
) -> None:
    with pytest.raises(ConnectionError):
        await connect_transport(
            closing_server.url, lambda frame: None, lambda exc: None
        )

    assert len(client_sessions) == 1
    assert client_sessions[0].closed


async def test_probe_handshake_timeout_closes_session(
    silent_server: SilentZiggurat,
    client_sessions: list[aiohttp.ClientSession],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(transport_module, "HANDSHAKE_TIMEOUT", 0.1)

    with pytest.raises(TimeoutError):
        await connect_transport(silent_server.url, lambda frame: None, lambda exc: None)

    assert len(client_sessions) == 1
    assert client_sessions[0].closed

    # The socket itself is gone too, not just the session object.
    async with asyncio.timeout(1):
        await silent_server.disconnected.wait()


async def test_probe_connect_refused_closes_session(
    client_sessions: list[aiohttp.ClientSession],
) -> None:
    # A bound socket that never listens refuses connections, and holding it keeps
    # anything else from taking the port meanwhile.
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]

        with pytest.raises(aiohttp.ClientConnectionError):
            await connect_transport(
                f"ws://127.0.0.1:{port}/", lambda frame: None, lambda exc: None
            )

    assert len(client_sessions) == 1
    assert client_sessions[0].closed


async def test_spinel_transport_roundtrip(spinel_rcp: SyntheticSpinelRcp) -> None:
    frames: list[bytes] = []
    lost: list[BaseException | None] = []
    transport = await connect_transport(spinel_rcp.url, frames.append, lost.append)
    assert isinstance(transport, SpinelTransport)

    await transport.send_frame(b"\x01\x02\x03")
    await _wait_for(spinel_rcp.tunnel_writes)
    assert spinel_rcp.tunnel_writes == [b"\x01\x02\x03"]

    await spinel_rcp.push_stream_frame(b"\xaa\xbb")
    await _wait_for(frames)
    assert frames == [b"\xaa\xbb"]

    await transport.disconnect()
    # A clean close surfaces as connection_lost with no error.
    assert lost == [None]


async def test_spinel_stream_frame_handler_error(
    spinel_rcp: SyntheticSpinelRcp,
) -> None:
    attempts: list[bytes] = []

    def boom(frame: bytes) -> None:
        attempts.append(frame)
        raise RuntimeError("handler blew up")

    transport = await connect_transport(spinel_rcp.url, boom, lambda exc: None)
    try:
        # The receive loop must survive a handler raising on a delivered frame.
        await spinel_rcp.push_stream_frame(b"\x01")
        await _wait_for(attempts)
        assert attempts == [b"\x01"]
    finally:
        await transport.disconnect()


async def test_spinel_connect_rejects_foreign_firmware() -> None:
    rcp = SyntheticSpinelRcp(get_status=Status.PROP_NOT_FOUND)
    await rcp.start()
    try:
        with pytest.raises(ConnectionError, match="does not embed"):
            await connect_transport(rcp.url, lambda frame: None, lambda exc: None)

        # The rejected port is closed again, so the next attempt can open it.
        assert rcp.connections == 1
        await rcp.wait_closed_by_client()
    finally:
        await rcp.stop()


async def test_spinel_tunnel_write_rejected() -> None:
    rcp = SyntheticSpinelRcp(set_status=Status.INVALID_ARGUMENT)
    await rcp.start()
    transport = await connect_transport(rcp.url, lambda frame: None, lambda exc: None)
    try:
        with pytest.raises(PropertyStatusError):
            await transport.send_frame(b"\x01")
    finally:
        await transport.disconnect()
        await rcp.stop()


async def test_websocket_frame_handler_error(server: SyntheticZiggurat) -> None:
    attempts: list[bytes] = []

    def boom(frame: bytes) -> None:
        attempts.append(frame)
        raise RuntimeError("handler blew up")

    transport = await connect_transport(server.url, boom, lambda exc: None)
    try:
        # The receive loop must survive a handler raising on a delivered frame.
        await transport.send_frame(p.encode_request(p.Shutdown(), 1))
        await _wait_for(attempts)
        await transport.send_frame(p.encode_request(p.Shutdown(), 2))
        await _wait_for(attempts, count=2)
    finally:
        await transport.disconnect()


async def test_websocket_send_after_disconnect(server: SyntheticZiggurat) -> None:
    transport = await connect_transport(
        server.url, lambda frame: None, lambda exc: None
    )
    await transport.disconnect()
    with pytest.raises(ConnectionError, match="Not connected"):
        await transport.send_frame(p.encode_request(p.Shutdown(), 1))


async def test_websocket_receive_loop_error(
    protocol_error_server: ProtocolErrorWebSocket,
) -> None:
    lost: list[BaseException | None] = []
    lost_event = asyncio.Event()

    def on_lost(exc: BaseException | None) -> None:
        lost.append(exc)
        lost_event.set()

    transport = await connect_transport(
        protocol_error_server.url, lambda frame: None, on_lost
    )
    try:
        async with asyncio.timeout(2):
            await lost_event.wait()
        # The malformed frame ends the receive loop, reporting the loss once.
        assert len(lost) == 1
    finally:
        await transport.disconnect()
