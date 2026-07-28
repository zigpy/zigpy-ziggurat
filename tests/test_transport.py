"""Tests for `connect_transport`, which probes a WebSocket for its protocol, and for
the transports it returns. The legacy JSON transcoding shim is covered separately in
`test_legacy.py`."""

import asyncio

import aiospinel
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
from zigpy_ziggurat.zigbee import protocol as p
from zigpy_ziggurat.zigbee.transport import (
    SpinelTransport,
    WebSocketTransport,
    connect_transport,
)


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
) -> None:
    with pytest.raises(ConnectionError):
        await connect_transport(
            closing_server.url, lambda frame: None, lambda exc: None
        )


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
    rcp = SyntheticSpinelRcp(get_prop_id=aiospinel.PackedUInt21(0x0001))
    await rcp.start()
    try:
        with pytest.raises(ConnectionError, match="does not embed"):
            await connect_transport(rcp.url, lambda frame: None, lambda exc: None)
    finally:
        await rcp.stop()


async def test_spinel_tunnel_write_rejected() -> None:
    rcp = SyntheticSpinelRcp(set_prop_id=aiospinel.PackedUInt21(0x0001))
    await rcp.start()
    transport = await connect_transport(rcp.url, lambda frame: None, lambda exc: None)
    try:
        with pytest.raises(ConnectionError, match="Tunnel write rejected"):
            await transport.send_frame(b"\x01")
    finally:
        await transport.disconnect()
        await rcp.stop()


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
