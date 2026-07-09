"""Tests for `connect_transport`, which probes a WebSocket for its protocol."""

import asyncio

import pytest

from tests.common import (
    ClosingZiggurat,
    SyntheticBinaryZiggurat,
    SyntheticZiggurat,
    binary_server,
    closing_server,
    server,
)
from zigpy_ziggurat.zigbee import protocol as p
from zigpy_ziggurat.zigbee.transport import (
    LegacyWebSocketTransport,
    WebSocketTransport,
    connect_transport,
)


async def _wait_for(frames: list[bytes], count: int = 1) -> None:
    async with asyncio.timeout(2):
        while len(frames) < count:
            await asyncio.sleep(0.01)


async def test_probe_selects_binary(binary_server: SyntheticBinaryZiggurat) -> None:
    frames: list[bytes] = []
    transport = await connect_transport(
        binary_server.url, frames.append, lambda exc: None
    )
    try:
        assert isinstance(transport, WebSocketTransport)
        await transport.send_frame(p.encode_request(p.Ping(), 1))
        await _wait_for(frames)
    finally:
        await transport.disconnect()

    # The opening hello is consumed by the probe, so the only frame is the response.
    assert len(frames) == 1
    header, body = p.FrameHeader.deserialize(frames[0])
    assert header.frame_type == p.FrameType.RESPONSE
    assert header.command == p.CommandId.PING
    assert body == bytes([p.Status.OK])
    assert isinstance(binary_server.requests[0], p.Ping)


async def test_probe_selects_legacy(server: SyntheticZiggurat) -> None:
    transport = await connect_transport(
        server.url, lambda frame: None, lambda exc: None
    )
    try:
        assert isinstance(transport, LegacyWebSocketTransport)
    finally:
        await transport.disconnect()


async def test_probe_rejects_unexpected_handshake(
    closing_server: ClosingZiggurat,
) -> None:
    with pytest.raises(ConnectionError):
        await connect_transport(
            closing_server.url, lambda frame: None, lambda exc: None
        )
