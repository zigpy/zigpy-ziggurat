"""Tests for `connect_transport`, which probes a WebSocket for its protocol."""

import asyncio
import json
import logging

import aiospinel
import pytest
import zigpy.types as t

from tests.common import (
    COORDINATOR_IEEE,
    ClosingZiggurat,
    ProtocolErrorWebSocket,
    SyntheticBinaryZiggurat,
    SyntheticSpinelRcp,
    SyntheticZiggurat,
    binary_server,
    closing_server,
    protocol_error_server,
    server,
    spinel_rcp,
)
from zigpy_ziggurat.zigbee import legacy as commands, protocol as p
from zigpy_ziggurat.zigbee.transport import (
    LegacyWebSocketTransport,
    SpinelTransport,
    WebSocketTransport,
    connect_transport,
)


async def _legacy(
    server: SyntheticZiggurat,
) -> tuple[LegacyWebSocketTransport, list[bytes]]:
    frames: list[bytes] = []
    transport = await connect_transport(server.url, frames.append, lambda exc: None)
    assert isinstance(transport, LegacyWebSocketTransport)
    return transport, frames


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
    header, body = p.ReplyHeader.deserialize(frames[0])
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


async def test_websocket_send_after_disconnect(
    binary_server: SyntheticBinaryZiggurat,
) -> None:
    transport = await connect_transport(
        binary_server.url, lambda frame: None, lambda exc: None
    )
    await transport.disconnect()
    with pytest.raises(ConnectionError, match="Not connected"):
        await transport.send_frame(p.encode_request(p.Ping(), 1))


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


# -- legacy JSON transcoding -----------------------------------------------------


async def test_legacy_encodes_packet_capture(server: SyntheticZiggurat) -> None:
    server.handlers["packet_capture"] = server.on_status
    server.handlers["packet_capture_change_channel"] = server.on_status
    transport, _ = await _legacy(server)
    try:
        await transport.send_frame(
            p.encode_request(p.PacketCapture(channel=t.uint8_t(15)), 1)
        )
        await server.wait_for(commands.PacketCapture)
        await transport.send_frame(
            p.encode_request(p.PacketCaptureChannel(channel=t.uint8_t(20)), 2)
        )
        captured = await server.wait_for(commands.PacketCaptureChangeChannel)
        assert captured.channel == 20
    finally:
        await transport.disconnect()


async def test_legacy_rejects_untranscodable_command(
    server: SyntheticZiggurat,
) -> None:
    transport, _ = await _legacy(server)
    try:
        with pytest.raises(ValueError, match="Cannot transcode"):
            await transport.send_frame(p.encode_request(p.GetFirmwareInfo(), 1))
    finally:
        await transport.disconnect()


async def test_legacy_decodes_captured_packet(server: SyntheticZiggurat) -> None:
    transport, frames = await _legacy(server)
    try:
        # An unknown event is dropped; the captured packet is transcoded to an event.
        await server.send_event_data(7, "not_a_real_event", {})
        await server.send_event_data(
            7,
            "captured_packet",
            {"channel": 15, "rssi": -80, "lqi": 200, "data": "aabbcc"},
        )
        await _wait_for(frames)
        assert len(frames) == 1
        header, body = p.ReplyHeader.deserialize(frames[0])
        assert header.frame_type == p.FrameType.EVENT
        assert header.command == p.CommandId.PACKET_CAPTURE
        packet = p.CapturedPacket.deserialize(body)[0]
        assert bytes(packet.psdu) == b"\xaa\xbb\xcc"
    finally:
        await transport.disconnect()


async def test_legacy_forwards_firmware_log(
    server: SyntheticZiggurat, caplog: pytest.LogCaptureFixture
) -> None:
    transport, _ = await _legacy(server)
    try:
        with caplog.at_level(logging.WARNING, logger="ziggurat.fw.foo.bar"):
            await server.send_raw(
                json.dumps(
                    {
                        "type": "notification",
                        "event": "log",
                        "data": {
                            "level": "WARN",
                            "target": "foo::bar",
                            "message": "something happened",
                        },
                    }
                )
            )
            async with asyncio.timeout(2):
                while "something happened" not in caplog.text:
                    await asyncio.sleep(0.01)
    finally:
        await transport.disconnect()


async def test_legacy_transmitted_becomes_send_confirm(
    server: SyntheticZiggurat,
) -> None:
    transport, frames = await _legacy(server)
    try:
        # The real server signals a send handoff with a bare `transmitted` event
        # that carries no `data`; it must become a SEND_CONFIRM, not crash.
        await server.send_event(9, "transmitted")
        await _wait_for(frames)
        header, body = p.ReplyHeader.deserialize(frames[0])
        assert header.frame_type == p.FrameType.NOTIFICATION
        assert header.command == p.CommandId.SEND_CONFIRM
        assert header.request_id == 9
        assert p.SendConfirm.deserialize(body)[0].confirmed
    finally:
        await transport.disconnect()


async def test_legacy_decodes_send_confirm_next_hop(
    server: SyntheticZiggurat,
) -> None:
    transport, frames = await _legacy(server)
    try:
        await server.send_confirm(1, next_hop="0x1234")
        await _wait_for(frames)
        header, body = p.ReplyHeader.deserialize(frames[0])
        assert header.command == p.CommandId.SEND_CONFIRM
        confirm = p.SendConfirm.deserialize(body)[0]
        assert confirm.next_hop == 0x1234
    finally:
        await transport.disconnect()


async def test_legacy_decodes_decrypt_failure_known_key(
    server: SyntheticZiggurat,
) -> None:
    transport, frames = await _legacy(server)
    try:
        await server.send_notification(
            commands.ApsDecryptionFailure(
                source=t.NWK(0x1234),
                source_ieee=COORDINATOR_IEEE,
                frame_counter=t.uint32_t(42),
                key_id="network",
            )
        )
        await _wait_for(frames)
        header, body = p.ReplyHeader.deserialize(frames[0])
        assert header.command == p.CommandId.APS_DECRYPT_FAILURE
        failure = p.ApsDecryptFailure.deserialize(body)[0]
        assert failure.key_id == p.KeyId.NETWORK
    finally:
        await transport.disconnect()


async def test_legacy_ignores_binary_and_unknown_response(
    server: SyntheticZiggurat,
) -> None:
    transport, frames = await _legacy(server)
    try:
        # A binary frame and a response for an unknown id are both dropped; a
        # following confirm still transcodes, proving the loop kept going.
        await server.ws.send_bytes(b"\x00\x01\x02")
        await server.send_raw(
            json.dumps({"type": "response", "id": 9999, "result": {}})
        )
        await server.send_confirm(1)
        await _wait_for(frames)
        assert len(frames) == 1
        header, _ = p.ReplyHeader.deserialize(frames[0])
        assert header.command == p.CommandId.SEND_CONFIRM
    finally:
        await transport.disconnect()
