"""Tests for the `ZigguratApi` request/response layer, against the synthetic server."""

import asyncio
from collections.abc import AsyncIterator

import pytest
from zigpy.exceptions import DeliveryError
import zigpy.types as t

from tests.common import RpcError, SyntheticZiggurat, server
from zigpy_ziggurat.zigbee import commands
from zigpy_ziggurat.zigbee.application import ZigguratApi

SEND_APS = commands.SendAps(
    delivery_mode="unicast",
    destination_eui64=None,
    destination=t.NWK(0x1234),
    profile_id=0x0104,
    cluster_id=0x0006,
    src_ep=1,
    dst_ep=1,
    aps_ack=True,
    aps_seq=55,
    radius=30,
    aps_encryption=False,
    priority=0,
    data=b"\x01\x02",
)


class RecordingApi(ZigguratApi):
    """A `ZigguratApi` whose callbacks record into plain lists."""

    def __init__(self, url: str) -> None:
        self.notifications: list[commands.Notification] = []
        self.disconnects: list[BaseException | None] = []
        super().__init__(url, self.notifications.append, self.disconnects.append)


@pytest.fixture
async def api(server: SyntheticZiggurat) -> AsyncIterator[RecordingApi]:
    instance = RecordingApi(server.url)
    await instance.connect()

    yield instance

    await instance.disconnect()


async def test_request(api: RecordingApi) -> None:
    status = await api.request(commands.Ping())
    assert status == commands.Status(status="pong")


async def test_error_response(api: RecordingApi, server: SyntheticZiggurat) -> None:
    async def fail(command: commands.Ping, request_id: int) -> commands.Status:
        raise RpcError("serial_port_error", "it burned down")

    server.handlers["ping"] = fail

    with pytest.raises(DeliveryError, match="serial_port_error: it burned down"):
        await api.request(commands.Ping())


async def test_request_transmitted(
    api: RecordingApi, server: SyntheticZiggurat
) -> None:
    await api.request_transmitted(SEND_APS)
    assert server.sent(commands.SendAps)[-1].aps_seq == 55


async def test_request_transmitted_failure_before_transmission(
    api: RecordingApi, server: SyntheticZiggurat
) -> None:
    async def fail(command: commands.SendAps, request_id: int) -> commands.Status:
        raise RpcError("transmit_failed", "channel busy")

    server.handlers["send_aps"] = fail

    with pytest.raises(DeliveryError, match="transmit_failed"):
        await api.request_transmitted(SEND_APS)


async def test_late_delivery_failure_is_logged(
    api: RecordingApi, server: SyntheticZiggurat, caplog: pytest.LogCaptureFixture
) -> None:
    async def ack_timeout(
        command: commands.SendAps, request_id: int
    ) -> commands.Status:
        await server.send_event(request_id, "transmitted")
        raise RpcError("aps_ack_timeout", "no ack")

    server.handlers["send_aps"] = ack_timeout

    # Resolves at the `transmitted` stage; the terminal failure arrives later and is
    # logged instead of raised
    await api.request_transmitted(SEND_APS)

    async with asyncio.timeout(1):
        while "Delivery failed after transmission" not in caplog.text:
            await asyncio.sleep(0.01)


async def test_unsolicited_messages_are_ignored(
    api: RecordingApi, server: SyntheticZiggurat, caplog: pytest.LogCaptureFixture
) -> None:
    await server.send_raw("not json")
    await server.send_raw('{"type": "response", "id": 9999, "result": {}}')
    await server.send_raw('{"type": "event", "id": 9999, "event": "transmitted"}')

    # A `transmitted` event for a request that did not ask for one
    async def eager(command: commands.Ping, request_id: int) -> commands.Status:
        await server.send_event(request_id, "transmitted")
        return commands.Status(status="pong")

    server.handlers["ping"] = eager

    # The connection survives all of it
    status = await api.request(commands.Ping())
    assert status == commands.Status(status="pong")
    assert "Failed to handle message" in caplog.text


async def test_notifications(api: RecordingApi, server: SyntheticZiggurat) -> None:
    sent: list[commands.Notification] = [
        commands.ReceivedApsCommand(
            source=t.NWK(0xAB12),
            destination=t.NWK(0x0000),
            group=None,
            profile_id=t.uint16_t(0x0104),
            cluster_id=t.uint16_t(0x0006),
            src_ep=t.uint8_t(1),
            dst_ep=t.uint8_t(1),
            lqi=t.uint8_t(255),
            rssi=t.int8s(-40),
            data=b"\x01\x02",
        ),
        commands.FrameCounterUpdate(frame_counter=t.uint32_t(1000)),
        commands.LinkKeyUpdate(
            ieee=t.EUI64.convert("aa:aa:aa:aa:aa:aa:aa:aa"),
            key=t.KeyData.convert("00:11:22:33:44:55:66:77:88:99:aa:bb:cc:dd:ee:ff"),
        ),
        commands.DeviceJoined(
            nwk=t.NWK(0xAB12),
            ieee=t.EUI64.convert("aa:aa:aa:aa:aa:aa:aa:aa"),
            parent=t.NWK(0x0000),
        ),
        commands.DeviceLeft(
            nwk=t.NWK(0xAB12),
            ieee=None,
            reason=commands.DeviceLeaveReason.ROUTER_REPORTED,
            router=t.NWK(0x0000),
            router_ieee=t.EUI64.convert("aa:aa:aa:aa:aa:aa:aa:aa"),
        ),
    ]

    for notification in sent:
        await server.send_notification(notification)

    async with asyncio.timeout(1):
        while len(api.notifications) < len(sent):
            await asyncio.sleep(0.01)

    assert api.notifications == sent


async def test_connection_lost_fails_pending_requests(
    api: RecordingApi, server: SyntheticZiggurat
) -> None:
    async def withhold(command: commands.Ping, request_id: int) -> None:
        return None

    server.handlers["ping"] = withhold

    request = asyncio.ensure_future(api.request(commands.Ping()))
    await server.wait_for(commands.Ping)
    await server.ws.close()

    with pytest.raises(ConnectionError):
        await request

    assert api.disconnects == [None]


async def test_protocol_error_disconnects(
    api: RecordingApi, server: SyntheticZiggurat
) -> None:
    # A malformed frame (reserved opcode) surfaces as a websocket protocol error
    server.transport.write(b"\x8f\x00")

    async with asyncio.timeout(1):
        while not api.disconnects:
            await asyncio.sleep(0.01)

    assert len(api.disconnects) == 1


async def test_timed_out_request_failed_late(
    api: RecordingApi, server: SyntheticZiggurat
) -> None:
    async def withhold(command: commands.Ping, request_id: int) -> None:
        return None

    server.handlers["ping"] = withhold

    # The caller gave up before any response arrived (zigpy wraps requests in
    # timeouts); disconnecting must tolerate the abandoned, cancelled future
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(api.request(commands.Ping()), 0.05)

    await api.disconnect()
    await asyncio.sleep(0)
