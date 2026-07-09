"""Tests for the `ZigguratApi` request/response layer against a fake transport."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import TypeVar, cast

import pytest
from zigpy.exceptions import DeliveryError
import zigpy.types as t

from zigpy_ziggurat.zigbee import api as api_module, protocol as p
from zigpy_ziggurat.zigbee.api import ZigguratApi

_Bytes = t.LVList[t.uint8_t, t.uint16_t]
RequestT = TypeVar("RequestT", bound=p.Request)
Handler = Callable[[p.Request, int], Awaitable[None]]


def _send_aps(*, aps_ack: bool) -> p.SendAps:
    return p.SendAps.build(
        delivery_mode=p.DeliveryMode.UNICAST,
        destination=t.NWK(0x1234),
        destination_eui64=None,
        aps_ack=aps_ack,
        aps_encryption=False,
        profile_id=0x0104,
        cluster_id=0x0006,
        src_ep=1,
        dst_ep=1,
        aps_seq=55,
        radius=30,
        priority=0,
        asdu=b"\x01\x02",
    )


class SyntheticBinaryTransport:
    """A fake firmware: parses request frames, records them, and replies with the
    binary frames its per-command handlers produce."""

    def __init__(self) -> None:
        self._on_frame: Callable[[bytes], None] = lambda frame: None
        self._on_lost: Callable[[BaseException | None], None] = lambda exc: None
        self.requests: list[p.Request] = []
        self.hw_ieee = t.EUI64.convert("11:22:33:44:55:66:77:88")
        self.handlers: dict[p.CommandId, Handler] = {
            p.CommandId.PING: self._empty_ok,
            p.CommandId.RESET: self._empty_ok,
            p.CommandId.SHUTDOWN: self._empty_ok,
            p.CommandId.PERMIT_JOINS: self._empty_ok,
            p.CommandId.GET_HW_ADDRESS: self._hw_address,
            p.CommandId.SEND_APS: self._send_aps,
            p.CommandId.ENERGY_SCAN: self._energy_scan,
        }

    async def factory(
        self,
        url: str,
        on_frame: Callable[[bytes], None],
        on_lost: Callable[[BaseException | None], None],
        *,
        baudrate: int = 115200,
        flow_control: str | None = None,
    ) -> "SyntheticBinaryTransport":
        self._on_frame = on_frame
        self._on_lost = on_lost
        return self

    async def disconnect(self) -> None:
        pass

    async def send_frame(self, frame: bytes) -> None:
        command = p.CommandId(frame[0])
        request_id = int.from_bytes(frame[1:3], "little")
        request = p.REQUESTS[command].deserialize(frame[3:])[0]
        self.requests.append(request)
        await self.handlers[command](request, request_id)

    def sent(self, request_type: type[RequestT]) -> list[RequestT]:
        return [r for r in self.requests if isinstance(r, request_type)]

    # -- frame injection -----------------------------------------------------------

    def ok(
        self, command: p.CommandId, request_id: int, payload: p.Response | None = None
    ) -> None:
        body = bytes([p.Status.OK]) + (payload.serialize() if payload else b"")
        self._on_frame(p.encode_reply(p.FrameType.RESPONSE, command, request_id, body))

    def error(
        self, command: p.CommandId, request_id: int, status: p.Status, message: str = ""
    ) -> None:
        body = p.Error(
            status=status, message=t.LongCharacterString(message)
        ).serialize()
        self._on_frame(p.encode_reply(p.FrameType.RESPONSE, command, request_id, body))

    def event(self, command: p.CommandId, request_id: int, payload: p.Response) -> None:
        self._on_frame(
            p.encode_reply(p.FrameType.EVENT, command, request_id, payload.serialize())
        )

    def notify(
        self, command: p.CommandId, request_id: int, payload: p.Notification
    ) -> None:
        self._on_frame(
            p.encode_reply(
                p.FrameType.NOTIFICATION, command, request_id, payload.serialize()
            )
        )

    def send_confirm(
        self, request_id: int, *, confirmed: bool = True, reason: str = ""
    ) -> None:
        self.notify(
            p.CommandId.SEND_CONFIRM,
            request_id,
            p.SendConfirm(
                confirmed=t.Bool(confirmed),
                next_hop=t.NWK(0xFFFF),
                reason=t.LongCharacterString(reason),
            ),
        )

    def aps_ack_confirm(
        self, request_id: int, *, acked: bool = True, reason: str = ""
    ) -> None:
        self.notify(
            p.CommandId.APS_ACK_CONFIRM,
            request_id,
            p.ApsAckConfirm(acked=t.Bool(acked), reason=t.LongCharacterString(reason)),
        )

    def lose(self, exc: BaseException | None = None) -> None:
        self._on_lost(exc)

    def raw(self, frame: bytes) -> None:
        self._on_frame(frame)

    # -- default handlers ----------------------------------------------------------

    async def _empty_ok(self, request: p.Request, request_id: int) -> None:
        self.ok(request.command, request_id)

    async def _hw_address(self, request: p.Request, request_id: int) -> None:
        self.ok(request.command, request_id, p.HwAddress(ieee=self.hw_ieee))

    async def _send_aps(self, request: p.Request, request_id: int) -> None:
        self.ok(request.command, request_id)
        self.send_confirm(request_id)
        if request.aps_ack:  # type: ignore[attr-defined]
            self.aps_ack_confirm(request_id)

    async def _energy_scan(self, request: p.Request, request_id: int) -> None:
        for channel in request.channels:  # type: ignore[attr-defined]
            self.event(
                p.CommandId.ENERGY_SCAN,
                request_id,
                p.EnergyResult(channel=t.uint8_t(channel), rssi=t.int8s(-85)),
            )
        self.ok(request.command, request_id)


class RecordingApi(ZigguratApi):
    """A `ZigguratApi` whose callbacks record into plain lists."""

    def __init__(self, url: str) -> None:
        self.notifications: list[p.Notification] = []
        self.disconnects: list[BaseException | None] = []
        super().__init__(url, self.notifications.append, self.disconnects.append)


@pytest.fixture
def transport(monkeypatch: pytest.MonkeyPatch) -> SyntheticBinaryTransport:
    server = SyntheticBinaryTransport()
    monkeypatch.setattr(api_module, "connect_transport", server.factory)
    return server


@pytest.fixture
async def api(transport: SyntheticBinaryTransport) -> AsyncIterator[RecordingApi]:
    instance = RecordingApi("binary://test")
    await instance.connect()

    yield instance

    await instance.disconnect()


async def test_request(api: RecordingApi, transport: SyntheticBinaryTransport) -> None:
    # An empty OK reply returns None
    assert await api.request(p.Ping()) is None

    hw = await api.request(p.GetHwAddress())
    assert isinstance(hw, p.HwAddress)
    assert hw.ieee == transport.hw_ieee


async def test_shutdown(api: RecordingApi, transport: SyntheticBinaryTransport) -> None:
    assert await api.request(p.Shutdown()) is None
    assert isinstance(transport.sent(p.Shutdown)[-1], p.Shutdown)


async def test_error_response(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def fail(request: p.Request, request_id: int) -> None:
        transport.error(
            request.command, request_id, p.Status.RADIO_ERROR, "it burned down"
        )

    transport.handlers[p.CommandId.PING] = fail

    with pytest.raises(DeliveryError, match="radio_error: it burned down"):
        await api.request(p.Ping())


async def test_request_confirmed(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    """An APS-ack send resolves once the end-to-end APS ack arrives."""
    await api.request_confirmed(_send_aps(aps_ack=True))
    assert transport.sent(p.SendAps)[-1].aps_seq == 55


async def test_request_confirmed_next_hop(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    """A no-ack unicast resolves on the local handoff."""
    await api.request_confirmed(_send_aps(aps_ack=False))


async def test_request_confirmed_rejected(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    """The stack rejects the frame, so the send raises before any confirm."""

    async def reject(request: p.Request, request_id: int) -> None:
        transport.error(
            request.command, request_id, p.Status.TRANSMIT_FAILED, "channel busy"
        )

    transport.handlers[p.CommandId.SEND_APS] = reject

    with pytest.raises(DeliveryError, match="transmit_failed"):
        await api.request_confirmed(_send_aps(aps_ack=True))


async def test_request_confirmed_failure(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    """The frame is handed off but the end-to-end APS ack never arrives."""

    async def ack_timeout(request: p.Request, request_id: int) -> None:
        transport.ok(request.command, request_id)
        transport.send_confirm(request_id, confirmed=True)
        transport.aps_ack_confirm(request_id, acked=False, reason="APS ack timed out")

    transport.handlers[p.CommandId.SEND_APS] = ack_timeout

    with pytest.raises(DeliveryError, match="APS ack timed out"):
        await api.request_confirmed(_send_aps(aps_ack=True))


async def test_request_stream(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    results: list[p.EnergyResult] = []
    async for item in api.request_stream(
        p.EnergyScan(channels=_Bytes([15, 20]), duration_per_channel_ms=t.uint16_t(100))
    ):
        results.append(cast(p.EnergyResult, item))
    assert [(r.channel, r.rssi) for r in results] == [(15, -85), (20, -85)]


async def test_notifications(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    transport.notify(
        p.CommandId.RECEIVED_APS,
        0,
        p.ReceivedAps(
            source=t.NWK(0xAB12),
            destination=t.NWK(0x0000),
            has_group=t.Bool(False),
            group=t.uint16_t(0),
            profile_id=t.uint16_t(0x0104),
            cluster_id=t.uint16_t(0x0006),
            src_ep=t.uint8_t(1),
            dst_ep=t.uint8_t(1),
            lqi=t.uint8_t(255),
            rssi=t.int8s(-40),
            data=t.LongOctetString(b"\x01\x02"),
        ),
    )
    transport.notify(
        p.CommandId.FRAME_COUNTER, 0, p.FrameCounter(frame_counter=t.uint32_t(1000))
    )
    transport.notify(
        p.CommandId.DEVICE_JOINED,
        0,
        p.DeviceJoined(
            nwk=t.NWK(0xAB12),
            ieee=t.EUI64.convert("aa:aa:aa:aa:aa:aa:aa:aa"),
            parent=t.NWK(0x0000),
        ),
    )

    assert [type(n) for n in api.notifications] == [
        p.ReceivedAps,
        p.FrameCounter,
        p.DeviceJoined,
    ]
    received = api.notifications[0]
    assert isinstance(received, p.ReceivedAps)
    assert received.data == b"\x01\x02"


async def test_unsolicited_frames_are_ignored(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    # A response and an event for an unknown request id
    transport.ok(p.CommandId.PING, 9999)
    transport.event(
        p.CommandId.ENERGY_SCAN,
        9999,
        p.EnergyResult(channel=t.uint8_t(1), rssi=t.int8s(-10)),
    )
    # A frame with an unknown command byte
    transport._on_frame(bytes([p.FrameType.NOTIFICATION, 0xEE, 0x00, 0x00]))

    # The connection survives all of it
    assert await api.request(p.Ping()) is None


async def test_connection_lost_fails_pending_requests(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def withhold(request: p.Request, request_id: int) -> None:
        return None

    transport.handlers[p.CommandId.PING] = withhold

    request = asyncio.ensure_future(api.request(p.Ping()))
    await asyncio.sleep(0)
    transport.lose(None)

    with pytest.raises(ConnectionError):
        await request

    assert api.disconnects == [None]


async def test_hello_reported_as_disconnect(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def withhold(request: p.Request, request_id: int) -> None:
        return None

    transport.handlers[p.CommandId.PING] = withhold
    request = asyncio.ensure_future(api.request(p.Ping()))
    await asyncio.sleep(0)

    # A firmware reboot (`hello`) wipes the stack, so it must surface as a disconnect
    # that fails in-flight requests, not as an ordinary notification.
    transport.notify(
        p.CommandId.HELLO,
        0,
        p.Hello(protocol_version=t.uint8_t(1), configured=t.Bool(False)),
    )

    with pytest.raises(ConnectionError):
        await request

    assert len(api.disconnects) == 1
    assert isinstance(api.disconnects[0], ConnectionError)
    assert api.notifications == []


async def test_timed_out_request_failed_late(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def withhold(request: p.Request, request_id: int) -> None:
        return None

    transport.handlers[p.CommandId.PING] = withhold

    # The caller gave up before any response arrived (zigpy wraps requests in
    # timeouts); disconnecting must tolerate the abandoned, cancelled future
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(api.request(p.Ping()), 0.05)

    await api.disconnect()
    await asyncio.sleep(0)


async def test_confirmed_send_delivery_failure(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def failed_confirm(request: p.Request, request_id: int) -> None:
        transport.ok(request.command, request_id)
        transport.send_confirm(request_id, confirmed=False, reason="no route")

    transport.handlers[p.CommandId.SEND_APS] = failed_confirm

    with pytest.raises(DeliveryError, match="no route"):
        await api.request_confirmed(_send_aps(aps_ack=False))


async def test_connection_lost_fails_pending_confirm(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def accept_only(request: p.Request, request_id: int) -> None:
        # Accept the send but never confirm, leaving a pending confirmation.
        transport.ok(request.command, request_id)

    transport.handlers[p.CommandId.SEND_APS] = accept_only

    request = asyncio.ensure_future(api.request_confirmed(_send_aps(aps_ack=False)))
    while not transport.sent(p.SendAps):
        await asyncio.sleep(0)
    transport.lose(None)

    with pytest.raises(ConnectionError):
        await request


async def test_unknown_notification_command_ignored(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    frame = p.FrameHeader(
        frame_type=p.FrameType.NOTIFICATION,
        command=t.uint8_t(0x06),
        request_id=t.uint16_t(0),
    ).serialize()
    transport.raw(frame)
    assert api.notifications == []


async def test_send_confirm_without_pending_ignored(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    # A confirm for a request we aren't tracking is dropped, not misrouted.
    transport.send_confirm(9999)
    assert api.notifications == []


async def test_last_reset_logged(
    api: RecordingApi,
    transport: SyntheticBinaryTransport,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level("WARNING", logger="ziggurat.fw"):
        transport.notify(
            p.CommandId.LAST_RESET,
            0,
            p.LastReset(message=t.LongCharacterString("brownout")),
        )
    assert "brownout" in caplog.text
    assert api.notifications == []
