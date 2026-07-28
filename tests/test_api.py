"""Tests for the `ZigguratApi` request/response layer against a fake transport."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from datetime import timedelta
from typing import TypeVar, cast

import pytest
from zigpy.exceptions import DeliveryError
import zigpy.types as t

from zigpy_ziggurat.zigbee import api as api_module, protocol as p
from zigpy_ziggurat.zigbee.api import ZigguratApi

_Bytes = t.LVList[t.uint8_t, t.uint16_t]
RequestT = TypeVar("RequestT", bound=p.Request)
Handler = Callable[[p.Request, int], Awaitable[None]]


def _send_aps(*, aps_ack: bool) -> p.SendUnicast:
    return p.SendUnicast.build(
        destination=t.NWK(0x1234),
        destination_eui64=None,
        aps_ack=aps_ack,
        aps_encryption=False,
        sleepy_destination=False,
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
        self.request_ids: list[int] = []
        self.hw_ieee = t.EUI64.convert("11:22:33:44:55:66:77:88")
        self.handlers: dict[p.RequestCommand, Handler] = {
            p.RequestCommand.RESET: self._empty_ok,
            p.RequestCommand.SHUTDOWN: self._empty_ok,
            p.RequestCommand.PERMIT_JOINS: self._empty_ok,
            p.RequestCommand.SET_TUNABLE: self._empty_ok,
            p.RequestCommand.GET_HW_ADDRESS: self._hw_address,
            p.RequestCommand.SEND_UNICAST: self._send_aps,
            p.RequestCommand.ENERGY_SCAN: self._energy_scan,
            p.RequestCommand.CANCEL_REQUEST: self._cancel_request,
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
        header, body = p.Header.deserialize(frame)
        assert header.frame_type == p.FrameType.REQUEST
        command = p.RequestCommand(header.command)
        request_id = int(header.request_id)
        request = p.REQUESTS[command].deserialize(body)[0]
        self.requests.append(request)
        self.request_ids.append(request_id)
        await self.handlers[command](request, request_id)

    def sent(self, request_type: type[RequestT]) -> list[RequestT]:
        return [r for r in self.requests if isinstance(r, request_type)]

    # -- frame injection -----------------------------------------------------------

    def ok(
        self,
        command: p.RequestCommand,
        request_id: int,
        payload: p.Response | None = None,
    ) -> None:
        body = bytes([p.Status.OK]) + (payload.serialize() if payload else b"")
        self._on_frame(p.encode_reply(p.FrameType.RESPONSE, command, request_id, body))

    def error(
        self, command: p.RequestCommand, request_id: int, status: p.Status
    ) -> None:
        body = bytes([status])
        self._on_frame(p.encode_reply(p.FrameType.RESPONSE, command, request_id, body))

    def rate_limited(
        self, command: p.RequestCommand, request_id: int, retry_in_ms: int
    ) -> None:
        body = p.RateLimitedPayload(
            status=p.Status.RATE_LIMITED, retry_in_ms=t.uint32_t(retry_in_ms)
        ).serialize()
        self._on_frame(p.encode_reply(p.FrameType.RESPONSE, command, request_id, body))

    def event(
        self, command: p.RequestCommand, request_id: int, payload: p.Response
    ) -> None:
        self._on_frame(
            p.encode_reply(p.FrameType.EVENT, command, request_id, payload.serialize())
        )

    def notify(
        self, command: p.NotificationCommand, request_id: int, payload: p.Notification
    ) -> None:
        self._on_frame(
            p.encode_reply(
                p.FrameType.NOTIFICATION, command, request_id, payload.serialize()
            )
        )

    def send_confirm(
        self, request_id: int, *, status: p.SendStatus = p.SendStatus.SUCCESS
    ) -> None:
        self.notify(
            p.NotificationCommand.SEND_CONFIRM,
            request_id,
            p.SendConfirm(status=status),
        )

    def aps_ack_confirm(
        self, request_id: int, *, status: p.SendStatus = p.SendStatus.SUCCESS
    ) -> None:
        self.notify(
            p.NotificationCommand.APS_ACK_CONFIRM,
            request_id,
            p.ApsAckConfirm(status=status),
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

    async def _cancel_request(self, request: p.Request, request_id: int) -> None:
        self.ok(request.command, request_id, p.CancelResult(cancelled=t.Bool(True)))

    async def _energy_scan(self, request: p.Request, request_id: int) -> None:
        for channel in request.channels:  # type: ignore[attr-defined]
            self.event(
                p.RequestCommand.ENERGY_SCAN,
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
    assert await api.request(p.Shutdown()) is None

    hw = await api.request(p.GetHwAddress())
    assert isinstance(hw, p.HwAddress)
    assert hw.ieee == transport.hw_ieee


async def test_shutdown(api: RecordingApi, transport: SyntheticBinaryTransport) -> None:
    assert await api.request(p.Shutdown()) is None
    assert isinstance(transport.sent(p.Shutdown)[-1], p.Shutdown)


async def test_set_tunable(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    await api.set_tunable("aps_ack_timeout", timedelta(seconds=5))

    sent = transport.sent(p.SetTunable)[-1]
    assert sent.name == b"aps_ack_timeout"
    assert sent.value == 5_000_000


async def test_error_response(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def fail(request: p.Request, request_id: int) -> None:
        transport.error(request.command, request_id, p.Status.RADIO_ERROR)

    transport.handlers[p.RequestCommand.SHUTDOWN] = fail

    with pytest.raises(DeliveryError, match="radio_error") as exc:
        await api.request(p.Shutdown())

    assert isinstance(exc.value, p.ProtocolError)
    assert exc.value.status == p.Status.RADIO_ERROR


async def test_rate_limited_response(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def rate_limit(request: p.Request, request_id: int) -> None:
        transport.rate_limited(request.command, request_id, retry_in_ms=1800)

    transport.handlers[p.RequestCommand.SEND_UNICAST] = rate_limit

    with pytest.raises(p.RateLimitedError, match="rate_limited: retry in 1.8s") as exc:
        await api.request_confirmed(_send_aps(aps_ack=False))

    assert exc.value.status == p.Status.RATE_LIMITED
    assert exc.value.retry_in == timedelta(milliseconds=1800)


async def test_request_confirmed(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    """An APS-ack send resolves once the end-to-end APS ack arrives."""
    await api.request_confirmed(_send_aps(aps_ack=True))
    assert transport.sent(p.SendUnicast)[-1].aps_seq == 55


async def test_cancel_on_abandon(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    """Cancelling a send awaiting confirmation cancels it on the firmware."""
    send_ids: list[int] = []

    async def accept_only(request: p.Request, request_id: int) -> None:
        send_ids.append(request_id)
        transport.ok(request.command, request_id)  # accepted, never confirmed

    transport.handlers[p.RequestCommand.SEND_UNICAST] = accept_only

    task = asyncio.create_task(api.request_confirmed(_send_aps(aps_ack=False)))
    while not transport.sent(p.SendUnicast):
        await asyncio.sleep(0)

    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(10):
        await asyncio.sleep(0)

    cancels = transport.sent(p.CancelRequest)
    assert len(cancels) == 1
    assert cancels[0].request_id == send_ids[0]


async def test_cancel_on_timeout(
    api: RecordingApi,
    transport: SyntheticBinaryTransport,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A confirmation timeout cancels the still-in-flight send on the firmware."""
    monkeypatch.setattr(api_module, "CONFIRM_TIMEOUT", 0.01)
    send_ids: list[int] = []

    async def accept_only(request: p.Request, request_id: int) -> None:
        send_ids.append(request_id)
        transport.ok(request.command, request_id)  # accepted, never confirmed

    transport.handlers[p.RequestCommand.SEND_UNICAST] = accept_only

    with pytest.raises(TimeoutError):
        await api.request_confirmed(_send_aps(aps_ack=False))

    for _ in range(10):
        await asyncio.sleep(0)

    cancels = transport.sent(p.CancelRequest)
    assert len(cancels) == 1
    assert cancels[0].request_id == send_ids[0]


async def test_cancel_skipped_while_closing(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    """Once we are tearing the connection down there is nothing left to cancel on."""
    send_ids: list[int] = []

    async def accept_only(request: p.Request, request_id: int) -> None:
        send_ids.append(request_id)
        transport.ok(request.command, request_id)  # accepted, never confirmed

    transport.handlers[p.RequestCommand.SEND_UNICAST] = accept_only

    task = asyncio.create_task(api.request_confirmed(_send_aps(aps_ack=False)))
    while not transport.sent(p.SendUnicast):
        await asyncio.sleep(0)

    await api.disconnect()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    for _ in range(10):
        await asyncio.sleep(0)

    assert send_ids
    assert transport.sent(p.CancelRequest) == []


async def test_confirmed_success_sends_no_cancel(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    """A send that confirms normally is never cancelled."""
    await api.request_confirmed(_send_aps(aps_ack=False))
    for _ in range(10):
        await asyncio.sleep(0)
    assert transport.sent(p.CancelRequest) == []


async def test_request_confirmed_rejected(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    """The stack rejects the frame, so the send raises before any confirm."""

    async def reject(request: p.Request, request_id: int) -> None:
        transport.error(request.command, request_id, p.Status.PAYLOAD_TOO_LONG)

    transport.handlers[p.RequestCommand.SEND_UNICAST] = reject

    with pytest.raises(DeliveryError, match="payload_too_long"):
        await api.request_confirmed(_send_aps(aps_ack=True))


async def test_request_confirmed_failure(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    """The frame is handed off but the end-to-end APS ack never arrives."""

    async def ack_timeout(request: p.Request, request_id: int) -> None:
        transport.ok(request.command, request_id)
        transport.send_confirm(request_id, status=p.SendStatus.SUCCESS)
        transport.aps_ack_confirm(request_id, status=p.SendStatus.APS_ACK_TIMEOUT)

    transport.handlers[p.RequestCommand.SEND_UNICAST] = ack_timeout

    with pytest.raises(DeliveryError, match="APS_ACK_TIMEOUT"):
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
        p.NotificationCommand.RECEIVED_APS,
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
        p.NotificationCommand.FRAME_COUNTER,
        0,
        p.FrameCounter(frame_counter=t.uint32_t(1000)),
    )
    transport.notify(
        p.NotificationCommand.DEVICE_JOINED,
        0,
        p.DeviceJoined(
            nwk=t.NWK(0xAB12),
            ieee=t.EUI64.convert("aa:aa:aa:aa:aa:aa:aa:aa"),
            parent=t.NWK(0x0000),
            rx_on_when_idle=t.uint1_t(1),
            device_type=p.ChildDeviceType.END_DEVICE,
            reserved=t.uint5_t(0),
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
    transport.ok(p.RequestCommand.SHUTDOWN, 9999)
    transport.event(
        p.RequestCommand.ENERGY_SCAN,
        9999,
        p.EnergyResult(channel=t.uint8_t(1), rssi=t.int8s(-10)),
    )
    # A frame with an unknown command byte
    transport.raw(
        p.Header(
            command=t.uint8_t(0xEE),
            frame_type=p.FrameType.NOTIFICATION,
            request_id=t.uint16_t(0),
        ).serialize()
    )

    # The connection survives all of it
    assert await api.request(p.Shutdown()) is None


async def test_connection_lost_fails_pending_requests(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def withhold(request: p.Request, request_id: int) -> None:
        return None

    transport.handlers[p.RequestCommand.SHUTDOWN] = withhold

    request = asyncio.ensure_future(api.request(p.Shutdown()))
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

    transport.handlers[p.RequestCommand.SHUTDOWN] = withhold
    request = asyncio.ensure_future(api.request(p.Shutdown()))
    await asyncio.sleep(0)

    # A firmware reboot (`hello`) wipes the stack, so it must surface as a disconnect
    # that fails in-flight requests, not as an ordinary notification.
    transport.notify(
        p.NotificationCommand.HELLO,
        0,
        p.Hello(
            protocol_version=t.uint8_t(p.PROTOCOL_VERSION), configured=t.Bool(False)
        ),
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

    transport.handlers[p.RequestCommand.SHUTDOWN] = withhold

    # The caller gave up before any response arrived (zigpy wraps requests in
    # timeouts); disconnecting must tolerate the abandoned, cancelled future
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(api.request(p.Shutdown()), 0.05)

    await api.disconnect()
    await asyncio.sleep(0)


async def test_confirmed_send_delivery_failure(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def failed_confirm(request: p.Request, request_id: int) -> None:
        transport.ok(request.command, request_id)
        transport.send_confirm(request_id, status=p.SendStatus.ROUTE_DISCOVERY_TIMEOUT)

    transport.handlers[p.RequestCommand.SEND_UNICAST] = failed_confirm

    with pytest.raises(DeliveryError, match="ROUTE_DISCOVERY_TIMEOUT"):
        await api.request_confirmed(_send_aps(aps_ack=False))


async def test_connection_lost_fails_pending_confirm(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    async def accept_only(request: p.Request, request_id: int) -> None:
        # Accept the send but never confirm, leaving a pending confirmation.
        transport.ok(request.command, request_id)

    transport.handlers[p.RequestCommand.SEND_UNICAST] = accept_only

    request = asyncio.ensure_future(api.request_confirmed(_send_aps(aps_ack=False)))
    while not transport.sent(p.SendUnicast):
        await asyncio.sleep(0)
    transport.lose(None)

    with pytest.raises(ConnectionError):
        await request


async def test_unknown_notification_command_ignored(
    api: RecordingApi, transport: SyntheticBinaryTransport
) -> None:
    frame = p.Header(
        command=t.uint8_t(0x06),
        frame_type=p.FrameType.NOTIFICATION,
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
            p.NotificationCommand.LAST_RESET,
            0,
            p.LastReset(message=t.LongCharacterString("brownout")),
        )
    assert "brownout" in caplog.text
    assert api.notifications == []
