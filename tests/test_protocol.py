"""Tests for the convenience accessors on the binary protocol structs."""

from datetime import timedelta

import pytest
import zigpy.types as t

from zigpy_ziggurat.zigbee import protocol as p

_IEEE = t.EUI64.convert("00:11:22:33:44:55:66:77")


def _device_left(
    reason: p.LeaveReason,
    *,
    rejoin: int = 0,
    has_router_ieee: int = 0,
    router: int = 0xFFFF,
) -> p.DeviceLeft:
    return p.DeviceLeft(
        nwk=t.NWK(0x1234),
        has_ieee=t.uint1_t(1),
        rejoin=t.uint1_t(rejoin),
        has_router_ieee=t.uint1_t(has_router_ieee),
        reserved=t.uint5_t(0),
        ieee=_IEEE,
        reason=reason,
        router=t.NWK(router),
        router_ieee=_IEEE,
    )


def test_device_left_announced() -> None:
    left = _device_left(p.LeaveReason.ANNOUNCED, rejoin=1)
    assert left.rejoin_or_none is True
    assert left.router_or_none is None
    assert left.router_ieee_or_none is None


def test_device_left_router_reported() -> None:
    left = _device_left(p.LeaveReason.ROUTER_REPORTED, has_router_ieee=1, router=0x5678)
    assert left.rejoin_or_none is None
    assert left.router_or_none == 0x5678
    assert left.router_ieee_or_none == _IEEE


def test_device_left_router_reported_without_ieee() -> None:
    left = _device_left(p.LeaveReason.ROUTER_REPORTED, has_router_ieee=0)
    assert left.router_ieee_or_none is None


def test_set_tunable_build() -> None:
    integer = p.SetTunable.build("unicast_retries", 5)
    assert integer.name == b"unicast_retries"
    assert integer.value == 5

    duration = p.SetTunable.build("aps_ack_timeout", timedelta(milliseconds=1500))
    assert duration.value == 1_500_000

    flag = p.SetTunable.build("allow_unsecured_rejoins", True)
    assert flag.value == 1


@pytest.mark.parametrize(
    "notification",
    [
        p.RouteRecord(destination=t.NWK(0x1234), relays=[t.NWK(0x0002), t.NWK(0x0003)]),
        p.RouteRecord(destination=t.NWK(0x1234), relays=[]),
        p.ApsFrameCounter(frame_counter=t.uint32_t(123456)),
        p.DeviceJoined(
            nwk=t.NWK(0xAB12),
            ieee=_IEEE,
            parent=t.NWK(0x0000),
            rx_on_when_idle=t.uint1_t(1),
            device_type=p.ChildDeviceType.ROUTER,
            reserved=t.uint5_t(0),
        ),
    ],
)
def test_notification_round_trip(notification: p.Notification) -> None:
    parsed, rest = type(notification).deserialize(notification.serialize())
    assert rest == b""
    assert parsed == notification


@pytest.mark.parametrize(
    "request_obj",
    [
        p.LoadRouteTable(
            entries=t.LVList[p.RouteEntry, t.uint16_t](
                [
                    p.RouteEntry(
                        destination=t.NWK(0x1234),
                        next_hop=t.NWK(0x5678),
                        path_cost=t.uint8_t(0xFF),
                    )
                ]
            )
        ),
        p.LoadSourceRoutes(
            entries=t.LVList[p.SourceRouteEntry, t.uint16_t](
                [
                    p.SourceRouteEntry(
                        destination=t.NWK(0x1234),
                        relays=t.LVList[t.NWK, t.uint8_t](
                            [t.NWK(0x0002), t.NWK(0x0003)]
                        ),
                    ),
                    p.SourceRouteEntry(
                        destination=t.NWK(0xABCD),
                        relays=t.LVList[t.NWK, t.uint8_t]([]),
                    ),
                ]
            )
        ),
    ],
)
def test_load_request_round_trip(request_obj: p.Request) -> None:
    parsed, rest = type(request_obj).deserialize(request_obj.serialize())
    assert rest == b""
    assert parsed == request_obj
