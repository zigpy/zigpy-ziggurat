"""Tests for the convenience accessors on the binary protocol structs."""

from datetime import timedelta

import zigpy.types as t

from zigpy_ziggurat.zigbee import protocol as p

_IEEE = t.EUI64.convert("00:11:22:33:44:55:66:77")


def test_send_confirm_next_hop_or_none() -> None:
    known = p.SendConfirm(
        confirmed=t.Bool(True), next_hop=t.NWK(0x1234), reason=t.LongCharacterString("")
    )
    assert known.next_hop_or_none == 0x1234
    unknown = p.SendConfirm(
        confirmed=t.Bool(True),
        next_hop=t.NWK(0xFFFF),
        reason=t.LongCharacterString(""),
    )
    assert unknown.next_hop_or_none is None


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
