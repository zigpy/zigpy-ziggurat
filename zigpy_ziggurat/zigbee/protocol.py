"""The binary Ziggurat control protocol."""

from __future__ import annotations

from datetime import timedelta
from typing import ClassVar

from zigpy.exceptions import DeliveryError
import zigpy.types as t

from zigpy_ziggurat.zigbee import wire

# Re-export the generated wire types for the rest of zigpy-ziggurat to use.
from zigpy_ziggurat.zigbee.wire import (
    PROTOCOL_VERSION as PROTOCOL_VERSION,
    ChildDeviceType as ChildDeviceType,
    DeliveryMode as DeliveryMode,
    FrameType as FrameType,
    Header as Header,
    KeyId as KeyId,
    LeaveReason as LeaveReason,
    NetworkState as NetworkState,
    NodeRole as NodeRole,
    NotificationCommand as NotificationCommand,
    RateLimitedPayload as RateLimitedPayload,
    RequestCommand as RequestCommand,
    RouteControl as RouteControl,
    SendStatus as SendStatus,
    Status as Status,
    TclkFlavorId as TclkFlavorId,
)


class Response(t.Struct):
    """A device -> host reply payload: a response or a streamed scan/event item."""


class Notification(t.Struct):
    """An unsolicited device -> host frame."""


class Request(t.Struct):
    """A host -> device request."""

    command: ClassVar[RequestCommand]
    # The OK-body response type; None when the OK reply is empty.
    response: ClassVar[type[Response] | None] = None
    # The streamed item type for a scan/stream request; None for plain request/response.
    event: ClassVar[type[Response] | None] = None


# -- table entries (streamed by scans, loaded by the load requests) --------------


class KeyEntry(Response, wire.KeyEntry):
    pass


class ChildEntry(Response, wire.ChildEntry):
    pass


class AddressEntry(Response, wire.AddressEntry):
    pass


class RouteEntry(Response, wire.RouteEntry):
    pass


class SourceRouteEntry(Response, wire.SourceRouteEntry):
    pass


# -- responses / streamed events -------------------------------------------------


class FirmwareInfo(Response, wire.FirmwareInfoPayload):
    pass


class HwAddress(Response, wire.HwAddressPayload):
    pass


class NetworkInfo(Response, wire.NetworkInfoPayload):
    pass


class ScanCount(Response, wire.ScanCountPayload):
    pass


class CancelResult(Response, wire.CancelResultPayload):
    # Whether a still-cancellable (pre-delivery) send was found and removed.
    pass


class EnergyResult(Response, wire.EnergyResultPayload):
    pass


class Beacon(Response, wire.BeaconPayload):
    @property
    def source_or_none(self) -> t.NWK | None:
        return self.source if self.source != t.NWK(0xFFFF) else None


class CapturedPacket(Response, wire.CapturedPacketPayload):
    pass


# -- requests --------------------------------------------------------------------


class Reset(Request, wire.ResetPayload):
    command = RequestCommand.RESET


class Shutdown(Request):
    command = RequestCommand.SHUTDOWN


class GetFirmwareInfo(Request):
    command = RequestCommand.GET_FIRMWARE_INFO
    response = FirmwareInfo


class GetHwAddress(Request):
    command = RequestCommand.GET_HW_ADDRESS
    response = HwAddress


class Configure(Request, wire.ConfigurePayload):
    command = RequestCommand.CONFIGURE


class LoadKeyTable(Request, wire.LoadKeyTablePayload):
    command = RequestCommand.LOAD_KEY_TABLE


class LoadChildren(Request, wire.LoadChildrenPayload):
    command = RequestCommand.LOAD_CHILDREN


class LoadAddressCache(Request, wire.LoadAddressCachePayload):
    command = RequestCommand.LOAD_ADDRESS_CACHE


class LoadRouteTable(Request, wire.LoadRouteTablePayload):
    command = RequestCommand.LOAD_ROUTE_TABLE


class LoadSourceRoutes(Request, wire.LoadSourceRoutesPayload):
    command = RequestCommand.LOAD_SOURCE_ROUTES


class StartNetwork(Request):
    command = RequestCommand.START_NETWORK


class GetNetworkInfo(Request):
    command = RequestCommand.GET_NETWORK_INFO
    response = NetworkInfo


class ScanKeyTable(Request):
    command = RequestCommand.SCAN_KEY_TABLE
    response = ScanCount
    event = KeyEntry


class ScanChildren(Request):
    command = RequestCommand.SCAN_CHILDREN
    response = ScanCount
    event = ChildEntry


class ScanAddressCache(Request):
    command = RequestCommand.SCAN_ADDRESS_CACHE
    response = ScanCount
    event = AddressEntry


class ScanRouteTable(Request):
    command = RequestCommand.SCAN_ROUTE_TABLE
    response = ScanCount
    event = RouteEntry


class SendUnicast(Request, wire.SendUnicastPayload):
    command = RequestCommand.SEND_UNICAST

    @classmethod
    def build(
        cls,
        *,
        destination: t.NWK | None,
        destination_eui64: t.EUI64 | None,
        aps_ack: bool,
        aps_encryption: bool,
        sleepy_destination: bool,
        profile_id: int,
        cluster_id: int,
        src_ep: int,
        dst_ep: int,
        aps_seq: int,
        radius: int,
        priority: int,
        asdu: bytes,
        route_control: RouteControl = RouteControl.STACK_DECIDES,
        next_hop: t.NWK | None = None,
        relays: list[t.NWK] | None = None,
    ) -> SendUnicast:
        return cls(
            has_eui64=t.uint1_t(destination_eui64 is not None),
            aps_ack=t.uint1_t(aps_ack),
            aps_encryption=t.uint1_t(aps_encryption),
            sleepy_destination=t.uint1_t(sleepy_destination),
            reserved=t.uint4_t(0),
            # 0xFFFE stands in for "no short address"; the firmware resolves the EUI64.
            destination=destination if destination is not None else t.NWK(0xFFFE),
            destination_eui64=destination_eui64 or t.EUI64([0] * 8),
            profile_id=t.uint16_t(profile_id),
            cluster_id=t.uint16_t(cluster_id),
            src_ep=t.uint8_t(src_ep),
            dst_ep=t.uint8_t(dst_ep),
            aps_seq=t.uint8_t(aps_seq),
            radius=t.uint8_t(radius),
            priority=t.int8s(priority),
            route=route_control,
            next_hop=next_hop,
            relays=(
                wire.SourceRouteRelays(relays=t.LVList[t.NWK, t.uint8_t](relays))
                if relays is not None
                else None
            ),
            asdu=t.LongOctetString(asdu),
        )


class SendBroadcast(Request, wire.SendBroadcastPayload):
    command = RequestCommand.SEND_BROADCAST

    @classmethod
    def build(
        cls,
        *,
        destination: t.NWK,
        profile_id: int,
        cluster_id: int,
        src_ep: int,
        dst_ep: int,
        aps_seq: int,
        radius: int,
        priority: int,
        asdu: bytes,
    ) -> SendBroadcast:
        return cls(
            reserved=t.uint8_t(0),
            destination=destination,
            profile_id=t.uint16_t(profile_id),
            cluster_id=t.uint16_t(cluster_id),
            src_ep=t.uint8_t(src_ep),
            dst_ep=t.uint8_t(dst_ep),
            aps_seq=t.uint8_t(aps_seq),
            radius=t.uint8_t(radius),
            priority=t.int8s(priority),
            asdu=t.LongOctetString(asdu),
        )


class SendGroupcast(Request, wire.SendGroupcastPayload):
    command = RequestCommand.SEND_GROUPCAST

    @classmethod
    def build(
        cls,
        *,
        group_id: int,
        profile_id: int,
        cluster_id: int,
        src_ep: int,
        aps_seq: int,
        radius: int,
        priority: int,
        asdu: bytes,
    ) -> SendGroupcast:
        return cls(
            reserved=t.uint8_t(0),
            group_id=t.uint16_t(group_id),
            profile_id=t.uint16_t(profile_id),
            cluster_id=t.uint16_t(cluster_id),
            src_ep=t.uint8_t(src_ep),
            aps_seq=t.uint8_t(aps_seq),
            radius=t.uint8_t(radius),
            priority=t.int8s(priority),
            asdu=t.LongOctetString(asdu),
        )


class PermitJoins(Request, wire.PermitJoinsPayload):
    command = RequestCommand.PERMIT_JOINS


class SetChannel(Request, wire.ChannelPayload):
    command = RequestCommand.SET_CHANNEL


class SetNwkUpdateId(Request, wire.NwkUpdateIdPayload):
    command = RequestCommand.SET_NWK_UPDATE_ID


class SetProvisionalKey(Request, wire.ProvisionalKeyPayload):
    command = RequestCommand.SET_PROVISIONAL_KEY


class EnergyScan(Request, wire.ScanRequestPayload):
    command = RequestCommand.ENERGY_SCAN
    event = EnergyResult


class NetworkScan(Request, wire.ScanRequestPayload):
    command = RequestCommand.NETWORK_SCAN
    event = Beacon


class PacketCapture(Request, wire.ChannelPayload):
    command = RequestCommand.PACKET_CAPTURE
    event = CapturedPacket


class PacketCaptureChannel(Request, wire.ChannelPayload):
    command = RequestCommand.PACKET_CAPTURE_CHANNEL


# The tunable name is a Rust field name of the stack's `Tunables` struct (see the
# `tunables!` block in ziggurat-zigbee). The value is type-punned into a u64:
# integers as-is, bools as 0/1, durations in microseconds, enums as their
# discriminant; the firmware rejects unknown names and out-of-range values.
class SetTunable(Request, wire.SetTunablePayload):
    command = RequestCommand.SET_TUNABLE

    @classmethod
    def build(cls, name: str, value: int | timedelta) -> SetTunable:
        if isinstance(value, timedelta):
            value = value // timedelta(microseconds=1)
        return cls(name=t.LVBytes(name.encode("ascii")), value=t.uint64_t(value))


class CancelRequest(Request, wire.CancelRequestPayload):
    command = RequestCommand.CANCEL_REQUEST
    response = CancelResult


# -- notifications ---------------------------------------------------------------


class Hello(Notification, wire.HelloPayload):
    pass


class LastReset(Notification, wire.LastResetPayload):
    pass


class ReceivedAps(Notification, wire.ReceivedApsPayload):
    @property
    def group_id(self) -> int | None:
        return int(self.group) if self.has_group else None


class SendConfirm(Notification, wire.SendConfirmPayload):
    pass


class ApsAckConfirm(Notification, wire.ApsAckConfirmPayload):
    pass


class BroadcastConfirm(Notification, wire.BroadcastConfirmPayload):
    pass


class DeviceJoined(Notification, wire.DeviceJoinedPayload):
    pass


class DeviceLeft(Notification, wire.DeviceLeftPayload):
    @property
    def ieee_or_none(self) -> t.EUI64 | None:
        return self.ieee if self.has_ieee else None

    @property
    def rejoin_or_none(self) -> bool | None:
        # `rejoin` is only meaningful for a self-announced leave.
        return bool(self.rejoin) if self.reason == LeaveReason.ANNOUNCED else None

    @property
    def router_or_none(self) -> t.NWK | None:
        return self.router if self.reason == LeaveReason.ROUTER_REPORTED else None

    @property
    def router_ieee_or_none(self) -> t.EUI64 | None:
        if self.reason == LeaveReason.ROUTER_REPORTED and self.has_router_ieee:
            return self.router_ieee
        return None


class FrameCounter(Notification, wire.FrameCounterPayload):
    pass


class LinkKey(Notification, wire.LinkKeyPayload):
    pass


class ApsDecryptFailure(Notification, wire.ApsDecryptFailPayload):
    pass


class RouteRecord(Notification, wire.RouteRecordPayload):
    pass


class ApsFrameCounter(Notification, wire.ApsFrameCounterPayload):
    pass


# Notification id -> struct, for decoding unsolicited frames. `SendConfirm`,
# `ApsAckConfirm` and `BroadcastConfirm` are handled specially (they resolve a pending
# send by request id).
NOTIFICATIONS: dict[NotificationCommand, type[Notification]] = {
    NotificationCommand.HELLO: Hello,
    NotificationCommand.LAST_RESET: LastReset,
    NotificationCommand.RECEIVED_APS: ReceivedAps,
    NotificationCommand.SEND_CONFIRM: SendConfirm,
    NotificationCommand.APS_ACK_CONFIRM: ApsAckConfirm,
    NotificationCommand.BROADCAST_CONFIRM: BroadcastConfirm,
    NotificationCommand.DEVICE_JOINED: DeviceJoined,
    NotificationCommand.DEVICE_LEFT: DeviceLeft,
    NotificationCommand.FRAME_COUNTER: FrameCounter,
    NotificationCommand.LINK_KEY: LinkKey,
    NotificationCommand.APS_DECRYPT_FAILURE: ApsDecryptFailure,
    NotificationCommand.ROUTE_RECORD: RouteRecord,
    NotificationCommand.APS_FRAME_COUNTER: ApsFrameCounter,
}


def encode_request(request: Request, request_id: int) -> bytes:
    """Serialize a request frame (3-byte header, then the payload)."""
    header = Header(
        command=t.uint8_t(request.command),
        frame_type=FrameType.REQUEST,
        request_id=t.uint16_t(request_id),
    )
    return header.serialize() + request.serialize()


def encode_reply(
    frame_type: FrameType,
    command: RequestCommand | NotificationCommand,
    request_id: int,
    body: bytes = b"",
) -> bytes:
    """Serialize a device -> host frame (3-byte header, then the body)."""
    header = Header(
        command=t.uint8_t(command),
        frame_type=frame_type,
        request_id=t.uint16_t(request_id),
    )
    return header.serialize() + body


# Command id -> request type, for parsing an outbound frame back into a struct.
REQUESTS: dict[RequestCommand, type[Request]] = {
    cls.command: cls for cls in Request.__subclasses__()
}


class ProtocolError(DeliveryError):
    """A firmware error response (non-OK status)."""

    # `detail` is client-side context (e.g. the rate-limit retry delay); the wire
    # carries only the status code.
    def __init__(self, status: Status, detail: str = "") -> None:
        code = status.name.lower()
        super().__init__(f"{code}: {detail}" if detail else code)
        self.status = status


class RateLimitedError(ProtocolError):
    """A broadcast rejected by the firmware's rate limit, carrying when to retry."""

    def __init__(self, retry_in: timedelta) -> None:
        super().__init__(
            Status.RATE_LIMITED, f"retry in {retry_in.total_seconds():.1f}s"
        )
        self.retry_in = retry_in
