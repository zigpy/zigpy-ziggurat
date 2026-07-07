"""The binary Ziggurat control protocol."""

from __future__ import annotations

from typing import ClassVar

from zigpy.exceptions import DeliveryError
import zigpy.types as t


# Device -> host only: host -> device frames are always requests and carry no
# frame type (see `encode_request`).
class FrameType(t.enum8):
    RESPONSE = 1
    EVENT = 2
    NOTIFICATION = 3


class Status(t.enum8):
    OK = 0
    PARSE = 1
    UNKNOWN_COMMAND = 2
    INVALID_STATE = 3
    NOT_CONFIGURED = 4
    RADIO_ERROR = 5
    NETWORK_START_FAILED = 6
    TRANSMIT_FAILED = 7
    SCAN_FAILED = 8
    INVALID_REQUEST = 9


class CommandId(t.enum8):
    # Notifications (device -> host, unsolicited)
    HELLO = 0x00
    # Requests (host -> device)
    PING = 0x01
    RESET = 0x02
    GET_FIRMWARE_INFO = 0x03
    GET_HW_ADDRESS = 0x04
    CONFIGURE = 0x10
    LOAD_KEY_TABLE = 0x11
    LOAD_CHILDREN = 0x12
    LOAD_ADDRESS_CACHE = 0x13
    START_NETWORK = 0x14
    GET_NETWORK_INFO = 0x18
    SCAN_KEY_TABLE = 0x19
    SCAN_CHILDREN = 0x1A
    SCAN_ADDRESS_CACHE = 0x1B
    SCAN_ROUTE_TABLE = 0x1C
    SEND_APS = 0x20
    PERMIT_JOINS = 0x21
    SET_CHANNEL = 0x22
    SET_NWK_UPDATE_ID = 0x23
    SET_PROVISIONAL_KEY = 0x24
    ENERGY_SCAN = 0x25
    NETWORK_SCAN = 0x26
    PACKET_CAPTURE = 0x27
    PACKET_CAPTURE_CHANNEL = 0x28
    # More notifications
    RECEIVED_APS = 0x30
    SEND_CONFIRM = 0x31
    APS_ACK_CONFIRM = 0x32
    DEVICE_JOINED = 0x33
    DEVICE_LEFT = 0x34
    FRAME_COUNTER = 0x35
    LINK_KEY = 0x36
    APS_DECRYPT_FAILURE = 0x37
    LAST_RESET = 0x38


class NodeRole(t.enum8):
    COORDINATOR = 0
    ROUTER = 1


class TclkFlavor(t.enum8):
    ZSTACK = 0
    EZSP = 1


class ChildDeviceType(t.enum2):
    UNKNOWN = 0
    ROUTER = 1
    END_DEVICE = 2


class KeyId(t.enum8):
    DATA = 0
    NETWORK = 1
    KEY_TRANSPORT = 2
    KEY_LOAD = 3


class LeaveReason(t.enum8):
    ANNOUNCED = 0
    ROUTER_REPORTED = 1
    KEEPALIVE_TIMEOUT = 2


class DeliveryMode(t.enum2):
    UNICAST = 0
    BROADCAST = 2
    MULTICAST = 3


class FrameHeader(t.Struct):
    """The 4-byte header of every device -> host frame."""

    frame_type: FrameType
    command: t.uint8_t  # raw CommandId byte (unknown ids still parse the header)
    request_id: t.uint16_t


class Response(t.Struct):
    """A device -> host reply payload: a response or a streamed scan/event item."""


class Notification(t.Struct):
    """An unsolicited device -> host frame."""


# -- shared sub-structures -------------------------------------------------------


class NetworkState(t.Struct):
    """The persistent network state shared by `Configure` and `NetworkInfo`."""

    channel: t.uint8_t
    nwk_update_id: t.uint8_t
    pan_id: t.PanId
    extended_pan_id: t.ExtendedPanId
    nwk_address: t.NWK
    ieee_address: t.EUI64
    network_key: t.KeyData
    network_key_seq: t.uint8_t
    network_key_tx_counter: t.uint32_t
    tc_link_key: t.KeyData
    has_tclk_seed: t.Bool
    tclk_seed: t.KeyData
    tclk_flavor: TclkFlavor
    tx_power: t.int8s
    aps_frame_counter: t.uint32_t


# -- table entries (streamed by scans, loaded by the load requests) --------------


class KeyEntry(Response):
    key: t.KeyData
    tx_counter: t.uint32_t
    rx_counter: t.uint32_t
    seq: t.uint8_t
    partner_ieee: t.EUI64


class ChildEntry(Response):
    ieee: t.EUI64
    nwk: t.NWK
    rx_on_when_idle: t.uint1_t
    device_type: ChildDeviceType
    reserved: t.uint5_t


class AddressEntry(Response):
    ieee: t.EUI64
    nwk: t.NWK


class RouteEntry(Response):
    destination: t.NWK
    next_hop: t.NWK
    path_cost: t.uint8_t


# -- responses / streamed events -------------------------------------------------


class FirmwareInfo(Response):
    protocol_version: t.uint8_t
    version: t.LVList[t.uint8_t, t.uint16_t]

    @property
    def version_text(self) -> str:
        return bytes(self.version).decode()


class HwAddress(Response):
    ieee: t.EUI64


class NetworkInfo(Response):
    state: NetworkState
    key_count: t.uint16_t
    started: t.Bool


class ScanCount(Response):
    count: t.uint16_t


class EnergyResult(Response):
    channel: t.uint8_t
    rssi: t.int8s


class Beacon(Response):
    channel: t.uint8_t
    source: t.NWK  # 0xFFFF when the beacon had no short source
    pan_id: t.PanId
    extended_pan_id: t.ExtendedPanId
    permit_joining: t.uint1_t
    router_capacity: t.uint1_t
    end_device_capacity: t.uint1_t
    reserved: t.uint5_t
    stack_profile: t.uint8_t
    protocol_version: t.uint8_t
    device_depth: t.uint8_t
    update_id: t.uint8_t
    lqi: t.uint8_t
    rssi: t.int8s

    @property
    def source_or_none(self) -> t.NWK | None:
        return self.source if self.source != t.NWK(0xFFFF) else None


class CapturedPacket(Response):
    channel: t.uint8_t
    rssi: t.int8s
    lqi: t.uint8_t
    psdu: t.LVList[t.uint8_t, t.uint16_t]

    @property
    def psdu_bytes(self) -> bytes:
        return bytes(self.psdu)


class Error(t.Struct):
    """The body of a failed response: a non-OK status and a diagnostic message."""

    status: Status
    message: t.LVList[t.uint8_t, t.uint16_t]

    @property
    def message_text(self) -> str:
        return bytes(self.message).decode()


# -- requests --------------------------------------------------------------------


class Request(t.Struct):
    """A host -> device request."""

    command: ClassVar[CommandId]
    # The OK-body response type; None when the OK reply is empty.
    response: ClassVar[type[Response] | None] = None
    # The streamed item type for a scan/stream request; None for plain request/response.
    event: ClassVar[type[Response] | None] = None


class Ping(Request):
    command = CommandId.PING


class Reset(Request):
    command = CommandId.RESET

    hard: t.Bool


class GetFirmwareInfo(Request):
    command = CommandId.GET_FIRMWARE_INFO
    response = FirmwareInfo


class GetHwAddress(Request):
    command = CommandId.GET_HW_ADDRESS
    response = HwAddress


class Configure(Request):
    command = CommandId.CONFIGURE

    role: NodeRole
    source_routing: t.Bool
    state: NetworkState


class LoadKeyTable(Request):
    command = CommandId.LOAD_KEY_TABLE

    entries: t.LVList[KeyEntry, t.uint16_t]


class LoadChildren(Request):
    command = CommandId.LOAD_CHILDREN

    entries: t.LVList[ChildEntry, t.uint16_t]


class LoadAddressCache(Request):
    command = CommandId.LOAD_ADDRESS_CACHE

    entries: t.LVList[AddressEntry, t.uint16_t]


class StartNetwork(Request):
    command = CommandId.START_NETWORK


class GetNetworkInfo(Request):
    command = CommandId.GET_NETWORK_INFO
    response = NetworkInfo


class ScanKeyTable(Request):
    command = CommandId.SCAN_KEY_TABLE
    response = ScanCount
    event = KeyEntry


class ScanChildren(Request):
    command = CommandId.SCAN_CHILDREN
    response = ScanCount
    event = ChildEntry


class ScanAddressCache(Request):
    command = CommandId.SCAN_ADDRESS_CACHE
    response = ScanCount
    event = AddressEntry


class ScanRouteTable(Request):
    command = CommandId.SCAN_ROUTE_TABLE
    response = ScanCount
    event = RouteEntry


class SendAps(Request):
    command = CommandId.SEND_APS

    has_eui64: t.uint1_t
    aps_ack: t.uint1_t
    aps_encryption: t.uint1_t
    delivery_mode: DeliveryMode
    reserved: t.uint3_t
    destination: t.NWK
    destination_eui64: t.EUI64
    profile_id: t.uint16_t
    cluster_id: t.uint16_t
    src_ep: t.uint8_t
    dst_ep: t.uint8_t
    aps_seq: t.uint8_t
    radius: t.uint8_t
    priority: t.int8s
    asdu: t.LVList[t.uint8_t, t.uint16_t]

    @classmethod
    def build(
        cls,
        *,
        delivery_mode: DeliveryMode,
        destination: t.NWK | None,
        destination_eui64: t.EUI64 | None,
        aps_ack: bool,
        aps_encryption: bool,
        profile_id: int,
        cluster_id: int,
        src_ep: int,
        dst_ep: int,
        aps_seq: int,
        radius: int,
        priority: int,
        asdu: bytes,
    ) -> SendAps:
        return cls(
            has_eui64=t.uint1_t(destination_eui64 is not None),
            aps_ack=t.uint1_t(aps_ack),
            aps_encryption=t.uint1_t(aps_encryption),
            delivery_mode=delivery_mode,
            reserved=t.uint3_t(0),
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
            asdu=t.LVList[t.uint8_t, t.uint16_t](asdu),
        )


class PermitJoins(Request):
    command = CommandId.PERMIT_JOINS

    duration: t.uint16_t
    accept_direct_joins: t.Bool


class SetChannel(Request):
    command = CommandId.SET_CHANNEL

    channel: t.uint8_t


class SetNwkUpdateId(Request):
    command = CommandId.SET_NWK_UPDATE_ID

    nwk_update_id: t.uint8_t


class SetProvisionalKey(Request):
    command = CommandId.SET_PROVISIONAL_KEY

    ieee: t.EUI64
    key: t.KeyData


class EnergyScan(Request):
    command = CommandId.ENERGY_SCAN
    event = EnergyResult

    channels: t.LVList[t.uint8_t, t.uint16_t]
    duration_per_channel_ms: t.uint16_t


class NetworkScan(Request):
    command = CommandId.NETWORK_SCAN
    event = Beacon

    channels: t.LVList[t.uint8_t, t.uint16_t]
    duration_per_channel_ms: t.uint16_t


class PacketCapture(Request):
    command = CommandId.PACKET_CAPTURE
    event = CapturedPacket

    channel: t.uint8_t


class PacketCaptureChannel(Request):
    command = CommandId.PACKET_CAPTURE_CHANNEL

    channel: t.uint8_t


# -- notifications ---------------------------------------------------------------


class Hello(Notification):
    protocol_version: t.uint8_t
    configured: t.Bool


class LastReset(Notification):
    message: t.LVList[t.uint8_t, t.uint16_t]

    @property
    def message_text(self) -> str:
        return bytes(self.message).decode()


class ReceivedAps(Notification):
    source: t.NWK
    destination: t.NWK
    has_group: t.Bool
    group: t.uint16_t
    profile_id: t.uint16_t
    cluster_id: t.uint16_t
    src_ep: t.uint8_t
    dst_ep: t.uint8_t
    lqi: t.uint8_t
    rssi: t.int8s
    data: t.LVList[t.uint8_t, t.uint16_t]

    @property
    def group_id(self) -> int | None:
        return int(self.group) if self.has_group else None

    @property
    def data_bytes(self) -> bytes:
        return bytes(self.data)


class SendConfirm(Notification):
    confirmed: t.Bool
    next_hop: t.NWK  # 0xFFFF when unknown
    reason: t.LVList[t.uint8_t, t.uint16_t]

    @property
    def next_hop_or_none(self) -> t.NWK | None:
        return self.next_hop if self.next_hop != t.NWK(0xFFFF) else None

    @property
    def reason_text(self) -> str:
        return bytes(self.reason).decode()


class ApsAckConfirm(Notification):
    acked: t.Bool
    reason: t.LVList[t.uint8_t, t.uint16_t]

    @property
    def reason_text(self) -> str:
        return bytes(self.reason).decode()


class DeviceJoined(Notification):
    nwk: t.NWK
    ieee: t.EUI64
    parent: t.NWK


class DeviceLeft(Notification):
    nwk: t.NWK
    has_ieee: t.uint1_t
    rejoin: t.uint1_t
    has_router_ieee: t.uint1_t
    reserved: t.uint5_t
    ieee: t.EUI64
    reason: LeaveReason
    router: t.NWK  # 0xFFFF when not router_reported
    router_ieee: t.EUI64

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


class FrameCounter(Notification):
    frame_counter: t.uint32_t


class LinkKey(Notification):
    ieee: t.EUI64
    key: t.KeyData


class ApsDecryptFailure(Notification):
    source: t.NWK
    source_ieee: t.EUI64
    frame_counter: t.uint32_t
    key_id: KeyId


# Notification id -> struct, for decoding unsolicited frames. `SendConfirm` and
# `ApsAckConfirm` are handled specially (they resolve a pending send by request id).
NOTIFICATIONS: dict[CommandId, type[Notification]] = {
    CommandId.HELLO: Hello,
    CommandId.LAST_RESET: LastReset,
    CommandId.RECEIVED_APS: ReceivedAps,
    CommandId.SEND_CONFIRM: SendConfirm,
    CommandId.APS_ACK_CONFIRM: ApsAckConfirm,
    CommandId.DEVICE_JOINED: DeviceJoined,
    CommandId.DEVICE_LEFT: DeviceLeft,
    CommandId.FRAME_COUNTER: FrameCounter,
    CommandId.LINK_KEY: LinkKey,
    CommandId.APS_DECRYPT_FAILURE: ApsDecryptFailure,
}


def encode_request(request: Request, request_id: int) -> bytes:
    """Serialize a request frame (3-byte header: command, request id)."""
    return (
        bytes([request.command])
        + t.uint16_t(request_id).serialize()
        + request.serialize()
    )


def encode_reply(
    frame_type: FrameType, command: CommandId, request_id: int, body: bytes = b""
) -> bytes:
    """Serialize a device -> host frame (4-byte header, then the body)."""
    header = FrameHeader(
        frame_type=frame_type,
        command=t.uint8_t(command),
        request_id=t.uint16_t(request_id),
    )
    return header.serialize() + body


# Command id -> request type, for parsing an outbound frame back into a struct.
REQUESTS: dict[CommandId, type[Request]] = {
    cls.command: cls for cls in Request.__subclasses__()
}


class ProtocolError(DeliveryError):
    """A firmware error response (non-OK status)."""

    # The message keeps the JSON path's "<code>: <message>" format (code = lowercased
    # status name) so existing str(exc).startswith(...) checks still work.
    def __init__(self, status: Status, message: str) -> None:
        code = status.name.lower()
        super().__init__(f"{code}: {message}" if message else f"{code}: ")
        self.status = status
        self.message = message
