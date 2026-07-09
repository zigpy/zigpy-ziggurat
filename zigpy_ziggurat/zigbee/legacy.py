"""Legacy JSON-RPC wire protocol for the WebSocket transport."""

from dataclasses import dataclass
import enum
from typing import ClassVar, Generic, TypeVar

from mashumaro import DataClassDictMixin
from mashumaro.config import BaseConfig
from mashumaro.types import SerializationStrategy
import zigpy.types as t


class BigEndianHexNwk(SerializationStrategy):
    """`1a2b`-style hex, the network address format of requests and responses."""

    def serialize(self, value: t.NWK) -> str:
        return f"{int(value):04x}"

    def deserialize(self, value: str) -> t.NWK:
        return t.NWK(int(value, 16))


class BigEndianHexPanId(SerializationStrategy):
    def serialize(self, value: t.PanId) -> str:
        return f"{int(value):04x}"

    def deserialize(self, value: str) -> t.PanId:
        return t.PanId(int(value, 16))


class LittleEndianHexNwk(SerializationStrategy):
    """`2b1a`-style hex, the network address format of notifications."""

    def serialize(self, value: t.NWK) -> str:
        return value.serialize().hex()

    def deserialize(self, value: str) -> t.NWK:
        return t.NWK.deserialize(bytes.fromhex(value))[0]


class ColonHexEui64(SerializationStrategy):
    def serialize(self, value: t.EUI64) -> str:
        return str(value)

    def deserialize(self, value: str) -> t.EUI64:
        return t.EUI64.convert(value)


class ColonHexExtendedPanId(SerializationStrategy):
    def serialize(self, value: t.ExtendedPanId) -> str:
        return str(value)

    def deserialize(self, value: str) -> t.ExtendedPanId:
        return t.ExtendedPanId(t.EUI64.convert(value))


class ColonHexKey(SerializationStrategy):
    def serialize(self, value: t.KeyData) -> str:
        return str(value)

    def deserialize(self, value: str) -> t.KeyData:
        return t.KeyData.convert(value)


class HexBytes(SerializationStrategy):
    def serialize(self, value: bytes) -> str:
        return value.hex()

    def deserialize(self, value: str) -> bytes:
        return bytes.fromhex(value)


class SizedInt(SerializationStrategy):
    """Plain JSON integers, validated into zigpy's sized integer types."""

    def __init__(self, int_type: type[int]) -> None:
        self._int_type = int_type

    def serialize(self, value: int) -> int:
        return int(value)

    def deserialize(self, value: int) -> int:
        return self._int_type(value)


class _WireConfig(BaseConfig):
    serialization_strategy = {
        t.NWK: BigEndianHexNwk(),
        t.PanId: BigEndianHexPanId(),
        t.EUI64: ColonHexEui64(),
        t.ExtendedPanId: ColonHexExtendedPanId(),
        t.KeyData: ColonHexKey(),
        bytes: HexBytes(),
        t.uint8_t: SizedInt(t.uint8_t),
        t.uint16_t: SizedInt(t.uint16_t),
        t.uint32_t: SizedInt(t.uint32_t),
        t.int8s: SizedInt(t.int8s),
    }


class _NotificationConfig(_WireConfig):
    serialization_strategy = {
        **_WireConfig.serialization_strategy,
        t.NWK: LittleEndianHexNwk(),
    }


@dataclass
class WireModel(DataClassDictMixin):
    class Config(_WireConfig): ...


@dataclass
class Response(WireModel): ...


@dataclass
class Status(Response):
    status: str


RESPONSE_T = TypeVar("RESPONSE_T", bound=Response)
EVENT_T = TypeVar("EVENT_T", bound=Response)


@dataclass
class Request(WireModel, Generic[RESPONSE_T]):
    method: ClassVar[str]
    response_type: ClassVar[type[Response]]


@dataclass
class StreamingRequest(Request[RESPONSE_T], Generic[RESPONSE_T, EVENT_T]):
    """A request answered by a stream of `event_name` events (each an `event_type`)
    before the terminal `response_type`."""

    event_type: ClassVar[type[Response]]
    event_name: ClassVar[str]


@dataclass
class KeyTableEntry(WireModel):
    partner_ieee: t.EUI64
    key: t.KeyData


@dataclass
class Ping(Request[Status]):
    method = "ping"
    response_type = Status


@dataclass
class Configure(Request[Status]):
    method = "configure"
    response_type = Status

    channel: int
    nwk_update_id: int
    pan_id: t.PanId
    extended_pan_id: t.ExtendedPanId
    nwk_address: t.NWK
    ieee_address: t.EUI64
    network_key: t.KeyData
    network_key_seq: int
    network_key_tx_counter: int
    tc_link_key: t.KeyData
    source_routing: bool
    # None means "pick automatically": the server applies its safe default
    tx_power: int | None
    # Unique trust center link keys negotiated in earlier sessions
    key_table: list[KeyTableEntry]
    # A TCLK seed carried over from a microcontroller stack, passed verbatim as the
    # source stack's plain hex string. Requires `tclk_flavor`.
    tclk_seed: str | None
    tclk_flavor: str | None

    aps_frame_counter: int = 0
    started: bool = False


@dataclass
class NetworkInfo(Response):
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
    tx_power: int
    tclk_seed: str | None
    tclk_flavor: str | None
    key_table: list[KeyTableEntry]

    aps_frame_counter: int = 0
    started: bool = False


@dataclass
class GetNetworkInfo(Request[NetworkInfo]):
    method = "get_network_info"
    response_type = NetworkInfo


@dataclass
class HwAddress(Response):
    ieee_address: t.EUI64


@dataclass
class GetHwAddress(Request[HwAddress]):
    method = "get_hw_address"
    response_type = HwAddress


@dataclass
class SendAps(Request[Status]):
    method = "send_aps"
    response_type = Status

    delivery_mode: str
    # Resolved by the server through its address map; takes precedence over
    # `destination` and selects the link key when `aps_encryption` is set
    destination_eui64: t.EUI64 | None
    destination: t.NWK | None
    profile_id: int
    cluster_id: int
    src_ep: int
    dst_ep: int
    aps_ack: bool
    aps_seq: int
    radius: int
    aps_encryption: bool
    priority: int
    data: bytes


@dataclass
class EnergyScanResult(Response):
    channel: t.uint8_t
    rssi: t.int8s


@dataclass
class EnergyScan(StreamingRequest[Status, EnergyScanResult]):
    method = "energy_scan"
    response_type = Status
    event_type = EnergyScanResult
    event_name = "energy_result"

    channels: list[int]
    duration_per_channel_ms: int


@dataclass
class NetworkBeaconEvent(Response):
    channel: t.uint8_t
    # Absent when the beacon's MAC source was not a short address
    source: t.NWK | None
    pan_id: t.PanId
    extended_pan_id: t.ExtendedPanId
    permit_joining: bool
    stack_profile: t.uint8_t
    protocol_version: t.uint8_t
    router_capacity: bool
    end_device_capacity: bool
    device_depth: t.uint8_t
    update_id: t.uint8_t
    lqi: t.uint8_t
    rssi: t.int8s


@dataclass
class NetworkScan(StreamingRequest[Status, NetworkBeaconEvent]):
    method = "network_scan"
    response_type = Status
    event_type = NetworkBeaconEvent
    event_name = "network_found"

    channels: list[int]
    duration_per_channel_ms: int


@dataclass
class PermitJoins(Request[Status]):
    method = "permit_joins"
    response_type = Status

    duration: int
    # Whether the coordinator also opens its own beacon for direct joins. False opens
    # only the trust center's authorization window, steering joins through routers.
    accept_direct_joins: bool = True


@dataclass
class SetProvisionalKey(Request[Status]):
    method = "set_provisional_key"
    response_type = Status

    ieee: t.EUI64
    key: t.KeyData


@dataclass
class SetChannel(Request[Status]):
    method = "set_channel"
    response_type = Status

    channel: int


@dataclass
class CapturedPacketEvent(Response):
    channel: t.uint8_t
    rssi: t.int8s
    lqi: t.uint8_t
    # Hex-encoded 802.15.4 MAC frame (FCS stripped)
    data: str


@dataclass
class PacketCapture(StreamingRequest[Status, CapturedPacketEvent]):
    method = "packet_capture"
    response_type = Status
    event_type = CapturedPacketEvent
    event_name = "captured_packet"

    channel: int


@dataclass
class PacketCaptureChangeChannel(Request[Status]):
    method = "packet_capture_change_channel"
    response_type = Status

    channel: int


@dataclass
class SetNwkUpdateId(Request[Status]):
    method = "set_nwk_update_id"
    response_type = Status

    nwk_update_id: int


@dataclass
class Notification(DataClassDictMixin):
    class Config(_NotificationConfig): ...


@dataclass
class ReceivedApsCommand(Notification):
    source: t.NWK
    destination: t.NWK
    group: int | None
    profile_id: t.uint16_t
    cluster_id: t.uint16_t
    src_ep: t.uint8_t
    dst_ep: t.uint8_t
    lqi: t.uint8_t
    rssi: t.int8s
    data: bytes


@dataclass
class FrameCounterUpdate(Notification):
    frame_counter: t.uint32_t


@dataclass
class LinkKeyUpdate(Notification):
    ieee: t.EUI64
    key: t.KeyData


@dataclass
class DeviceJoined(Notification):
    nwk: t.NWK
    ieee: t.EUI64
    parent: t.NWK


class DeviceLeaveReason(enum.StrEnum):
    """How the server learned that a device left the network."""

    # The device itself broadcast a NWK Leave announcement (`rejoin` is set)
    ANNOUNCED = "announced"
    # A parent router relayed an APS Update-Device "Device Left" (`router`/
    # `router_ieee` are set)
    ROUTER_REPORTED = "router_reported"
    # A sleepy child aged out of the neighbor table without a keepalive
    KEEPALIVE_TIMEOUT = "keepalive_timeout"


@dataclass
class DeviceLeft(Notification):
    nwk: t.NWK
    # Unknown when the leaving device never made it into the server's address map
    ieee: t.EUI64 | None
    # How the server learned of the departure
    reason: DeviceLeaveReason
    # Set only for ANNOUNCED: whether the device intends to rejoin
    rejoin: bool | None = None
    # Set only for ROUTER_REPORTED: the router that relayed the leave. The EUI64 is
    # unknown when the server could not resolve it from its address map.
    router: t.NWK | None = None
    router_ieee: t.EUI64 | None = None


@dataclass
class ApsDecryptionFailure(Notification):
    # An APS command frame from this device could not be decrypted with any key the
    # server holds. Its link key is almost certainly wrong or missing, which also
    # blocks joins routed through it (the trust center can't read its Update-Device).
    source: t.NWK
    source_ieee: t.EUI64
    frame_counter: t.uint32_t
    key_id: str


NOTIFICATIONS: dict[str, type[Notification]] = {
    "received_aps_command": ReceivedApsCommand,
    "frame_counter_update": FrameCounterUpdate,
    "link_key_update": LinkKeyUpdate,
    "device_joined": DeviceJoined,
    "device_left": DeviceLeft,
    "aps_decryption_failure": ApsDecryptionFailure,
}
