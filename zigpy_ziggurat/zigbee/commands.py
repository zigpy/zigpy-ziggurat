"""Typed models for the ziggurat JSON-RPC wire protocol, mirroring the server's
serde types. Requests and responses share one set of wire formats; notifications
encode network addresses little-endian."""

from dataclasses import dataclass
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


@dataclass
class Request(WireModel, Generic[RESPONSE_T]):
    method: ClassVar[str]
    response_type: ClassVar[type[Response]]


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
    data: bytes
    aps_encryption: bool


@dataclass
class EnergyScanResults(Response):
    results: dict[int, float]


@dataclass
class EnergyScan(Request[EnergyScanResults]):
    method = "energy_scan"
    response_type = EnergyScanResults

    channels: list[int]
    duration_per_channel_ms: int


@dataclass
class PermitJoins(Request[Status]):
    method = "permit_joins"
    response_type = Status

    duration: int


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


@dataclass
class DeviceLeft(Notification):
    nwk: t.NWK
    # Unknown when the leaving device never made it into the server's address map
    ieee: t.EUI64 | None


NOTIFICATIONS: dict[str, type[Notification]] = {
    "received_aps_command": ReceivedApsCommand,
    "frame_counter_update": FrameCounterUpdate,
    "link_key_update": LinkKeyUpdate,
    "device_joined": DeviceJoined,
    "device_left": DeviceLeft,
}
