"""Tests for the legacy JSON-RPC server and the transport shim that transcodes the
binary protocol to it. Both the shim and this file are temporary: when the legacy
server is retired, delete them together.

The synthetic JSON server lives here rather than in `tests/common.py` for the same
reason -- nothing else depends on it."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
import dataclasses
import json
import logging
from typing import Any, TypeVar

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest
import zigpy.device
import zigpy.endpoint
from zigpy.exceptions import DeliveryError, NetworkNotFormed
import zigpy.state
import zigpy.types as t
import zigpy.zdo.types as zdo_t

from tests.common import (
    COORDINATOR_IEEE,
    DEVICE_IEEE,
    DEVICE_NWK,
    LINK_KEY,
    NETWORK_KEY,
    flush,
    make_app_config,
)
from zigpy_ziggurat.zigbee import (
    application as application_module,
    legacy as commands,
    protocol as p,
)
from zigpy_ziggurat.zigbee.application import ControllerApplication
from zigpy_ziggurat.zigbee.transport import LegacyWebSocketTransport, connect_transport

REQUEST_T = TypeVar("REQUEST_T")


def _request_types() -> dict[str, type[commands.Request[Any]]]:
    """Every concrete request, walking past intermediate bases like
    `StreamingRequest` that declare `method` without assigning it."""
    result: dict[str, type[commands.Request[Any]]] = {}
    stack = list(commands.Request.__subclasses__())
    while stack:
        cls = stack.pop()
        stack.extend(cls.__subclasses__())
        if "method" in cls.__dict__:
            result[cls.method] = cls
    return result


REQUEST_TYPES: dict[str, type[commands.Request[Any]]] = _request_types()
NOTIFICATION_EVENTS: dict[type[commands.Notification], str] = {
    cls: name for name, cls in commands.NOTIFICATIONS.items()
}


class RpcError(Exception):
    """Raised by a handler to produce an error response."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(f"{code}: {message}")
        self.code = code
        self.message = message


def make_network_info() -> commands.NetworkInfo:
    return commands.NetworkInfo(
        channel=t.uint8_t(15),
        nwk_update_id=t.uint8_t(0),
        pan_id=t.PanId(0x1A2B),
        extended_pan_id=t.ExtendedPanId(t.EUI64.convert("aa:bb:cc:dd:ee:ff:00:11")),
        nwk_address=t.NWK(0x0000),
        ieee_address=COORDINATOR_IEEE,
        network_key=NETWORK_KEY,
        network_key_seq=t.uint8_t(0),
        network_key_tx_counter=t.uint32_t(1000),
        tc_link_key=t.KeyData(b"ZigBeeAlliance09"),
        tx_power=8,
        tclk_seed=None,
        tclk_flavor=None,
        key_table=[],
    )


class SyntheticLegacyZiggurat:
    """A real aiohttp websocket server speaking the legacy JSON protocol, with
    per-method handlers that tests can override."""

    def __init__(self) -> None:
        self.web_app = web.Application()
        self.web_app.router.add_get("/", self._handle_connection)
        self.url = ""
        self.connections = 0
        self._ws: web.WebSocketResponse | None = None
        self.requests: list[Any] = []
        self._configured: commands.Configure | None = None
        self.network_info = make_network_info()
        self.hw_address = t.EUI64.convert("11:22:33:44:55:66:77:88")
        self.handlers: dict[str, Callable[[Any, int], Awaitable[Any]]] = {
            "ping": self.on_ping,
            "configure": self.on_configure,
            "get_network_info": self.on_get_network_info,
            "get_hw_address": self.on_get_hw_address,
            "send_aps": self.on_send_aps,
            "energy_scan": self.on_energy_scan,
            "network_scan": self.on_network_scan,
            "permit_joins": self.on_status,
            "set_provisional_key": self.on_status,
            "set_channel": self.on_status,
            "set_nwk_update_id": self.on_status,
            "packet_capture": self.on_status,
            "packet_capture_change_channel": self.on_status,
        }

    @property
    def ws(self) -> web.WebSocketResponse:
        assert self._ws is not None
        return self._ws

    @property
    def configured(self) -> commands.Configure:
        assert self._configured is not None
        return self._configured

    async def _handle_connection(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        self._ws = ws

        await ws.send_json({"type": "hello", "version": 1, "state": "running"})

        async for msg in ws:
            data = json.loads(msg.data)
            command = REQUEST_TYPES[data["method"]].from_dict(data["params"])
            self.requests.append(command)
            await ws.send_json({"type": "event", "id": data["id"], "event": "accepted"})

            try:
                response = await self.handlers[data["method"]](command, data["id"])
            except RpcError as exc:
                await ws.send_json(
                    {
                        "type": "response",
                        "id": data["id"],
                        "error": {"code": exc.code, "message": exc.message},
                    }
                )
            else:
                # `None` deliberately withholds the response
                if response is not None:
                    await ws.send_json(
                        {
                            "type": "response",
                            "id": data["id"],
                            "result": response.to_dict(),
                        }
                    )

        return ws

    async def send_event(self, request_id: int, event: str) -> None:
        await self.ws.send_json({"type": "event", "id": request_id, "event": event})

    async def send_event_data(
        self, request_id: int, event: str, data: dict[str, Any]
    ) -> None:
        await self.ws.send_json(
            {"type": "event", "id": request_id, "event": event, "data": data}
        )

    async def send_confirm(self, request_id: int, *, reason: str | None = None) -> None:
        if reason is not None:
            data: dict[str, Any] = {
                "id": request_id,
                "status": "failed",
                "reason": reason,
            }
        else:
            data = {"id": request_id, "status": "confirmed", "next_hop": None}

        await self.ws.send_json(
            {"type": "notification", "event": "send_confirm", "data": data}
        )

    async def aps_ack_confirm(
        self, request_id: int, *, reason: str | None = None
    ) -> None:
        if reason is not None:
            data: dict[str, Any] = {
                "id": request_id,
                "status": "failed",
                "reason": reason,
            }
        else:
            data = {"id": request_id, "status": "confirmed"}

        await self.ws.send_json(
            {"type": "notification", "event": "aps_ack_confirm", "data": data}
        )

    async def send_notification(self, notification: commands.Notification) -> None:
        await self.ws.send_json(
            {
                "type": "notification",
                "event": NOTIFICATION_EVENTS[type(notification)],
                "data": notification.to_dict(),
            }
        )

    async def send_raw(self, text: str) -> None:
        await self.ws.send_str(text)

    def sent(self, request_type: type[REQUEST_T]) -> list[REQUEST_T]:
        return [r for r in self.requests if isinstance(r, request_type)]

    async def wait_for(
        self, request_type: type[REQUEST_T], count: int = 1
    ) -> REQUEST_T:
        async with asyncio.timeout(2):
            while len(self.sent(request_type)) < count:
                await asyncio.sleep(0.01)

        return self.sent(request_type)[count - 1]

    async def on_ping(self, command: commands.Ping, request_id: int) -> commands.Status:
        return commands.Status(status="pong")

    async def on_status(self, command: Any, request_id: int) -> commands.Status:
        return commands.Status(status="success")

    async def on_configure(
        self, command: commands.Configure, request_id: int
    ) -> commands.Status:
        self._configured = command
        return commands.Status(status="success")

    async def on_get_network_info(
        self, command: commands.GetNetworkInfo, request_id: int
    ) -> commands.NetworkInfo:
        return self.network_info

    async def on_get_hw_address(
        self, command: commands.GetHwAddress, request_id: int
    ) -> commands.HwAddress:
        return commands.HwAddress(ieee_address=self.hw_address)

    async def on_send_aps(
        self, command: commands.SendAps, request_id: int
    ) -> commands.Status:
        await self.send_confirm(request_id)
        if command.aps_ack:
            await self.aps_ack_confirm(request_id)
        return commands.Status(status="accepted")

    async def on_energy_scan(
        self, command: commands.EnergyScan, request_id: int
    ) -> commands.Status:
        for channel in command.channels:
            await self.send_event_data(
                request_id,
                "energy_result",
                commands.EnergyScanResult(
                    channel=t.uint8_t(channel), rssi=t.int8s(-85)
                ).to_dict(),
            )
        return commands.Status(status="complete")

    async def on_network_scan(
        self, command: commands.NetworkScan, request_id: int
    ) -> commands.Status:
        return commands.Status(status="complete")


@pytest.fixture
async def legacy_server() -> AsyncIterator[SyntheticLegacyZiggurat]:
    ziggurat = SyntheticLegacyZiggurat()
    test_server = TestServer(ziggurat.web_app)
    await test_server.start_server()
    ziggurat.url = f"ws://localhost:{test_server.port}/"

    yield ziggurat

    await test_server.close()


@pytest.fixture
async def legacy_connected_app(
    legacy_server: SyntheticLegacyZiggurat,
) -> AsyncIterator[ControllerApplication]:
    app = ControllerApplication(make_app_config(legacy_server.url))
    await app.connect()

    yield app

    await app.shutdown(db=False)


@pytest.fixture
async def legacy_app(
    legacy_connected_app: ControllerApplication,
) -> ControllerApplication:
    await legacy_connected_app.start_network()
    return legacy_connected_app


async def _legacy(
    server: SyntheticLegacyZiggurat,
) -> tuple[LegacyWebSocketTransport, list[bytes]]:
    frames: list[bytes] = []
    transport = await connect_transport(server.url, frames.append, lambda exc: None)
    assert isinstance(transport, LegacyWebSocketTransport)
    return transport, frames


async def _wait_for(frames: list[bytes], count: int = 1) -> None:
    async with asyncio.timeout(2):
        while len(frames) < count:
            await asyncio.sleep(0.01)


def add_initialized_device(app: ControllerApplication) -> zigpy.device.Device:
    device = app.add_device(DEVICE_IEEE, DEVICE_NWK)
    device.node_desc = app.get_device(nwk=t.NWK(0x0000)).node_desc
    device.status = zigpy.device.Status.ENDPOINTS_INIT
    device.add_endpoint(1).status = zigpy.endpoint.Status.ZDO_INIT
    return device


# -- protocol probing ------------------------------------------------------------


async def test_probe_selects_legacy(legacy_server: SyntheticLegacyZiggurat) -> None:
    transport = await connect_transport(
        legacy_server.url, lambda frame: None, lambda exc: None
    )
    try:
        assert isinstance(transport, LegacyWebSocketTransport)
    finally:
        await transport.disconnect()


# -- application against the legacy server ---------------------------------------


async def test_legacy_connect(
    legacy_connected_app: ControllerApplication,
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    assert legacy_server.connections == 1

    await legacy_connected_app.permit_ncp(1)
    permit = legacy_server.sent(commands.PermitJoins)[-1]
    assert permit.duration == 1
    assert permit.accept_direct_joins is True


async def test_legacy_start_network(
    legacy_connected_app: ControllerApplication,
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    """The binary `Configure` + `LoadKeyTable`* + `StartNetwork` sequence coalesces
    into the single JSON `configure` call the legacy server takes."""
    await legacy_connected_app.start_network()

    assert legacy_server.configured.channel == 15
    assert legacy_server.configured.pan_id == t.PanId(0x1A2B)
    assert legacy_server.configured.network_key == NETWORK_KEY
    assert legacy_server.configured.aps_frame_counter == 0
    # One JSON call, however many binary frames it was split across
    assert len(legacy_server.sent(commands.Configure)) == 1
    assert legacy_connected_app.backups[-1].network_info.pan_id == t.PanId(0x1A2B)


async def test_legacy_load_network_info(
    legacy_connected_app: ControllerApplication,
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    app = legacy_connected_app

    async def not_configured(
        command: commands.GetNetworkInfo, request_id: int
    ) -> commands.NetworkInfo:
        raise RpcError("not_configured", "no stack is running")

    # A stateless server with no network running and no local backup: no network
    legacy_server.handlers["get_network_info"] = not_configured
    with pytest.raises(NetworkNotFormed):
        await app.load_network_info()

    # Unrelated errors propagate. The unknown JSON code has no binary status, so
    # it degrades to a generic invalid-request; the detail goes to the log.
    async def serial_error(
        command: commands.GetNetworkInfo, request_id: int
    ) -> commands.NetworkInfo:
        raise RpcError("serial_port_error", "it burned down")

    legacy_server.handlers["get_network_info"] = serial_error
    with pytest.raises(DeliveryError, match="invalid_request"):
        await app.load_network_info()

    legacy_server.handlers["get_network_info"] = legacy_server.on_get_network_info
    await app.load_network_info()
    assert app.state.node_info.ieee == COORDINATOR_IEEE
    assert app.state.network_info.channel == 15
    assert app.state.network_info.network_key.key == NETWORK_KEY
    assert app.state.network_info.stack_specific == {}
    # The JSON server surfaces no children, address cache or route table
    assert app.state.network_info.children == []
    assert app.state.network_info.nwk_addresses == {}

    # TCLK seeds map to the stack_specific layout of their source stack
    legacy_server.network_info.tclk_seed = "ab" * 16
    legacy_server.network_info.tclk_flavor = "zstack"
    await app.load_network_info()
    assert app.state.network_info.stack_specific == {"zstack": {"tclk_seed": "ab" * 16}}

    legacy_server.network_info.tclk_flavor = "ezsp"
    await app.load_network_info()
    assert app.state.network_info.stack_specific == {"ezsp": {"hashed_tclk": "ab" * 16}}

    # The key table the JSON `get_network_info` returns inline is replayed as the
    # events of the binary `ScanKeyTable` that follows
    legacy_server.network_info.key_table = [
        commands.KeyTableEntry(partner_ieee=DEVICE_IEEE, key=LINK_KEY)
    ]
    await app.load_network_info()
    assert app.state.network_info.key_table == [
        zigpy.state.Key(key=LINK_KEY, partner_ieee=DEVICE_IEEE)
    ]


async def test_legacy_write_network_info(
    legacy_connected_app: ControllerApplication,
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    app = legacy_connected_app
    await app.load_network_info()
    network_info = app.state.network_info
    node_info = app.state.node_info

    # A zstack TCLK seed rides along verbatim
    await app.write_network_info(
        network_info=network_info.replace(
            stack_specific={"zstack": {"tclk_seed": "cd" * 16}},
            key_table=[zigpy.state.Key(key=LINK_KEY, partner_ieee=DEVICE_IEEE)],
        ),
        node_info=node_info,
    )
    assert legacy_server.configured.tclk_seed == "cd" * 16
    assert legacy_server.configured.tclk_flavor == "zstack"
    assert legacy_server.configured.key_table == [
        commands.KeyTableEntry(partner_ieee=DEVICE_IEEE, key=LINK_KEY)
    ]

    # An ezsp seed likewise
    await app.write_network_info(
        network_info=network_info.replace(
            stack_specific={"ezsp": {"hashed_tclk": "ef" * 16}}
        ),
        node_info=node_info,
    )
    assert legacy_server.configured.tclk_seed == "ef" * 16
    assert legacy_server.configured.tclk_flavor == "ezsp"

    # When zigpy forms a fresh network it leaves the IEEE address unspecified,
    # deferring to the radio's hardware address
    await app.write_network_info(
        network_info=network_info,
        node_info=node_info.replace(ieee=t.EUI64.UNKNOWN),  # type: ignore[attr-defined]
    )
    assert legacy_server.sent(commands.GetHwAddress)
    assert legacy_server.configured.ieee_address == legacy_server.hw_address
    assert app.state.node_info.ieee == legacy_server.hw_address


async def test_legacy_permits(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    await legacy_app.permit_with_link_key(
        node=DEVICE_IEEE, link_key=LINK_KEY, time_s=12
    )

    provisional = legacy_server.sent(commands.SetProvisionalKey)[-1]
    assert provisional.ieee == DEVICE_IEEE
    assert provisional.key == LINK_KEY

    # `super().permit()` broadcasts Mgmt_Permit_Joining_req and calls `permit_ncp`
    broadcast = legacy_server.sent(commands.SendAps)[-1]
    assert broadcast.delivery_mode == "broadcast"
    assert broadcast.cluster_id == zdo_t.ZDOCmd.Mgmt_Permit_Joining_req
    assert legacy_server.sent(commands.PermitJoins)[-1].duration == 12


async def test_legacy_move_network_to_channel(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    await legacy_app._move_network_to_channel(new_channel=20, new_nwk_update_id=1)

    methods = [type(r) for r in legacy_server.requests]
    assert methods.index(commands.SetNwkUpdateId) < methods.index(commands.SetChannel)
    assert legacy_server.sent(commands.SetNwkUpdateId)[-1].nwk_update_id == 1
    assert legacy_server.sent(commands.SetChannel)[-1].channel == 20


async def test_legacy_watchdog_feed(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    # The legacy server has no firmware-info call: it is probed with a JSON `ping`
    await legacy_app._watchdog_feed()
    assert isinstance(legacy_server.requests[-1], commands.Ping)


async def test_legacy_reset_network_info(legacy_app: ControllerApplication) -> None:
    # The legacy server has no shutdown; the shim OKs it locally
    await legacy_app.reset_network_info()


@pytest.mark.parametrize(
    ("dst", "tx_options", "expected"),
    [
        (
            t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK),
            t.TransmitOptions.ACK,
            {
                "delivery_mode": "unicast",
                "destination": DEVICE_NWK,
                "destination_eui64": DEVICE_IEEE,
                "aps_ack": True,
                "aps_encryption": False,
            },
        ),
        (
            t.AddrModeAddress(addr_mode=t.AddrMode.IEEE, address=DEVICE_IEEE),
            t.TransmitOptions.NONE,
            {
                "delivery_mode": "unicast",
                # 0xFFFE on the wire means "no short address"
                "destination": None,
                "destination_eui64": DEVICE_IEEE,
            },
        ),
        (
            t.AddrModeAddress(addr_mode=t.AddrMode.Group, address=t.Group(0x0002)),
            t.TransmitOptions.NONE,
            {"delivery_mode": "multicast", "destination": t.NWK(0x0002), "dst_ep": 0},
        ),
        (
            t.AddrModeAddress(
                addr_mode=t.AddrMode.Broadcast,
                address=t.BroadcastAddress.ALL_ROUTERS_AND_COORDINATOR,
            ),
            t.TransmitOptions.NONE,
            {"delivery_mode": "broadcast", "destination": t.NWK(0xFFFC)},
        ),
    ],
)
async def test_legacy_send_packet(
    legacy_app: ControllerApplication,
    legacy_server: SyntheticLegacyZiggurat,
    dst: t.AddrModeAddress,
    tx_options: t.TransmitOptions,
    expected: dict[str, Any],
) -> None:
    legacy_app.add_device(DEVICE_IEEE, DEVICE_NWK)

    await legacy_app.send_packet(
        t.ZigbeePacket(
            src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x0000)),
            src_ep=t.uint8_t(1),
            dst=dst,
            dst_ep=t.uint8_t(1),
            tsn=t.uint8_t(33),
            profile_id=t.uint16_t(0x0104),
            cluster_id=t.uint16_t(0x0006),
            data=t.SerializableBytes(b"\x01\x02\x03"),
            tx_options=tx_options,
        )
    )

    request = legacy_server.sent(commands.SendAps)[-1]
    assert request.data == b"\x01\x02\x03"
    assert request.aps_seq == 33
    assert request.radius == 30
    for field, value in expected.items():
        assert getattr(request, field) == value


async def test_legacy_send_packet_delivery_failure(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    async def fail(command: commands.SendAps, request_id: int) -> commands.Status:
        raise RpcError("transmit_failed", "radio unavailable")

    legacy_server.handlers["send_aps"] = fail

    # The legacy `transmit_failed` code maps to the binary RADIO_ERROR status
    with pytest.raises(DeliveryError, match="radio_error"):
        await legacy_app.send_packet(
            t.ZigbeePacket(
                src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x0000)),
                src_ep=t.uint8_t(1),
                dst=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK),
                dst_ep=t.uint8_t(1),
                tsn=t.uint8_t(34),
                profile_id=t.uint16_t(0x0104),
                cluster_id=t.uint16_t(0x0006),
                data=t.SerializableBytes(b"\x04"),
            )
        )


async def test_legacy_send_confirm_failure(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    """The JSON protocol carries no failure kind, so a failed confirm becomes the
    least-wrong binary stand-in."""

    async def fail_confirm(
        command: commands.SendAps, request_id: int
    ) -> commands.Status:
        await legacy_server.send_confirm(request_id, reason="no_route")
        return commands.Status(status="accepted")

    legacy_server.handlers["send_aps"] = fail_confirm

    with pytest.raises(DeliveryError, match="TRANSMIT_FAILED"):
        await legacy_app.send_packet(
            t.ZigbeePacket(
                src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x0000)),
                src_ep=t.uint8_t(1),
                dst=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK),
                dst_ep=t.uint8_t(1),
                tsn=t.uint8_t(35),
                profile_id=t.uint16_t(0x0104),
                cluster_id=t.uint16_t(0x0006),
                data=t.SerializableBytes(b"\x05"),
            )
        )


async def test_legacy_aps_ack_confirm_failure(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    async def fail_ack(command: commands.SendAps, request_id: int) -> commands.Status:
        await legacy_server.send_confirm(request_id)
        await legacy_server.aps_ack_confirm(request_id, reason="timeout")
        return commands.Status(status="accepted")

    legacy_server.handlers["send_aps"] = fail_ack

    with pytest.raises(DeliveryError, match="APS_ACK_TIMEOUT"):
        await legacy_app.send_packet(
            t.ZigbeePacket(
                src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x0000)),
                src_ep=t.uint8_t(1),
                dst=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK),
                dst_ep=t.uint8_t(1),
                tsn=t.uint8_t(36),
                profile_id=t.uint16_t(0x0104),
                cluster_id=t.uint16_t(0x0006),
                data=t.SerializableBytes(b"\x06"),
                tx_options=t.TransmitOptions.ACK,
            )
        )


async def test_legacy_energy_scan(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    energies = await legacy_app.energy_scan(
        # zigpy mis-annotates the classmethod's `cls` as an instance
        channels=t.Channels.from_channel_list([11, 15]),  # type: ignore[misc]
        duration_exp=2,
        count=1,
    )

    scan = legacy_server.sent(commands.EnergyScan)[-1]
    assert scan.channels == [11, 15]
    # 0.016 ms/symbol * 960 symbols * (2**2 + 1)
    assert scan.duration_per_channel_ms == 77
    assert sorted(energies) == [11, 15]


async def test_legacy_network_scan(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    beacon = commands.NetworkBeaconEvent(
        channel=t.uint8_t(11),
        source=t.NWK(0x0000),
        pan_id=t.PanId(0x1A2B),
        extended_pan_id=t.ExtendedPanId(t.EUI64.convert("aa:bb:cc:dd:ee:ff:00:11")),
        permit_joining=True,
        stack_profile=t.uint8_t(2),
        protocol_version=t.uint8_t(2),
        router_capacity=True,
        end_device_capacity=True,
        device_depth=t.uint8_t(0),
        update_id=t.uint8_t(0),
        lqi=t.uint8_t(200),
        rssi=t.int8s(-60),
    )

    async def scan(command: commands.NetworkScan, request_id: int) -> commands.Status:
        await legacy_server.send_event_data(
            request_id, "network_found", beacon.to_dict()
        )
        # A beacon whose MAC source was not a short address
        await legacy_server.send_event_data(
            request_id,
            "network_found",
            dataclasses.replace(beacon, source=None).to_dict(),
        )
        return commands.Status(status="complete")

    legacy_server.handlers["network_scan"] = scan

    found = [
        result
        async for result in legacy_app.network_scan(
            # zigpy mis-annotates the classmethod's `cls` as an instance
            channels=t.Channels.from_channel_list([11]),  # type: ignore[misc]
            duration_exp=2,
        )
    ]

    assert [f.src for f in found] == [t.NWK(0x0000), None]
    assert found[0].pan_id == t.PanId(0x1A2B)
    assert found[0].lqi == 200
    assert found[0].permit_joining is True


async def test_legacy_packet_capture(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    async def capture(
        command: commands.PacketCapture, request_id: int
    ) -> commands.Status:
        await legacy_server.send_event_data(
            request_id,
            "captured_packet",
            commands.CapturedPacketEvent(
                channel=t.uint8_t(15),
                rssi=t.int8s(-80),
                lqi=t.uint8_t(200),
                data="aabbcc",
            ).to_dict(),
        )
        return commands.Status(status="complete")

    legacy_server.handlers["packet_capture"] = capture

    packets = [packet async for packet in legacy_app.packet_capture(15)]
    assert len(packets) == 1
    assert packets[0].data == b"\xaa\xbb\xcc"
    assert legacy_server.sent(commands.PacketCapture)[0].channel == 15

    await legacy_app.packet_capture_change_channel(20)
    assert legacy_server.sent(commands.PacketCaptureChangeChannel)[0].channel == 20


async def test_legacy_received_aps_notification(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    add_initialized_device(legacy_app)

    # A ZDO request arriving over the wire is answered end-to-end
    await legacy_server.send_notification(
        commands.ReceivedApsCommand(
            source=DEVICE_NWK,
            destination=t.NWK(0x0000),
            group=None,
            profile_id=t.uint16_t(0x0000),
            cluster_id=t.uint16_t(zdo_t.ZDOCmd.Node_Desc_req),
            src_ep=t.uint8_t(0),
            dst_ep=t.uint8_t(0),
            lqi=t.uint8_t(255),
            rssi=t.int8s(-40),
            data=b"\x77" + t.NWK(0x0000).serialize(),
        )
    )
    reply = await legacy_server.wait_for(commands.SendAps)
    assert reply.cluster_id == zdo_t.ZDOCmd.Node_Desc_rsp
    assert reply.data[0] == 0x77

    # A group-addressed frame carries its group id through the transcoder
    await legacy_server.send_notification(
        commands.ReceivedApsCommand(
            source=DEVICE_NWK,
            destination=t.NWK(0x0000),
            group=2,
            profile_id=t.uint16_t(0x0104),
            cluster_id=t.uint16_t(0x0006),
            src_ep=t.uint8_t(1),
            dst_ep=t.uint8_t(255),
            lqi=t.uint8_t(255),
            rssi=t.int8s(-40),
            data=b"\x01",
        )
    )
    await flush(legacy_app)


async def test_legacy_frame_counter_notification(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    await legacy_server.send_notification(
        commands.FrameCounterUpdate(frame_counter=t.uint32_t(123456))
    )
    await flush(legacy_app)

    assert legacy_app.state.network_info.network_key.tx_counter == 123456


async def test_legacy_link_key_notification(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    await legacy_server.send_notification(
        commands.LinkKeyUpdate(ieee=DEVICE_IEEE, key=LINK_KEY)
    )
    await flush(legacy_app)

    assert legacy_app.state.network_info.key_table == [
        zigpy.state.Key(key=LINK_KEY, partner_ieee=DEVICE_IEEE)
    ]


async def test_legacy_device_joined_notification(
    legacy_app: ControllerApplication,
    legacy_server: SyntheticLegacyZiggurat,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(application_module, "DEVICE_JOIN_MAX_DELAY", 0.05)

    await legacy_server.send_notification(
        commands.DeviceJoined(nwk=DEVICE_NWK, ieee=DEVICE_IEEE, parent=t.NWK(0x0000))
    )
    await flush(legacy_app)
    await asyncio.sleep(0.1)

    assert legacy_app.get_device(ieee=DEVICE_IEEE).nwk == DEVICE_NWK


async def test_legacy_device_left_notification(
    legacy_app: ControllerApplication, legacy_server: SyntheticLegacyZiggurat
) -> None:
    left: list[zigpy.device.Device] = []

    class Listener:
        def device_left(self, device: zigpy.device.Device) -> None:
            left.append(device)

    legacy_app.add_listener(Listener())
    device = legacy_app.add_device(DEVICE_IEEE, DEVICE_NWK)

    # Each leave reason maps onto its binary counterpart
    await legacy_server.send_notification(
        commands.DeviceLeft(
            nwk=DEVICE_NWK,
            ieee=DEVICE_IEEE,
            reason=commands.DeviceLeaveReason.ANNOUNCED,
            rejoin=False,
        )
    )
    await flush(legacy_app)
    assert left == [device]

    # A parent router relayed the leave; the IEEE is resolved through the registry
    await legacy_server.send_notification(
        commands.DeviceLeft(
            nwk=DEVICE_NWK,
            ieee=None,
            reason=commands.DeviceLeaveReason.ROUTER_REPORTED,
            router=t.NWK(0x1234),
            router_ieee=t.EUI64.convert("bb:bb:bb:bb:bb:bb:bb:bb"),
        )
    )
    await flush(legacy_app)
    assert left == [device, device]

    await legacy_server.send_notification(
        commands.DeviceLeft(
            nwk=t.NWK(0xBEEF),
            ieee=None,
            reason=commands.DeviceLeaveReason.KEEPALIVE_TIMEOUT,
        )
    )
    await flush(legacy_app)
    assert left == [device, device]


async def test_legacy_aps_decryption_failure_notification(
    legacy_app: ControllerApplication,
    legacy_server: SyntheticLegacyZiggurat,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        await legacy_server.send_notification(
            commands.ApsDecryptionFailure(
                source=t.NWK(0x1234),
                source_ieee=DEVICE_IEEE,
                frame_counter=t.uint32_t(42),
                # An unknown key id degrades to the network key
                key_id="tc_link_key",
            )
        )
        await flush(legacy_app)

    assert "Could not decrypt an APS command" in caplog.text


# -- transcoding at the transport level -------------------------------------------


async def test_legacy_acknowledges_restore_loads(
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    """The legacy server re-learns its topology tables, so the restore loads are
    acknowledged locally and never reach it."""
    transport, frames = await _legacy(legacy_server)
    try:
        loads: list[p.Request] = [
            p.LoadChildren(entries=t.LVList[p.ChildEntry, t.uint16_t]([])),
            p.LoadAddressCache(entries=t.LVList[p.AddressEntry, t.uint16_t]([])),
            p.LoadRouteTable(entries=t.LVList[p.RouteEntry, t.uint16_t]([])),
            p.LoadSourceRoutes(entries=t.LVList[p.SourceRouteEntry, t.uint16_t]([])),
        ]
        for request_id, load in enumerate(loads, start=1):
            await transport.send_frame(p.encode_request(load, request_id))

        await _wait_for(frames, count=len(loads))
        assert len(frames) == len(loads)
        for frame, load in zip(frames, loads, strict=True):
            header, body = p.Header.deserialize(frame)
            assert header.frame_type == p.FrameType.RESPONSE
            assert header.command == load.command
            assert body == bytes([p.Status.OK])

        assert legacy_server.requests == []
    finally:
        await transport.disconnect()


async def test_legacy_drops_cancel_request(
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    """The legacy server has no request-cancel concept, so a best-effort cancel is
    dropped rather than sent as an unknown method."""
    transport, frames = await _legacy(legacy_server)
    try:
        await transport.send_frame(
            p.encode_request(p.CancelRequest(request_id=t.uint16_t(7)), 1)
        )
        # Nothing is emitted locally and nothing reaches the server
        await asyncio.sleep(0.05)
        assert frames == []
        assert legacy_server.requests == []
    finally:
        await transport.disconnect()


async def test_legacy_rejects_untranscodable_command(
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    """A binary request with no JSON equivalent fails loudly instead of being
    silently dropped."""
    transport, _ = await _legacy(legacy_server)
    try:
        with pytest.raises(ValueError, match="Cannot transcode"):
            await transport.send_frame(
                p.encode_request(p.SetTunable.build("aps_ack_timeout", 5), 1)
            )
    finally:
        await transport.disconnect()


async def test_legacy_rejects_unknown_command(
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    transport, _ = await _legacy(legacy_server)
    try:
        # An unknown command byte fails loudly instead of silently vanishing.
        frame = p.Header(
            command=t.uint8_t(0xEE),
            frame_type=p.FrameType.REQUEST,
            request_id=t.uint16_t(1),
        ).serialize()
        with pytest.raises(KeyError):
            await transport.send_frame(frame)
    finally:
        await transport.disconnect()


async def test_legacy_firmware_info_via_ping(
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    transport, frames = await _legacy(legacy_server)
    try:
        # The legacy server has no firmware-info call: the shim probes it with a
        # JSON `ping` and fabricates the response payload.
        await transport.send_frame(p.encode_request(p.GetFirmwareInfo(), 1))
        await legacy_server.wait_for(commands.Ping)
        await _wait_for(frames)
        header, body = p.Header.deserialize(frames[0])
        assert header.frame_type == p.FrameType.RESPONSE
        assert header.command == p.RequestCommand.GET_FIRMWARE_INFO
        assert body[0] == p.Status.OK
        info = p.FirmwareInfo.deserialize(body[1:])[0]
        assert info.protocol_version == p.PROTOCOL_VERSION
    finally:
        await transport.disconnect()


async def test_legacy_decodes_captured_packet(
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    transport, frames = await _legacy(legacy_server)
    try:
        # An unknown event is dropped; the captured packet is transcoded to an event.
        await legacy_server.send_event_data(7, "not_a_real_event", {})
        await legacy_server.send_event_data(
            7,
            "captured_packet",
            {"channel": 15, "rssi": -80, "lqi": 200, "data": "aabbcc"},
        )
        await _wait_for(frames)
        assert len(frames) == 1
        header, body = p.Header.deserialize(frames[0])
        assert header.frame_type == p.FrameType.EVENT
        assert header.command == p.RequestCommand.PACKET_CAPTURE
        packet = p.CapturedPacket.deserialize(body)[0]
        assert bytes(packet.psdu) == b"\xaa\xbb\xcc"
    finally:
        await transport.disconnect()


async def test_legacy_forwards_firmware_log(
    legacy_server: SyntheticLegacyZiggurat, caplog: pytest.LogCaptureFixture
) -> None:
    transport, _ = await _legacy(legacy_server)
    try:
        with caplog.at_level(logging.WARNING, logger="ziggurat.fw.foo.bar"):
            await legacy_server.send_raw(
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
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    transport, frames = await _legacy(legacy_server)
    try:
        # The real server signals a send handoff with a bare `transmitted` event
        # that carries no `data`; it must become a SEND_CONFIRM, not crash.
        await legacy_server.send_event(9, "transmitted")
        await _wait_for(frames)
        header, body = p.Header.deserialize(frames[0])
        assert header.frame_type == p.FrameType.NOTIFICATION
        assert header.command == p.NotificationCommand.SEND_CONFIRM
        assert header.request_id == 9
        assert p.SendConfirm.deserialize(body)[0].status == p.SendStatus.SUCCESS
    finally:
        await transport.disconnect()


async def test_legacy_decodes_decrypt_failure_known_key(
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    transport, frames = await _legacy(legacy_server)
    try:
        await legacy_server.send_notification(
            commands.ApsDecryptionFailure(
                source=t.NWK(0x1234),
                source_ieee=COORDINATOR_IEEE,
                frame_counter=t.uint32_t(42),
                key_id="network",
            )
        )
        await _wait_for(frames)
        header, body = p.Header.deserialize(frames[0])
        assert header.command == p.NotificationCommand.APS_DECRYPT_FAILURE
        failure = p.ApsDecryptFailure.deserialize(body)[0]
        assert failure.key_id == p.KeyId.NETWORK
    finally:
        await transport.disconnect()


async def test_legacy_ignores_binary_and_unknown_response(
    legacy_server: SyntheticLegacyZiggurat,
) -> None:
    transport, frames = await _legacy(legacy_server)
    try:
        # A binary frame and a response for an unknown id are both dropped; a
        # following confirm still transcodes, proving the loop kept going.
        await legacy_server.ws.send_bytes(b"\x00\x01\x02")
        await legacy_server.send_raw(
            json.dumps({"type": "response", "id": 9999, "result": {}})
        )
        await legacy_server.send_confirm(1)
        await _wait_for(frames)
        assert len(frames) == 1
        header, _ = p.Header.deserialize(frames[0])
        assert header.command == p.NotificationCommand.SEND_CONFIRM
    finally:
        await transport.disconnect()
