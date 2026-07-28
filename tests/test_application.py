"""One async test per public zigpy method of `ControllerApplication`, all running
against the synthetic binary websocket server."""

import asyncio
from datetime import timedelta
import logging
import os
from typing import Any

from aiohttp import web
import pytest
import zigpy.config
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
    StatusError,
    SyntheticZiggurat,
    app,
    connected_app,
    flush,
    make_app_config,
    server,
)
from zigpy_ziggurat.config import CONF_TUNABLES, CONF_ZIGGURAT_CONFIG
from zigpy_ziggurat.zigbee import application as application_module, protocol as p
from zigpy_ziggurat.zigbee.application import (
    ControllerApplication,
    ZigguratCoordinator,
    map_rssi_to_energy,
)


class RecordingApplication(ControllerApplication):
    """Records the packets zigpy is notified of, for tests that assert on the decoded
    `ZigbeePacket` itself rather than on zigpy's reaction to it."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self.packets: list[t.ZigbeePacket] = []

    def packet_received(self, packet: t.ZigbeePacket) -> None:
        self.packets.append(packet)
        super().packet_received(packet)


def add_initialized_device(
    app: ControllerApplication,
    ieee: t.EUI64 = DEVICE_IEEE,
    nwk: t.NWK = DEVICE_NWK,
) -> zigpy.device.Device:
    """An initialized device: zigpy does not interview it when packets arrive."""
    device = app.add_device(ieee, nwk)
    device.node_desc = app.get_device(nwk=t.NWK(0x0000)).node_desc
    device.status = zigpy.device.Status.ENDPOINTS_INIT
    device.add_endpoint(1).status = zigpy.endpoint.Status.ZDO_INIT
    return device


def zdo_packet(cluster_id: int, data: bytes, src: t.NWK = DEVICE_NWK) -> t.ZigbeePacket:
    return t.ZigbeePacket(
        src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=src),
        src_ep=t.uint8_t(0),
        dst=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x0000)),
        dst_ep=t.uint8_t(0),
        tsn=t.uint8_t(data[0]),
        profile_id=t.uint16_t(0x0000),
        cluster_id=t.uint16_t(cluster_id),
        data=t.SerializableBytes(data),
        lqi=t.uint8_t(255),
        rssi=t.int8s(-40),
    )


def aps_packet(
    dst: t.AddrModeAddress,
    *,
    tsn: int = 33,
    src_ep: int = 1,
    dst_ep: int = 1,
    tx_options: t.TransmitOptions = t.TransmitOptions.NONE,
    data: bytes = b"\x01\x02\x03",
) -> t.ZigbeePacket:
    return t.ZigbeePacket(
        src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x0000)),
        src_ep=t.uint8_t(src_ep),
        dst=dst,
        dst_ep=t.uint8_t(dst_ep),
        tsn=t.uint8_t(tsn),
        profile_id=t.uint16_t(0x0104),
        cluster_id=t.uint16_t(0x0006),
        data=t.SerializableBytes(data),
        tx_options=tx_options,
    )


async def test_connect(
    connected_app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    assert server.connections == 1

    # Any transient radio state left by a previous client is cleared on connect
    assert not server.sent(p.Reset)[-1].hard

    # Requests round-trip over the socket
    await connected_app.permit_ncp(1)
    assert server.sent(p.PermitJoins)[-1].duration == 1


async def test_connect_unix_socket() -> None:
    socket_path = f"/tmp/zigpy-ziggurat-test-{os.getpid()}.sock"
    ziggurat = SyntheticZiggurat()
    runner = web.AppRunner(ziggurat.web_app)
    await runner.setup()
    site = web.UnixSite(runner, socket_path)
    await site.start()

    app = ControllerApplication(make_app_config(f"ws+unix://{socket_path}"))
    await app.connect()
    await app.permit_ncp(2)
    assert ziggurat.sent(p.PermitJoins)[-1].duration == 2

    await app.shutdown(db=False)
    await runner.cleanup()


async def test_disconnect(
    connected_app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await connected_app.disconnect()
    await connected_app.disconnect()  # idempotent without a connection

    async with asyncio.timeout(1):
        while not server.ws.closed:
            await asyncio.sleep(0.01)


async def test_start_network(
    connected_app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await connected_app.start_network()

    coordinator = connected_app.get_device(nwk=t.NWK(0x0000))
    assert isinstance(coordinator, ZigguratCoordinator)
    assert coordinator.ieee == COORDINATOR_IEEE
    assert coordinator.node_desc is not None
    assert coordinator.node_desc.logical_type == zdo_t.LogicalType.Coordinator
    assert coordinator.node_desc.server_mask == 0x2C01
    assert 1 in coordinator.endpoints

    # The loaded settings were written back to the server and backed up locally
    assert server.configured.channel == 15
    assert server.configured.pan_id == t.PanId(0x1A2B)
    assert server.configured.network_key == NETWORK_KEY
    assert connected_app.backups[-1].network_info.pan_id == t.PanId(0x1A2B)

    # The network only starts once the state and tables have been loaded
    commands = [type(r) for r in server.requests]
    assert commands.index(p.Configure) < commands.index(p.StartNetwork)


async def test_start_network_applies_tunables(
    server: SyntheticZiggurat,
) -> None:
    """Tunables are a debug/experiment surface, applied once the network is up."""
    app = ControllerApplication(
        make_app_config(
            server.url,
            **{
                CONF_ZIGGURAT_CONFIG: {
                    CONF_TUNABLES: {"aps_ack_timeout": 5, "max_broadcast_jitter": 100}
                }
            },
        )
    )
    await app.connect()
    await app.start_network()

    assert [
        (bytes(s.name).decode(), int(s.value)) for s in server.sent(p.SetTunable)
    ] == [
        ("aps_ack_timeout", 5),
        ("max_broadcast_jitter", 100),
    ]
    # They are applied to a started network, not folded into the configuration
    commands = [type(r) for r in server.requests]
    assert commands.index(p.StartNetwork) < commands.index(p.SetTunable)

    await app.shutdown(db=False)


async def test_coordinator_device(app: ControllerApplication) -> None:
    coordinator = app.get_device(nwk=t.NWK(0x0000))
    coordinator.manufacturer = "ignored"
    coordinator.model = "ignored"
    assert coordinator.manufacturer == "Ziggurat"
    assert coordinator.model == "Coordinator"


async def test_load_network_info(
    connected_app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    app = connected_app

    async def not_configured(request: p.Request, request_id: int) -> None:
        raise StatusError(p.Status.NOT_CONFIGURED)

    # A stateless server with no network running and no local backup: no network
    server.handlers[p.RequestCommand.GET_NETWORK_INFO] = not_configured
    with pytest.raises(NetworkNotFormed):
        await app.load_network_info()

    # Unrelated errors propagate
    async def radio_error(request: p.Request, request_id: int) -> None:
        raise StatusError(p.Status.RADIO_ERROR)

    server.handlers[p.RequestCommand.GET_NETWORK_INFO] = radio_error
    with pytest.raises(DeliveryError, match="radio_error"):
        await app.load_network_info()

    # The server has a running network
    server.handlers[p.RequestCommand.GET_NETWORK_INFO] = server.on_get_network_info
    await app.load_network_info()
    assert app.state.node_info.ieee == COORDINATOR_IEEE
    assert app.state.network_info.channel == 15
    # zigpy mis-annotates the classmethod's `cls` as an instance
    assert app.state.network_info.channel_mask == t.Channels.from_channel_list([15])  # type: ignore[misc]
    assert app.state.network_info.network_key.key == NETWORK_KEY
    assert app.state.network_info.network_key.tx_counter == 1000
    # The APS outgoing frame counter lives on the TC link key by convention
    assert app.state.network_info.tc_link_key.tx_counter == 2000
    assert app.state.network_info.stack_specific == {}

    # A restarted, stateless server: the latest backup is restored with both frame
    # counters jumped past their stale values
    await app.write_network_info(
        network_info=app.state.network_info, node_info=app.state.node_info
    )
    counter = app.state.network_info.network_key.tx_counter
    aps_counter = app.state.network_info.tc_link_key.tx_counter
    server.handlers[p.RequestCommand.GET_NETWORK_INFO] = not_configured
    await app.load_network_info()
    margin = application_module.FRAME_COUNTER_RESTORE_MARGIN
    assert app.state.network_info.network_key.tx_counter == counter + margin
    assert app.state.network_info.tc_link_key.tx_counter == aps_counter + margin


@pytest.mark.parametrize(
    ("flavor", "expected"),
    [
        (p.TclkFlavorId.Z_STACK, {"zstack": {"tclk_seed": "ab" * 16}}),
        (p.TclkFlavorId.EZSP, {"ezsp": {"hashed_tclk": "ab" * 16}}),
    ],
)
async def test_load_network_info_tclk_seed(
    connected_app: ControllerApplication,
    server: SyntheticZiggurat,
    flavor: p.TclkFlavorId,
    expected: dict[str, Any],
) -> None:
    """A seed carried over from a microcontroller stack maps onto the stack_specific
    layout of the stack it came from."""
    server.network_state.has_tclk_seed = t.Bool(True)
    server.network_state.tclk_seed = t.KeyData(bytes.fromhex("ab" * 16))
    server.network_state.tclk_flavor = flavor

    await connected_app.load_network_info()
    assert connected_app.state.network_info.stack_specific == expected


async def test_load_network_info_tables(
    connected_app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    """The key table, children, address cache and route table each stream in as the
    events of their own scan request."""
    child_ieee = t.EUI64.convert("bb:bb:bb:bb:bb:bb:bb:bb")
    server.key_table = [
        p.KeyEntry(
            key=LINK_KEY,
            tx_counter=t.uint32_t(7),
            rx_counter=t.uint32_t(9),
            seq=t.uint8_t(1),
            partner_ieee=DEVICE_IEEE,
        )
    ]
    server.children = [
        p.ChildEntry(
            ieee=DEVICE_IEEE,
            nwk=DEVICE_NWK,
            rx_on_when_idle=t.uint1_t(0),
            device_type=p.ChildDeviceType.END_DEVICE,
            reserved=t.uint5_t(0),
        ),
        p.ChildEntry(
            ieee=child_ieee,
            nwk=t.NWK(0x1234),
            rx_on_when_idle=t.uint1_t(1),
            device_type=p.ChildDeviceType.ROUTER,
            reserved=t.uint5_t(0),
        ),
    ]
    server.address_cache = [
        p.AddressEntry(ieee=DEVICE_IEEE, nwk=DEVICE_NWK),
        p.AddressEntry(ieee=child_ieee, nwk=t.NWK(0x1234)),
    ]
    server.route_table = [
        p.RouteEntry(
            destination=DEVICE_NWK, next_hop=t.NWK(0x1234), path_cost=t.uint8_t(3)
        )
    ]

    await connected_app.load_network_info()
    network_info = connected_app.state.network_info

    assert network_info.key_table == [
        zigpy.state.Key(
            key=LINK_KEY,
            partner_ieee=DEVICE_IEEE,
            tx_counter=t.uint32_t(7),
            rx_counter=t.uint32_t(9),
            seq=t.uint8_t(1),
        )
    ]
    assert network_info.children == [DEVICE_IEEE, child_ieee]
    assert network_info.nwk_addresses == {
        DEVICE_IEEE: DEVICE_NWK,
        child_ieee: t.NWK(0x1234),
    }
    assert network_info.stack_specific == {
        "ziggurat": {
            "routes": [
                {
                    "destination": DEVICE_NWK,
                    "next_hop": t.NWK(0x1234),
                    "path_cost": 3,
                }
            ]
        }
    }


async def test_load_network_info_no_routes(
    connected_app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    # An empty route table adds no `ziggurat` section at all
    await connected_app.load_network_info()
    assert connected_app.state.network_info.stack_specific == {}


async def test_write_network_info(
    connected_app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    app = connected_app
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
    zstack = server.configured
    assert zstack.has_tclk_seed
    assert bytes(zstack.tclk_seed).hex() == "cd" * 16
    assert zstack.tclk_flavor == p.TclkFlavorId.Z_STACK
    assert list(server.sent(p.LoadKeyTable)[-1].entries) == [
        p.KeyEntry(
            key=LINK_KEY,
            tx_counter=t.uint32_t(0),
            rx_counter=t.uint32_t(0),
            seq=t.uint8_t(0),
            partner_ieee=DEVICE_IEEE,
        )
    ]

    # An ezsp seed likewise
    await app.write_network_info(
        network_info=network_info.replace(
            stack_specific={"ezsp": {"hashed_tclk": "ef" * 16}}
        ),
        node_info=node_info,
    )
    ezsp = server.configured
    assert bytes(ezsp.tclk_seed).hex() == "ef" * 16
    assert ezsp.tclk_flavor == p.TclkFlavorId.EZSP

    # With no seed at all one is generated, so unique link keys can still be issued
    await app.write_network_info(network_info=network_info, node_info=node_info)
    assert not server.configured.has_tclk_seed
    assert bytes(server.configured.tclk_seed) != bytes(16)

    # When zigpy forms a fresh network it leaves the IEEE address unspecified,
    # deferring to the radio's hardware address
    await app.write_network_info(
        network_info=network_info,
        node_info=node_info.replace(ieee=t.EUI64.UNKNOWN),  # type: ignore[attr-defined]
    )
    assert server.sent(p.GetHwAddress)
    assert server.configured.ieee_address == server.hw_address
    assert app.state.node_info.ieee == server.hw_address

    # Every write is recorded as a backup: zigpy's database is the network's NVRAM
    assert app.backups[-1].node_info.ieee == server.hw_address


async def test_write_network_info_default_tx_power(
    connected_app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await connected_app.load_network_info()

    # `None` means "pick automatically", which becomes a safe default
    await connected_app.write_network_info(
        network_info=connected_app.state.network_info.replace(tx_power=None),
        node_info=connected_app.state.node_info,
    )
    assert server.configured.tx_power == 8


async def test_write_network_info_restores_children(
    connected_app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    """Children are reloaded so the stack can route to sleepy end devices before
    they check in again."""
    app = connected_app
    await app.load_network_info()

    unknown_ieee = t.EUI64.convert("cc:cc:cc:cc:cc:cc:cc:cc")
    children = [t.EUI64([n] * 8) for n in range(1, 16)]
    nwk_addresses = {ieee: t.NWK(0x1000 + n) for n, ieee in enumerate(children)}

    await app.write_network_info(
        network_info=app.state.network_info.replace(
            # The unknown child has no address, so it cannot be loaded
            children=[*children, unknown_ieee],
            nwk_addresses=nwk_addresses,
        ),
        node_info=app.state.node_info,
    )

    # 15 children in batches of 12
    loads = server.sent(p.LoadChildren)
    assert [len(load.entries) for load in loads] == [12, 3]

    entries = [entry for load in loads for entry in load.entries]
    assert [entry.ieee for entry in entries] == children
    assert [entry.nwk for entry in entries] == list(nwk_addresses.values())
    # The backup carries no capability, so the device type is unknown
    assert {entry.device_type for entry in entries} == {p.ChildDeviceType.UNKNOWN}


async def test_permit_ncp(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.permit_ncp(42)
    permit = server.sent(p.PermitJoins)[-1]
    assert permit.duration == 42
    # Permitting on the coordinator opens its own beacon for direct joins
    assert permit.accept_direct_joins


async def test_permit_steered_to_router(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    device = add_initialized_device(app)
    permits: list[int] = []

    # The unicast Mgmt_Permit_Joining_req awaits a ZDO reply no synthetic device sends
    async def permit(duration: int, *args: Any, **kwargs: Any) -> None:
        permits.append(duration)

    device.zdo.permit = permit  # type: ignore[method-assign,assignment]

    await app.permit(time_s=30, node=DEVICE_IEEE)

    assert permits == [30]
    # The trust center window opens without advertising the coordinator as a parent
    permit_joins = server.sent(p.PermitJoins)[-1]
    assert permit_joins.duration == 30
    assert not permit_joins.accept_direct_joins


async def test_permit_with_link_key(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.permit_with_link_key(node=DEVICE_IEEE, link_key=LINK_KEY, time_s=12)

    provisional = server.sent(p.SetProvisionalKey)[-1]
    assert provisional.ieee == DEVICE_IEEE
    assert provisional.key == LINK_KEY

    # `super().permit()` broadcasts Mgmt_Permit_Joining_req and calls `permit_ncp`
    broadcast = server.sent(p.SendBroadcast)[-1]
    assert broadcast.cluster_id == zdo_t.ZDOCmd.Mgmt_Permit_Joining_req
    assert server.sent(p.PermitJoins)[-1].duration == 12


async def test_permit_node_conversion_and_all_routers(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.permit(time_s=20, node="aa:bb:cc:dd:11:22:33:44")
    steered = server.sent(p.PermitJoins)[-1]
    assert steered.duration == 20
    assert not steered.accept_direct_joins

    # No node falls through to the base broadcast, which opens the coordinator too
    await app.permit(time_s=30)
    opened = server.sent(p.PermitJoins)[-1]
    assert opened.duration == 30
    assert opened.accept_direct_joins


async def test_energy_scan(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    rssis = iter([-90, -80, -70])

    async def scan(request: p.EnergyScan, request_id: int) -> None:
        rssi = next(rssis)
        for channel in request.channels:
            await server.send_event(
                p.RequestCommand.ENERGY_SCAN,
                request_id,
                p.EnergyResult(channel=t.uint8_t(channel), rssi=t.int8s(rssi)),
            )

    server.handlers[p.RequestCommand.ENERGY_SCAN] = scan

    energies = await app.energy_scan(
        # zigpy mis-annotates the classmethod's `cls` as an instance
        channels=t.Channels.from_channel_list([11, 15]),  # type: ignore[misc]
        duration_exp=2,
        count=3,
    )

    scans = server.sent(p.EnergyScan)
    assert len(scans) == 3
    assert list(scans[0].channels) == [11, 15]
    # 0.016 ms/symbol * 960 symbols * (2**2 + 1)
    assert scans[0].duration_per_channel_ms == 77

    # Each channel's RSSI readings are averaged, then mapped onto 0-255
    assert energies == {
        11: pytest.approx(map_rssi_to_energy(-80.0)),
        15: pytest.approx(map_rssi_to_energy(-80.0)),
    }
    assert 0 < energies[11] < 255


async def test_network_scan(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    server.beacons = [
        p.Beacon(
            channel=t.uint8_t(11),
            source=t.NWK(0x0000),
            pan_id=t.PanId(0x1A2B),
            extended_pan_id=t.ExtendedPanId(t.EUI64.convert("aa:bb:cc:dd:ee:ff:00:11")),
            permit_joining=t.uint1_t(1),
            router_capacity=t.uint1_t(1),
            end_device_capacity=t.uint1_t(1),
            reserved=t.uint5_t(0),
            stack_profile=t.uint8_t(2),
            protocol_version=t.uint8_t(2),
            device_depth=t.uint8_t(0),
            update_id=t.uint8_t(0),
            lqi=t.uint8_t(200),
            rssi=t.int8s(-60),
        ),
        # A beacon whose MAC source was not a short address
        p.Beacon(
            channel=t.uint8_t(15),
            source=t.NWK(0xFFFF),
            pan_id=t.PanId(0x4C5D),
            extended_pan_id=t.ExtendedPanId(t.EUI64.convert("01:02:03:04:05:06:07:08")),
            permit_joining=t.uint1_t(0),
            router_capacity=t.uint1_t(0),
            end_device_capacity=t.uint1_t(0),
            reserved=t.uint5_t(0),
            stack_profile=t.uint8_t(2),
            protocol_version=t.uint8_t(2),
            device_depth=t.uint8_t(2),
            update_id=t.uint8_t(1),
            lqi=t.uint8_t(120),
            rssi=t.int8s(-80),
        ),
    ]

    found = [
        beacon
        async for beacon in app.network_scan(
            # zigpy mis-annotates the classmethod's `cls` as an instance
            channels=t.Channels.from_channel_list([11, 15]),  # type: ignore[misc]
            duration_exp=2,
        )
    ]

    scans = server.sent(p.NetworkScan)
    assert len(scans) == 1
    assert list(scans[0].channels) == [11, 15]
    # 0.016 ms/symbol * 960 symbols * (2**2 + 1)
    assert scans[0].duration_per_channel_ms == 77

    assert found == [
        t.NetworkBeacon(
            pan_id=t.PanId(0x1A2B),
            extended_pan_id=t.ExtendedPanId(t.EUI64.convert("aa:bb:cc:dd:ee:ff:00:11")),
            channel=t.uint8_t(11),
            permit_joining=True,
            stack_profile=t.uint8_t(2),
            nwk_update_id=t.uint8_t(0),
            lqi=t.uint8_t(200),
            src=t.NWK(0x0000),
            rssi=t.int8s(-60),
            depth=t.uint8_t(0),
            router_capacity=True,
            device_capacity=True,
            protocol_version=t.uint8_t(2),
        ),
        t.NetworkBeacon(
            pan_id=t.PanId(0x4C5D),
            extended_pan_id=t.ExtendedPanId(t.EUI64.convert("01:02:03:04:05:06:07:08")),
            channel=t.uint8_t(15),
            permit_joining=False,
            stack_profile=t.uint8_t(2),
            nwk_update_id=t.uint8_t(1),
            lqi=t.uint8_t(120),
            src=None,
            rssi=t.int8s(-80),
            depth=t.uint8_t(2),
            router_capacity=False,
            device_capacity=False,
            protocol_version=t.uint8_t(2),
        ),
    ]


async def test_send_packet_unicast(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    app.add_device(DEVICE_IEEE, DEVICE_NWK)

    await app.send_packet(
        aps_packet(
            t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK),
            tx_options=t.TransmitOptions.ACK,
        )
    )

    send = server.sent(p.SendUnicast)[-1]
    assert send.destination == DEVICE_NWK
    assert send.has_eui64
    # The link key is selected by EUI64, resolved from the device registry
    assert send.destination_eui64 == DEVICE_IEEE
    assert send.aps_ack
    assert not send.aps_encryption
    assert not send.sleepy_destination
    assert send.profile_id == 0x0104
    assert send.aps_seq == 33
    assert send.radius == 30
    assert send.priority == 0
    assert bytes(send.asdu) == b"\x01\x02\x03"


async def test_send_packet_unicast_by_ieee(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.send_packet(
        aps_packet(t.AddrModeAddress(addr_mode=t.AddrMode.IEEE, address=DEVICE_IEEE))
    )

    send = server.sent(p.SendUnicast)[-1]
    # 0xFFFE stands in for "no short address": the server resolves the EUI64
    assert send.destination == t.NWK(0xFFFE)
    assert send.has_eui64
    assert send.destination_eui64 == DEVICE_IEEE


async def test_send_packet_unicast_unknown_device(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    # An address with no device behind it carries no EUI64 at all
    await app.send_packet(
        aps_packet(t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x9999)))
    )

    send = server.sent(p.SendUnicast)[-1]
    assert send.destination == t.NWK(0x9999)
    assert not send.has_eui64


async def test_send_packet_encrypted(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    app.add_device(DEVICE_IEEE, DEVICE_NWK)

    await app.send_packet(
        aps_packet(
            t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK),
            tx_options=t.TransmitOptions.ACK | t.TransmitOptions.APS_Encryption,
        )
    )

    send = server.sent(p.SendUnicast)[-1]
    assert send.aps_encryption
    assert send.destination_eui64 == DEVICE_IEEE


async def test_send_packet_encrypted_without_eui64(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    """APS encryption selects the link key by EUI64, so a destination that resolves
    to no device cannot be encrypted."""
    with pytest.raises(DeliveryError, match="without a destination EUI64"):
        await app.send_packet(
            aps_packet(
                t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x9999)),
                tx_options=t.TransmitOptions.APS_Encryption,
            )
        )

    assert server.sent(p.SendUnicast) == []


async def test_send_packet_broadcast(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.send_packet(
        aps_packet(
            t.AddrModeAddress(
                addr_mode=t.AddrMode.Broadcast,
                address=t.BroadcastAddress.ALL_ROUTERS_AND_COORDINATOR,
            ),
            dst_ep=255,
        )
    )

    send = server.sent(p.SendBroadcast)[-1]
    assert send.destination == t.NWK(0xFFFC)
    assert send.dst_ep == 255
    assert send.aps_seq == 33
    assert bytes(send.asdu) == b"\x01\x02\x03"


async def test_send_packet_groupcast(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.send_packet(
        aps_packet(
            t.AddrModeAddress(addr_mode=t.AddrMode.Group, address=t.Group(0x0002)),
            dst_ep=255,
        )
    )

    send = server.sent(p.SendGroupcast)[-1]
    assert send.group_id == 0x0002
    assert send.aps_seq == 33
    assert bytes(send.asdu) == b"\x01\x02\x03"


async def test_send_packet_sleepy_destination(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.send_packet(
        t.ZigbeePacket(
            src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x0000)),
            src_ep=t.uint8_t(1),
            dst=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK),
            dst_ep=t.uint8_t(1),
            tsn=t.uint8_t(1),
            profile_id=t.uint16_t(0x0104),
            cluster_id=t.uint16_t(0x0006),
            data=t.SerializableBytes(b"\x01"),
            extended_timeout=True,
            priority=t.PacketPriority.HIGH,
            radius=t.uint8_t(5),
        )
    )

    send = server.sent(p.SendUnicast)[-1]
    assert send.sleepy_destination
    assert int(send.priority) == int(t.PacketPriority.HIGH)
    assert send.radius == 5


@pytest.mark.parametrize(
    ("relays", "source_routing", "route", "next_hop", "expected_relays"),
    [
        # Nothing is known about the route: the stack decides
        (None, False, p.RouteControl.STACK_DECIDES, None, None),
        # A direct child is one hop away
        ([], False, p.RouteControl.HINT_NEXT_HOP, DEVICE_NWK, None),
        # zigpy stores relays in received order, so the next hop is the last one
        ([t.NWK(0x1111)], False, p.RouteControl.HINT_NEXT_HOP, t.NWK(0x1111), None),
        (
            [t.NWK(0x1111), t.NWK(0x2222)],
            False,
            p.RouteControl.HINT_NEXT_HOP,
            t.NWK(0x2222),
            None,
        ),
        # With source routing enabled the whole path is supplied instead
        (
            [t.NWK(0x1111), t.NWK(0x2222)],
            True,
            p.RouteControl.HINT_SOURCE_ROUTE,
            None,
            [t.NWK(0x2222), t.NWK(0x1111)],
        ),
        # A device with no known relays gives source routing nothing to work with
        (None, True, p.RouteControl.STACK_DECIDES, None, None),
    ],
)
async def test_send_packet_route_hints(
    server: SyntheticZiggurat,
    relays: list[t.NWK] | None,
    source_routing: bool,
    route: p.RouteControl,
    next_hop: t.NWK | None,
    expected_relays: list[t.NWK] | None,
) -> None:
    """Within the startup window the app hints the route it already knows, to keep
    the stack from rediscovering every path at once."""
    app = ControllerApplication(
        make_app_config(
            server.url, **{zigpy.config.CONF_SOURCE_ROUTING: source_routing}
        )
    )
    await app.connect()
    await app.start_network()

    device = app.add_device(DEVICE_IEEE, DEVICE_NWK)
    device.relays = t.Relays(relays) if relays is not None else None

    await app.send_packet(
        aps_packet(t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK))
    )

    send = server.sent(p.SendUnicast)[-1]
    assert send.route == route
    assert send.next_hop == next_hop
    if expected_relays is None:
        assert send.relays is None
    else:
        assert send.relays is not None
        assert list(send.relays.relays) == expected_relays

    await app.shutdown(db=False)


async def test_send_packet_no_route_hints_after_startup(
    app: ControllerApplication,
    server: SyntheticZiggurat,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Past the startup window the stack's own routing table is authoritative."""
    monkeypatch.setattr(application_module, "ROUTE_HINT_DURATION", timedelta(0))

    device = app.add_device(DEVICE_IEEE, DEVICE_NWK)
    device.relays = t.Relays([t.NWK(0x1111)])

    await app.send_packet(
        aps_packet(t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK))
    )

    send = server.sent(p.SendUnicast)[-1]
    assert send.route == p.RouteControl.STACK_DECIDES
    assert send.next_hop is None


async def test_send_packet_delivery_failure(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    async def fail(request: p.SendUnicast, request_id: int) -> None:
        raise StatusError(p.Status.RADIO_ERROR)

    server.handlers[p.RequestCommand.SEND_UNICAST] = fail

    with pytest.raises(DeliveryError, match="radio_error"):
        await app.send_packet(
            aps_packet(t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK))
        )


async def test_send_packet_confirm_failure(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    async def no_route(request: p.SendUnicast, request_id: int) -> None:
        await server.send_notification(
            p.SendConfirm(status=p.SendStatus.ROUTE_DISCOVERY_TIMEOUT), request_id
        )

    server.handlers[p.RequestCommand.SEND_UNICAST] = no_route

    with pytest.raises(DeliveryError, match="ROUTE_DISCOVERY_TIMEOUT"):
        await app.send_packet(
            aps_packet(t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK))
        )


async def test_add_endpoint(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    requests_before = len(server.requests)

    await app.add_endpoint(
        zdo_t.SimpleDescriptor(
            endpoint=12,
            profile=0x0104,
            device_type=0x0008,
            device_version=1,
            input_clusters=[0x0006],
            output_clusters=[0x0019],
        )
    )

    endpoint = app.get_device(nwk=t.NWK(0x0000)).endpoints[12]
    assert isinstance(endpoint, zigpy.endpoint.Endpoint)
    assert endpoint.profile_id == 0x0104
    assert 0x0006 in endpoint.in_clusters
    assert 0x0019 in endpoint.out_clusters

    # The endpoint exists only on the static coordinator device: nothing is sent
    assert len(server.requests) == requests_before


async def test_force_remove(app: ControllerApplication) -> None:
    # A no-op: the server keeps no device registry of its own
    await app.force_remove(app.get_device(nwk=t.NWK(0x0000)))


async def test_reset_network_info(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.reset_network_info()
    assert server.sent(p.Shutdown)


async def test_watchdog_feed(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app._watchdog_feed()
    assert isinstance(server.requests[-1], p.GetFirmwareInfo)


async def test_move_network_to_channel(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app._move_network_to_channel(new_channel=20, new_nwk_update_id=1)

    # The update id goes first so no beacon on the new channel ever advertises the
    # old network instance
    commands = [type(r) for r in server.requests]
    assert commands.index(p.SetNwkUpdateId) < commands.index(p.SetChannel)
    assert server.sent(p.SetNwkUpdateId)[-1].nwk_update_id == 1
    assert server.sent(p.SetChannel)[-1].channel == 20


async def test_packet_received(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    add_initialized_device(app)
    our_nwk = t.NWK(0x0000).serialize()

    # Node_Desc_req: answered locally, advertising the trust center's revision
    app.packet_received(zdo_packet(zdo_t.ZDOCmd.Node_Desc_req, b"\x10" + our_nwk))
    reply = await server.wait_for(p.SendUnicast)
    assert reply.cluster_id == zdo_t.ZDOCmd.Node_Desc_rsp
    assert reply.destination == DEVICE_NWK
    assert reply.asdu[0] == 0x10
    assert reply.asdu[1] == zdo_t.Status.SUCCESS

    # Active_EP_req
    app.packet_received(zdo_packet(zdo_t.ZDOCmd.Active_EP_req, b"\x11" + our_nwk))
    reply = await server.wait_for(p.SendUnicast, count=2)
    assert reply.cluster_id == zdo_t.ZDOCmd.Active_EP_rsp
    assert reply.asdu[1] == zdo_t.Status.SUCCESS
    endpoint_count = reply.asdu[4]
    assert 1 in reply.asdu[5 : 5 + endpoint_count]

    # Simple_Desc_req for a registered endpoint
    app.packet_received(
        zdo_packet(zdo_t.ZDOCmd.Simple_Desc_req, b"\x12" + our_nwk + b"\x01")
    )
    reply = await server.wait_for(p.SendUnicast, count=3)
    assert reply.cluster_id == zdo_t.ZDOCmd.Simple_Desc_rsp
    assert reply.asdu[1] == zdo_t.Status.SUCCESS

    # Requests that are not answered locally. zigpy may originate its own requests
    # (e.g. IEEE_addr_req for an unknown sender), so only count ZDO responses.
    def zdo_replies() -> list[p.SendUnicast]:
        return [r for r in server.sent(p.SendUnicast) if r.cluster_id & 0x8000]

    replies_before = len(zdo_replies())
    for packet in [
        # The ZDO endpoint itself
        zdo_packet(zdo_t.ZDOCmd.Simple_Desc_req, b"\x13" + our_nwk + b"\x00"),
        # An unregistered endpoint
        zdo_packet(zdo_t.ZDOCmd.Simple_Desc_req, b"\x14" + our_nwk + b"\x63"),
        # Another node's descriptor
        zdo_packet(zdo_t.ZDOCmd.Node_Desc_req, b"\x15" + t.NWK(0x1234).serialize()),
        # Not a request we answer
        zdo_packet(zdo_t.ZDOCmd.Mgmt_Lqi_req, b"\x16\x00"),
        # A truncated payload
        zdo_packet(zdo_t.ZDOCmd.Node_Desc_req, b"\x17"),
        # An unknown cluster
        zdo_packet(0xFF00, b"\x18"),
        # An unknown sender
        zdo_packet(zdo_t.ZDOCmd.Node_Desc_req, b"\x19" + our_nwk, src=t.NWK(0xDEAD)),
    ]:
        app.packet_received(packet)

    await asyncio.sleep(0.05)
    assert len(zdo_replies()) == replies_before


async def test_packet_received_aqara_node_desc_override(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    our_nwk = t.NWK(0x0000).serialize()
    coordinator = app.get_device(nwk=t.NWK(0x0000))
    assert coordinator.node_desc is not None
    assert coordinator.node_desc.manufacturer_code == application_module.DEFAULT_MFG_ID

    def reported_mfg_code(reply: p.SendUnicast) -> int:
        node_desc, _ = zdo_t.NodeDescriptor.deserialize(bytes(reply.asdu[4:]))
        return node_desc.manufacturer_code

    # A Lumi/Aqara device is answered with the Xiaomi manufacturer code so it pairs
    aqara_nwk = t.NWK(0x1234)
    add_initialized_device(
        app, ieee=t.EUI64.convert("54:ef:44:00:00:00:00:01"), nwk=aqara_nwk
    )
    app.packet_received(
        zdo_packet(zdo_t.ZDOCmd.Node_Desc_req, b"\x20" + our_nwk, src=aqara_nwk)
    )
    reply = await server.wait_for(p.SendUnicast)
    assert reply.cluster_id == zdo_t.ZDOCmd.Node_Desc_rsp
    assert reported_mfg_code(reply) == 0x115F

    # The coordinator's stored descriptor is untouched; other devices see the default
    assert coordinator.node_desc.manufacturer_code == application_module.DEFAULT_MFG_ID
    add_initialized_device(app)
    app.packet_received(
        zdo_packet(zdo_t.ZDOCmd.Node_Desc_req, b"\x21" + our_nwk, src=DEVICE_NWK)
    )
    reply = await server.wait_for(p.SendUnicast, count=2)
    assert reported_mfg_code(reply) == application_module.DEFAULT_MFG_ID


async def test_on_notification_received_aps(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    add_initialized_device(app)

    # A ZDO request arriving over the wire is answered end-to-end
    await server.send_notification(
        p.ReceivedAps(
            source=DEVICE_NWK,
            destination=t.NWK(0x0000),
            has_group=t.Bool(False),
            group=t.uint16_t(0),
            profile_id=t.uint16_t(0x0000),
            cluster_id=t.uint16_t(zdo_t.ZDOCmd.Node_Desc_req),
            src_ep=t.uint8_t(0),
            dst_ep=t.uint8_t(0),
            lqi=t.uint8_t(255),
            rssi=t.int8s(-40),
            data=t.LongOctetString(b"\x77" + t.NWK(0x0000).serialize()),
        )
    )
    reply = await server.wait_for(p.SendUnicast)
    assert reply.cluster_id == zdo_t.ZDOCmd.Node_Desc_rsp
    assert reply.asdu[0] == 0x77


@pytest.mark.parametrize(
    ("destination", "group", "expected_dst"),
    [
        (
            t.NWK(0x0000),
            2,
            t.AddrModeAddress(addr_mode=t.AddrMode.Group, address=t.Group(2)),
        ),
        (
            t.NWK(0xFFFD),
            None,
            t.AddrModeAddress(
                addr_mode=t.AddrMode.Broadcast, address=t.BroadcastAddress(0xFFFD)
            ),
        ),
        (
            t.NWK(0x0000),
            None,
            t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x0000)),
        ),
    ],
)
async def test_on_notification_received_aps_address_modes(
    server: SyntheticZiggurat,
    destination: t.NWK,
    group: int | None,
    expected_dst: t.AddrModeAddress,
) -> None:
    """A group id, a broadcast address and a plain network address each decode into
    their own zigpy address mode."""
    app = RecordingApplication(make_app_config(server.url))
    await app.connect()
    await app.start_network()
    add_initialized_device(app)

    await server.send_notification(
        p.ReceivedAps(
            source=DEVICE_NWK,
            destination=destination,
            has_group=t.Bool(group is not None),
            group=t.uint16_t(group or 0),
            profile_id=t.uint16_t(0x0104),
            cluster_id=t.uint16_t(0x0006),
            src_ep=t.uint8_t(1),
            dst_ep=t.uint8_t(1),
            lqi=t.uint8_t(200),
            rssi=t.int8s(-70),
            # A ZCL Read Attributes of the OnOff attribute
            data=t.LongOctetString(b"\x00\x01\x00\x00\x00"),
        )
    )
    await flush(app)

    assert len(app.packets) == 1
    packet = app.packets[0]
    assert packet.dst == expected_dst
    assert packet.src == t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK)
    assert packet.lqi == 200
    assert packet.rssi == -70
    assert packet.data.serialize() == b"\x00\x01\x00\x00\x00"

    await app.shutdown(db=False)


async def test_on_notification_frame_counter(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await server.send_notification(p.FrameCounter(frame_counter=t.uint32_t(123456)))
    await flush(app)

    assert app.state.network_info.network_key.tx_counter == 123456
    assert app.backups[-1].network_info.network_key.tx_counter == 123456


async def test_on_notification_aps_frame_counter(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    """The APS outgoing frame counter lives on the TC link key by convention."""
    assert app.state.network_info.tc_link_key.tx_counter != 654321

    await server.send_notification(p.ApsFrameCounter(frame_counter=t.uint32_t(654321)))
    await flush(app)

    assert app.state.network_info.tc_link_key.tx_counter == 654321
    assert app.backups[-1].network_info.tc_link_key.tx_counter == 654321


async def test_on_notification_route_record(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    """A route record reveals the path a device's frames took to reach us, which
    zigpy stores as the relay list to send back along."""
    device = add_initialized_device(app)
    assert device.relays is None

    relays = [t.NWK(0x1111), t.NWK(0x2222)]
    await server.send_notification(
        p.RouteRecord(destination=DEVICE_NWK, relays=t.LVList[t.NWK, t.uint8_t](relays))
    )
    await flush(app)

    assert device.relays == relays


async def test_on_notification_link_key(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    new_key = t.KeyData.convert("ff:ee:dd:cc:bb:aa:99:88:77:66:55:44:33:22:11:00")

    await server.send_notification(p.LinkKey(ieee=DEVICE_IEEE, key=LINK_KEY))
    await flush(app)
    assert app.state.network_info.key_table == [
        zigpy.state.Key(key=LINK_KEY, partner_ieee=DEVICE_IEEE)
    ]

    # A renegotiated key replaces the previous entry instead of duplicating it
    await server.send_notification(p.LinkKey(ieee=DEVICE_IEEE, key=new_key))
    await flush(app)
    assert app.state.network_info.key_table == [
        zigpy.state.Key(key=new_key, partner_ieee=DEVICE_IEEE)
    ]
    assert app.backups[-1].network_info.key_table == app.state.network_info.key_table


async def test_on_notification_device_joined(
    app: ControllerApplication,
    server: SyntheticZiggurat,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(application_module, "DEVICE_JOIN_MAX_DELAY", 0.05)

    def joined(nwk: t.NWK, ieee: t.EUI64) -> p.DeviceJoined:
        return p.DeviceJoined(
            nwk=nwk,
            ieee=ieee,
            parent=t.NWK(0x0000),
            rx_on_when_idle=t.uint1_t(1),
            device_type=p.ChildDeviceType.ROUTER,
            reserved=t.uint5_t(0),
        )

    # A brand-new device only joins after the announcement grace period
    await server.send_notification(joined(DEVICE_NWK, DEVICE_IEEE))
    await flush(app)
    with pytest.raises(KeyError):
        app.get_device(ieee=DEVICE_IEEE)
    await asyncio.sleep(0.1)
    assert app.get_device(ieee=DEVICE_IEEE).nwk == DEVICE_NWK

    # A device that announced itself within the grace period is not joined again
    ieee2 = t.EUI64.convert("bb:bb:bb:bb:bb:bb:bb:bb")
    await server.send_notification(joined(t.NWK(0x5678), ieee2))
    await flush(app)
    device2 = app.add_device(ieee2, t.NWK(0x5678))  # the announcement's effect
    await asyncio.sleep(0.1)
    assert app.get_device(ieee=ieee2) is device2

    # A known, initialized device rejoining with a new address skips the delay
    ieee3 = t.EUI64.convert("cc:cc:cc:cc:cc:cc:cc:cc")
    device3 = app.add_device(ieee3, t.NWK(0x9999))
    device3.node_desc = app.get_device(nwk=t.NWK(0x0000)).node_desc
    device3.status = zigpy.device.Status.ENDPOINTS_INIT
    await server.send_notification(joined(t.NWK(0x9AAA), ieee3))
    await flush(app)
    assert app.get_device(ieee=ieee3).nwk == t.NWK(0x9AAA)


async def test_on_notification_device_left(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    left: list[zigpy.device.Device] = []

    class Listener:
        def device_left(self, device: zigpy.device.Device) -> None:
            left.append(device)

    app.add_listener(Listener())
    device = app.add_device(DEVICE_IEEE, DEVICE_NWK)

    def device_left(
        nwk: t.NWK, ieee: t.EUI64 | None, reason: p.LeaveReason
    ) -> p.DeviceLeft:
        return p.DeviceLeft(
            nwk=nwk,
            has_ieee=t.uint1_t(ieee is not None),
            rejoin=t.uint1_t(0),
            has_router_ieee=t.uint1_t(0),
            reserved=t.uint5_t(0),
            ieee=ieee if ieee is not None else t.EUI64([0] * 8),
            reason=reason,
            router=t.NWK(0xFFFF),
            router_ieee=t.EUI64([0] * 8),
        )

    # The device announced its own departure
    await server.send_notification(
        device_left(DEVICE_NWK, DEVICE_IEEE, p.LeaveReason.ANNOUNCED)
    )
    await flush(app)
    assert left == [device]

    # A parent router relayed the leave; the IEEE is resolved through the registry
    await server.send_notification(
        device_left(DEVICE_NWK, None, p.LeaveReason.ROUTER_REPORTED)
    )
    await flush(app)
    assert left == [device, device]

    # An entirely unknown device is dropped
    await server.send_notification(
        device_left(t.NWK(0xBEEF), None, p.LeaveReason.KEEPALIVE_TIMEOUT)
    )
    await flush(app)
    assert left == [device, device]


async def test_on_notification_aps_decryption_failure(
    app: ControllerApplication,
    server: SyntheticZiggurat,
    caplog: pytest.LogCaptureFixture,
) -> None:
    with caplog.at_level(logging.WARNING):
        await server.send_notification(
            p.ApsDecryptFailure(
                source=t.NWK(0x1234),
                source_ieee=DEVICE_IEEE,
                frame_counter=t.uint32_t(42),
                key_id=p.KeyId.KEY_TRANSPORT,
            )
        )
        await flush(app)

    assert "Could not decrypt an APS command" in caplog.text
    assert str(DEVICE_IEEE) in caplog.text


async def test_connection_lost(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    lost: list[BaseException | None] = []

    class Listener:
        def connection_lost(self, exc: BaseException | None) -> None:
            lost.append(exc)

    app.add_listener(Listener())
    await server.ws.close()

    async with asyncio.timeout(1):
        while not lost:
            await asyncio.sleep(0.01)

    assert lost == [None]


async def test_packet_capture(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    server.captured_packets = [
        p.CapturedPacket(
            channel=t.uint8_t(15),
            rssi=t.int8s(-80),
            lqi=t.uint8_t(200),
            psdu=t.LongOctetString(b"\xaa\xbb\xcc"),
        )
    ]

    packets = [packet async for packet in app.packet_capture(15)]

    assert len(packets) == 1
    assert packets[0].channel == 15
    assert packets[0].rssi == -80
    assert packets[0].lqi == 200
    assert packets[0].data == b"\xaa\xbb\xcc"
    assert server.sent(p.PacketCapture)[0].channel == 15


async def test_packet_capture_change_channel(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.packet_capture_change_channel(20)

    assert server.sent(p.PacketCaptureChannel)[0].channel == 20


def test_max_concurrent_requests() -> None:
    assert application_module._max_concurrent_requests("ws://host/") == 128
    assert application_module._max_concurrent_requests("ws+unix:///run/z.sock") == 128
    assert application_module._max_concurrent_requests("/dev/ttyUSB0") == 32
