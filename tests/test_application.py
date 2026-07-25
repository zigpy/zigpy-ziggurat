"""One async test per public zigpy method of `ControllerApplication`, all running
against the synthetic websocket server."""

import asyncio
import logging
import os
from typing import Any
from unittest.mock import AsyncMock

from aiohttp import web
import pytest
import zigpy.device
import zigpy.endpoint
from zigpy.exceptions import DeliveryError, NetworkNotFormed
import zigpy.state
import zigpy.types as t
import zigpy.zdo.types as zdo_t

from tests.common import (
    COORDINATOR_IEEE,
    NETWORK_KEY,
    RpcError,
    SyntheticZiggurat,
    app,
    connected_app,
    make_app_config,
    server,
)
from zigpy_ziggurat.zigbee import application as application_module, legacy as commands
from zigpy_ziggurat.zigbee.application import (
    ControllerApplication,
    ZigguratCoordinator,
    map_rssi_to_energy,
)

DEVICE_IEEE = t.EUI64.convert("aa:aa:aa:aa:aa:aa:aa:aa")
DEVICE_NWK = t.NWK(0xAB12)
LINK_KEY = t.KeyData.convert("00:11:22:33:44:55:66:77:88:99:aa:bb:cc:dd:ee:ff")


async def flush(app: ControllerApplication) -> None:
    """Round-trip a request: the websocket is ordered, so by the time the response
    arrives every previously sent notification has been processed."""
    await app._watchdog_feed()


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


async def test_connect(
    connected_app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    assert server.connections == 1

    # Requests round-trip over the socket
    await connected_app.permit_ncp(1)
    assert server.sent(commands.PermitJoins)[-1].duration == 1


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
    assert ziggurat.sent(commands.PermitJoins)[-1].duration == 2

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

    async def not_configured(
        command: commands.GetNetworkInfo, request_id: int
    ) -> commands.NetworkInfo:
        raise RpcError("not_configured", "no stack is running")

    # A stateless server with no network running and no local backup: no network
    server.handlers["get_network_info"] = not_configured
    with pytest.raises(NetworkNotFormed):
        await app.load_network_info()

    # Unrelated errors propagate. The unknown JSON code has no binary status, so
    # it degrades to a generic invalid-request; the detail goes to the log.
    async def serial_error(
        command: commands.GetNetworkInfo, request_id: int
    ) -> commands.NetworkInfo:
        raise RpcError("serial_port_error", "it burned down")

    server.handlers["get_network_info"] = serial_error
    with pytest.raises(DeliveryError, match="invalid_request"):
        await app.load_network_info()

    # The server has a running network
    server.handlers["get_network_info"] = server.on_get_network_info
    await app.load_network_info()
    assert app.state.node_info.ieee == COORDINATOR_IEEE
    assert app.state.network_info.channel == 15
    # zigpy mis-annotates the classmethod's `cls` as an instance
    assert app.state.network_info.channel_mask == t.Channels.from_channel_list([15])  # type: ignore[misc]
    assert app.state.network_info.network_key.key == NETWORK_KEY
    assert app.state.network_info.stack_specific == {}

    # TCLK seeds map to the stack_specific layout of their source stack
    server.network_info.tclk_seed = "ab" * 16
    server.network_info.tclk_flavor = "zstack"
    await app.load_network_info()
    assert app.state.network_info.stack_specific == {"zstack": {"tclk_seed": "ab" * 16}}

    server.network_info.tclk_flavor = "ezsp"
    await app.load_network_info()
    assert app.state.network_info.stack_specific == {"ezsp": {"hashed_tclk": "ab" * 16}}

    # Negotiated link keys come back as the key table
    server.network_info.key_table = [
        commands.KeyTableEntry(partner_ieee=DEVICE_IEEE, key=LINK_KEY)
    ]
    await app.load_network_info()
    assert app.state.network_info.key_table == [
        zigpy.state.Key(key=LINK_KEY, partner_ieee=DEVICE_IEEE)
    ]

    # A restarted, stateless server: the latest backup is restored with the frame
    # counter jumped past the stale value
    await app.write_network_info(
        network_info=app.state.network_info, node_info=app.state.node_info
    )
    counter = app.state.network_info.network_key.tx_counter
    aps_counter = app.state.network_info.tc_link_key.tx_counter
    server.handlers["get_network_info"] = not_configured
    await app.load_network_info()
    margin = application_module.FRAME_COUNTER_RESTORE_MARGIN
    assert app.state.network_info.network_key.tx_counter == counter + margin
    assert app.state.network_info.tc_link_key.tx_counter == aps_counter + margin


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
    assert server.configured.tclk_seed == "cd" * 16
    assert server.configured.tclk_flavor == "zstack"
    assert server.configured.key_table == [
        commands.KeyTableEntry(partner_ieee=DEVICE_IEEE, key=LINK_KEY)
    ]

    # An ezsp seed likewise
    await app.write_network_info(
        network_info=network_info.replace(
            stack_specific={"ezsp": {"hashed_tclk": "ef" * 16}}
        ),
        node_info=node_info,
    )
    assert server.configured.tclk_seed == "ef" * 16
    assert server.configured.tclk_flavor == "ezsp"

    # When zigpy forms a fresh network it leaves the IEEE address unspecified,
    # deferring to the radio's hardware address
    await app.write_network_info(
        network_info=network_info,
        node_info=node_info.replace(ieee=t.EUI64.UNKNOWN),  # type: ignore[attr-defined]
    )
    assert server.sent(commands.GetHwAddress)
    assert server.configured.ieee_address == server.hw_address
    assert app.state.node_info.ieee == server.hw_address

    # Every write is recorded as a backup: zigpy's database is the network's NVRAM
    assert app.backups[-1].node_info.ieee == server.hw_address


async def test_permit_ncp(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.permit_ncp(42)
    permit = server.sent(commands.PermitJoins)[-1]
    assert permit.duration == 42
    # Permitting on the coordinator opens its own beacon for direct joins
    assert permit.accept_direct_joins is True


async def test_permit_steered_to_router(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    device = app.add_device(DEVICE_IEEE, DEVICE_NWK)
    # The unicast Mgmt_Permit_Joining_req awaits a ZDO reply no synthetic device sends
    device.zdo.permit = AsyncMock()  # type: ignore[method-assign]

    await app.permit(time_s=30, node=DEVICE_IEEE)

    assert device.zdo.permit.mock_calls == [((30,), {})]
    # The trust center window opens without advertising the coordinator as a parent
    permit = server.sent(commands.PermitJoins)[-1]
    assert permit.duration == 30
    assert permit.accept_direct_joins is False


async def test_permit_with_link_key(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.permit_with_link_key(node=DEVICE_IEEE, link_key=LINK_KEY, time_s=12)

    provisional = server.sent(commands.SetProvisionalKey)[-1]
    assert provisional.ieee == DEVICE_IEEE
    assert provisional.key == LINK_KEY

    # `super().permit()` broadcasts Mgmt_Permit_Joining_req and calls `permit_ncp`
    broadcast = server.sent(commands.SendAps)[-1]
    assert broadcast.delivery_mode == "broadcast"
    assert broadcast.cluster_id == zdo_t.ZDOCmd.Mgmt_Permit_Joining_req
    assert server.sent(commands.PermitJoins)[-1].duration == 12


async def test_permit_node_conversion_and_all_routers(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app.permit(time_s=20, node="aa:bb:cc:dd:11:22:33:44")
    steered = server.sent(commands.PermitJoins)[-1]
    assert steered.duration == 20
    assert steered.accept_direct_joins is False

    # No node falls through to the base broadcast, which opens the coordinator too
    await app.permit(time_s=30)
    opened = server.sent(commands.PermitJoins)[-1]
    assert opened.duration == 30
    assert opened.accept_direct_joins is True


async def test_energy_scan(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    rssis = iter([-90, -80, -70])

    async def scan(command: commands.EnergyScan, request_id: int) -> commands.Status:
        rssi = next(rssis)
        for channel in command.channels:
            await server.send_event_data(
                request_id,
                "energy_result",
                commands.EnergyScanResult(
                    channel=t.uint8_t(channel), rssi=t.int8s(rssi)
                ).to_dict(),
            )
        return commands.Status(status="complete")

    server.handlers["energy_scan"] = scan

    energies = await app.energy_scan(
        # zigpy mis-annotates the classmethod's `cls` as an instance
        channels=t.Channels.from_channel_list([11, 15]),  # type: ignore[misc]
        duration_exp=2,
        count=3,
    )

    scans = server.sent(commands.EnergyScan)
    assert len(scans) == 3
    assert scans[0].channels == [11, 15]
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
    beacons = [
        commands.NetworkBeaconEvent(
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
        ),
        # A beacon whose MAC source was not a short address
        commands.NetworkBeaconEvent(
            channel=t.uint8_t(15),
            source=None,
            pan_id=t.PanId(0x4C5D),
            extended_pan_id=t.ExtendedPanId(t.EUI64.convert("01:02:03:04:05:06:07:08")),
            permit_joining=False,
            stack_profile=t.uint8_t(2),
            protocol_version=t.uint8_t(2),
            router_capacity=False,
            end_device_capacity=False,
            device_depth=t.uint8_t(2),
            update_id=t.uint8_t(1),
            lqi=t.uint8_t(120),
            rssi=t.int8s(-80),
        ),
    ]

    async def scan(command: commands.NetworkScan, request_id: int) -> commands.Status:
        for beacon in beacons:
            await server.send_event_data(request_id, "network_found", beacon.to_dict())
        return commands.Status(status="complete")

    server.handlers["network_scan"] = scan

    found = [
        beacon
        async for beacon in app.network_scan(
            # zigpy mis-annotates the classmethod's `cls` as an instance
            channels=t.Channels.from_channel_list([11, 15]),  # type: ignore[misc]
            duration_exp=2,
        )
    ]

    scans = server.sent(commands.NetworkScan)
    assert len(scans) == 1
    assert scans[0].channels == [11, 15]
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


@pytest.mark.parametrize(
    ("dst", "tx_options", "src_ep", "dst_ep", "expected"),
    [
        (
            t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK),
            t.TransmitOptions.ACK,
            1,
            1,
            {
                "delivery_mode": "unicast",
                "destination": DEVICE_NWK,
                "destination_eui64": DEVICE_IEEE,
                "profile_id": 0x0104,
                "aps_ack": True,
                "aps_encryption": False,
            },
        ),
        (
            t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK),
            t.TransmitOptions.NONE,
            0,
            0,
            {
                "delivery_mode": "unicast",
                "profile_id": 0x0104,
                "aps_ack": False,
            },
        ),
        (
            t.AddrModeAddress(addr_mode=t.AddrMode.IEEE, address=DEVICE_IEEE),
            t.TransmitOptions.NONE,
            1,
            1,
            {
                "delivery_mode": "unicast",
                "destination": None,
                "destination_eui64": DEVICE_IEEE,
            },
        ),
        (
            t.AddrModeAddress(addr_mode=t.AddrMode.Group, address=t.Group(0x0002)),
            t.TransmitOptions.NONE,
            1,
            255,
            {
                "delivery_mode": "multicast",
                "destination": t.NWK(0x0002),
            },
        ),
        (
            t.AddrModeAddress(
                addr_mode=t.AddrMode.Broadcast,
                address=t.BroadcastAddress.ALL_ROUTERS_AND_COORDINATOR,
            ),
            t.TransmitOptions.NONE,
            1,
            255,
            {
                "delivery_mode": "broadcast",
                "destination": t.NWK(0xFFFC),
            },
        ),
        (
            t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=DEVICE_NWK),
            t.TransmitOptions.ACK | t.TransmitOptions.APS_Encryption,
            1,
            1,
            {
                "delivery_mode": "unicast",
                "destination": DEVICE_NWK,
                # The link key is selected by EUI64, resolved from the device registry
                "destination_eui64": DEVICE_IEEE,
                "aps_encryption": True,
            },
        ),
    ],
)
async def test_send_packet(
    app: ControllerApplication,
    server: SyntheticZiggurat,
    dst: t.AddrModeAddress,
    tx_options: t.TransmitOptions,
    src_ep: int,
    dst_ep: int,
    expected: dict[str, Any],
) -> None:
    app.add_device(DEVICE_IEEE, DEVICE_NWK)

    await app.send_packet(
        t.ZigbeePacket(
            src=t.AddrModeAddress(addr_mode=t.AddrMode.NWK, address=t.NWK(0x0000)),
            src_ep=t.uint8_t(src_ep),
            dst=dst,
            dst_ep=t.uint8_t(dst_ep),
            tsn=t.uint8_t(33),
            profile_id=t.uint16_t(0x0104),
            cluster_id=t.uint16_t(0x0006),
            data=t.SerializableBytes(b"\x01\x02\x03"),
            tx_options=tx_options,
        )
    )

    request = server.sent(commands.SendAps)[-1]
    assert request.data == b"\x01\x02\x03"
    assert request.aps_seq == 33
    assert request.radius == 30
    for field, value in expected.items():
        assert getattr(request, field) == value


async def test_send_packet_delivery_failure(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    async def fail(command: commands.SendAps, request_id: int) -> commands.Status:
        raise RpcError("transmit_failed", "radio unavailable")

    server.handlers["send_aps"] = fail

    # The legacy `transmit_failed` code maps to the binary RADIO_ERROR status.
    with pytest.raises(DeliveryError, match="radio_error"):
        await app.send_packet(
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


async def test_reset_network_info(app: ControllerApplication) -> None:
    await app.reset_network_info()


async def test_watchdog_feed(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app._watchdog_feed()
    assert isinstance(server.requests[-1], commands.Ping)


async def test_move_network_to_channel(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await app._move_network_to_channel(new_channel=20, new_nwk_update_id=1)

    # The update id goes first so no beacon on the new channel ever advertises the
    # old network instance
    methods = [type(r) for r in server.requests]
    assert methods.index(commands.SetNwkUpdateId) < methods.index(commands.SetChannel)
    assert server.sent(commands.SetNwkUpdateId)[-1].nwk_update_id == 1
    assert server.sent(commands.SetChannel)[-1].channel == 20


async def test_packet_received(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    add_initialized_device(app)
    our_nwk = t.NWK(0x0000).serialize()

    # Node_Desc_req: answered locally, advertising the trust center's revision
    app.packet_received(zdo_packet(zdo_t.ZDOCmd.Node_Desc_req, b"\x10" + our_nwk))
    reply = await server.wait_for(commands.SendAps)
    assert reply.cluster_id == zdo_t.ZDOCmd.Node_Desc_rsp
    assert reply.destination == DEVICE_NWK
    assert reply.data[0] == 0x10
    assert reply.data[1] == zdo_t.Status.SUCCESS

    # Active_EP_req
    app.packet_received(zdo_packet(zdo_t.ZDOCmd.Active_EP_req, b"\x11" + our_nwk))
    reply = await server.wait_for(commands.SendAps, count=2)
    assert reply.cluster_id == zdo_t.ZDOCmd.Active_EP_rsp
    assert reply.data[1] == zdo_t.Status.SUCCESS
    endpoint_count = reply.data[4]
    assert 1 in reply.data[5 : 5 + endpoint_count]

    # Simple_Desc_req for a registered endpoint
    app.packet_received(
        zdo_packet(zdo_t.ZDOCmd.Simple_Desc_req, b"\x12" + our_nwk + b"\x01")
    )
    reply = await server.wait_for(commands.SendAps, count=3)
    assert reply.cluster_id == zdo_t.ZDOCmd.Simple_Desc_rsp
    assert reply.data[1] == zdo_t.Status.SUCCESS

    # Requests that are not answered locally. zigpy may originate its own requests
    # (e.g. IEEE_addr_req for an unknown sender), so only count ZDO responses.
    def zdo_replies() -> list[commands.SendAps]:
        return [r for r in server.sent(commands.SendAps) if r.cluster_id & 0x8000]

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


async def test_on_notification_received_aps_command(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    add_initialized_device(app)

    # A ZDO request arriving over the wire is answered end-to-end
    await server.send_notification(
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
    reply = await server.wait_for(commands.SendAps)
    assert reply.cluster_id == zdo_t.ZDOCmd.Node_Desc_rsp
    assert reply.data[0] == 0x77

    # Group- and broadcast-addressed frames parse into their address modes
    await server.send_notification(
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
    await server.send_notification(
        commands.ReceivedApsCommand(
            source=DEVICE_NWK,
            destination=t.NWK(0xFFFD),
            group=None,
            profile_id=t.uint16_t(0x0104),
            cluster_id=t.uint16_t(0x0006),
            src_ep=t.uint8_t(1),
            dst_ep=t.uint8_t(1),
            lqi=t.uint8_t(255),
            rssi=t.int8s(-40),
            data=b"\x02",
        )
    )
    await flush(app)


async def test_on_notification_frame_counter_update(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    await server.send_notification(
        commands.FrameCounterUpdate(frame_counter=t.uint32_t(123456))
    )
    await flush(app)

    assert app.state.network_info.network_key.tx_counter == 123456
    assert app.backups[-1].network_info.network_key.tx_counter == 123456


async def test_on_notification_link_key_update(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    new_key = t.KeyData.convert("ff:ee:dd:cc:bb:aa:99:88:77:66:55:44:33:22:11:00")

    await server.send_notification(
        commands.LinkKeyUpdate(ieee=DEVICE_IEEE, key=LINK_KEY)
    )
    await flush(app)
    assert app.state.network_info.key_table == [
        zigpy.state.Key(key=LINK_KEY, partner_ieee=DEVICE_IEEE)
    ]

    # A renegotiated key replaces the previous entry instead of duplicating it
    await server.send_notification(
        commands.LinkKeyUpdate(ieee=DEVICE_IEEE, key=new_key)
    )
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

    # A brand-new device only joins after the announcement grace period
    await server.send_notification(
        commands.DeviceJoined(nwk=DEVICE_NWK, ieee=DEVICE_IEEE, parent=t.NWK(0x0000))
    )
    await flush(app)
    with pytest.raises(KeyError):
        app.get_device(ieee=DEVICE_IEEE)
    await asyncio.sleep(0.1)
    assert app.get_device(ieee=DEVICE_IEEE).nwk == DEVICE_NWK

    # A device that announced itself within the grace period is not joined again
    ieee2 = t.EUI64.convert("bb:bb:bb:bb:bb:bb:bb:bb")
    await server.send_notification(
        commands.DeviceJoined(nwk=t.NWK(0x5678), ieee=ieee2, parent=t.NWK(0x0000))
    )
    await flush(app)
    device2 = app.add_device(ieee2, t.NWK(0x5678))  # the announcement's effect
    await asyncio.sleep(0.1)
    assert app.get_device(ieee=ieee2) is device2

    # A known, initialized device rejoining with a new address skips the delay
    ieee3 = t.EUI64.convert("cc:cc:cc:cc:cc:cc:cc:cc")
    device3 = app.add_device(ieee3, t.NWK(0x9999))
    device3.node_desc = app.get_device(nwk=t.NWK(0x0000)).node_desc
    device3.status = zigpy.device.Status.ENDPOINTS_INIT
    await server.send_notification(
        commands.DeviceJoined(nwk=t.NWK(0x9AAA), ieee=ieee3, parent=t.NWK(0x0000))
    )
    await flush(app)
    assert app.get_device(ieee=ieee3).nwk == t.NWK(0x9AAA)


async def test_packet_received_aqara_node_desc_override(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    our_nwk = t.NWK(0x0000).serialize()
    coordinator = app.get_device(nwk=t.NWK(0x0000))
    assert coordinator.node_desc is not None
    assert coordinator.node_desc.manufacturer_code == application_module.DEFAULT_MFG_ID

    def reported_mfg_code(reply: commands.SendAps) -> int:
        node_desc, _ = zdo_t.NodeDescriptor.deserialize(bytes(reply.data[4:]))
        return node_desc.manufacturer_code

    # A Lumi/Aqara device is answered with the Xiaomi manufacturer code so it pairs
    aqara_nwk = t.NWK(0x1234)
    add_initialized_device(
        app, ieee=t.EUI64.convert("54:ef:44:00:00:00:00:01"), nwk=aqara_nwk
    )
    app.packet_received(
        zdo_packet(zdo_t.ZDOCmd.Node_Desc_req, b"\x20" + our_nwk, src=aqara_nwk)
    )
    reply = await server.wait_for(commands.SendAps)
    assert reply.cluster_id == zdo_t.ZDOCmd.Node_Desc_rsp
    assert reported_mfg_code(reply) == 0x115F

    # The coordinator's stored descriptor is untouched; other devices see the default
    assert coordinator.node_desc.manufacturer_code == application_module.DEFAULT_MFG_ID
    add_initialized_device(app)
    app.packet_received(
        zdo_packet(zdo_t.ZDOCmd.Node_Desc_req, b"\x21" + our_nwk, src=DEVICE_NWK)
    )
    reply = await server.wait_for(commands.SendAps, count=2)
    assert reported_mfg_code(reply) == application_module.DEFAULT_MFG_ID


async def test_on_notification_device_left(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    left: list[zigpy.device.Device] = []

    class Listener:
        def device_left(self, device: zigpy.device.Device) -> None:
            left.append(device)

    app.add_listener(Listener())
    device = app.add_device(DEVICE_IEEE, DEVICE_NWK)

    # The device announced its own departure
    await server.send_notification(
        commands.DeviceLeft(
            nwk=DEVICE_NWK,
            ieee=DEVICE_IEEE,
            reason=commands.DeviceLeaveReason.ANNOUNCED,
            rejoin=False,
        )
    )
    await flush(app)
    assert left == [device]

    # A parent router relayed the leave; the IEEE is resolved through the registry
    await server.send_notification(
        commands.DeviceLeft(
            nwk=DEVICE_NWK,
            ieee=None,
            reason=commands.DeviceLeaveReason.ROUTER_REPORTED,
            router=t.NWK(0x1234),
            router_ieee=t.EUI64.convert("bb:bb:bb:bb:bb:bb:bb:bb"),
        )
    )
    await flush(app)
    assert left == [device, device]

    # An entirely unknown device is dropped
    await server.send_notification(
        commands.DeviceLeft(
            nwk=t.NWK(0xBEEF),
            ieee=None,
            reason=commands.DeviceLeaveReason.KEEPALIVE_TIMEOUT,
        )
    )
    await flush(app)
    assert left == [device, device]


async def test_on_notification_aps_decryption_failure(
    app: ControllerApplication,
    server: SyntheticZiggurat,
    caplog: pytest.LogCaptureFixture,
) -> None:
    source_ieee = t.EUI64.convert("aa:aa:aa:aa:aa:aa:aa:aa")
    with caplog.at_level(logging.WARNING):
        await server.send_notification(
            commands.ApsDecryptionFailure(
                source=t.NWK(0x1234),
                source_ieee=source_ieee,
                frame_counter=t.uint32_t(42),
                key_id="tc_link_key",
            )
        )
        await flush(app)

    assert "Could not decrypt an APS command" in caplog.text
    assert str(source_ieee) in caplog.text


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
    async def capture(
        command: commands.PacketCapture, request_id: int
    ) -> commands.Status:
        await server.send_event_data(
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

    server.handlers["packet_capture"] = capture

    packets = [packet async for packet in app.packet_capture(15)]

    assert len(packets) == 1
    assert packets[0].channel == 15
    assert packets[0].data == b"\xaa\xbb\xcc"
    assert server.sent(commands.PacketCapture)[0].channel == 15


async def test_packet_capture_change_channel(
    app: ControllerApplication, server: SyntheticZiggurat
) -> None:
    server.handlers["packet_capture_change_channel"] = server.on_status

    await app.packet_capture_change_channel(20)

    assert server.sent(commands.PacketCaptureChangeChannel)[0].channel == 20


def test_max_concurrent_requests() -> None:
    assert application_module._max_concurrent_requests("ws://host/") == 128
    assert application_module._max_concurrent_requests("ws+unix:///run/z.sock") == 128
    assert application_module._max_concurrent_requests("/dev/ttyUSB0") == 32
