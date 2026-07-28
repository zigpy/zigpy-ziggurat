from __future__ import annotations

import asyncio
from collections.abc import AsyncGenerator
from datetime import datetime, timedelta, timezone
import logging
import math
import os
import statistics
from typing import Any, cast

import zigpy.application
import zigpy.backups
import zigpy.config
import zigpy.device
import zigpy.endpoint
from zigpy.exceptions import DeliveryError, NetworkNotFormed
import zigpy.state
import zigpy.types as t
import zigpy.zdo.types as zdo_t

from zigpy_ziggurat.config import CONF_TUNABLES, CONF_ZIGGURAT_CONFIG, CONFIG_SCHEMA
from zigpy_ziggurat.zigbee import protocol as p
from zigpy_ziggurat.zigbee.api import ZigguratApi

_LOGGER = logging.getLogger(__name__)

RSSI_MIN = -92
RSSI_MAX = -5

# How long a freshly-joined device gets to announce itself before zigpy is told about
# the join. Some devices do not tolerate being interviewed mid-join (see zigpy-znp).
DEVICE_JOIN_MAX_DELAY = 5

DEFAULT_MFG_ID = 0x134B  # Open Home Foundation
MFG_ID_OVERRIDES = {
    "04:CF:8C": 0x115F,  # Xiaomi
    "54:EF:44": 0x115F,  # Lumi
}


def _max_concurrent_requests(url: str) -> int:
    if url.startswith(("ws://", "wss://", "ws+unix://")):
        return 128
    return 32


# 802.15.4 6.3.1: time spent scanning each channel is
# aBaseSuperframeDuration * (2^n + 1) symbols, at 16 us per symbol
SYMBOL_PERIOD_MS = 0.016
BASE_SUPERFRAME_DURATION_SYMBOLS = 960

KEY_BATCH_SIZE = 12

# On a stateless restart we resume from the last persisted frame counters, which trail
# the radio's true counters by up to the persist stride (plus any commit lag). Jump both
# the NWK and APS outgoing counters past that gap so a restart can never roll back and
# make peers reject our secured frames.
FRAME_COUNTER_RESTORE_MARGIN = 1000

# How long route hints are supplied to the stack, to reduce startup churn.
ROUTE_HINT_DURATION = timedelta(minutes=10)


def logistic(x: float, *, L: float = 1, x_0: float = 0, k: float = 1) -> float:
    """Logistic function."""
    return L / (1 + math.exp(-k * (x - x_0)))


def map_rssi_to_energy(rssi: float) -> float:
    """Remaps RSSI (in dBm) to Energy (0-255), same curve as bellows."""
    return logistic(
        x=rssi,
        L=255,
        x_0=RSSI_MIN + 0.45 * (RSSI_MAX - RSSI_MIN),
        k=0.13,
    )


class ZigguratCoordinator(zigpy.device.Device):
    """The coordinator device, constructed statically (Ziggurat has no loopback ZDO)."""

    @property
    def manufacturer(self) -> str:
        return "Ziggurat"

    @manufacturer.setter
    def manufacturer(self, value: str) -> None:
        pass

    @property
    def model(self) -> str:
        return "Coordinator"

    @model.setter
    def model(self, value: str) -> None:
        pass


class ControllerApplication(zigpy.application.ControllerApplication):
    DISPLAY_NAME = "Ziggurat"
    DESCRIPTION = "Ziggurat: An open source, host-side Zigbee stack in Rust"
    SCHEMA = CONFIG_SCHEMA

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._api: ZigguratApi | None = None
        self._start_time: datetime | None = None

    async def connect(self) -> None:
        # The device path is either the WebSocket URL of a ziggurat server or the
        # serial port of a ziggurat firmware (e.g. an ESP32-C6 over USB-Serial-JTAG).
        device = self._config[zigpy.config.CONF_DEVICE]
        url = device[zigpy.config.CONF_DEVICE_PATH]

        # zigpy types `connection_lost` as Exception-only but handles None fine
        api = ZigguratApi(
            url,
            self.on_notification,
            self.connection_lost,  # type: ignore[arg-type]
            baudrate=device[zigpy.config.CONF_DEVICE_BAUDRATE],
            flow_control=device[zigpy.config.CONF_DEVICE_FLOW_CONTROL],
        )
        await api.connect()
        self._api = api

        # Clear any transient radio state left by a previous client (e.g. a packet
        # capture still streaming on the firmware) so this session starts from idle.
        await api.request(p.Reset(hard=t.Bool(False)))

    async def disconnect(self) -> None:
        self._start_time = None

        if self._api is not None:
            try:
                await self._api.disconnect()
            finally:
                self._api = None

    async def start_network(self) -> None:
        await self.load_network_info()
        await self.write_network_info(
            network_info=self.state.network_info, node_info=self.state.node_info
        )

        assert self._api is not None

        for name, value in self._config[CONF_ZIGGURAT_CONFIG][CONF_TUNABLES].items():
            await self._api.set_tunable(name, value)

        self._register_coordinator_device()
        await self.register_endpoints()

        url = self._config[zigpy.config.CONF_DEVICE][zigpy.config.CONF_DEVICE_PATH]
        self._concurrent_requests_semaphore.max_concurrency = _max_concurrent_requests(
            url
        )

        self._start_time = datetime.now(timezone.utc)

    def _register_coordinator_device(self) -> None:
        coordinator = ZigguratCoordinator(
            self, self.state.node_info.ieee, self.state.node_info.nwk
        )

        # Remote devices read this via ZDO Node_Desc_req, which zigpy answers with the
        # device's node descriptor. The server mask advertises a primary trust center
        # with stack compliance revision 22: joiners check it to decide whether to
        # perform the trust center link key exchange.
        coordinator.node_desc = zdo_t.NodeDescriptor(
            logical_type=zdo_t.LogicalType.Coordinator,
            complex_descriptor_available=0,
            user_descriptor_available=0,
            reserved=0,
            aps_flags=0,
            frequency_band=zdo_t.NodeDescriptor.FrequencyBand.Freq2400MHz,
            mac_capability_flags=(
                zdo_t.NodeDescriptor.MACCapabilityFlags.FullFunctionDevice
                | zdo_t.NodeDescriptor.MACCapabilityFlags.MainsPowered
                | zdo_t.NodeDescriptor.MACCapabilityFlags.RxOnWhenIdle
                | zdo_t.NodeDescriptor.MACCapabilityFlags.AllocateAddress
            ),
            manufacturer_code=DEFAULT_MFG_ID,
            maximum_buffer_size=82,
            maximum_incoming_transfer_size=128,
            server_mask=0x2C01,  # Primary Trust Center, revision 22
            maximum_outgoing_transfer_size=128,
            descriptor_capability_field=zdo_t.NodeDescriptor.DescriptorCapability.NONE,
        )
        coordinator.status = zigpy.device.Status.ENDPOINTS_INIT

        self.devices[self.state.node_info.ieee] = coordinator

    async def load_network_info(self, *, load_devices: bool = False) -> None:
        assert self._api is not None

        try:
            info = cast(p.NetworkInfo, await self._api.request(p.GetNetworkInfo()))
        except p.ProtocolError as exc:
            if exc.status != p.Status.NOT_CONFIGURED:
                raise

            # The server is stateless and has no network running (e.g. it just
            # restarted): the most recent zigpy database backup is authoritative
            self._get_network_settings()
            return

        state = info.state

        key_table: list[zigpy.state.Key] = []
        async for key_entry in self._api.request_stream(p.ScanKeyTable()):
            key_entry = cast(p.KeyEntry, key_entry)
            key_table.append(
                zigpy.state.Key(
                    key=key_entry.key,
                    partner_ieee=key_entry.partner_ieee,
                    tx_counter=key_entry.tx_counter,
                    rx_counter=key_entry.rx_counter,
                    seq=key_entry.seq,
                )
            )

        stack_specific: dict[str, Any] = {}
        if state.has_tclk_seed:
            seed_hex = bytes(state.tclk_seed).hex()

            if state.tclk_flavor == p.TclkFlavorId.Z_STACK:
                stack_specific = {"zstack": {"tclk_seed": seed_hex}}
            else:
                stack_specific = {"ezsp": {"hashed_tclk": seed_hex}}

        children: list[t.EUI64] = []
        async for child_entry in self._api.request_stream(p.ScanChildren()):
            child_entry = cast(p.ChildEntry, child_entry)
            children.append(child_entry.ieee)

        nwk_addresses: dict[t.EUI64, t.NWK] = {}
        async for addr_entry in self._api.request_stream(p.ScanAddressCache()):
            addr_entry = cast(p.AddressEntry, addr_entry)
            nwk_addresses[addr_entry.ieee] = addr_entry.nwk

        routes = []
        async for route_entry in self._api.request_stream(p.ScanRouteTable()):
            route_entry = cast(p.RouteEntry, route_entry)
            routes.append(
                {
                    "destination": route_entry.destination,
                    "next_hop": route_entry.next_hop,
                    "path_cost": route_entry.path_cost,
                }
            )
        if routes:
            stack_specific["ziggurat"] = {"routes": routes}

        self.state.node_info = zigpy.state.NodeInfo(
            nwk=state.nwk_address,
            ieee=state.ieee_address,
            logical_type=zdo_t.LogicalType.Coordinator,
            manufacturer="Ziggurat",
            model="Coordinator",
        )
        self.state.network_info = zigpy.state.NetworkInfo(
            extended_pan_id=state.extended_pan_id,
            pan_id=state.pan_id,
            nwk_update_id=state.nwk_update_id,
            nwk_manager_id=t.NWK(0x0000),
            channel=state.channel,
            tx_power=state.tx_power,
            # zigpy mis-annotates the classmethod's `cls` as an instance
            channel_mask=t.Channels.from_channel_list([state.channel]),  # type: ignore[misc]
            security_level=t.uint8_t(5),
            network_key=zigpy.state.Key(
                key=state.network_key,
                seq=state.network_key_seq,
                tx_counter=state.network_key_tx_counter,
            ),
            tc_link_key=zigpy.state.Key(
                key=state.tc_link_key,
                partner_ieee=self.state.node_info.ieee,
                # The APS outgoing frame counter lives on the TC link key by convention
                tx_counter=state.aps_frame_counter,
            ),
            key_table=key_table,
            children=children,
            nwk_addresses=nwk_addresses,
            stack_specific=stack_specific,
        )

    def _get_network_settings(self) -> None:
        try:
            latest_backup = self.backups[-1]
        except IndexError as exc:
            raise NetworkNotFormed() from exc

        # The backup's counters trail the radio's true counters by however many frames
        # were sent after the last counter notification: jump both past that gap so a
        # restart never rolls back the NWK or APS outgoing frame counter.
        network_key = latest_backup.network_info.network_key
        tc_link_key = latest_backup.network_info.tc_link_key
        self.state.network_info = latest_backup.network_info.replace(
            network_key=network_key.replace(
                tx_counter=network_key.tx_counter + FRAME_COUNTER_RESTORE_MARGIN
            ),
            tc_link_key=tc_link_key.replace(
                tx_counter=tc_link_key.tx_counter + FRAME_COUNTER_RESTORE_MARGIN
            ),
        )
        self.state.node_info = latest_backup.node_info

    async def force_remove(self, dev: zigpy.device.Device) -> None:
        _LOGGER.debug("Not implemented")

    async def add_endpoint(self, descriptor: zdo_t.SimpleDescriptor) -> None:
        # There is no firmware to register the endpoint with: it exists only on the
        # static coordinator device, which ZDO requests are answered from
        endpoint = self._device.add_endpoint(descriptor.endpoint)
        endpoint.status = zigpy.endpoint.Status.ZDO_INIT
        endpoint.profile_id = descriptor.profile
        # zigpy stores the raw value too, converting to the profile's enum lazily
        endpoint.device_type = descriptor.device_type  # type: ignore[assignment]

        for cluster_id in descriptor.input_clusters:
            endpoint.add_input_cluster(cluster_id)

        for cluster_id in descriptor.output_clusters:
            endpoint.add_output_cluster(cluster_id)

    async def _move_network_to_channel(
        self, new_channel: int, new_nwk_update_id: int
    ) -> None:
        # zigpy has already broadcast the migration to the network; this is the
        # coordinator's own move. The update id goes first so no beacon on the new
        # channel ever advertises the old network instance.
        assert self._api is not None
        await self._api.request(
            p.SetNwkUpdateId(nwk_update_id=t.uint8_t(new_nwk_update_id))
        )
        await self._api.request(p.SetChannel(channel=t.uint8_t(new_channel)))

    async def permit(self, time_s: int = 60, node: t.EUI64 | str | None = None) -> None:
        if node is not None:
            if not isinstance(node, t.EUI64):
                node = t.EUI64.convert(node)
            if node != self.state.node_info.ieee:
                # The base sends a unicast Mgmt_Permit_Joining_req to the target
                # router to steer joins through it. Open our trust center window too,
                # without advertising the coordinator itself as a join target.
                await super().permit(time_s, node=node)
                assert self._api is not None
                await self._api.request(
                    p.PermitJoins(
                        duration=t.uint16_t(time_s), accept_direct_joins=t.Bool(False)
                    )
                )
                return

        await super().permit(time_s, node=node)

    async def permit_ncp(self, time_s: int = 60) -> None:
        assert self._api is not None
        await self._api.request(
            p.PermitJoins(duration=t.uint16_t(time_s), accept_direct_joins=t.Bool(True))
        )

    async def permit_with_link_key(
        self, node: t.EUI64, link_key: t.KeyData, time_s: int = 60
    ) -> None:
        assert self._api is not None
        await self._api.request(p.SetProvisionalKey(ieee=node, key=link_key))

        await super().permit(time_s)

    async def energy_scan(
        self, channels: t.Channels, duration_exp: int, count: int
    ) -> dict[int, float]:
        duration_per_channel_ms = round(
            SYMBOL_PERIOD_MS * BASE_SUPERFRAME_DURATION_SYMBOLS * (2**duration_exp + 1)
        )

        all_results: dict[int, list[float]] = {}

        assert self._api is not None
        for _ in range(count):
            async for result in self._api.request_stream(
                p.EnergyScan(
                    channels=list(channels),
                    duration_per_channel_ms=duration_per_channel_ms,
                )
            ):
                result = cast(p.EnergyResult, result)
                all_results.setdefault(result.channel, []).append(result.rssi)

        return {
            channel: map_rssi_to_energy(statistics.mean(all_results[channel]))
            for channel in list(channels)
        }

    async def _network_scan(
        self, channels: t.Channels, duration_exp: int
    ) -> AsyncGenerator[t.NetworkBeacon, None]:
        duration_per_channel_ms = round(
            SYMBOL_PERIOD_MS * BASE_SUPERFRAME_DURATION_SYMBOLS * (2**duration_exp + 1)
        )

        assert self._api is not None
        async for beacon in self._api.request_stream(
            p.NetworkScan(
                channels=list(channels),
                duration_per_channel_ms=duration_per_channel_ms,
            )
        ):
            beacon = cast(p.Beacon, beacon)
            yield t.NetworkBeacon(
                pan_id=beacon.pan_id,
                extended_pan_id=beacon.extended_pan_id,
                channel=beacon.channel,
                permit_joining=bool(beacon.permit_joining),
                stack_profile=beacon.stack_profile,
                nwk_update_id=beacon.update_id,
                lqi=beacon.lqi,
                src=beacon.source_or_none,
                rssi=beacon.rssi,
                depth=beacon.device_depth,
                router_capacity=bool(beacon.router_capacity),
                device_capacity=bool(beacon.end_device_capacity),
                protocol_version=beacon.protocol_version,
            )

    async def _packet_capture(
        self, channel: int
    ) -> AsyncGenerator[t.CapturedPacket, None]:
        assert self._api is not None
        async for packet in self._api.request_stream(
            p.PacketCapture(channel=t.uint8_t(channel))
        ):
            packet = cast(p.CapturedPacket, packet)
            yield t.CapturedPacket(
                timestamp=datetime.now(timezone.utc),
                rssi=packet.rssi,
                lqi=packet.lqi,
                channel=packet.channel,
                data=packet.psdu,
            )

    async def _packet_capture_change_channel(self, channel: int) -> None:
        assert self._api is not None
        await self._api.request(p.PacketCaptureChannel(channel=t.uint8_t(channel)))

    async def write_network_info(
        self,
        *,
        network_info: zigpy.state.NetworkInfo,
        node_info: zigpy.state.NodeInfo,
    ) -> None:
        assert self._api is not None

        # A TCLK seed carried over from a microcontroller stack: ziggurat derives the
        # unique link keys the previous stack issued to devices from it. Both stacks
        # already store the seed as a plain hex string.
        stack_specific = network_info.stack_specific
        tclk_seed = None
        tclk_flavor = p.TclkFlavorId.EZSP

        if "zstack" in stack_specific and "tclk_seed" in stack_specific["zstack"]:
            tclk_seed = stack_specific["zstack"]["tclk_seed"]
            tclk_flavor = p.TclkFlavorId.Z_STACK
        elif "ezsp" in stack_specific and "hashed_tclk" in stack_specific["ezsp"]:
            tclk_seed = stack_specific["ezsp"]["hashed_tclk"]
            tclk_flavor = p.TclkFlavorId.EZSP

        # `UNKNOWN` is assigned after the class body, where mypy cannot see it
        if node_info.ieee == t.EUI64.UNKNOWN:  # type: ignore[attr-defined]
            # zigpy leaves the IEEE address unspecified when forming a new network,
            # deferring to the radio's hardware address
            rsp = cast(p.HwAddress, await self._api.request(p.GetHwAddress()))
            node_info = node_info.replace(ieee=rsp.ieee)

        state = p.NetworkState(
            channel=t.uint8_t(network_info.channel),
            nwk_update_id=t.uint8_t(network_info.nwk_update_id),
            pan_id=network_info.pan_id,
            extended_pan_id=network_info.extended_pan_id,
            nwk_address=node_info.nwk,
            ieee_address=node_info.ieee,
            network_key=network_info.network_key.key,
            network_key_seq=t.uint8_t(network_info.network_key.seq),
            network_key_tx_counter=t.uint32_t(network_info.network_key.tx_counter),
            tc_link_key=network_info.tc_link_key.key,
            has_tclk_seed=t.Bool(tclk_seed is not None),
            tclk_seed=t.KeyData(
                bytes.fromhex(tclk_seed) if tclk_seed is not None else os.urandom(16)
            ),
            tclk_flavor=tclk_flavor,
            # None means "pick automatically": apply a safe default
            tx_power=t.int8s(
                network_info.tx_power if network_info.tx_power is not None else 8
            ),
            aps_frame_counter=t.uint32_t(network_info.tc_link_key.tx_counter),
        )
        await self._api.request(
            p.Configure(
                role=p.NodeRole.COORDINATOR,
                source_routing=t.Bool(self.config[zigpy.config.CONF_SOURCE_ROUTING]),
                state=state,
            )
        )

        # Unique trust center link keys negotiated in earlier sessions
        entries = [
            p.KeyEntry(
                key=key.key,
                tx_counter=t.uint32_t(key.tx_counter),
                rx_counter=t.uint32_t(key.rx_counter),
                seq=t.uint8_t(key.seq),
                partner_ieee=key.partner_ieee,
            )
            for key in network_info.key_table
        ]
        for start in range(0, len(entries), KEY_BATCH_SIZE):
            await self._api.request(
                p.LoadKeyTable(
                    entries=t.LVList[p.KeyEntry, t.uint16_t](
                        entries[start : start + KEY_BATCH_SIZE]
                    )
                )
            )

        await self._restore_children(network_info)

        await self._api.request(p.StartNetwork())

        # Ziggurat has no persistent storage of its own: zigpy's backup database is
        # the network's NVRAM, so the settings just written are recorded there for
        # `start_network` to find
        self.state.network_info = network_info
        self.state.node_info = node_info
        self.backups.add_backup(
            zigpy.backups.NetworkBackup(network_info=network_info, node_info=node_info)
        )

    async def _restore_children(self, network_info: zigpy.state.NetworkInfo) -> None:
        assert self._api is not None
        # The backup carries no capability, so device type is Unknown (restored as a
        # sleepy end device); children without a known NWK address can't be loaded.
        entries = [
            p.ChildEntry(
                ieee=ieee,
                nwk=network_info.nwk_addresses[ieee],
                rx_on_when_idle=t.uint1_t(1),
                device_type=p.ChildDeviceType.UNKNOWN,
                reserved=t.uint5_t(0),
            )
            for ieee in network_info.children
            if ieee in network_info.nwk_addresses
        ]
        for start in range(0, len(entries), KEY_BATCH_SIZE):
            await self._api.request(
                p.LoadChildren(
                    entries=t.LVList[p.ChildEntry, t.uint16_t](
                        entries[start : start + KEY_BATCH_SIZE]
                    )
                )
            )

    async def reset_network_info(self) -> None:
        assert self._api is not None
        await self._api.request(p.Shutdown())

    async def _watchdog_feed(self) -> None:
        assert self._api is not None
        await self._api.request(p.GetFirmwareInfo())

    def packet_received(self, packet: t.ZigbeePacket) -> None:
        # ZDO requests addressed to the coordinator have to be answered here: there is
        # no firmware ZDO underneath Ziggurat, and zigpy itself only handles a subset
        # (NWK_addr_req, IEEE_addr_req, Match_Desc_req)
        if (
            packet.profile_id == 0x0000
            and packet.src_ep == 0
            and packet.dst_ep == 0
            and packet.src is not None
            and packet.src.addr_mode == t.AddrMode.NWK
        ):
            self._maybe_handle_local_zdo_request(packet)

        super().packet_received(packet)

    def _maybe_handle_local_zdo_request(self, packet: t.ZigbeePacket) -> None:
        assert packet.src is not None

        try:
            device = self.get_device(nwk=t.NWK(packet.src.address))
        except KeyError:
            return

        try:
            hdr, args = device.zdo.deserialize(
                packet.cluster_id, packet.data.serialize()
            )
        except (ValueError, KeyError):
            return

        if hdr.command_id not in (
            zdo_t.ZDOCmd.Node_Desc_req,
            zdo_t.ZDOCmd.Active_EP_req,
            zdo_t.ZDOCmd.Simple_Desc_req,
        ):
            return

        # The address of interest must be us
        if args[0] != self.state.node_info.nwk:
            return

        coordinator = self._device
        nwk = self.state.node_info.nwk

        if hdr.command_id == zdo_t.ZDOCmd.Node_Desc_req:
            # Joining devices read our node descriptor to learn the trust center's
            # stack compliance revision before attempting the link key exchange
            node_desc = coordinator.node_desc
            assert node_desc is not None

            # Aqara/Xiaomi/Lumi devices only finish joining if we report the Xiaomi
            # manufacturer code; answer the requester with the code its OUI expects
            mfg_id = MFG_ID_OVERRIDES.get(str(device.ieee)[:8].upper())
            if mfg_id is not None:
                node_desc = cast(
                    zdo_t.NodeDescriptor,
                    node_desc.replace(manufacturer_code=t.uint16_t(mfg_id)),  # type: ignore[arg-type]
                )

            device.zdo.create_catching_task(
                device.zdo.Node_Desc_rsp(
                    zdo_t.Status.SUCCESS,
                    nwk,
                    node_desc,
                    tsn=hdr.tsn,
                )
            )
        elif hdr.command_id == zdo_t.ZDOCmd.Active_EP_req:
            endpoints = [t.uint8_t(ep) for ep in coordinator.endpoints if ep != 0]
            device.zdo.create_catching_task(
                device.zdo.Active_EP_rsp(
                    zdo_t.Status.SUCCESS,
                    nwk,
                    endpoints,
                    tsn=hdr.tsn,
                )
            )
        elif hdr.command_id == zdo_t.ZDOCmd.Simple_Desc_req:
            endpoint = coordinator.endpoints.get(args[1])

            if endpoint is None or args[1] == 0:
                return

            descriptor = zdo_t.SizePrefixedSimpleDescriptor(
                endpoint=endpoint.endpoint_id,
                profile=endpoint.profile_id,
                device_type=endpoint.device_type,
                device_version=1,
                input_clusters=list(endpoint.in_clusters),
                output_clusters=list(endpoint.out_clusters),
            )
            device.zdo.create_catching_task(
                device.zdo.Simple_Desc_rsp(
                    zdo_t.Status.SUCCESS,
                    nwk,
                    descriptor,
                    tsn=hdr.tsn,
                )
            )

    def _handle_device_joined(
        self, nwk: t.NWK, ieee: t.EUI64, parent_nwk: t.NWK
    ) -> None:
        try:
            self.get_device(ieee=ieee)
        except KeyError:
            pass
        else:
            # A known device rejoined, possibly with a new network address
            self.handle_join(nwk=nwk, ieee=ieee, parent_nwk=parent_nwk)
            return

        # Give a new device a chance to announce itself before the join starts the
        # interview: the announcement creates the device through `packet_received`
        # and a later `handle_join` would cancel and restart the interview
        def join_if_still_unannounced() -> None:
            try:
                self.get_device(ieee=ieee)
            except KeyError:
                self.handle_join(nwk=nwk, ieee=ieee, parent_nwk=parent_nwk)

        asyncio.get_running_loop().call_later(
            DEVICE_JOIN_MAX_DELAY, join_if_still_unannounced
        )

    def on_notification(self, notification: p.Notification) -> None:
        match notification:
            case p.ReceivedAps():
                self._handle_received_aps_command(notification)
            case p.FrameCounter():
                _LOGGER.debug(
                    "NWK frame counter updated to %d", notification.frame_counter
                )
                self.state.network_info.network_key.tx_counter = (
                    notification.frame_counter
                )
                self.backups.add_backup(self.backups.from_network_state())
            case p.ApsFrameCounter():
                _LOGGER.debug(
                    "APS frame counter updated to %d", notification.frame_counter
                )
                self.state.network_info.tc_link_key.tx_counter = (
                    notification.frame_counter
                )
                self.backups.add_backup(self.backups.from_network_state())
            case p.RouteRecord():
                self.handle_relays(
                    nwk=notification.destination, relays=list(notification.relays)
                )
            case p.DeviceJoined():
                self._handle_device_joined(
                    notification.nwk, notification.ieee, notification.parent
                )
            case p.DeviceLeft():
                ieee = notification.ieee_or_none
                if ieee is None:
                    try:
                        ieee = self.get_device(nwk=notification.nwk).ieee
                    except KeyError:
                        return

                _LOGGER.debug(
                    "Device %s (%s) left the network: %s",
                    notification.nwk,
                    ieee,
                    notification.reason.name.lower(),
                )
                self.handle_leave(nwk=notification.nwk, ieee=ieee)
            case p.LinkKey():
                key = zigpy.state.Key(
                    key=notification.key,
                    partner_ieee=notification.ieee,
                )
                _LOGGER.debug("Link key updated for %s", key.partner_ieee)

                self.state.network_info.key_table = [
                    k
                    for k in self.state.network_info.key_table
                    if k.partner_ieee != key.partner_ieee
                ] + [key]
                self.backups.add_backup(
                    zigpy.backups.NetworkBackup(
                        network_info=self.state.network_info,
                        node_info=self.state.node_info,
                    )
                )
            case p.ApsDecryptFailure():
                _LOGGER.warning(
                    "Could not decrypt an APS command from %s (%s): its trust center "
                    "link key is wrong or missing.",
                    notification.source_ieee,
                    notification.source,
                )

    def _handle_received_aps_command(self, command: p.ReceivedAps) -> None:
        group = command.group_id
        if group is not None:
            dst = t.AddrModeAddress(
                addr_mode=t.AddrMode.Group,
                address=t.Group(group),
            )
        elif command.destination >= 0xFFF8:
            dst = t.AddrModeAddress(
                addr_mode=t.AddrMode.Broadcast,
                address=t.BroadcastAddress(command.destination),
            )
        else:
            dst = t.AddrModeAddress(
                addr_mode=t.AddrMode.NWK,
                address=command.destination,
            )

        packet = t.ZigbeePacket(
            src=t.AddrModeAddress(
                addr_mode=t.AddrMode.NWK,
                address=command.source,
            ),
            dst=dst,
            src_ep=command.src_ep,
            dst_ep=command.dst_ep,
            profile_id=command.profile_id,
            cluster_id=command.cluster_id,
            lqi=command.lqi,
            rssi=command.rssi,
            data=t.SerializableBytes(command.data),
        )
        self.packet_received(packet)

    async def send_packet(self, packet: t.ZigbeePacket) -> None:
        dst = packet.dst
        assert dst is not None and dst.address is not None

        try:
            device = self.get_device_with_address(dst)
        except (KeyError, ValueError):
            device = None

        destination: t.NWK | None = None
        destination_eui64 = device.ieee if device is not None else None

        if dst.addr_mode == t.AddrMode.IEEE:
            # The server resolves the EUI64 to a network address
            destination_eui64 = cast(t.EUI64, dst.address)
        else:
            destination = t.NWK(dst.address)

        if (
            t.TransmitOptions.APS_Encryption in packet.tx_options
            and destination_eui64 is None
        ):
            raise DeliveryError(
                "Cannot send an encrypted packet without a destination EUI64"
            )

        # Resolves once the send is confirmed: passive-ack quorum for a broadcast or
        # groupcast, next-hop acceptance for a no-ack unicast, or the end-to-end APS
        # ack. A rejected or failed send raises `DeliveryError`.
        assert self._api is not None
        priority = packet.priority if packet.priority is not None else 0
        radius = packet.radius or 30
        asdu = packet.data.serialize()

        send: p.SendUnicast | p.SendBroadcast | p.SendGroupcast
        async with self._limit_concurrency(priority=packet.priority):
            if dst.addr_mode == t.AddrMode.Group:
                assert destination is not None
                send = p.SendGroupcast.build(
                    group_id=int(destination),
                    profile_id=packet.profile_id,
                    cluster_id=packet.cluster_id or 0x0000,
                    src_ep=packet.src_ep or 0,
                    aps_seq=packet.tsn,
                    radius=radius,
                    priority=priority,
                    asdu=asdu,
                )
            elif dst.addr_mode == t.AddrMode.Broadcast:
                assert destination is not None
                send = p.SendBroadcast.build(
                    destination=destination,
                    profile_id=packet.profile_id,
                    cluster_id=packet.cluster_id or 0x0000,
                    src_ep=packet.src_ep or 0,
                    dst_ep=packet.dst_ep or 0,
                    aps_seq=packet.tsn,
                    radius=radius,
                    priority=priority,
                    asdu=asdu,
                )
            else:
                route_control = p.RouteControl.STACK_DECIDES
                next_hop = None
                relays = None

                # Within the network startup period, provide route hints to the
                # stack to reduce routing congestion
                if device is not None and (
                    self._start_time is None
                    or datetime.now(timezone.utc) - self._start_time
                    < ROUTE_HINT_DURATION
                ):
                    maybe_relays = self.build_source_route_to(device)

                    if maybe_relays is None:
                        maybe_next_hop = None
                    elif not maybe_relays:
                        maybe_next_hop = device.nwk
                    else:
                        maybe_next_hop = maybe_relays[0]

                    if self.config[zigpy.config.CONF_SOURCE_ROUTING] and maybe_relays:
                        route_control = p.RouteControl.HINT_SOURCE_ROUTE
                        relays = maybe_relays
                    elif maybe_next_hop is not None:
                        route_control = p.RouteControl.HINT_NEXT_HOP
                        next_hop = maybe_next_hop

                send = p.SendUnicast.build(
                    destination=destination,
                    destination_eui64=destination_eui64,
                    aps_ack=t.TransmitOptions.ACK in packet.tx_options,
                    aps_encryption=(
                        t.TransmitOptions.APS_Encryption in packet.tx_options
                    ),
                    sleepy_destination=packet.extended_timeout,
                    profile_id=packet.profile_id,
                    cluster_id=packet.cluster_id or 0x0000,
                    src_ep=packet.src_ep or 0,
                    dst_ep=packet.dst_ep or 0,
                    aps_seq=packet.tsn,
                    radius=radius,
                    priority=priority,
                    route_control=route_control,
                    next_hop=next_hop,
                    relays=relays,
                    asdu=asdu,
                )

            await self._api.request_confirmed(send)
