import asyncio
import json
import logging
import math
import statistics

import zigpy.application
import zigpy.backups
import zigpy.device
import zigpy.endpoint
from zigpy.exceptions import DeliveryError
import zigpy.serial
import zigpy.state
import zigpy.types as t
import zigpy.zdo.types as zdo_t

_LOGGER = logging.getLogger(__name__)

RSSI_MIN = -92
RSSI_MAX = -5

# 802.15.4 6.3.1: time spent scanning each channel is
# aBaseSuperframeDuration * (2^n + 1) symbols, at 16 us per symbol
SYMBOL_PERIOD_MS = 0.016
BASE_SUPERFRAME_DURATION_SYMBOLS = 960


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


FALLBACK_NETWORK_SETTINGS = zigpy.backups.NetworkBackup.from_dict(
    {
        "version": 1,
        "backup_time": "2025-06-29T03:35:11.850787+00:00",
        "network_info": {
            "extended_pan_id": "3a:9f:44:01:0b:3c:cb:93",
            "pan_id": "4072",
            "nwk_update_id": 0,
            "nwk_manager_id": "0000",
            "channel": 11,
            "channel_mask": [11],
            "security_level": 5,
            "network_key": {
                "key": "ee:83:0c:e4:85:57:9c:8c:b1:3f:87:00:b6:5d:4b:e8",
                "tx_counter": 0,
                "rx_counter": 0,
                "seq": 0,
                "partner_ieee": "ff:ff:ff:ff:ff:ff:ff:ff",
            },
            "tc_link_key": {
                "key": "5a:69:67:42:65:65:41:6c:6c:69:61:6e:63:65:30:39",
                "tx_counter": 0,
                "rx_counter": 0,
                "seq": 0,
                "partner_ieee": "bc:02:6e:ff:fe:24:db:90",
            },
            "key_table": [],
            "children": [],
            "nwk_addresses": {},
            "stack_specific": {},
            "metadata": {},
            "source": None,
        },
        "node_info": {
            "nwk": "0000",
            "ieee": "bc:02:6e:ff:fe:24:db:90",
            "logical_type": "coordinator",
            "model": None,
            "manufacturer": None,
            "version": None,
        },
    }
)


class ZigguratProtocol(zigpy.serial.SerialProtocol):
    def __init__(self, on_async_event, on_disconnect):
        super().__init__()

        self.on_async_event = on_async_event
        self.on_disconnect = on_disconnect
        self.tid = 1
        self.pending_requests: dict[int, asyncio.Future] = {}

    def data_received(self, data: bytes):
        super().data_received(data)

        while b"\n" in self._buffer:
            line, self._buffer = self._buffer.split(b"\n", 1)
            line = line.strip()
            if not line:
                continue

            # Parse JSON
            try:
                msg = json.loads(line.decode("utf-8"))
            except json.JSONDecodeError as e:
                _LOGGER.debug("Failed to parse line as JSON: %r: %r", line, e)
                continue

            try:
                self.handle_message(msg)
            except Exception:
                _LOGGER.exception("Failed to handle message: %r", msg)
                continue

    def handle_message(self, message: dict):
        tid = message.get("tid", 0)
        _LOGGER.debug("Received: %r", message)

        if tid == 0:
            # Asynchronous event
            self.on_async_event(message)
            return

        # Response to a pending request
        fut = self.pending_requests.pop(tid, None)
        if not fut or fut.done():
            _LOGGER.debug(
                f"Received response for unknown or finished TID={tid}: {message}"
            )
            return

        fut.set_result(message)

    def connection_lost(self, exc):
        self.on_disconnect(exc)

        for fut in self.pending_requests.values():
            if not fut.done():
                fut.set_exception(ConnectionError("Connection lost"))

        self.pending_requests.clear()
        super().connection_lost(exc)

    async def send_command(self, cmd: str, data: dict) -> dict:
        tid = self.tid
        self.tid = (self.tid + 1) & 0xFFFFFFFF

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self.pending_requests[tid] = fut

        message = {
            "tid": tid,
            "cmd": cmd,
            "data": data,
        }
        line = json.dumps(message) + "\n"
        _LOGGER.debug("Sending: %r", line)
        self._transport.write(line.encode("utf-8"))

        rsp = await fut

        if rsp["data"]["status"] == "error":
            reason = rsp["data"].get("reason") or "unknown error"
            raise DeliveryError(f"Error sending command: {reason}")

        return rsp


class ZigguratCoordinator(zigpy.device.Device):
    """Zigpy device representing the coordinator. Ziggurat has no loopback ZDO, so the
    device is constructed statically instead of being interviewed over the air."""

    @property
    def manufacturer(self) -> str:
        return "Ziggurat"

    @manufacturer.setter
    def manufacturer(self, value) -> None:
        pass

    @property
    def model(self) -> str:
        return "Coordinator"

    @model.setter
    def model(self, value) -> None:
        pass


class ControllerApplication(zigpy.application.ControllerApplication):
    def __init__(self, config):
        super().__init__(config)
        self._api = None

    async def connect(self):
        _, api = await zigpy.serial.create_serial_connection(
            loop=asyncio.get_running_loop(),
            protocol_factory=lambda: ZigguratProtocol(
                self.on_async_event, self.connection_lost
            ),
            url=self._config[zigpy.config.CONF_DEVICE][zigpy.config.CONF_DEVICE_PATH],
        )
        await api.wait_until_connected()
        self._api = api

    async def disconnect(self):
        if self._api is not None:
            try:
                await self._api.disconnect()
            finally:
                self._api = None

    async def start_network(self):
        backup = self._get_network_settings()

        # Our frame counter shouldn't be off by more than 100. Keep it in sync.
        backup.network_info.network_key.tx_counter += 500
        self.backups.add_backup(backup)

        await self.write_network_info(
            network_info=self.state.network_info, node_info=self.state.node_info
        )

        self._register_coordinator_device()
        await self.register_endpoints()

    def _register_coordinator_device(self):
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
            manufacturer_code=0xFFFF,
            maximum_buffer_size=82,
            maximum_incoming_transfer_size=128,
            server_mask=0x2C01,  # Primary Trust Center, revision 22
            maximum_outgoing_transfer_size=128,
            descriptor_capability_field=zdo_t.NodeDescriptor.DescriptorCapability.NONE,
        )
        coordinator.status = zigpy.device.Status.ENDPOINTS_INIT

        self.devices[self.state.node_info.ieee] = coordinator

    async def load_network_info(self, *, load_devices=False):
        self._get_network_settings()

    def _get_network_settings(self):
        try:
            # Use the most recent backup from the zigpy database, if supported
            latest_backup = self.backups[-1]
        except IndexError:
            latest_backup = FALLBACK_NETWORK_SETTINGS
        else:
            latest_backup = latest_backup

        self.state.network_info = latest_backup.network_info
        self.state.node_info = latest_backup.node_info

        return latest_backup

    async def force_remove(self, dev):
        _LOGGER.debug("Not implemented")

    async def add_endpoint(self, descriptor: zdo_t.SimpleDescriptor) -> None:
        # There is no firmware to register the endpoint with: it exists only on the
        # static coordinator device, which ZDO requests are answered from
        endpoint = self._device.add_endpoint(descriptor.endpoint)
        endpoint.status = zigpy.endpoint.Status.ZDO_INIT
        endpoint.profile_id = descriptor.profile
        endpoint.device_type = descriptor.device_type

        for cluster_id in descriptor.input_clusters:
            endpoint.add_input_cluster(cluster_id)

        for cluster_id in descriptor.output_clusters:
            endpoint.add_output_cluster(cluster_id)

    async def permit_ncp(self, time_s: int = 60):
        await self._api.send_command(
            "permit_joins",
            {
                "duration": time_s,
            },
        )

    async def permit_with_link_key(self, node, link_key, time_s: int = 60):
        _LOGGER.debug("Not implemented")

    async def energy_scan(
        self, channels: t.Channels, duration_exp: int, count: int
    ) -> dict[int, float]:
        duration_per_channel_ms = round(
            SYMBOL_PERIOD_MS * BASE_SUPERFRAME_DURATION_SYMBOLS * (2**duration_exp + 1)
        )

        all_results: dict[int, list[float]] = {}

        for _ in range(count):
            rsp = await self._api.send_command(
                "energy_scan",
                {
                    "channels": list(channels),
                    "duration_per_channel_ms": duration_per_channel_ms,
                },
            )

            for channel, rssi in rsp["data"]["results"].items():
                all_results.setdefault(int(channel), []).append(rssi)

        return {
            channel: map_rssi_to_energy(statistics.mean(all_results[channel]))
            for channel in list(channels)
        }

    async def write_network_info(self, *, network_info, node_info):
        await self._api.send_command(
            "set_network_settings",
            {
                "channel": network_info.channel,
                "nwk_update_id": network_info.nwk_update_id,
                "pan_id": network_info.pan_id.serialize()[::-1].hex(),
                "extended_pan_id": str(network_info.extended_pan_id),
                "nwk_address": node_info.nwk.serialize()[::-1].hex(),
                "ieee_address": str(node_info.ieee),
                "network_key": str(network_info.network_key.key),
                "network_key_seq": network_info.network_key.seq,
                # To avoid persisting state while also preventing counter rollback,
                # just base the counter on the current time
                "network_key_tx_counter": network_info.network_key.tx_counter,
                "tc_link_key": str(network_info.tc_link_key.key),
                # Unique trust center link keys negotiated in earlier sessions
                "key_table": [
                    {
                        "partner_ieee": str(key.partner_ieee),
                        "key": str(key.key),
                    }
                    for key in network_info.key_table
                ],
            },
        )

    async def reset_network_info(self):
        pass

    def packet_received(self, packet):
        # ZDO requests addressed to the coordinator have to be answered here: there is
        # no firmware ZDO underneath Ziggurat, and zigpy itself only handles a subset
        # (NWK_addr_req, IEEE_addr_req, Match_Desc_req)
        if (
            packet.profile_id == 0x0000
            and packet.src_ep == 0
            and packet.dst_ep == 0
            and packet.src.addr_mode == t.AddrMode.NWK
        ):
            self._maybe_handle_local_zdo_request(packet)

        super().packet_received(packet)

    def _maybe_handle_local_zdo_request(self, packet):
        try:
            device = self.get_device(nwk=packet.src.address)
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
            device.zdo.create_catching_task(
                device.zdo.Node_Desc_rsp(
                    zdo_t.Status.SUCCESS,
                    nwk,
                    coordinator.node_desc,
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

    def on_async_event(self, event):
        if event["cmd"] == "received_aps_command":
            data = event["data"]

            if data.get("group") is not None:
                dst = t.AddrModeAddress(
                    addr_mode=t.AddrMode.Group,
                    address=t.Group(data["group"]),
                )
            else:
                dst_nwk, _ = t.NWK.deserialize(bytes.fromhex(data["destination"]))

                if dst_nwk >= 0xFFF8:
                    dst = t.AddrModeAddress(
                        addr_mode=t.AddrMode.Broadcast,
                        address=t.BroadcastAddress(dst_nwk),
                    )
                else:
                    dst = t.AddrModeAddress(
                        addr_mode=t.AddrMode.NWK,
                        address=dst_nwk,
                    )

            packet = t.ZigbeePacket(
                src=t.AddrModeAddress(
                    addr_mode=t.AddrMode.NWK,
                    address=t.NWK.deserialize(bytes.fromhex(data["source"]))[0],
                ),
                dst=dst,
                src_ep=data["src_ep"],
                dst_ep=data["dst_ep"],
                profile_id=data["profile_id"],
                cluster_id=data["cluster_id"],
                lqi=data["lqi"],
                rssi=data["rssi"],
                data=t.SerializableBytes(bytes.fromhex(data["data"])),
            )
            self.packet_received(packet)
        elif event["cmd"] == "frame_counter_update":
            self.state.network_info.network_key.tx_counter = event["data"][
                "frame_counter"
            ]
            _LOGGER.debug(
                "Frame counter updated to %d",
                self.state.network_info.network_key.tx_counter,
            )
            self.backups.add_backup(
                zigpy.backups.NetworkBackup(
                    network_info=self.state.network_info,
                    node_info=self.state.node_info,
                )
            )
        elif event["cmd"] == "link_key_update":
            key = zigpy.state.Key.from_dict(
                {
                    "key": event["data"]["key"],
                    "tx_counter": 0,
                    "rx_counter": 0,
                    "seq": 0,
                    "partner_ieee": event["data"]["ieee"],
                }
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

    async def send_packet(self, packet):
        profile_id = 0x0000

        if packet.src_ep != 0 or packet.dst_ep != 0:
            profile_id = 0x0104

        if packet.dst.addr_mode == t.AddrMode.IEEE:
            # The server resolves the EUI64 to a network address
            addressing = {"destination_eui64": str(packet.dst.address)}
            delivery_mode = "unicast"
        else:
            addressing = {"destination": packet.dst.address.serialize()[::-1].hex()}
            delivery_mode = {
                t.AddrMode.NWK: "unicast",
                t.AddrMode.Group: "multicast",
                t.AddrMode.Broadcast: "broadcast",
            }[packet.dst.addr_mode]

        await self._api.send_command(
            "send_aps_command",
            {
                "delivery_mode": delivery_mode,
                **addressing,
                "profile_id": profile_id,
                "cluster_id": packet.cluster_id or 0x0000,
                "src_ep": packet.src_ep,
                "dst_ep": packet.dst_ep or 0,
                "aps_ack": t.TransmitOptions.ACK in packet.tx_options,
                "radius": packet.radius or 30,
                "aps_seq": packet.tsn,
                "data": packet.data.serialize().hex(),
            },
        )
