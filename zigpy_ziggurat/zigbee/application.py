import asyncio
import json
import logging
import math
import statistics

import aiohttp
import zigpy.application
import zigpy.backups
import zigpy.config
import zigpy.device
import zigpy.endpoint
from zigpy.exceptions import DeliveryError
import zigpy.state
import zigpy.types as t
import zigpy.zdo.types as zdo_t

_LOGGER = logging.getLogger(__name__)

RSSI_MIN = -92
RSSI_MAX = -5

# How long a freshly-joined device gets to announce itself before zigpy is told about
# the join. Some devices do not tolerate being interviewed mid-join (see zigpy-znp).
DEVICE_JOIN_MAX_DELAY = 5

# 802.15.4 6.3.1: time spent scanning each channel is
# aBaseSuperframeDuration * (2^n + 1) symbols, at 16 us per symbol
SYMBOL_PERIOD_MS = 0.016
BASE_SUPERFRAME_DURATION_SYMBOLS = 960

WEBSOCKET_HEARTBEAT = 15


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


class PendingRequest:
    """The in-flight state of one request: an optional `transmitted` stage future and
    the terminal `response` future."""

    def __init__(self, *, want_transmitted: bool) -> None:
        loop = asyncio.get_running_loop()
        self.response: asyncio.Future = loop.create_future()
        self.transmitted: asyncio.Future | None = (
            loop.create_future() if want_transmitted else None
        )

    def fail(self, exc: BaseException) -> None:
        if self.transmitted is not None and not self.transmitted.done():
            self.transmitted.set_exception(exc)

        if not self.response.done():
            self.response.set_exception(exc)


def _make_late_failure_logger(pending: "PendingRequest"):
    """Consume the terminal result of a request that already resolved at the
    `transmitted` stage, so delivery failures are visible but not raised. Failures
    from before transmission were already raised to the caller and are not logged."""

    def log_late_failure(fut: asyncio.Future) -> None:
        if fut.cancelled():
            return

        exc = fut.exception()
        if exc is None:
            return

        transmitted = (
            pending.transmitted is not None
            and pending.transmitted.done()
            and pending.transmitted.exception() is None
        )

        if transmitted:
            _LOGGER.warning("Delivery failed after transmission: %s", exc)

    return log_late_failure


class ZigguratApi:
    """The Ziggurat WebSocket API: concurrent requests correlated by id, with
    lifecycle events (`accepted`, `transmitted`) preceding each terminal response."""

    def __init__(self, url: str, on_notification, on_disconnect) -> None:
        self._url = url
        self._on_notification = on_notification
        self._on_disconnect = on_disconnect

        self._session: aiohttp.ClientSession | None = None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._receiver_task: asyncio.Task | None = None
        self._request_id = 1
        self._pending: dict[int, PendingRequest] = {}

    async def connect(self) -> None:
        self._session = aiohttp.ClientSession()
        self._websocket = await self._session.ws_connect(
            self._url, heartbeat=WEBSOCKET_HEARTBEAT
        )

        hello = json.loads(await self._websocket.receive_str())
        _LOGGER.debug("Connected to ziggurat: %r", hello)

        self._receiver_task = asyncio.create_task(self._receive_loop())

    async def disconnect(self) -> None:
        if self._receiver_task is not None:
            self._receiver_task.cancel()
            self._receiver_task = None

        if self._websocket is not None:
            await self._websocket.close()
            self._websocket = None

        if self._session is not None:
            await self._session.close()
            self._session = None

    async def _receive_loop(self) -> None:
        exc: BaseException | None = None

        try:
            async for msg in self._websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        self._handle_message(json.loads(msg.data))
                    except Exception:
                        _LOGGER.exception("Failed to handle message: %r", msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    exc = self._websocket.exception()
                    break
        except asyncio.CancelledError:
            raise
        except Exception as e:
            exc = e
        finally:
            self._fail_pending(ConnectionError("Connection lost"))
            self._on_disconnect(exc)

    def _fail_pending(self, exc: BaseException) -> None:
        for pending in self._pending.values():
            pending.response.add_done_callback(_make_late_failure_logger(pending))
            pending.fail(exc)

        self._pending.clear()

    def _handle_message(self, msg: dict) -> None:
        _LOGGER.debug("Received: %r", msg)
        msg_type = msg["type"]

        if msg_type == "notification":
            self._on_notification(msg["event"], msg["data"])
        elif msg_type == "event":
            pending = self._pending.get(msg["id"])

            if (
                pending is not None
                and msg["event"] == "transmitted"
                and pending.transmitted is not None
                and not pending.transmitted.done()
            ):
                pending.transmitted.set_result(None)
        elif msg_type == "response":
            pending = self._pending.pop(msg["id"], None)

            if pending is None:
                _LOGGER.debug("Response for unknown request: %r", msg)
                return

            if "error" in msg:
                error = msg["error"]
                pending.fail(DeliveryError(f"{error['code']}: {error['message']}"))
            elif not pending.response.done():
                pending.response.set_result(msg["result"])

    async def request(
        self, method: str, params: dict, *, resolve_on: str = "response"
    ) -> dict | None:
        request_id = self._request_id
        self._request_id = (self._request_id + 1) % 2**32 or 1

        pending = PendingRequest(want_transmitted=(resolve_on == "transmitted"))
        self._pending[request_id] = pending

        message = {"id": request_id, "method": method, "params": params}
        _LOGGER.debug("Sending: %r", message)
        await self._websocket.send_str(json.dumps(message))

        if resolve_on == "transmitted":
            # The terminal response continues in the background; an end-to-end
            # delivery failure after transmission is logged, not raised
            pending.response.add_done_callback(_make_late_failure_logger(pending))
            await pending.transmitted
            return None

        return await pending.response


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
    DISPLAY_NAME = "Ziggurat"
    DESCRIPTION = "Ziggurat: An open source, host-side Zigbee stack in Rust"

    def __init__(self, config):
        super().__init__(config)
        self._api = None

    async def connect(self):
        device_path = self._config[zigpy.config.CONF_DEVICE][
            zigpy.config.CONF_DEVICE_PATH
        ]

        # ZHA entries predating the WebSocket API use `socket://host:port`
        url = device_path.replace("socket://", "ws://", 1)

        api = ZigguratApi(url, self.on_notification, self.connection_lost)
        await api.connect()
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
        await self._api.request(
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
            result = await self._api.request(
                "energy_scan",
                {
                    "channels": list(channels),
                    "duration_per_channel_ms": duration_per_channel_ms,
                },
            )

            for channel, rssi in result["results"].items():
                all_results.setdefault(int(channel), []).append(rssi)

        return {
            channel: map_rssi_to_energy(statistics.mean(all_results[channel]))
            for channel in list(channels)
        }

    async def write_network_info(self, *, network_info, node_info):
        await self._api.request(
            "configure",
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

    def on_notification(self, event: str, data: dict):
        if event == "received_aps_command":
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
        elif event == "frame_counter_update":
            self.state.network_info.network_key.tx_counter = data["frame_counter"]
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
        elif event == "device_joined":
            nwk, _ = t.NWK.deserialize(bytes.fromhex(data["nwk"]))
            ieee = t.EUI64.convert(data["ieee"])
            parent_nwk, _ = t.NWK.deserialize(bytes.fromhex(data["parent"]))
            self._handle_device_joined(nwk, ieee, parent_nwk)
        elif event == "device_left":
            nwk, _ = t.NWK.deserialize(bytes.fromhex(data["nwk"]))

            if data["ieee"] is not None:
                ieee = t.EUI64.convert(data["ieee"])
            else:
                try:
                    ieee = self.get_device(nwk=nwk).ieee
                except KeyError:
                    return

            self.handle_leave(nwk=nwk, ieee=ieee)
        elif event == "link_key_update":
            key = zigpy.state.Key.from_dict(
                {
                    "key": data["key"],
                    "tx_counter": 0,
                    "rx_counter": 0,
                    "seq": 0,
                    "partner_ieee": data["ieee"],
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

        # Resolves once the frame is on the air (EZSP `messageSent` parity); the
        # APS-ack delivery result arrives later and is logged by the API layer
        await self._api.request(
            "send_aps",
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
            resolve_on="transmitted",
        )
