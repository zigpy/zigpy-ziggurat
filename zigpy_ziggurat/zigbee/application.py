import asyncio
from collections.abc import Callable
import json
import logging
import math
import statistics
from typing import Any, cast

import aiohttp
import zigpy.application
import zigpy.backups
import zigpy.config
import zigpy.device
import zigpy.endpoint
from zigpy.exceptions import DeliveryError, NetworkNotFormed
import zigpy.state
import zigpy.types as t
import zigpy.zdo.types as zdo_t

from zigpy_ziggurat.zigbee.commands import (
    NOTIFICATIONS,
    RESPONSE_T,
    Configure,
    DeviceJoined,
    DeviceLeft,
    EnergyScan,
    FrameCounterUpdate,
    GetHwAddress,
    GetNetworkInfo,
    KeyTableEntry,
    LinkKeyUpdate,
    Notification,
    PermitJoins,
    Ping,
    ReceivedApsCommand,
    Request,
    SendAps,
    SetChannel,
    SetNwkUpdateId,
    SetProvisionalKey,
)

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


class PendingRequest:
    """The in-flight state of one request: an optional `transmitted` stage future and
    the terminal `response` future."""

    def __init__(self, *, want_transmitted: bool) -> None:
        loop = asyncio.get_running_loop()
        self.response: asyncio.Future[dict[str, Any]] = loop.create_future()
        self.transmitted: asyncio.Future[None] | None = (
            loop.create_future() if want_transmitted else None
        )

    def fail(self, exc: BaseException) -> None:
        if self.transmitted is not None and not self.transmitted.done():
            self.transmitted.set_exception(exc)

        if not self.response.done():
            self.response.set_exception(exc)


def _make_late_failure_logger(
    pending: PendingRequest,
) -> Callable[[asyncio.Future[dict[str, Any]]], None]:
    """Consume the terminal result of a request that already resolved at the
    `transmitted` stage, so delivery failures are visible but not raised. Failures
    from before transmission were already raised to the caller and are not logged."""

    def log_late_failure(fut: asyncio.Future[dict[str, Any]]) -> None:
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

    def __init__(
        self,
        url: str,
        on_notification: Callable[[Notification], None],
        on_disconnect: Callable[[BaseException | None], None],
    ) -> None:
        self._url = url
        self._on_notification = on_notification
        self._on_disconnect = on_disconnect

        self._session: aiohttp.ClientSession | None = None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._receiver_task: asyncio.Task[None] | None = None
        self._request_id = 1
        self._pending: dict[int, PendingRequest] = {}

    async def connect(self) -> None:
        if self._url.startswith("ws+unix://"):
            # The URL's path is the socket path; the HTTP-level host is a placeholder
            connector = aiohttp.UnixConnector(path=self._url.removeprefix("ws+unix://"))
            url = "ws://localhost/"
        else:
            connector = None
            url = self._url

        self._session = aiohttp.ClientSession(connector=connector)
        self._websocket = await self._session.ws_connect(
            url, heartbeat=WEBSOCKET_HEARTBEAT
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
        websocket = self._websocket
        assert websocket is not None

        exc: BaseException | None = None

        try:
            async for msg in websocket:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        self._handle_message(json.loads(msg.data))
                    except Exception:
                        _LOGGER.exception("Failed to handle message: %r", msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    exc = websocket.exception()
                    break
        except asyncio.CancelledError:
            # A deliberate `disconnect()`, not a connection loss
            self._fail_pending(ConnectionError("Connection closed"))
            raise
        except Exception as e:  # pragma: no cover
            # aiohttp surfaces connection failures as `ERROR` messages or by ending
            # the iterator, never by raising; kept as a guard for other versions
            exc = e

        self._fail_pending(ConnectionError("Connection lost"))
        self._on_disconnect(exc)

    def _fail_pending(self, exc: BaseException) -> None:
        for pending in self._pending.values():
            pending.response.add_done_callback(_make_late_failure_logger(pending))
            pending.fail(exc)

        self._pending.clear()

    def _handle_message(self, msg: dict[str, Any]) -> None:
        _LOGGER.debug("Received: %r", msg)
        msg_type = msg["type"]

        if msg_type == "notification":
            self._on_notification(NOTIFICATIONS[msg["event"]].from_dict(msg["data"]))
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

    async def request(self, command: Request[RESPONSE_T]) -> RESPONSE_T:
        result = await self._send_request(command, want_transmitted=False)
        assert result is not None

        # `response_type` is a plain ClassVar: it cannot carry the type variable
        return cast(RESPONSE_T, command.response_type.from_dict(result))

    async def request_transmitted(self, command: Request[Any]) -> None:
        """Resolve once the frame is on the air instead of waiting for delivery."""
        await self._send_request(command, want_transmitted=True)

    async def _send_request(
        self, command: Request[Any], *, want_transmitted: bool
    ) -> dict[str, Any] | None:
        request_id = self._request_id
        self._request_id = (self._request_id + 1) % 2**32 or 1

        pending = PendingRequest(want_transmitted=want_transmitted)
        self._pending[request_id] = pending

        message = {
            "id": request_id,
            "method": command.method,
            "params": command.to_dict(),
        }
        _LOGGER.debug("Sending: %r", message)
        assert self._websocket is not None
        await self._websocket.send_str(json.dumps(message))

        if want_transmitted:
            # The terminal response continues in the background; an end-to-end
            # delivery failure after transmission is logged, not raised
            pending.response.add_done_callback(_make_late_failure_logger(pending))
            assert pending.transmitted is not None
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

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__(config)
        self._api: ZigguratApi | None = None

    async def connect(self) -> None:
        # The device path is the WebSocket URL of the ziggurat server
        url = self._config[zigpy.config.CONF_DEVICE][zigpy.config.CONF_DEVICE_PATH]

        # zigpy types `connection_lost` as Exception-only but handles None fine
        api = ZigguratApi(
            url,
            self.on_notification,
            self.connection_lost,  # type: ignore[arg-type]
        )
        await api.connect()
        self._api = api

    async def disconnect(self) -> None:
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

        self._register_coordinator_device()
        await self.register_endpoints()

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
            manufacturer_code=0xFFFF,
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
            info = await self._api.request(GetNetworkInfo())
        except DeliveryError as exc:
            if not str(exc).startswith("not_configured"):
                raise

            # The server is stateless and has no network running (e.g. it just
            # restarted): the most recent zigpy database backup is authoritative
            self._get_network_settings()
            return

        stack_specific = {}
        if info.tclk_seed is not None:
            if info.tclk_flavor == "zstack":
                stack_specific = {"zstack": {"tclk_seed": info.tclk_seed}}
            else:
                stack_specific = {"ezsp": {"hashed_tclk": info.tclk_seed}}

        self.state.node_info = zigpy.state.NodeInfo(
            nwk=info.nwk_address,
            ieee=info.ieee_address,
            logical_type=zdo_t.LogicalType.Coordinator,
            manufacturer="Ziggurat",
            model="Coordinator",
        )
        self.state.network_info = zigpy.state.NetworkInfo(
            extended_pan_id=info.extended_pan_id,
            pan_id=info.pan_id,
            nwk_update_id=info.nwk_update_id,
            nwk_manager_id=t.NWK(0x0000),
            channel=info.channel,
            tx_power=info.tx_power,
            # zigpy mis-annotates the classmethod's `cls` as an instance
            channel_mask=t.Channels.from_channel_list([info.channel]),  # type: ignore[misc]
            security_level=t.uint8_t(5),
            network_key=zigpy.state.Key(
                key=info.network_key,
                seq=info.network_key_seq,
                tx_counter=info.network_key_tx_counter,
            ),
            tc_link_key=zigpy.state.Key(
                key=info.tc_link_key,
                partner_ieee=self.state.node_info.ieee,
            ),
            key_table=[
                zigpy.state.Key(key=entry.key, partner_ieee=entry.partner_ieee)
                for entry in info.key_table
            ],
            stack_specific=stack_specific,
        )

    def _get_network_settings(self) -> None:
        try:
            latest_backup = self.backups[-1]
        except IndexError as exc:
            raise NetworkNotFormed() from exc

        # The backup's frame counter trails the radio's true counter by however many
        # frames were sent after the last counter update notification: jump past it
        network_key = latest_backup.network_info.network_key
        self.state.network_info = latest_backup.network_info.replace(
            network_key=network_key.replace(tx_counter=network_key.tx_counter + 500)
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
        await self._api.request(SetNwkUpdateId(nwk_update_id=new_nwk_update_id))
        await self._api.request(SetChannel(channel=new_channel))

    async def permit_ncp(self, time_s: int = 60) -> None:
        assert self._api is not None
        await self._api.request(PermitJoins(duration=time_s))

    async def permit_with_link_key(
        self, node: t.EUI64, link_key: t.KeyData, time_s: int = 60
    ) -> None:
        assert self._api is not None
        await self._api.request(SetProvisionalKey(ieee=node, key=link_key))

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
            result = await self._api.request(
                EnergyScan(
                    channels=list(channels),
                    duration_per_channel_ms=duration_per_channel_ms,
                )
            )

            for channel, rssi in result.results.items():
                all_results.setdefault(channel, []).append(rssi)

        return {
            channel: map_rssi_to_energy(statistics.mean(all_results[channel]))
            for channel in list(channels)
        }

    async def write_network_info(
        self,
        *,
        network_info: zigpy.state.NetworkInfo,
        node_info: zigpy.state.NodeInfo,
    ) -> None:
        # A TCLK seed carried over from a microcontroller stack: ziggurat derives the
        # unique link keys the previous stack issued to devices from it. Both stacks
        # already store the seed as a plain hex string.
        stack_specific = network_info.stack_specific
        tclk_seed = None
        tclk_flavor = None

        if "zstack" in stack_specific and "tclk_seed" in stack_specific["zstack"]:
            tclk_seed = stack_specific["zstack"]["tclk_seed"]
            tclk_flavor = "zstack"
        elif "ezsp" in stack_specific and "hashed_tclk" in stack_specific["ezsp"]:
            tclk_seed = stack_specific["ezsp"]["hashed_tclk"]
            tclk_flavor = "ezsp"

        assert self._api is not None

        # `UNKNOWN` is assigned after the class body, where mypy cannot see it
        if node_info.ieee == t.EUI64.UNKNOWN:  # type: ignore[attr-defined]
            # zigpy leaves the IEEE address unspecified when forming a new network,
            # deferring to the radio's hardware address
            rsp = await self._api.request(GetHwAddress())
            node_info = node_info.replace(ieee=rsp.ieee_address)

        await self._api.request(
            Configure(
                channel=network_info.channel,
                # None means "pick automatically": the server applies its safe default
                tx_power=network_info.tx_power,
                nwk_update_id=network_info.nwk_update_id,
                pan_id=network_info.pan_id,
                extended_pan_id=network_info.extended_pan_id,
                nwk_address=node_info.nwk,
                ieee_address=node_info.ieee,
                network_key=network_info.network_key.key,
                network_key_seq=network_info.network_key.seq,
                network_key_tx_counter=network_info.network_key.tx_counter,
                tc_link_key=network_info.tc_link_key.key,
                source_routing=self.config[zigpy.config.CONF_SOURCE_ROUTING],
                # Unique trust center link keys negotiated in earlier sessions
                key_table=[
                    KeyTableEntry(partner_ieee=key.partner_ieee, key=key.key)
                    for key in network_info.key_table
                ],
                tclk_seed=tclk_seed,
                tclk_flavor=tclk_flavor,
            )
        )

        # Ziggurat has no persistent storage of its own: zigpy's backup database is
        # the network's NVRAM, so the settings just written are recorded there for
        # `start_network` to find
        self.state.network_info = network_info
        self.state.node_info = node_info
        self.backups.add_backup(
            zigpy.backups.NetworkBackup(network_info=network_info, node_info=node_info)
        )

    async def reset_network_info(self) -> None:
        pass

    async def _watchdog_feed(self) -> None:
        assert self._api is not None
        await self._api.request(Ping())

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

    def on_notification(self, notification: Notification) -> None:
        match notification:
            case ReceivedApsCommand():
                self._handle_received_aps_command(notification)
            case FrameCounterUpdate():
                self.state.network_info.network_key.tx_counter = (
                    notification.frame_counter
                )
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
            case DeviceJoined():
                self._handle_device_joined(
                    notification.nwk, notification.ieee, notification.parent
                )
            case DeviceLeft():
                if notification.ieee is not None:
                    ieee = notification.ieee
                else:
                    try:
                        ieee = self.get_device(nwk=notification.nwk).ieee
                    except KeyError:
                        return

                self.handle_leave(nwk=notification.nwk, ieee=ieee)
            case LinkKeyUpdate():
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

    def _handle_received_aps_command(self, command: ReceivedApsCommand) -> None:
        if command.group is not None:
            dst = t.AddrModeAddress(
                addr_mode=t.AddrMode.Group,
                address=t.Group(command.group),
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
        aps_encryption = t.TransmitOptions.APS_Encryption in packet.tx_options

        dst = packet.dst
        assert dst is not None and dst.address is not None

        destination: t.NWK | None = None
        destination_eui64: t.EUI64 | None = None

        if dst.addr_mode == t.AddrMode.IEEE:
            # The server resolves the EUI64 to a network address
            destination_eui64 = cast(t.EUI64, dst.address)
            delivery_mode = "unicast"
        else:
            destination = t.NWK(dst.address)
            delivery_mode = {
                t.AddrMode.NWK: "unicast",
                t.AddrMode.Group: "multicast",
                t.AddrMode.Broadcast: "broadcast",
            }[dst.addr_mode]

            if aps_encryption:
                # The server selects the link key by EUI64
                destination_eui64 = self.get_device(nwk=destination).ieee

        # Resolves once the frame is on the air (EZSP `messageSent` parity); the
        # APS-ack delivery result arrives later and is logged by the API layer
        assert self._api is not None
        await self._api.request_transmitted(
            SendAps(
                delivery_mode=delivery_mode,
                destination_eui64=destination_eui64,
                destination=destination,
                profile_id=packet.profile_id,
                cluster_id=packet.cluster_id or 0x0000,
                src_ep=packet.src_ep or 0,
                dst_ep=packet.dst_ep or 0,
                aps_ack=t.TransmitOptions.ACK in packet.tx_options,
                aps_encryption=aps_encryption,
                radius=packet.radius or 30,
                aps_seq=packet.tsn,
                priority=packet.priority if packet.priority is not None else 0,
                data=packet.data.serialize(),
            )
        )
