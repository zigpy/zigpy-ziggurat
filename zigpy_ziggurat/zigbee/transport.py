"""Frame transports for the binary protocol: serial (Spinel tunnel), binary
WebSocket, and a JSON-transcoding WebSocket for early users on the legacy server."""

from __future__ import annotations

import asyncio
from collections.abc import Callable
import json
import logging
from typing import Any, Protocol, cast

import aiohttp
import aiospinel
import zigpy.serial
import zigpy.types as t

from zigpy_ziggurat.zigbee import legacy, protocol as p

_LOGGER = logging.getLogger(__name__)

WEBSOCKET_HEARTBEAT = 15

PROP_VENDOR_ZIGGURAT = aiospinel.PackedUInt21(0x3D5A)

# The callback the API installs to receive a device -> host binary frame.
OnFrame = Callable[[bytes], None]
OnLost = Callable[[BaseException | None], None]


# The server must announce itself with a hello within this window.
HANDSHAKE_TIMEOUT = 5


class Transport(Protocol):
    """Moves binary protocol frames between the API and a device, once connected."""

    async def disconnect(self) -> None: ...

    async def send_frame(self, frame: bytes) -> None: ...


async def connect_transport(
    url: str,
    on_frame: OnFrame,
    on_lost: OnLost,
    *,
    baudrate: int = 115200,
    flow_control: str | None = None,
) -> Transport:
    """Open a connected transport for `url`, probing a WebSocket for its protocol."""
    if not url.startswith(("ws://", "wss://", "ws+unix://")):
        spinel = SpinelTransport(
            url, on_frame, on_lost, baudrate=baudrate, flow_control=flow_control
        )
        await spinel.connect()
        return spinel

    return await _probe_websocket(url, on_frame, on_lost)


# -- serial (Spinel tunnel) ------------------------------------------------------


class _SpinelProtocol(aiospinel.SpinelProtocol):
    """Tunnels binary frames over the vendor Spinel stream property."""

    def __init__(self, on_frame: OnFrame, on_lost: OnLost) -> None:
        super().__init__()
        self._on_frame = on_frame
        self._on_lost = on_lost
        self.add_property_listener(PROP_VENDOR_ZIGGURAT, self._stream_frame_received)

    def connection_lost(self, exc: BaseException | None) -> None:
        super().connection_lost(exc)
        self._on_lost(exc)

    def _stream_frame_received(self, data: bytes) -> None:
        # Responses to our own property SETs also land here, with no payload.
        if len(data) < 2:
            return
        length = int.from_bytes(data[:2], "little")
        try:
            self._on_frame(data[2 : 2 + length])
        except Exception:
            _LOGGER.exception("Failed to handle frame: %r", data)

    async def start_ziggurat(self) -> None:
        rsp = await self.send_command(
            aiospinel.CommandID.PROP_VALUE_GET,
            PROP_VENDOR_ZIGGURAT.serialize(),
        )
        prop_id, _ = aiospinel.PackedUInt21.deserialize(rsp.data)
        if prop_id != PROP_VENDOR_ZIGGURAT:
            raise ConnectionError(
                f"Firmware does not embed the Ziggurat stack: {rsp!r}"
            )
        _LOGGER.debug("Embedded Ziggurat firmware detected")

    async def tunnel_send(self, frame: bytes) -> None:
        # No retries: a timed-out tunnel write must not resend the request (the first
        # copy may already have been processed).
        rsp = await self.send_command(
            aiospinel.CommandID.PROP_VALUE_SET,
            (
                PROP_VENDOR_ZIGGURAT.serialize()
                + len(frame).to_bytes(2, "little")
                + frame
            ),
            retries=0,
        )
        prop_id, _ = aiospinel.PackedUInt21.deserialize(rsp.data)
        if prop_id != PROP_VENDOR_ZIGGURAT:
            raise ConnectionError(f"Tunnel write rejected: {rsp!r}")


class SpinelTransport:
    """The binary protocol tunneled over a serial OpenThread RCP's Spinel stream."""

    def __init__(
        self,
        url: str,
        on_frame: OnFrame,
        on_lost: OnLost,
        *,
        baudrate: int = 115200,
        flow_control: str | None = None,
    ) -> None:
        self._url = url
        self._on_frame = on_frame
        self._on_lost = on_lost
        self._baudrate = baudrate
        self._flow_control = flow_control
        self._protocol: _SpinelProtocol | None = None

    async def connect(self) -> None:
        _, protocol = await zigpy.serial.create_serial_connection(
            loop=asyncio.get_running_loop(),
            protocol_factory=lambda: _SpinelProtocol(self._on_frame, self._on_lost),
            url=self._url,
            baudrate=self._baudrate,
            flow_control=cast(Any, self._flow_control),
        )
        self._protocol = cast(_SpinelProtocol, protocol)
        await self._protocol.wait_until_connected()
        await self._protocol.start_ziggurat()

    async def disconnect(self) -> None:
        if self._protocol is not None:
            self._protocol.close()
            await self._protocol.wait_until_closed()
            self._protocol = None

    async def send_frame(self, frame: bytes) -> None:
        assert self._protocol is not None
        await self._protocol.tunnel_send(frame)


# -- WebSocket -------------------------------------------------------------------


async def _open_websocket(
    url: str,
) -> tuple[aiohttp.ClientSession, aiohttp.ClientWebSocketResponse]:
    if url.startswith("ws+unix://"):
        # The URL's path is the socket path; the HTTP host is a placeholder.
        connector: aiohttp.BaseConnector | None = aiohttp.UnixConnector(
            path=url.removeprefix("ws+unix://")
        )
        ws_url = "ws://localhost/"
    else:
        connector = None
        ws_url = url

    session = aiohttp.ClientSession(connector=connector)
    websocket = await session.ws_connect(ws_url, heartbeat=WEBSOCKET_HEARTBEAT)
    return session, websocket


async def _probe_websocket(url: str, on_frame: OnFrame, on_lost: OnLost) -> Transport:
    """Pick the transport from the server's opening hello: binary frame or JSON text."""
    session, websocket = await _open_websocket(url)
    async with asyncio.timeout(HANDSHAKE_TIMEOUT):
        hello = await websocket.receive()

    if hello.type == aiohttp.WSMsgType.BINARY:
        transport: _WebSocketBase = WebSocketTransport(on_frame, on_lost)
        _LOGGER.debug("Detected binary WebSocket protocol")
    elif hello.type == aiohttp.WSMsgType.TEXT:
        transport = LegacyWebSocketTransport(on_frame, on_lost)
        _LOGGER.debug("Detected legacy JSON WebSocket protocol: %s", hello.data)
        _LOGGER.warning(
            "The legacy JSON WebSocket protocol will be removed soon. Please upgrade"
            " the Ziggurat app to switch to the new binary protocol."
        )
    else:
        await session.close()
        raise ConnectionError(f"Unexpected handshake from ziggurat: {hello!r}")

    transport._adopt(session, websocket)
    return transport


class _WebSocketBase:
    """Shared aiohttp WebSocket plumbing, driven from a socket passed to `_adopt`."""

    def __init__(self, on_frame: OnFrame, on_lost: OnLost) -> None:
        self._on_frame = on_frame
        self._on_lost = on_lost
        self._session: aiohttp.ClientSession | None = None
        self._websocket: aiohttp.ClientWebSocketResponse | None = None
        self._receiver_task: asyncio.Task[None] | None = None

    def _adopt(
        self,
        session: aiohttp.ClientSession,
        websocket: aiohttp.ClientWebSocketResponse,
    ) -> None:
        self._session = session
        self._websocket = websocket
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
                if msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.BINARY):
                    try:
                        self._handle_message(msg)
                    except Exception:
                        _LOGGER.exception("Failed to handle message: %r", msg.data)
                elif msg.type == aiohttp.WSMsgType.ERROR:
                    exc = websocket.exception()
                    break
        except asyncio.CancelledError:
            # A deliberate disconnect; the API is already tearing down.
            self._on_lost(None)
            raise
        self._on_lost(exc)

    def _handle_message(self, msg: aiohttp.WSMessage) -> None:
        raise NotImplementedError

    async def send_frame(self, frame: bytes) -> None:
        raise NotImplementedError

    async def _send(self, data: bytes | str) -> None:
        if self._websocket is None:
            raise ConnectionError("Not connected")
        if isinstance(data, str):
            await self._websocket.send_str(data)
        else:
            await self._websocket.send_bytes(data)


class WebSocketTransport(_WebSocketBase):
    """The binary protocol carried as WebSocket binary frames."""

    def _handle_message(self, msg: aiohttp.WSMessage) -> None:
        if msg.type == aiohttp.WSMsgType.BINARY:
            self._on_frame(msg.data)

    async def send_frame(self, frame: bytes) -> None:
        await self._send(frame)


# JSON error code -> binary status
_STATUS_BY_CODE: dict[str, p.Status] = {
    "parse": p.Status.MALFORMED_PAYLOAD,
    "unknown_command": p.Status.UNKNOWN_COMMAND,
    # The legacy server's lone state error is a load after the network started.
    "invalid_state": p.Status.ALREADY_STARTED,
    "not_configured": p.Status.NOT_CONFIGURED,
    "radio_error": p.Status.RADIO_ERROR,
    "network_start_failed": p.Status.NETWORK_START_FAILED,
    "transmit_failed": p.Status.RADIO_ERROR,
    "scan_failed": p.Status.SCAN_FAILED,
    "invalid_request": p.Status.INVALID_REQUEST,
}

_LEAVE_REASONS: dict[legacy.DeviceLeaveReason, p.LeaveReason] = {
    legacy.DeviceLeaveReason.ANNOUNCED: p.LeaveReason.ANNOUNCED,
    legacy.DeviceLeaveReason.ROUTER_REPORTED: p.LeaveReason.ROUTER_REPORTED,
    legacy.DeviceLeaveReason.KEEPALIVE_TIMEOUT: p.LeaveReason.KEEPALIVE_TIMEOUT,
}

_RUST_LOG_LEVELS = {
    "ERROR": logging.ERROR,
    "WARN": logging.WARNING,
    "INFO": logging.INFO,
    "DEBUG": logging.DEBUG,
    "TRACE": 5,
}


class LegacyWebSocketTransport(_WebSocketBase):
    """Transcodes the binary protocol to/from the legacy JSON-RPC server."""

    def __init__(self, on_frame: OnFrame, on_lost: OnLost) -> None:
        super().__init__(on_frame, on_lost)
        # request id -> command, so a JSON response builds the right binary reply
        self._pending_commands: dict[int, p.RequestCommand] = {}
        # The binary protocol splits `configure` (Configure + LoadKeyTable* +
        # StartNetwork) that the JSON server takes as one call; coalesce it.
        self._pending_configure: p.Configure | None = None
        self._pending_keys: list[p.KeyEntry] = []
        # The key table the JSON get_network_info returns inline, replayed as the
        # events of the ScanKeyTable that follows on the binary side.
        self._scan_keys: list[p.KeyEntry] = []

    # -- outbound: binary frame -> JSON request ------------------------------------

    async def send_frame(self, frame: bytes) -> None:
        header, body = p.Header.deserialize(frame)
        command = p.RequestCommand(header.command)
        request_id = int(header.request_id)
        request = p.REQUESTS[command].deserialize(body)[0]

        if command in (p.RequestCommand.SHUTDOWN, p.RequestCommand.RESET):
            # The legacy server has neither shutdown nor reset; it replaces the stack
            # on `configure`. OK them locally so callers don't depend on either.
            self._emit_ok(command, request_id)
        elif command == p.RequestCommand.CONFIGURE:
            self._pending_configure = cast(p.Configure, request)
            self._pending_keys = []
            self._emit_ok(command, request_id)
        elif command == p.RequestCommand.LOAD_KEY_TABLE:
            self._pending_keys.extend(cast(p.LoadKeyTable, request).entries)
            self._emit_ok(command, request_id)
        elif command in (
            p.RequestCommand.LOAD_CHILDREN,
            p.RequestCommand.LOAD_ADDRESS_CACHE,
            p.RequestCommand.LOAD_ROUTE_TABLE,
            p.RequestCommand.LOAD_SOURCE_ROUTES,
        ):
            # The legacy server re-learns its topology tables, so acknowledge these
            # restore loads locally and drop them.
            self._emit_ok(command, request_id)
        elif command == p.RequestCommand.START_NETWORK:
            assert self._pending_configure is not None
            params = self._configure_params(self._pending_configure, self._pending_keys)
            self._pending_configure = None
            self._pending_keys = []
            self._pending_commands[request_id] = command
            await self._send_json(request_id, "configure", params)
        elif command == p.RequestCommand.GET_NETWORK_INFO:
            self._pending_commands[request_id] = command
            await self._send_json(request_id, "get_network_info", {})
        elif command == p.RequestCommand.SCAN_KEY_TABLE:
            for entry in self._scan_keys:
                self._emit(p.FrameType.EVENT, command, request_id, entry.serialize())
            count = p.ScanCount(count=t.uint16_t(len(self._scan_keys)))
            self._emit_ok(command, request_id, count)
            self._scan_keys = []
        elif command in (
            p.RequestCommand.SCAN_CHILDREN,
            p.RequestCommand.SCAN_ADDRESS_CACHE,
            p.RequestCommand.SCAN_ROUTE_TABLE,
        ):
            # The JSON server surfaces only the key table (inline in get_network_info);
            # it has no children/address/route scans, so these stream empty. The app
            # re-learns that topology from join notifications during the transition.
            self._emit_ok(command, request_id, p.ScanCount(count=t.uint16_t(0)))
        elif command == p.RequestCommand.CANCEL_REQUEST:
            # The legacy server has no request-cancel concept, so the best-effort
            # cancel from `ZigguratApi._cancel_send` is dropped here.
            pass
        else:
            method, params = self._encode_request(command, request)
            self._pending_commands[request_id] = command
            await self._send_json(request_id, method, params)

    async def _send_json(
        self, request_id: int, method: str, params: dict[str, Any]
    ) -> None:
        await self._send(
            json.dumps({"id": request_id, "method": method, "params": params})
        )

    def _encode_request(
        self, command: p.RequestCommand, request: p.Request
    ) -> tuple[str, dict[str, Any]]:
        if command == p.RequestCommand.GET_FIRMWARE_INFO:
            # The legacy server has no firmware-info call; `ping` keeps the liveness
            # probe end-to-end and the response is fabricated in `_handle_response`.
            return "ping", {}
        if command == p.RequestCommand.GET_HW_ADDRESS:
            return "get_hw_address", {}
        if command == p.RequestCommand.PERMIT_JOINS:
            permit = cast(p.PermitJoins, request)
            return (
                "permit_joins",
                legacy.PermitJoins(
                    duration=int(permit.duration),
                    accept_direct_joins=bool(permit.accept_direct_joins),
                ).to_dict(),
            )
        if command == p.RequestCommand.SET_CHANNEL:
            channel = int(cast(p.SetChannel, request).channel)
            return "set_channel", legacy.SetChannel(channel=channel).to_dict()
        if command == p.RequestCommand.SET_NWK_UPDATE_ID:
            update_id = int(cast(p.SetNwkUpdateId, request).nwk_update_id)
            return (
                "set_nwk_update_id",
                legacy.SetNwkUpdateId(nwk_update_id=update_id).to_dict(),
            )
        if command == p.RequestCommand.SET_PROVISIONAL_KEY:
            key = cast(p.SetProvisionalKey, request)
            return (
                "set_provisional_key",
                legacy.SetProvisionalKey(ieee=key.ieee, key=key.key).to_dict(),
            )
        if command == p.RequestCommand.ENERGY_SCAN:
            scan = cast(p.EnergyScan, request)
            return (
                "energy_scan",
                legacy.EnergyScan(
                    channels=[int(c) for c in scan.channels],
                    duration_per_channel_ms=int(scan.duration_per_channel_ms),
                ).to_dict(),
            )
        if command == p.RequestCommand.NETWORK_SCAN:
            net_scan = cast(p.NetworkScan, request)
            return (
                "network_scan",
                legacy.NetworkScan(
                    channels=[int(c) for c in net_scan.channels],
                    duration_per_channel_ms=int(net_scan.duration_per_channel_ms),
                ).to_dict(),
            )
        if command == p.RequestCommand.PACKET_CAPTURE:
            channel = int(cast(p.PacketCapture, request).channel)
            return "packet_capture", legacy.PacketCapture(channel=channel).to_dict()
        if command == p.RequestCommand.PACKET_CAPTURE_CHANNEL:
            channel = int(cast(p.PacketCaptureChannel, request).channel)
            return (
                "packet_capture_change_channel",
                legacy.PacketCaptureChangeChannel(channel=channel).to_dict(),
            )
        if command == p.RequestCommand.SEND_UNICAST:
            return "send_aps", self._send_unicast_params(cast(p.SendUnicast, request))
        if command == p.RequestCommand.SEND_BROADCAST:
            return "send_aps", self._send_broadcast_params(
                cast(p.SendBroadcast, request)
            )
        if command == p.RequestCommand.SEND_GROUPCAST:
            return "send_aps", self._send_groupcast_params(
                cast(p.SendGroupcast, request)
            )
        raise ValueError(f"Cannot transcode {command!r} to JSON")

    def _send_unicast_params(self, request: p.SendUnicast) -> dict[str, Any]:
        destination = (
            None if request.destination == t.NWK(0xFFFE) else t.NWK(request.destination)
        )
        return legacy.SendAps(
            delivery_mode="unicast",
            destination_eui64=request.destination_eui64 if request.has_eui64 else None,
            destination=destination,
            profile_id=int(request.profile_id),
            cluster_id=int(request.cluster_id),
            src_ep=int(request.src_ep),
            dst_ep=int(request.dst_ep),
            aps_ack=bool(request.aps_ack),
            aps_seq=int(request.aps_seq),
            radius=int(request.radius),
            aps_encryption=bool(request.aps_encryption),
            priority=int(request.priority),
            data=bytes(request.asdu),
        ).to_dict()

    def _send_broadcast_params(self, request: p.SendBroadcast) -> dict[str, Any]:
        return legacy.SendAps(
            delivery_mode="broadcast",
            destination_eui64=None,
            destination=t.NWK(request.destination),
            profile_id=int(request.profile_id),
            cluster_id=int(request.cluster_id),
            src_ep=int(request.src_ep),
            dst_ep=int(request.dst_ep),
            aps_ack=False,
            aps_seq=int(request.aps_seq),
            radius=int(request.radius),
            aps_encryption=False,
            priority=int(request.priority),
            data=bytes(request.asdu),
        ).to_dict()

    def _send_groupcast_params(self, request: p.SendGroupcast) -> dict[str, Any]:
        # The legacy server carried the group id in `destination` for a multicast.
        return legacy.SendAps(
            delivery_mode="multicast",
            destination_eui64=None,
            destination=t.NWK(request.group_id),
            profile_id=int(request.profile_id),
            cluster_id=int(request.cluster_id),
            src_ep=int(request.src_ep),
            dst_ep=0,
            aps_ack=False,
            aps_seq=int(request.aps_seq),
            radius=int(request.radius),
            aps_encryption=False,
            priority=int(request.priority),
            data=bytes(request.asdu),
        ).to_dict()

    def _configure_params(
        self, configure: p.Configure, keys: list[p.KeyEntry]
    ) -> dict[str, Any]:
        state = configure.state
        seed = bytes(state.tclk_seed).hex() if state.has_tclk_seed else None
        flavor = None
        if state.has_tclk_seed:
            flavor = "zstack" if state.tclk_flavor == p.TclkFlavorId.Z_STACK else "ezsp"
        return legacy.Configure(
            channel=int(state.channel),
            nwk_update_id=int(state.nwk_update_id),
            pan_id=state.pan_id,
            extended_pan_id=state.extended_pan_id,
            nwk_address=state.nwk_address,
            ieee_address=state.ieee_address,
            network_key=state.network_key,
            network_key_seq=int(state.network_key_seq),
            network_key_tx_counter=int(state.network_key_tx_counter),
            tc_link_key=state.tc_link_key,
            source_routing=bool(configure.source_routing),
            tx_power=int(state.tx_power),
            key_table=[
                legacy.KeyTableEntry(partner_ieee=k.partner_ieee, key=k.key)
                for k in keys
            ],
            tclk_seed=seed,
            tclk_flavor=flavor,
            aps_frame_counter=int(state.aps_frame_counter),
        ).to_dict()

    # -- inbound: JSON message -> binary frame -------------------------------------

    def _handle_message(self, msg: aiohttp.WSMessage) -> None:
        if msg.type != aiohttp.WSMsgType.TEXT:
            return
        message = json.loads(msg.data)
        kind = message["type"]
        if kind == "response":
            self._handle_response(message)
        elif kind == "event":
            self._handle_event(message)
        elif kind == "notification":
            self._handle_notification(message)

    def _handle_response(self, message: dict[str, Any]) -> None:
        request_id = message["id"]
        if request_id not in self._pending_commands:
            return
        command = self._pending_commands.pop(request_id)

        if "error" in message:
            error = message["error"]
            code = error["code"]
            # A JSON code with no binary status (a host-side failure the firmware
            # can't produce) degrades to a generic invalid-request.
            status = _STATUS_BY_CODE.get(code, p.Status.INVALID_REQUEST)
            # The binary protocol carries only the status; the diagnostic text
            # becomes a log line, like the binary server's own warnings.
            _LOGGER.warning(
                "Legacy server error for %r (id=%d): %s: %s",
                command,
                request_id,
                code,
                error["message"],
            )
            self._emit(p.FrameType.RESPONSE, command, request_id, bytes([status]))
        elif command == p.RequestCommand.GET_FIRMWARE_INFO:
            # Transcoded to a JSON `ping`, which has no result: fabricate the payload.
            self._emit_ok(
                command,
                request_id,
                p.FirmwareInfo(
                    protocol_version=t.uint8_t(p.PROTOCOL_VERSION),
                    version=t.LongCharacterString("ziggurat/legacy"),
                ),
            )
        elif command == p.RequestCommand.GET_NETWORK_INFO:
            self._emit_ok(command, request_id, self._network_info(message["result"]))
        elif command == p.RequestCommand.GET_HW_ADDRESS:
            hw = legacy.HwAddress.from_dict(message["result"])
            self._emit_ok(command, request_id, p.HwAddress(ieee=hw.ieee_address))
        else:
            self._emit_ok(command, request_id)

    def _handle_event(self, message: dict[str, Any]) -> None:
        request_id = message["id"]
        event = message["event"]
        if event == "transmitted":
            # The legacy send handoff, delivered as a bare event; the binary protocol
            # models it as a `send_confirm` notification keyed by request id.
            self._emit_notification(
                p.NotificationCommand.SEND_CONFIRM,
                request_id,
                p.SendConfirm(status=p.SendStatus.SUCCESS),
            )
            return
        if event == "energy_result":
            result = legacy.EnergyScanResult.from_dict(message["data"])
            payload: p.Response = p.EnergyResult(
                channel=t.uint8_t(result.channel), rssi=t.int8s(result.rssi)
            )
            command = p.RequestCommand.ENERGY_SCAN
        elif event == "network_found":
            payload = self._beacon(message["data"])
            command = p.RequestCommand.NETWORK_SCAN
        elif event == "captured_packet":
            packet = legacy.CapturedPacketEvent.from_dict(message["data"])
            payload = p.CapturedPacket(
                channel=t.uint8_t(packet.channel),
                rssi=t.int8s(packet.rssi),
                lqi=t.uint8_t(packet.lqi),
                psdu=t.LongOctetString(bytes.fromhex(packet.data)),
            )
            command = p.RequestCommand.PACKET_CAPTURE
        else:
            # `accepted` and any other bare event have no binary equivalent.
            return
        self._emit(p.FrameType.EVENT, command, request_id, payload.serialize())

    def _handle_notification(self, message: dict[str, Any]) -> None:
        event = message["event"]
        data = message["data"]
        if event == "log":
            self._handle_log(data)
        elif event == "send_confirm":
            self._emit_notification(
                p.NotificationCommand.SEND_CONFIRM, data["id"], self._send_confirm(data)
            )
        elif event == "aps_ack_confirm":
            self._emit_notification(
                p.NotificationCommand.APS_ACK_CONFIRM,
                data["id"],
                self._aps_ack_confirm(data),
            )
        elif event == "received_aps_command":
            self._emit_notification(
                p.NotificationCommand.RECEIVED_APS, 0, self._received_aps(data)
            )
        elif event == "frame_counter_update":
            counter = legacy.FrameCounterUpdate.from_dict(data)
            self._emit_notification(
                p.NotificationCommand.FRAME_COUNTER,
                0,
                p.FrameCounter(frame_counter=t.uint32_t(counter.frame_counter)),
            )
        elif event == "link_key_update":
            link = legacy.LinkKeyUpdate.from_dict(data)
            self._emit_notification(
                p.NotificationCommand.LINK_KEY,
                0,
                p.LinkKey(ieee=link.ieee, key=link.key),
            )
        elif event == "device_joined":
            joined = legacy.DeviceJoined.from_dict(data)
            self._emit_notification(
                p.NotificationCommand.DEVICE_JOINED,
                0,
                p.DeviceJoined(
                    nwk=joined.nwk,
                    ieee=joined.ieee,
                    parent=joined.parent,
                    # The legacy JSON protocol carries no capability information
                    rx_on_when_idle=t.uint1_t(1),
                    device_type=p.ChildDeviceType.UNKNOWN,
                    reserved=t.uint5_t(0),
                ),
            )
        elif event == "device_left":
            self._emit_notification(
                p.NotificationCommand.DEVICE_LEFT, 0, self._device_left(data)
            )
        elif event == "aps_decryption_failure":
            self._emit_notification(
                p.NotificationCommand.APS_DECRYPT_FAILURE,
                0,
                self._aps_decrypt_failure(data),
            )

    def _handle_log(self, data: dict[str, Any]) -> None:
        level = _RUST_LOG_LEVELS.get(data["level"], logging.INFO)
        logger = logging.getLogger("ziggurat.fw." + data["target"].replace("::", "."))
        logger.log(level, "%s", data["message"])

    # -- inbound payload builders --------------------------------------------------

    def _network_info(self, result: dict[str, Any]) -> p.NetworkInfo:
        info = legacy.NetworkInfo.from_dict(result)
        self._scan_keys = [
            p.KeyEntry(
                key=entry.key,
                tx_counter=t.uint32_t(0),
                rx_counter=t.uint32_t(0),
                seq=t.uint8_t(0),
                partner_ieee=entry.partner_ieee,
            )
            for entry in info.key_table
        ]
        seed = info.tclk_seed
        state = p.NetworkState(
            channel=t.uint8_t(info.channel),
            nwk_update_id=t.uint8_t(info.nwk_update_id),
            pan_id=info.pan_id,
            extended_pan_id=info.extended_pan_id,
            nwk_address=info.nwk_address,
            ieee_address=info.ieee_address,
            network_key=info.network_key,
            network_key_seq=t.uint8_t(info.network_key_seq),
            network_key_tx_counter=t.uint32_t(info.network_key_tx_counter),
            tc_link_key=info.tc_link_key,
            has_tclk_seed=t.Bool(seed is not None),
            tclk_seed=t.KeyData(bytes.fromhex(seed) if seed is not None else bytes(16)),
            tclk_flavor=(
                p.TclkFlavorId.Z_STACK
                if info.tclk_flavor == "zstack"
                else p.TclkFlavorId.EZSP
            ),
            tx_power=t.int8s(info.tx_power),
            aps_frame_counter=t.uint32_t(info.aps_frame_counter),
        )
        return p.NetworkInfo(
            state=state,
            key_count=t.uint16_t(len(info.key_table)),
            started=t.Bool(info.started),
        )

    def _beacon(self, data: dict[str, Any]) -> p.Beacon:
        beacon = legacy.NetworkBeaconEvent.from_dict(data)
        return p.Beacon(
            channel=t.uint8_t(beacon.channel),
            source=beacon.source if beacon.source is not None else t.NWK(0xFFFF),
            pan_id=beacon.pan_id,
            extended_pan_id=beacon.extended_pan_id,
            permit_joining=t.uint1_t(beacon.permit_joining),
            router_capacity=t.uint1_t(beacon.router_capacity),
            end_device_capacity=t.uint1_t(beacon.end_device_capacity),
            reserved=t.uint5_t(0),
            stack_profile=t.uint8_t(beacon.stack_profile),
            protocol_version=t.uint8_t(beacon.protocol_version),
            device_depth=t.uint8_t(beacon.device_depth),
            update_id=t.uint8_t(beacon.update_id),
            lqi=t.uint8_t(beacon.lqi),
            rssi=t.int8s(beacon.rssi),
        )

    def _send_confirm(self, data: dict[str, Any]) -> p.SendConfirm:
        return p.SendConfirm(
            # The legacy JSON protocol carries no failure kind; a transmit failure is
            # the least-wrong stand-in.
            status=(
                p.SendStatus.SUCCESS
                if data["status"] == "confirmed"
                else p.SendStatus.TRANSMIT_FAILED
            ),
        )

    def _aps_ack_confirm(self, data: dict[str, Any]) -> p.ApsAckConfirm:
        return p.ApsAckConfirm(
            status=(
                p.SendStatus.SUCCESS
                if data["status"] == "confirmed"
                else p.SendStatus.APS_ACK_TIMEOUT
            ),
        )

    def _received_aps(self, data: dict[str, Any]) -> p.ReceivedAps:
        received = legacy.ReceivedApsCommand.from_dict(data)
        return p.ReceivedAps(
            source=received.source,
            destination=received.destination,
            has_group=t.Bool(received.group is not None),
            group=t.uint16_t(received.group or 0),
            profile_id=t.uint16_t(received.profile_id),
            cluster_id=t.uint16_t(received.cluster_id),
            src_ep=t.uint8_t(received.src_ep),
            dst_ep=t.uint8_t(received.dst_ep),
            lqi=t.uint8_t(received.lqi),
            rssi=t.int8s(received.rssi),
            data=t.LongOctetString(received.data),
        )

    def _device_left(self, data: dict[str, Any]) -> p.DeviceLeft:
        left = legacy.DeviceLeft.from_dict(data)
        return p.DeviceLeft(
            nwk=left.nwk,
            has_ieee=t.uint1_t(left.ieee is not None),
            rejoin=t.uint1_t(bool(left.rejoin)),
            has_router_ieee=t.uint1_t(left.router_ieee is not None),
            reserved=t.uint5_t(0),
            ieee=left.ieee if left.ieee is not None else t.EUI64([0] * 8),
            reason=_LEAVE_REASONS[left.reason],
            router=left.router if left.router is not None else t.NWK(0xFFFF),
            router_ieee=(
                left.router_ieee if left.router_ieee is not None else t.EUI64([0] * 8)
            ),
        )

    def _aps_decrypt_failure(self, data: dict[str, Any]) -> p.ApsDecryptFailure:
        failure = legacy.ApsDecryptionFailure.from_dict(data)
        key_id = p.KeyId.NETWORK
        name = failure.key_id.upper()
        if name in p.KeyId.__members__:
            key_id = p.KeyId[name]
        return p.ApsDecryptFailure(
            source=failure.source,
            source_ieee=failure.source_ieee,
            frame_counter=t.uint32_t(failure.frame_counter),
            key_id=key_id,
        )

    # -- frame emission ------------------------------------------------------------

    def _emit(
        self,
        frame_type: p.FrameType,
        command: p.RequestCommand | p.NotificationCommand,
        request_id: int,
        body: bytes = b"",
    ) -> None:
        self._on_frame(p.encode_reply(frame_type, command, request_id, body))

    def _emit_ok(
        self,
        command: p.RequestCommand,
        request_id: int,
        payload: p.Response | None = None,
    ) -> None:
        body = bytes([p.Status.OK]) + (
            payload.serialize() if payload is not None else b""
        )
        self._emit(p.FrameType.RESPONSE, command, request_id, body)

    def _emit_notification(
        self, command: p.NotificationCommand, request_id: int, payload: p.Notification
    ) -> None:
        self._emit(p.FrameType.NOTIFICATION, command, request_id, payload.serialize())
