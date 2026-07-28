"""A synthetic ziggurat server: a real aiohttp websocket server speaking the binary
wire protocol, with per-command handlers that tests can override. The harness decodes
request frames and encodes replies through the same structs as the client, so every
serialization strategy is exercised in both directions."""

import asyncio
import base64
from collections.abc import AsyncIterator, Awaitable, Callable, Sequence
import hashlib
from typing import Any, TypeVar

from aiohttp import web
from aiohttp.test_utils import TestServer
import aiospinel
import pytest
import zigpy.config
import zigpy.types as t

from zigpy_ziggurat.zigbee import protocol as p
from zigpy_ziggurat.zigbee.application import ControllerApplication
from zigpy_ziggurat.zigbee.transport import PROP_VENDOR_ZIGGURAT

COORDINATOR_IEEE = t.EUI64.convert("00:11:22:33:44:55:66:77")
NETWORK_KEY = t.KeyData.convert("11:22:33:44:55:66:77:88:99:aa:bb:cc:dd:ee:ff:00")
TC_LINK_KEY = t.KeyData(b"ZigBeeAlliance09")

DEVICE_IEEE = t.EUI64.convert("aa:aa:aa:aa:aa:aa:aa:aa")
DEVICE_NWK = t.NWK(0xAB12)
LINK_KEY = t.KeyData.convert("00:11:22:33:44:55:66:77:88:99:aa:bb:cc:dd:ee:ff")

REQUEST_T = TypeVar("REQUEST_T")

Handler = Callable[[Any, int], Awaitable[p.Response | None]]

# Notification struct -> command id, the inverse of the client's decode table.
NOTIFICATION_COMMANDS: dict[type[p.Notification], p.NotificationCommand] = {
    cls: command for command, cls in p.NOTIFICATIONS.items()
}


class StatusError(Exception):
    """Raised by a handler to reply with an error status instead of an OK."""

    def __init__(self, status: p.Status) -> None:
        super().__init__(status.name)
        self.status = status


def make_network_state() -> p.NetworkState:
    return p.NetworkState(
        channel=t.uint8_t(15),
        nwk_update_id=t.uint8_t(0),
        pan_id=t.PanId(0x1A2B),
        extended_pan_id=t.ExtendedPanId(t.EUI64.convert("aa:bb:cc:dd:ee:ff:00:11")),
        nwk_address=t.NWK(0x0000),
        ieee_address=COORDINATOR_IEEE,
        network_key=NETWORK_KEY,
        network_key_seq=t.uint8_t(0),
        network_key_tx_counter=t.uint32_t(1000),
        tc_link_key=TC_LINK_KEY,
        has_tclk_seed=t.Bool(False),
        tclk_seed=t.KeyData(bytes(16)),
        tclk_flavor=p.TclkFlavorId.EZSP,
        tx_power=t.int8s(8),
        aps_frame_counter=t.uint32_t(2000),
    )


class SyntheticZiggurat:
    """A websocket server speaking the binary protocol, driven by per-command
    handlers. A handler returns the OK response payload (`None` for an empty OK),
    emits any streamed events itself, and raises `StatusError` to reply with an
    error status."""

    def __init__(self) -> None:
        self.web_app = web.Application()
        self.web_app.router.add_get("/", self._handle_connection)
        self.url = ""
        self.connections = 0
        self._ws: web.WebSocketResponse | None = None
        self._transport: asyncio.Transport | None = None
        self.requests: list[p.Request] = []
        self.network_state = make_network_state()
        self.started = True
        self.hw_address = t.EUI64.convert("11:22:33:44:55:66:77:88")
        self.key_table: list[p.KeyEntry] = []
        self.children: list[p.ChildEntry] = []
        self.address_cache: list[p.AddressEntry] = []
        self.route_table: list[p.RouteEntry] = []
        self.beacons: list[p.Beacon] = []
        self.captured_packets: list[p.CapturedPacket] = []
        self.handlers: dict[p.RequestCommand, Handler] = {
            p.RequestCommand.RESET: self.on_empty_ok,
            p.RequestCommand.SHUTDOWN: self.on_empty_ok,
            p.RequestCommand.GET_FIRMWARE_INFO: self.on_get_firmware_info,
            p.RequestCommand.GET_HW_ADDRESS: self.on_get_hw_address,
            p.RequestCommand.GET_NETWORK_INFO: self.on_get_network_info,
            p.RequestCommand.SCAN_KEY_TABLE: self.on_scan_key_table,
            p.RequestCommand.SCAN_CHILDREN: self.on_scan_children,
            p.RequestCommand.SCAN_ADDRESS_CACHE: self.on_scan_address_cache,
            p.RequestCommand.SCAN_ROUTE_TABLE: self.on_scan_route_table,
            p.RequestCommand.CONFIGURE: self.on_empty_ok,
            p.RequestCommand.LOAD_KEY_TABLE: self.on_empty_ok,
            p.RequestCommand.LOAD_CHILDREN: self.on_empty_ok,
            p.RequestCommand.LOAD_ADDRESS_CACHE: self.on_empty_ok,
            p.RequestCommand.LOAD_ROUTE_TABLE: self.on_empty_ok,
            p.RequestCommand.LOAD_SOURCE_ROUTES: self.on_empty_ok,
            p.RequestCommand.START_NETWORK: self.on_empty_ok,
            p.RequestCommand.SEND_UNICAST: self.on_send_unicast,
            p.RequestCommand.SEND_BROADCAST: self.on_send_broadcast,
            p.RequestCommand.SEND_GROUPCAST: self.on_send_groupcast,
            p.RequestCommand.CANCEL_REQUEST: self.on_cancel_request,
            p.RequestCommand.PERMIT_JOINS: self.on_empty_ok,
            p.RequestCommand.SET_CHANNEL: self.on_empty_ok,
            p.RequestCommand.SET_NWK_UPDATE_ID: self.on_empty_ok,
            p.RequestCommand.SET_PROVISIONAL_KEY: self.on_empty_ok,
            p.RequestCommand.SET_TUNABLE: self.on_empty_ok,
            p.RequestCommand.ENERGY_SCAN: self.on_energy_scan,
            p.RequestCommand.NETWORK_SCAN: self.on_network_scan,
            p.RequestCommand.PACKET_CAPTURE: self.on_packet_capture,
            p.RequestCommand.PACKET_CAPTURE_CHANNEL: self.on_empty_ok,
        }

    @property
    def ws(self) -> web.WebSocketResponse:
        assert self._ws is not None
        return self._ws

    @property
    def transport(self) -> asyncio.Transport:
        assert self._transport is not None
        return self._transport

    @property
    def configured(self) -> p.NetworkState:
        return self.sent(p.Configure)[-1].state

    async def _handle_connection(
        self, http_request: web.Request
    ) -> web.WebSocketResponse:
        self.connections += 1
        self._transport = http_request.transport
        ws = web.WebSocketResponse()
        await ws.prepare(http_request)
        self._ws = ws

        await self.send_notification(
            p.Hello(
                protocol_version=t.uint8_t(p.PROTOCOL_VERSION), configured=t.Bool(True)
            )
        )

        async for msg in ws:
            header, body = p.Header.deserialize(msg.data)
            command = p.RequestCommand(header.command)
            request_id = int(header.request_id)
            request = p.REQUESTS[command].deserialize(body)[0]
            self.requests.append(request)

            try:
                response = await self.handlers[command](request, request_id)
            except StatusError as exc:
                await self._emit(
                    p.FrameType.RESPONSE, command, request_id, bytes([exc.status])
                )
            else:
                await self._emit(
                    p.FrameType.RESPONSE,
                    command,
                    request_id,
                    bytes([p.Status.OK])
                    + (response.serialize() if response is not None else b""),
                )

        return ws

    async def _emit(
        self,
        frame_type: p.FrameType,
        command: p.RequestCommand | p.NotificationCommand,
        request_id: int,
        body: bytes = b"",
    ) -> None:
        await self.ws.send_bytes(p.encode_reply(frame_type, command, request_id, body))

    async def send_event(
        self, command: p.RequestCommand, request_id: int, payload: p.Response
    ) -> None:
        await self._emit(p.FrameType.EVENT, command, request_id, payload.serialize())

    async def send_notification(
        self, notification: p.Notification, request_id: int = 0
    ) -> None:
        await self._emit(
            p.FrameType.NOTIFICATION,
            NOTIFICATION_COMMANDS[type(notification)],
            request_id,
            notification.serialize(),
        )

    async def send_raw(self, frame: bytes) -> None:
        await self.ws.send_bytes(frame)

    def sent(self, request_type: type[REQUEST_T]) -> list[REQUEST_T]:
        return [r for r in self.requests if isinstance(r, request_type)]

    async def wait_for(
        self, request_type: type[REQUEST_T], count: int = 1
    ) -> REQUEST_T:
        async with asyncio.timeout(2):
            while len(self.sent(request_type)) < count:
                await asyncio.sleep(0.01)

        return self.sent(request_type)[count - 1]

    # -- default handlers ----------------------------------------------------------

    async def on_empty_ok(self, request: p.Request, request_id: int) -> None:
        return None

    async def on_get_firmware_info(
        self, request: p.GetFirmwareInfo, request_id: int
    ) -> p.FirmwareInfo:
        return p.FirmwareInfo(
            protocol_version=t.uint8_t(p.PROTOCOL_VERSION),
            version=t.LongCharacterString("ziggurat/synthetic"),
        )

    async def on_get_hw_address(
        self, request: p.GetHwAddress, request_id: int
    ) -> p.HwAddress:
        return p.HwAddress(ieee=self.hw_address)

    async def on_get_network_info(
        self, request: p.GetNetworkInfo, request_id: int
    ) -> p.NetworkInfo:
        return p.NetworkInfo(
            state=self.network_state,
            key_count=t.uint16_t(len(self.key_table)),
            started=t.Bool(self.started),
        )

    async def _scan(
        self,
        command: p.RequestCommand,
        request_id: int,
        entries: Sequence[p.Response],
    ) -> p.ScanCount:
        for entry in entries:
            await self.send_event(command, request_id, entry)
        return p.ScanCount(count=t.uint16_t(len(entries)))

    async def on_scan_key_table(
        self, request: p.ScanKeyTable, request_id: int
    ) -> p.ScanCount:
        return await self._scan(
            p.RequestCommand.SCAN_KEY_TABLE, request_id, self.key_table
        )

    async def on_scan_children(
        self, request: p.ScanChildren, request_id: int
    ) -> p.ScanCount:
        return await self._scan(
            p.RequestCommand.SCAN_CHILDREN, request_id, self.children
        )

    async def on_scan_address_cache(
        self, request: p.ScanAddressCache, request_id: int
    ) -> p.ScanCount:
        return await self._scan(
            p.RequestCommand.SCAN_ADDRESS_CACHE, request_id, self.address_cache
        )

    async def on_scan_route_table(
        self, request: p.ScanRouteTable, request_id: int
    ) -> p.ScanCount:
        return await self._scan(
            p.RequestCommand.SCAN_ROUTE_TABLE, request_id, self.route_table
        )

    async def on_send_unicast(self, request: p.SendUnicast, request_id: int) -> None:
        # The local handoff is terminal for a no-ack unicast; an ack-requested one
        # is only confirmed by the end-to-end APS ack that follows.
        await self.send_notification(
            p.SendConfirm(status=p.SendStatus.SUCCESS), request_id
        )
        if request.aps_ack:
            await self.send_notification(
                p.ApsAckConfirm(status=p.SendStatus.SUCCESS), request_id
            )
        return None

    async def on_send_broadcast(
        self, request: p.SendBroadcast, request_id: int
    ) -> None:
        # A broadcast is confirmed by its passive-ack quorum, not by the handoff.
        await self.send_notification(
            p.BroadcastConfirm(status=p.SendStatus.SUCCESS), request_id
        )
        return None

    async def on_send_groupcast(
        self, request: p.SendGroupcast, request_id: int
    ) -> None:
        await self.send_notification(
            p.BroadcastConfirm(status=p.SendStatus.SUCCESS), request_id
        )
        return None

    async def on_cancel_request(
        self, request: p.CancelRequest, request_id: int
    ) -> p.CancelResult:
        return p.CancelResult(cancelled=t.Bool(True))

    async def on_energy_scan(self, request: p.EnergyScan, request_id: int) -> None:
        for channel in request.channels:
            await self.send_event(
                p.RequestCommand.ENERGY_SCAN,
                request_id,
                p.EnergyResult(channel=t.uint8_t(channel), rssi=t.int8s(-85)),
            )
        return None

    async def on_network_scan(self, request: p.NetworkScan, request_id: int) -> None:
        for beacon in self.beacons:
            await self.send_event(p.RequestCommand.NETWORK_SCAN, request_id, beacon)
        return None

    async def on_packet_capture(
        self, request: p.PacketCapture, request_id: int
    ) -> None:
        for packet in self.captured_packets:
            await self.send_event(p.RequestCommand.PACKET_CAPTURE, request_id, packet)
        return None


class ClosingZiggurat:
    """A server that closes without a hello, so the probe cannot pick a protocol."""

    def __init__(self) -> None:
        self.web_app = web.Application()
        self.web_app.router.add_get("/", self._handle_connection)
        self.url = ""

    async def _handle_connection(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)
        await ws.close()
        return ws


_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


class ProtocolErrorWebSocket:
    """A raw WebSocket server that sends a valid binary hello then a bad opcode."""

    def __init__(self) -> None:
        self.url = ""
        self._server: asyncio.Server | None = None

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"ws://127.0.0.1:{port}/"

    async def stop(self) -> None:
        assert self._server is not None
        self._server.close()
        await self._server.wait_closed()

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request = b""
        while b"\r\n\r\n" not in request:
            chunk = await reader.read(1024)
            if not chunk:
                return
            request += chunk

        key = ""
        for line in request.decode().split("\r\n"):
            if line.lower().startswith("sec-websocket-key:"):
                key = line.split(":", 1)[1].strip()
        accept = base64.b64encode(
            hashlib.sha1((key + _WS_GUID).encode()).digest()
        ).decode()

        writer.write(
            b"HTTP/1.1 101 Switching Protocols\r\n"
            b"Upgrade: websocket\r\n"
            b"Connection: Upgrade\r\n"
            b"Sec-WebSocket-Accept: " + accept.encode() + b"\r\n\r\n"
        )
        # A valid FIN+binary frame (the hello), then a frame using reserved opcode
        # 0x3, a protocol error the client surfaces as a WSMsgType.ERROR message.
        writer.write(b"\x82\x01\x00")
        writer.write(b"\x83\x00")
        await writer.drain()
        writer.close()


class SyntheticSpinelRcp:
    """A TCP server speaking Spinel, exposing the Ziggurat vendor property."""

    def __init__(
        self,
        *,
        get_prop_id: aiospinel.PackedUInt21 = PROP_VENDOR_ZIGGURAT,
        set_prop_id: aiospinel.PackedUInt21 = PROP_VENDOR_ZIGGURAT,
    ) -> None:
        self.url = ""
        self.tunnel_writes: list[bytes] = []
        self._get_prop_id = get_prop_id
        self._set_prop_id = set_prop_id
        self._server: asyncio.Server | None = None
        self._writers: list[asyncio.StreamWriter] = []

    async def start(self) -> None:
        self._server = await asyncio.start_server(self._serve, "127.0.0.1", 0)
        port = self._server.sockets[0].getsockname()[1]
        self.url = f"socket://127.0.0.1:{port}"

    async def stop(self) -> None:
        assert self._server is not None
        for writer in self._writers:
            writer.close()
        self._server.close()
        await self._server.wait_closed()

    async def _serve(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        self._writers.append(writer)
        buffer = bytearray()
        while True:
            data = await reader.read(1024)
            if not data:
                break
            buffer += data
            while True:
                chunk, flag, rest = buffer.partition(
                    bytes([aiospinel.HDLCSpecial.FLAG])
                )
                if not flag:
                    buffer = bytearray(chunk)
                    break
                buffer = bytearray(rest)
                if chunk:
                    self._handle(aiospinel.HDLCLiteFrame.from_bytes(chunk), writer)

    def _handle(
        self, hdlc: aiospinel.HDLCLiteFrame, writer: asyncio.StreamWriter
    ) -> None:
        frame = aiospinel.SpinelFrame.from_bytes(hdlc.data)
        tid = frame.header.transaction_id
        if frame.command_id == aiospinel.CommandID.PROP_VALUE_GET:
            self._respond(writer, tid, self._get_prop_id.serialize())
        elif frame.command_id == aiospinel.CommandID.PROP_VALUE_SET:
            _, rest = aiospinel.PackedUInt21.deserialize(frame.data)
            length = int.from_bytes(rest[:2], "little")
            self.tunnel_writes.append(rest[2 : 2 + length])
            self._respond(writer, tid, self._set_prop_id.serialize())

    def _respond(
        self, writer: asyncio.StreamWriter, tid: int | None, data: bytes
    ) -> None:
        frame = aiospinel.SpinelFrame(
            header=aiospinel.SpinelHeader(
                flag=0b10, network_link_id=0, transaction_id=tid
            ),
            command_id=aiospinel.CommandID.PROP_VALUE_IS,
            data=data,
        )
        writer.write(aiospinel.HDLCLiteFrame(data=frame.serialize()).serialize())

    async def push_stream_frame(self, payload: bytes) -> None:
        data = (
            PROP_VENDOR_ZIGGURAT.serialize()
            + len(payload).to_bytes(2, "little")
            + payload
        )
        frame = aiospinel.SpinelFrame(
            header=aiospinel.SpinelHeader(
                flag=0b10, network_link_id=0, transaction_id=0
            ),
            command_id=aiospinel.CommandID.PROP_VALUE_IS,
            data=data,
        )
        encoded = aiospinel.HDLCLiteFrame(data=frame.serialize()).serialize()
        for writer in self._writers:
            writer.write(encoded)
            await writer.drain()


def make_app_config(url: str, **extra: Any) -> dict[str, Any]:
    return {
        zigpy.config.CONF_DEVICE: {zigpy.config.CONF_DEVICE_PATH: url},
        **extra,
    }


async def flush(app: ControllerApplication) -> None:
    """Round-trip a request: the websocket is ordered, so by the time the response
    arrives every previously sent notification has been processed."""
    await app._watchdog_feed()


@pytest.fixture
async def server() -> AsyncIterator[SyntheticZiggurat]:
    ziggurat = SyntheticZiggurat()
    test_server = TestServer(ziggurat.web_app)
    await test_server.start_server()
    ziggurat.url = f"ws://localhost:{test_server.port}/"

    yield ziggurat

    await test_server.close()


@pytest.fixture
async def spinel_rcp() -> AsyncIterator[SyntheticSpinelRcp]:
    rcp = SyntheticSpinelRcp()
    await rcp.start()

    yield rcp

    await rcp.stop()


@pytest.fixture
async def closing_server() -> AsyncIterator[ClosingZiggurat]:
    ziggurat = ClosingZiggurat()
    test_server = TestServer(ziggurat.web_app)
    await test_server.start_server()
    ziggurat.url = f"ws://localhost:{test_server.port}/"

    yield ziggurat

    await test_server.close()


@pytest.fixture
async def protocol_error_server() -> AsyncIterator[ProtocolErrorWebSocket]:
    ziggurat = ProtocolErrorWebSocket()
    await ziggurat.start()

    yield ziggurat

    await ziggurat.stop()


@pytest.fixture
async def connected_app(
    server: SyntheticZiggurat,
) -> AsyncIterator[ControllerApplication]:
    app = ControllerApplication(make_app_config(server.url))
    await app.connect()

    yield app

    await app.shutdown(db=False)


@pytest.fixture
async def app(connected_app: ControllerApplication) -> ControllerApplication:
    await connected_app.start_network()
    return connected_app
