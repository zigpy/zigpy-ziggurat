"""A synthetic ziggurat server: a real aiohttp websocket server speaking the wire
protocol, with per-method handlers that tests can override. The harness decodes
incoming params and encodes responses through the same wire models as the client,
so every serialization strategy is exercised in both directions."""

import asyncio
import base64
from collections.abc import AsyncIterator, Awaitable, Callable
import hashlib
import json
from typing import Any, TypeVar

from aiohttp import web
from aiohttp.test_utils import TestServer
import aiospinel
import pytest
import zigpy.config
import zigpy.types as t

from zigpy_ziggurat.zigbee import legacy as commands, protocol as p
from zigpy_ziggurat.zigbee.application import ControllerApplication
from zigpy_ziggurat.zigbee.transport import PROP_VENDOR_ZIGGURAT


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

COORDINATOR_IEEE = t.EUI64.convert("00:11:22:33:44:55:66:77")
NETWORK_KEY = t.KeyData.convert("11:22:33:44:55:66:77:88:99:aa:bb:cc:dd:ee:ff:00")

REQUEST_T = TypeVar("REQUEST_T")


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


class SyntheticZiggurat:
    def __init__(self) -> None:
        self.web_app = web.Application()
        self.web_app.router.add_get("/", self._handle_connection)
        self.url = ""
        self.connections = 0
        self._ws: web.WebSocketResponse | None = None
        self._transport: asyncio.Transport | None = None
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
    def configured(self) -> commands.Configure:
        assert self._configured is not None
        return self._configured

    async def _handle_connection(self, request: web.Request) -> web.WebSocketResponse:
        self.connections += 1
        self._transport = request.transport
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


class SyntheticBinaryZiggurat:
    """A WebSocket server speaking the binary protocol, OK-ing every request frame."""

    def __init__(self) -> None:
        self.web_app = web.Application()
        self.web_app.router.add_get("/", self._handle_connection)
        self.url = ""
        self.requests: list[p.Request] = []

    async def _handle_connection(self, request: web.Request) -> web.WebSocketResponse:
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        hello = p.Hello(protocol_version=t.uint8_t(1), configured=t.Bool(False))
        await ws.send_bytes(
            p.encode_reply(
                p.FrameType.NOTIFICATION, p.CommandId.HELLO, 0, hello.serialize()
            )
        )

        async for msg in ws:
            command = p.CommandId(msg.data[0])
            request_id = int.from_bytes(msg.data[1:3], "little")
            self.requests.append(p.REQUESTS[command].deserialize(msg.data[3:])[0])
            await ws.send_bytes(
                p.encode_reply(
                    p.FrameType.RESPONSE, command, request_id, bytes([p.Status.OK])
                )
            )

        return ws


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


def make_app_config(url: str) -> dict[str, Any]:
    return {zigpy.config.CONF_DEVICE: {zigpy.config.CONF_DEVICE_PATH: url}}


@pytest.fixture
async def server() -> AsyncIterator[SyntheticZiggurat]:
    ziggurat = SyntheticZiggurat()
    test_server = TestServer(ziggurat.web_app)
    await test_server.start_server()
    ziggurat.url = f"ws://localhost:{test_server.port}/"

    yield ziggurat

    await test_server.close()


@pytest.fixture
async def binary_server() -> AsyncIterator[SyntheticBinaryZiggurat]:
    ziggurat = SyntheticBinaryZiggurat()
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
