"""A synthetic ziggurat server: a real aiohttp websocket server speaking the wire
protocol, with per-method handlers that tests can override. The harness decodes
incoming params and encodes responses through the same wire models as the client,
so every serialization strategy is exercised in both directions."""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
import json
from typing import Any, TypeVar

from aiohttp import web
from aiohttp.test_utils import TestServer
import pytest
import zigpy.config
import zigpy.types as t

from zigpy_ziggurat.zigbee import commands
from zigpy_ziggurat.zigbee.application import ControllerApplication


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
        await self.send_event(request_id, "transmitted")
        return commands.Status(status="delivered" if command.aps_ack else "sent")

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
