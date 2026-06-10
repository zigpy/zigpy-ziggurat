import asyncio
import json
import logging

import zigpy.application
import zigpy.backups
from zigpy.exceptions import DeliveryError
import zigpy.serial
import zigpy.state
import zigpy.types as t

_LOGGER = logging.getLogger(__name__)

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

    async def add_endpoint(self, descriptor):
        _LOGGER.debug("Not implemented")

    async def permit_ncp(self, time_s: int = 60):
        await self._api.send_command(
            "permit_joins",
            {
                "duration": time_s,
            },
        )

    async def permit_with_link_key(self, node, link_key, time_s: int = 60):
        _LOGGER.debug("Not implemented")

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
