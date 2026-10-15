"""MQTT output. Never lets a broker problem stop the detector."""
from __future__ import annotations

import json
import logging
from typing import Any

from logic.config import MqttCfg

logger = logging.getLogger(__name__)


def make_client(client_id: str = "") -> Any:
    """Create a paho MQTT client with either the 1.x or the 2.x constructor."""
    import paho.mqtt.client as mqtt

    if hasattr(mqtt, "CallbackAPIVersion"):  # paho-mqtt >= 2.0
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    return mqtt.Client(client_id=client_id)


class MqttPublisher:
    """Publishes JSON payloads with QoS from config.

    Connection happens in paho's background thread, which keeps retrying with
    backoff. Failures are logged as one warning until the next successful
    connection; ``publish`` never raises.
    """

    def __init__(self, host: str, port: int, cfg: MqttCfg, client: Any = None,
                 client_id: str = "safestride") -> None:
        self.cfg = cfg
        self.addr = f"{host}:{port}"
        self.connected = False
        self._warned = False
        self.client = client if client is not None else make_client(client_id)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        try:
            self.client.reconnect_delay_set(cfg.reconnect_min_s, cfg.reconnect_max_s)
            self.client.max_queued_messages_set(cfg.max_queued)
            self.client.connect_async(host, port, cfg.keepalive_s)
            self.client.loop_start()
        except Exception as exc:  # noqa: BLE001 - MQTT must never stop the detector
            self._warn(f"cannot start MQTT client for {self.addr}: {exc!r}")

    def _warn(self, msg: str) -> None:
        if not self._warned:
            logger.warning("%s (alerts still go to the incident log; retrying in the background)", msg)
            self._warned = True

    def _on_connect(self, _client: Any, _userdata: Any, _flags: Any, rc: Any, *_args: Any) -> None:
        if getattr(rc, "is_failure", rc != 0):
            self._warn(f"MQTT broker {self.addr} refused connection: {rc}")
            return
        self.connected, self._warned = True, False
        logger.info("MQTT connected to %s", self.addr)

    def _on_disconnect(self, *_args: Any) -> None:
        if self.connected:
            self.connected = False
            self._warn(f"MQTT broker {self.addr} disconnected")

    def publish(self, topic: str, payload: dict[str, Any]) -> bool:
        """Queue ``payload`` as JSON on ``topic``; True if paho accepted it while connected."""
        try:
            info = self.client.publish(topic, json.dumps(payload), qos=self.cfg.qos)
        except Exception as exc:  # noqa: BLE001
            self._warn(f"MQTT publish to {topic} failed: {exc!r}")
            return False
        if not self.connected or getattr(info, "rc", 0) != 0:
            self._warn(f"MQTT broker {self.addr} unreachable; message to {topic} queued or dropped")
            return False
        return True

    def close(self) -> None:
        """Stop the network thread."""
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MQTT close failed: %r", exc)
