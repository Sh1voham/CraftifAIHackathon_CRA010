"""MQTT helpers: a paho 1.x / 2.x compatible client constructor."""
from __future__ import annotations

from typing import Any


def make_client(client_id: str = "") -> Any:
    """Create a paho MQTT client with either the 1.x or the 2.x constructor."""
    import paho.mqtt.client as mqtt

    if hasattr(mqtt, "CallbackAPIVersion"):  # paho-mqtt >= 2.0
        return mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id=client_id)
    return mqtt.Client(client_id=client_id)
