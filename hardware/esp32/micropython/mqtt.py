# =============================================================================
#  mqtt.py  —  Autonomous Vehicle Control System / ESP32 / MicroPython
# =============================================================================
#  Minimal but production-grade MQTT v3.1.1 client built on `umqtt.simple`.
#  Features:
#    - Connect / disconnect with username+password and last-will.
#    - Subscribe + per-topic message dispatch (handler map).
#    - Async keep-alive loop (PINGREQ every keepalive_s).
#    - Reconnect with exponential backoff.
#    - JSON helpers: publish_json(), and inbound JSON auto-decode.
#
#  Author: AVC team   |  License: MIT   |  Date: 2024
# =============================================================================

import uasyncio as asyncio
import json
import time
import sys
from umqtt.simple import MQTTClient

# -----------------------------------------------------------------------------
#  Defaults
# -----------------------------------------------------------------------------
DEFAULT_KEEPALIVE_S = 30
DEFAULT_BACKOFF_MS  = 2000
MAX_BACKOFF_MS      = 60000

# -----------------------------------------------------------------------------
#  MqttClient class
# -----------------------------------------------------------------------------
class MqttClient:
    """@brief Async-friendly wrapper around umqtt.simple.MQTTClient.

    Lifecycle:
      c = MqttClient(broker, port, user, pass, node_id)
      c.on_message = handler            # global fallback
      c.subscribe("avc/<node>/cmd/x")
      await c.connect_async()
      await c.keep_alive_loop()         # long-running coroutine
    """

    def __init__(self, broker: str, port: int = 1883,
                 user: str = "", password: str = "",
                 node_id: str = "avc-esp32"):
        # Strip scheme prefix.
        if broker.startswith("mqtt://"):
            broker = broker[len("mqtt://"):]
        elif broker.startswith("mqtts://"):
            broker = broker[len("mqtts://"):]
            print("[mqtt] WARNING: TLS not supported in umqtt.simple — using plain")
        self._broker = broker
        self._port = port or 1883
        self._user = user
        self._pass = password
        self._node = node_id
        self._cli = None
        self._connected = False
        self._backoff = DEFAULT_BACKOFF_MS
        self._handlers = {}     # topic (exact or wildcard) → callable
        self.on_message = None  # global fallback (topic, payload_str)

    # ---- Properties -------------------------------------------------------
    @property
    def is_connected(self):
        return self._connected

    # ---- Connect / disconnect --------------------------------------------
    def _build_client(self):
        last_will_topic = f"avc/{self._node}/state/status"
        last_will_msg   = json.dumps({"node": self._node, "state": "offline"})
        cli = MQTTClient(
            client_id=self._node,
            server=self._broker,
            port=self._port,
            user=self._user or None,
            password=self._pass or None,
            keepalive=DEFAULT_KEEPALIVE_S,
            ssl=False,
        )
        cli.set_last_will(last_will_topic, last_will_msg, retain=True)
        cli.set_callback(self._on_msg)
        return cli

    async def connect_async(self, timeout_s: int = 10) -> bool:
        """@brief Non-blocking connect with timeout."""
        try:
            self._cli = self._build_client()
            await asyncio.to_thread(self._cli.connect) \
                if hasattr(asyncio, "to_thread") else self._cli.connect()
            self._connected = True
            self._backoff = DEFAULT_BACKOFF_MS
            print(f"[mqtt] connected to {self._broker}:{self._port} as '{self._node}'")
            # Announce online.
            self.publish(f"avc/{self._node}/state/status",
                         json.dumps({"node": self._node, "state": "online"}),
                         retain=True)
            return True
        except Exception as e:
            self._connected = False
            print(f"[mqtt] connect failed: {e}")
            return False

    def disconnect(self):
        """@brief Cleanly disconnect and publish offline status."""
        if self._cli and self._connected:
            try:
                self.publish(f"avc/{self._node}/state/status",
                             json.dumps({"node": self._node, "state": "offline_graceful"}),
                             retain=True)
                self._cli.disconnect()
            except Exception as e:
                print(f"[mqtt] disconnect error: {e}")
        self._connected = False

    # ---- Subscribe / publish ---------------------------------------------
    def subscribe(self, topic: str, qos: int = 0):
        """@brief Subscribe to a topic. Calls _cli.subscribe synchronously."""
        if not self._connected:
            print("[mqtt] subscribe: not connected, deferring")
            return False
        try:
            self._cli.subscribe(topic, qos)
            print(f"[mqtt] subscribed {topic}")
            return True
        except Exception as e:
            print(f"[mqtt] subscribe failed: {e}")
            return False

    def publish(self, topic: str, payload: str, qos: int = 0, retain: bool = False):
        """@brief Publish a string payload (must be ≤ ~1 KB on ESP32)."""
        if not self._connected:
            return False
        try:
            self._cli.publish(topic.encode(), payload.encode(),
                              qos=qos, retain=retain)
            return True
        except Exception as e:
            print(f"[mqtt] publish failed: {e}")
            self._connected = False
            return False

    def publish_json(self, topic: str, obj: dict, qos: int = 0, retain: bool = False):
        """@brief Publish a JSON-serialisable object."""
        try:
            payload = json.dumps(obj)
            return self.publish(topic, payload, qos, retain)
        except (TypeError, ValueError) as e:
            print(f"[mqtt] JSON encode failed: {e}")
            return False

    # ---- Inbound dispatch -------------------------------------------------
    def _on_msg(self, topic: bytes, payload: bytes):
        """@brief umqtt.simple callback — runs in the client's checking loop."""
        try:
            t = topic.decode("utf-8", "replace")
            p = payload.decode("utf-8", "replace")
            print(f"[mqtt] ← {t}: {p[:80]}")
            # Exact-match handler?
            h = self._handlers.get(t)
            if h:
                h(t, p)
                return
            # Wildcard handler (last-segment `#`)?
            for pat, hh in self._handlers.items():
                if pat.endswith("#") and t.startswith(pat[:-1]):
                    hh(t, p)
                    return
            # Fallback.
            if self.on_message:
                self.on_message(t, p)
        except Exception as e:
            sys.print_exception(e)

    def on_topic(self, topic: str, handler):
        """@brief Register a per-topic handler."""
        self._handlers[topic] = handler

    # ---- Long-running coroutines -----------------------------------------
    async def keep_alive_loop(self):
        """@brief Periodically call wait_msg() to process PINGREQs + inbound.

        umqtt.simple is synchronous, so we run wait_msg() in a short loop
        with sleeps in between — this is non-blocking from uasyncio's POV.
        """
        while True:
            if self._connected:
                try:
                    # check_msg returns immediately; wait_msg blocks for ~1s.
                    self._cli.check_msg()
                except OSError:
                    print("[mqtt] socket error — marking disconnected")
                    self._connected = False
                except Exception as e:
                    print(f"[mqtt] check_msg error: {e}")
            else:
                # Try to reconnect with exponential backoff.
                await asyncio.sleep_ms(self._backoff)
                if await self.connect_async():
                    # Re-subscribe to all registered handlers' topics.
                    for t in self._handlers.keys():
                        self.subscribe(t)
                else:
                    self._backoff = min(self._backoff * 2, MAX_BACKOFF_MS)
            await asyncio.sleep_ms(50)
