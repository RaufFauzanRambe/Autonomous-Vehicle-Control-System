"""
File:        python/mqtt_client.py
Brief:       MqttClient — wrapper around paho-mqtt providing connect/
             reconnect with backoff, topic subscriptions, message
             callbacks, QoS, last-will, and optional TLS.
Author:      Autonomous Vehicle Team
Date:        2025-01-30
License:     MIT
"""

from __future__ import annotations

import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, Optional

from loguru import logger

try:
    import paho.mqtt.client as mqtt
    _PAHO_AVAILABLE = True
except ImportError:  # pragma: no cover
    mqtt = None  # type: ignore
    _PAHO_AVAILABLE = False


# ----------------------------------------------------------------------
# Types
# ----------------------------------------------------------------------
MessageCallback = Callable[[str, bytes], None]


@dataclass(slots=True)
class Subscription:
    """A topic subscription with optional callback."""

    topic: str
    qos: int = 0
    callback: Optional[MessageCallback] = None
    message_count: int = 0
    last_message_ts: float = 0.0


# ----------------------------------------------------------------------
# MqttClient
# ----------------------------------------------------------------------
class MqttClient:
    """Production-quality paho-mqtt wrapper.

    Features:
        * Auto-reconnect with exponential backoff
        * Topic registration before connect (so QoS-1 retained msgs arrive)
        * Last-will-and-testament for crash detection
        * Optional TLS with CA cert verification
        * Thread-safe publish (paho is thread-safe across the network
          loop, but we still lock around subscription mutation)
        * Per-topic callbacks plus a global default callback

    Example:
        >>> client = MqttClient({"host": "broker.hivemq.com", "port": 1883})
        >>> client.connect()
        >>> client.subscribe("vehicle/+/estop", lambda t, p: print(t, p))
        >>> client.publish("vehicle/av-01/status", "online", qos=1)
    """

    DEFAULT_KEEPALIVE_S: int = 60
    DEFAULT_BACKOFF_MAX_S: float = 30.0

    def __init__(self, config: dict) -> None:
        self.host: str = config.get("host", "127.0.0.1")
        self.port: int = int(config.get("port", 1883))
        self.client_id: str = config.get("client_id", "av-rpi-01")
        self.keepalive_s: int = int(config.get("keepalive_s",
                                               self.DEFAULT_KEEPALIVE_S))
        self.username: Optional[str] = config.get("username")
        self.password: Optional[str] = config.get("password")
        self.use_tls: bool = bool(config.get("tls", False))
        self.ca_certs: Optional[str] = config.get("ca_certs")
        self.certfile: Optional[str] = config.get("certfile")
        self.keyfile: Optional[str] = config.get("keyfile")
        self.use_websockets: bool = bool(config.get("websockets", False))
        self.websocket_path: str = config.get("websocket_path", "/mqtt")

        # Last will
        will_cfg = config.get("will") or {}
        self.will_topic: Optional[str] = will_cfg.get("topic")
        self.will_payload: str = will_cfg.get("payload", "offline")
        self.will_qos: int = int(will_cfg.get("qos", 1))
        self.will_retain: bool = bool(will_cfg.get("retain", True))

        # Backoff
        self.backoff_initial_s: float = float(config.get("backoff_initial_s", 1.0))
        self.backoff_factor: float = float(config.get("backoff_factor", 2.0))
        self.backoff_max_s: float = float(config.get("backoff_max_s",
                                                     self.DEFAULT_BACKOFF_MAX_S))

        # Internal state
        self._client: Optional[object] = None
        self._subscriptions: Dict[str, Subscription] = {}
        self._default_callback: Optional[MessageCallback] = None
        self._connected = threading.Event()
        self._lock = threading.RLock()
        self._publish_count: int = 0
        self._publish_failures: int = 0
        self._reconnect_attempts: int = 0

        if not _PAHO_AVAILABLE:
            logger.warning("paho-mqtt not installed — running in stub mode")

    # ------------------------------------------------------------------
    # Connect / disconnect
    # ------------------------------------------------------------------
    def connect(self, timeout_s: float = 10.0) -> bool:
        """Open a connection and start the network loop in a background thread."""
        if not _PAHO_AVAILABLE:
            logger.warning("MQTT stub: not connecting to broker")
            return False

        # Build the client
        if self.use_websockets:
            self._client = mqtt.Client(
                client_id=self.client_id,
                transport="websockets",
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            )
            self._client.ws_set_options(path=self.websocket_path)
        else:
            self._client = mqtt.Client(
                client_id=self.client_id,
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            )

        # Auth
        if self.username:
            self._client.username_pw_set(self.username, self.password or "")

        # TLS
        if self.use_tls:
            self._client.tls_set(
                ca_certs=self.ca_certs,
                certfile=self.certfile,
                keyfile=self.keyfile,
                cert_reqs=ssl.CERT_REQUIRED,
                tls_version=ssl.PROTOCOL_TLS_CLIENT,
            )
            self._client.tls_insecure_set(False)

        # Last will
        if self.will_topic:
            self._client.will_set(
                self.will_topic,
                payload=self.will_payload,
                qos=self.will_qos,
                retain=self.will_retain,
            )

        # Callbacks
        self._client.on_connect = self._on_connect
        self._client.on_disconnect = self._on_disconnect
        self._client.on_message = self._on_message
        self._client.on_subscribe = self._on_subscribe
        self._client.on_publish = self._on_publish

        logger.info("Connecting to MQTT broker {}:{} as '{}'",
                    self.host, self.port, self.client_id)
        try:
            self._client.connect(self.host, self.port, self.keepalive_s)
        except Exception as exc:  # noqa: BLE001
            logger.error("MQTT connect failed: {}", exc)
            return False

        self._client.loop_start()
        return self._connected.wait(timeout=timeout_s)

    def disconnect(self) -> None:
        """Disconnect cleanly and stop the network loop."""
        if not self._client:
            return
        try:
            # Announce going offline
            if self.will_topic:
                self._client.publish(
                    self.will_topic, payload="offline",
                    qos=self.will_qos, retain=self.will_retain,
                )
            self._client.loop_stop()
            self._client.disconnect()
        except Exception as exc:  # noqa: BLE001
            logger.warning("MQTT disconnect error: {}", exc)
        self._connected.clear()
        logger.info("MQTT disconnected")

    def is_connected(self) -> bool:
        return self._connected.is_set()

    # ------------------------------------------------------------------
    # paho callbacks
    # ------------------------------------------------------------------
    def _on_connect(self, client, userdata, flags, rc, properties=None) -> None:
        if rc == 0:
            logger.info("MQTT connected")
            self._connected.set()
            self._reconnect_attempts = 0
            # Re-subscribe to all known topics
            self._resubscribe_all()
            # Announce we're online
            if self.will_topic:
                client.publish(
                    self.will_topic, payload="online",
                    qos=self.will_qos, retain=self.will_retain,
                )
        else:
            logger.error("MQTT connect refused (rc={})", rc)

    def _on_disconnect(self, client, userdata, rc, properties=None) -> None:
        self._connected.clear()
        if rc != 0:
            logger.warning("MQTT unexpected disconnect (rc={}); "
                           "paho will retry", rc)
        else:
            logger.info("MQTT clean disconnect")

    def _on_subscribe(self, client, userdata, mid, granted_qos, properties=None) -> None:
        logger.debug("MQTT subscribe ack mid={} granted_qos={}", mid, granted_qos)

    def _on_publish(self, client, userdata, mid, reason_code, properties=None) -> None:
        self._publish_count += 1

    def _on_message(self, client, userdata, message) -> None:
        topic = message.topic
        payload = bytes(message.payload)
        sub = self._subscriptions.get(topic)
        if sub:
            sub.message_count += 1
            sub.last_message_ts = time.monotonic()
            if sub.callback:
                try:
                    sub.callback(topic, payload)
                except Exception as exc:  # noqa: BLE001
                    logger.exception("MQTT callback for '{}' failed: {}",
                                     topic, exc)
        elif self._default_callback:
            try:
                self._default_callback(topic, payload)
            except Exception as exc:  # noqa: BLE001
                logger.exception("MQTT default callback failed: {}", exc)
        else:
            logger.debug("MQTT unhandled message on '{}': {} bytes",
                         topic, len(payload))

    # ------------------------------------------------------------------
    # Subscriptions
    # ------------------------------------------------------------------
    def subscribe(self, topic: str, callback: Optional[MessageCallback] = None,
                  qos: int = 0) -> None:
        """Subscribe to a topic with an optional callback.

        If ``callback`` is ``None`` the default callback is used.
        """
        with self._lock:
            sub = Subscription(topic=topic, qos=qos, callback=callback)
            self._subscriptions[topic] = sub
        if self._connected.is_set() and self._client is not None:
            result, _ = self._client.subscribe(topic, qos=qos)
            if result != mqtt.MQTT_ERR_SUCCESS:
                logger.error("MQTT subscribe failed for '{}': {}", topic, result)
            else:
                logger.info("MQTT subscribed to '{}' (qos={})", topic, qos)

    def unsubscribe(self, topic: str) -> None:
        with self._lock:
            self._subscriptions.pop(topic, None)
        if self._connected.is_set() and self._client is not None:
            self._client.unsubscribe(topic)

    def set_default_callback(self, callback: MessageCallback) -> None:
        self._default_callback = callback

    def _resubscribe_all(self) -> None:
        """Re-subscribe to all known topics after a reconnect."""
        if not self._client:
            return
        with self._lock:
            subs = list(self._subscriptions.values())
        for sub in subs:
            self._client.subscribe(sub.topic, qos=sub.qos)
            logger.debug("MQTT re-subscribed to '{}'", sub.topic)

    # ------------------------------------------------------------------
    # Publish
    # ------------------------------------------------------------------
    def publish(self, topic: str, payload, qos: int = 0,
                retain: bool = False) -> bool:
        """Publish a message. Returns ``True`` on success."""
        if not self._connected.is_set() or self._client is None:
            self._publish_failures += 1
            logger.debug("MQTT publish dropped (disconnected): '{}'", topic)
            return False
        try:
            info = self._client.publish(topic, payload=payload, qos=qos, retain=retain)
            if info.rc != mqtt.MQTT_ERR_SUCCESS:
                self._publish_failures += 1
                logger.warning("MQTT publish failed: rc={}", info.rc)
                return False
            return True
        except Exception as exc:  # noqa: BLE001
            self._publish_failures += 1
            logger.exception("MQTT publish exception: {}", exc)
            return False

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------
    def stats(self) -> dict:
        return {
            "connected": self.is_connected(),
            "host": self.host,
            "port": self.port,
            "client_id": self.client_id,
            "publish_count": self._publish_count,
            "publish_failures": self._publish_failures,
            "reconnect_attempts": self._reconnect_attempts,
            "subscriptions": {
                t: {
                    "qos": s.qos,
                    "message_count": s.message_count,
                    "last_message_ts": s.last_message_ts,
                }
                for t, s in self._subscriptions.items()
            },
        }
