"""
V2X Communication Node Module

This module implements the V2XNode class which manages all Vehicle-to-Everything
(V2X) communications for an autonomous vehicle. It supports V2V (vehicle-to-vehicle)
and V2I (vehicle-to-infrastructure) messaging with a publish/subscribe pattern,
priority-based message queuing, timeout handling, and connection lifecycle management.

References:
    - IEEE 802.11p / DSRC standard
    - SAE J2735 DSRC Message Set Dictionary
    - ETSI ITS-G5 European V2X standard
"""

from __future__ import annotations

import logging
import threading
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from enum import IntEnum
from queue import Empty, PriorityQueue
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


class MessageType(IntEnum):
    """Enumeration of supported V2X message types."""
    BSM = 1            # Basic Safety Message (SAE J2735)
    MAP = 2            # Map Data
    SPAT = 3           # Signal Phase and Timing
    CAM = 4            # Cooperative Awareness Message (ETSI)
    DENM = 5           # Decentralized Environmental Notification
    RSA = 6            # Road Side Alert
    PSM = 7            # Personal Safety Message
    CPM = 8            # Collective Perception Message
    CUSTOM = 99        # Application-defined custom message


class CommunicationMode(IntEnum):
    """V2X communication mode."""
    V2V = 1    # Vehicle-to-Vehicle
    V2I = 2    # Vehicle-to-Infrastructure
    V2P = 3    # Vehicle-to-Pedestrian
    V2N = 4    # Vehicle-to-Network
    V2C = 5    # Vehicle-to-Cloud


class MessagePriority(IntEnum):
    """Priority levels for V2X message queue ordering.

    Lower numeric value = higher priority. Messages with higher priority
    are transmitted first when the channel is congested.
    """
    CRITICAL = 0    # Emergency brake, collision imminent
    HIGH = 1        # Safety-critical: BSM, SPAT change
    NORMAL = 2      # Routine: periodic CAM, MAP updates
    LOW = 3         # Informational: traffic info, weather
    BACKGROUND = 4  # Non-urgent diagnostics, statistics


class NodeState(IntEnum):
    """V2X node operational state."""
    INITIALIZING = 0
    READY = 1
    CONNECTED = 2
    DEGRADED = 3
    OFFLINE = 4
    ERROR = 5


@dataclass
class V2XMessage:
    """Represents a V2X communication message with metadata.

    Attributes:
        msg_type: The type of V2X message.
        payload: Serialized message content (bytes or dict).
        source_id: Unique identifier of the sender.
        destination_id: Target identifier, or None for broadcast.
        mode: Communication mode (V2V, V2I, etc.).
        priority: Queue priority level.
        timestamp: Creation time in seconds since epoch.
        msg_id: Unique message identifier for tracking.
        ttl: Time-to-live in seconds; message expires after this.
        hops: Number of network hops the message has traversed.
        max_hops: Maximum allowed hops before discarding.
        signature: Optional cryptographic signature for authentication.
    """
    msg_type: MessageType
    payload: Any
    source_id: str
    destination_id: Optional[str] = None
    mode: CommunicationMode = CommunicationMode.V2V
    priority: MessagePriority = MessagePriority.NORMAL
    timestamp: float = field(default_factory=time.time)
    msg_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    ttl: float = 5.0
    hops: int = 0
    max_hops: int = 10
    signature: Optional[bytes] = None

    def __lt__(self, other: V2XMessage) -> bool:
        """Enable priority queue ordering by priority then timestamp."""
        if self.priority != other.priority:
            return self.priority < other.priority
        return self.timestamp < other.timestamp

    def is_expired(self) -> bool:
        """Check whether the message has exceeded its time-to-live."""
        return (time.time() - self.timestamp) > self.ttl

    def is_broadcast(self) -> bool:
        """Return True if the message is addressed to all recipients."""
        return self.destination_id is None

    def clone_for_rebroadcast(self) -> V2XMessage:
        """Create a copy of the message with incremented hop count."""
        return V2XMessage(
            msg_type=self.msg_type,
            payload=self.payload,
            source_id=self.source_id,
            destination_id=self.destination_id,
            mode=self.mode,
            priority=self.priority,
            timestamp=self.timestamp,
            msg_id=self.msg_id,
            ttl=self.ttl,
            hops=self.hops + 1,
            max_hops=self.max_hops,
            signature=self.signature,
        )


@dataclass
class PeerInfo:
    """Information about a communication peer (vehicle or infrastructure).

    Attributes:
        peer_id: Unique peer identifier.
        mode: Communication mode used with this peer.
        last_seen: Timestamp of last received message.
        msg_count: Number of messages exchanged.
        rssi: Received Signal Strength Indicator in dBm.
        latency: Measured round-trip latency in seconds.
    """
    peer_id: str
    mode: CommunicationMode
    last_seen: float = field(default_factory=time.time)
    msg_count: int = 0
    rssi: float = -70.0
    latency: float = 0.1


# Type alias for subscriber callbacks
SubscriberCallback = Callable[[V2XMessage], None]


class V2XNode:
    """V2X communication node managing all vehicle-to-everything messaging.

    The V2XNode acts as the central communication hub for the autonomous vehicle,
    handling message publish/subscribe, priority queue management, peer tracking,
    timeout handling, and connection lifecycle for both V2V and V2I links.

    Thread Safety:
        All public methods are thread-safe. Internal state is protected by
        a reentrant lock. Message dispatching runs on a dedicated daemon thread.

    Usage Example::

        node = V2XNode(node_id="AV-001")
        node.start()

        # Subscribe to BSM messages
        node.subscribe(MessageType.BSM, my_bsm_handler)

        # Publish a safety message
        msg = V2XMessage(
            msg_type=MessageType.BSM,
            payload={"speed": 22.5, "heading": 90.0},
            source_id="AV-001",
            priority=MessagePriority.HIGH,
        )
        node.publish(msg)
    """

    def __init__(
        self,
        node_id: str,
        max_queue_size: int = 1000,
        peer_timeout: float = 30.0,
        dispatch_interval: float = 0.01,
        enable_rebroadcast: bool = True,
    ) -> None:
        """Initialize the V2X communication node.

        Args:
            node_id: Unique identifier for this node in the V2X network.
            max_queue_size: Maximum number of messages in the outbound queue.
            peer_timeout: Seconds after which an inactive peer is considered offline.
            dispatch_interval: Seconds between message dispatch cycles.
            enable_rebroadcast: Whether to rebroadcast messages that allow it.
        """
        self.node_id: str = node_id
        self.max_queue_size: int = max_queue_size
        self.peer_timeout: float = peer_timeout
        self.dispatch_interval: float = dispatch_interval
        self.enable_rebroadcast: bool = enable_rebroadcast

        self._state: NodeState = NodeState.INITIALIZING
        self._lock: threading.RLock = threading.RLock()
        self._outbound_queue: PriorityQueue[V2XMessage] = PriorityQueue(maxsize=max_queue_size)
        self._subscribers: Dict[MessageType, List[SubscriberCallback]] = defaultdict(list)
        self._peers: Dict[str, PeerInfo] = {}
        self._peer_modes: Dict[str, Set[CommunicationMode]] = defaultdict(set)
        self._seen_msg_ids: Set[str] = set()
        self._seen_msg_ids_lock: threading.Lock = threading.Lock()
        self._stats: Dict[str, int] = {
            "published": 0,
            "received": 0,
            "dropped": 0,
            "expired": 0,
            "rebroadcast": 0,
        }

        self._dispatch_thread: Optional[threading.Thread] = None
        self._peer_cleanup_thread: Optional[threading.Thread] = None
        self._running: threading.Event = threading.Event()

    # -------------------------------------------------------------------------
    # Lifecycle management
    # -------------------------------------------------------------------------

    def start(self) -> None:
        """Start the V2X node and its background threads.

        Spawns a dispatch thread that continuously processes the outbound
        message queue and a peer cleanup thread that removes stale peers.
        """
        with self._lock:
            if self._state in (NodeState.READY, NodeState.CONNECTED):
                logger.warning("V2XNode %s is already running", self.node_id)
                return

            self._running.set()
            self._state = NodeState.READY
            logger.info("V2XNode %s starting", self.node_id)

        self._dispatch_thread = threading.Thread(
            target=self._dispatch_loop,
            name=f"v2x-dispatch-{self.node_id}",
            daemon=True,
        )
        self._dispatch_thread.start()

        self._peer_cleanup_thread = threading.Thread(
            target=self._peer_cleanup_loop,
            name=f"v2x-peer-cleanup-{self.node_id}",
            daemon=True,
        )
        self._peer_cleanup_thread.start()

    def stop(self) -> None:
        """Gracefully stop the V2X node.

        Signals all background threads to exit and waits for them to
        finish their current work cycle.
        """
        logger.info("V2XNode %s stopping", self.node_id)
        self._running.clear()

        if self._dispatch_thread is not None:
            self._dispatch_thread.join(timeout=5.0)
        if self._peer_cleanup_thread is not None:
            self._peer_cleanup_thread.join(timeout=5.0)

        with self._lock:
            self._state = NodeState.OFFLINE
            self._subscribers.clear()
            self._peers.clear()
            self._peer_modes.clear()

        logger.info("V2XNode %s stopped", self.node_id)

    @property
    def state(self) -> NodeState:
        """Return the current operational state of the node."""
        with self._lock:
            return self._state

    # -------------------------------------------------------------------------
    # Publish / Subscribe
    # -------------------------------------------------------------------------

    def subscribe(self, msg_type: MessageType, callback: SubscriberCallback) -> None:
        """Register a callback for a specific message type.

        Multiple callbacks can be registered for the same message type.
        Callbacks are invoked synchronously during message dispatch; avoid
        long-running or blocking operations inside callbacks.

        Args:
            msg_type: The message type to listen for.
            callback: Function called when a matching message is received.
        """
        with self._lock:
            if callback not in self._subscribers[msg_type]:
                self._subscribers[msg_type].append(callback)
                logger.debug(
                    "Subscribed callback %s to %s on node %s",
                    callback.__name__,
                    msg_type.name,
                    self.node_id,
                )

    def unsubscribe(self, msg_type: MessageType, callback: SubscriberCallback) -> None:
        """Remove a previously registered callback.

        Args:
            msg_type: The message type the callback was registered for.
            callback: The callback function to remove.
        """
        with self._lock:
            if callback in self._subscribers[msg_type]:
                self._subscribers[msg_type].remove(callback)
                logger.debug(
                    "Unsubscribed callback %s from %s on node %s",
                    callback.__name__,
                    msg_type.name,
                    self.node_id,
                )

    def publish(self, message: V2XMessage) -> bool:
        """Enqueue a message for transmission.

        The message is placed in the outbound priority queue. If the
        queue is full the message is dropped and the dropped counter
        is incremented.

        Args:
            message: The V2XMessage to publish.

        Returns:
            True if the message was successfully enqueued, False otherwise.
        """
        if self._state in (NodeState.OFFLINE, NodeState.ERROR):
            logger.error("Cannot publish: node %s is in state %s", self.node_id, self._state.name)
            return False

        if message.is_expired():
            logger.warning("Dropping expired message %s of type %s", message.msg_id, message.msg_type.name)
            self._stats["expired"] += 1
            return False

        try:
            self._outbound_queue.put_nowait(message)
            self._stats["published"] += 1
            logger.debug(
                "Published %s message %s (priority=%s)",
                message.msg_type.name,
                message.msg_id,
                message.priority.name,
            )
            return True
        except Exception:
            self._stats["dropped"] += 1
            logger.warning("Outbound queue full; dropping message %s", message.msg_id)
            return False

    # -------------------------------------------------------------------------
    # Inbound message handling
    # -------------------------------------------------------------------------

    def receive(self, message: V2XMessage) -> bool:
        """Process an inbound V2X message.

        Deduplication is performed using the message ID. After validation
        the message is delivered to all registered subscribers. If
        rebroadcasting is enabled and the message is a broadcast that
        hasn't exceeded its hop limit, it is re-queued for forwarding.

        Args:
            message: The received V2XMessage.

        Returns:
            True if the message was processed (not a duplicate), False otherwise.
        """
        # --- Deduplication ---
        with self._seen_msg_ids_lock:
            if message.msg_id in self._seen_msg_ids:
                logger.debug("Duplicate message %s ignored", message.msg_id)
                return False
            self._seen_msg_ids.add(message.msg_id)

        # --- Expiry check ---
        if message.is_expired():
            self._stats["expired"] += 1
            logger.debug("Received expired message %s", message.msg_id)
            return False

        # --- Update peer table ---
        self._update_peer(message)

        # --- Deliver to subscribers ---
        self._deliver_to_subscribers(message)
        self._stats["received"] += 1

        # --- Rebroadcast if applicable ---
        if self.enable_rebroadcast and message.is_broadcast() and message.hops < message.max_hops:
            rebroadcast_msg = message.clone_for_rebroadcast()
            self.publish(rebroadcast_msg)
            self._stats["rebroadcast"] += 1

        return True

    # -------------------------------------------------------------------------
    # Peer management
    # -------------------------------------------------------------------------

    def _update_peer(self, message: V2XMessage) -> None:
        """Update peer information based on a received message."""
        with self._lock:
            if message.source_id in self._peers:
                peer = self._peers[message.source_id]
                peer.last_seen = time.time()
                peer.msg_count += 1
            else:
                peer = PeerInfo(
                    peer_id=message.source_id,
                    mode=message.mode,
                )
                self._peers[message.source_id] = peer
                logger.info("New peer discovered: %s (mode=%s)", message.source_id, message.mode.name)

            self._peer_modes[message.source_id].add(message.mode)

            if self._state == NodeState.READY and len(self._peers) > 0:
                self._state = NodeState.CONNECTED

    def get_peers(self, mode: Optional[CommunicationMode] = None) -> List[PeerInfo]:
        """Return a list of known peers, optionally filtered by communication mode.

        Args:
            mode: If specified, only return peers that support this mode.

        Returns:
            List of PeerInfo objects for matching peers.
        """
        with self._lock:
            if mode is None:
                return list(self._peers.values())
            return [
                peer for peer_id, peer in self._peers.items()
                if mode in self._peer_modes.get(peer_id, set())
            ]

    def get_peer(self, peer_id: str) -> Optional[PeerInfo]:
        """Look up a specific peer by its identifier.

        Args:
            peer_id: The unique peer identifier.

        Returns:
            PeerInfo if found, None otherwise.
        """
        with self._lock:
            return self._peers.get(peer_id)

    def is_peer_online(self, peer_id: str) -> bool:
        """Check whether a peer is currently considered online.

        A peer is online if it has been seen within the configured timeout.

        Args:
            peer_id: The peer identifier to check.

        Returns:
            True if the peer is online, False otherwise.
        """
        with self._lock:
            peer = self._peers.get(peer_id)
            if peer is None:
                return False
            return (time.time() - peer.last_seen) < self.peer_timeout

    # -------------------------------------------------------------------------
    # Statistics and diagnostics
    # -------------------------------------------------------------------------

    def get_stats(self) -> Dict[str, int]:
        """Return message processing statistics."""
        return dict(self._stats)

    def get_queue_size(self) -> int:
        """Return the current number of messages in the outbound queue."""
        return self._outbound_queue.qsize()

    # -------------------------------------------------------------------------
    # Internal worker loops
    # -------------------------------------------------------------------------

    def _dispatch_loop(self) -> None:
        """Background loop that processes the outbound message queue.

        Messages are dequeued in priority order and passed to the
        platform-specific transmit handler. Expired messages are discarded.
        """
        logger.debug("Dispatch loop started for node %s", self.node_id)
        while self._running.is_set():
            try:
                message = self._outbound_queue.get(timeout=self.dispatch_interval)
            except Empty:
                continue

            if message.is_expired():
                self._stats["expired"] += 1
                logger.debug("Discarding expired outbound message %s", message.msg_id)
                continue

            self._transmit(message)
            self._outbound_queue.task_done()

        logger.debug("Dispatch loop stopped for node %s", self.node_id)

    def _peer_cleanup_loop(self) -> None:
        """Background loop that periodically removes stale peers.

        Peers that have not been seen within ``self.peer_timeout`` seconds
        are removed from the peer table.
        """
        logger.debug("Peer cleanup loop started for node %s", self.node_id)
        while self._running.is_set():
            time.sleep(self.peer_timeout / 2)
            self._cleanup_stale_peers()

        logger.debug("Peer cleanup loop stopped for node %s", self.node_id)

    def _cleanup_stale_peers(self) -> None:
        """Remove peers whose last-seen timestamp exceeds the timeout."""
        now = time.time()
        stale_ids: List[str] = []
        with self._lock:
            for peer_id, peer in self._peers.items():
                if (now - peer.last_seen) > self.peer_timeout:
                    stale_ids.append(peer_id)
            for peer_id in stale_ids:
                del self._peers[peer_id]
                self._peer_modes.pop(peer_id, None)
                logger.info("Peer %s timed out and was removed", peer_id)

            if not self._peers and self._state == NodeState.CONNECTED:
                self._state = NodeState.READY

    def _deliver_to_subscribers(self, message: V2XMessage) -> None:
        """Invoke all subscriber callbacks for the message type."""
        with self._lock:
            callbacks = list(self._subscribers.get(message.msg_type, []))

        for callback in callbacks:
            try:
                callback(message)
            except Exception:
                logger.exception(
                    "Subscriber callback %s raised an exception for message %s",
                    callback.__name__,
                    message.msg_id,
                )

    def _transmit(self, message: V2XMessage) -> None:
        """Platform-specific transmit hook.

        In a production deployment this method interfaces with the
        DSRC radio or C-V2X modem. The default implementation logs
        the transmission event.

        Args:
            message: The message to transmit over the air.
        """
        logger.info(
            "TX [%s] %s -> %s (type=%s, priority=%s, hops=%d)",
            self.node_id,
            message.source_id,
            message.destination_id or "BROADCAST",
            message.msg_type.name,
            message.priority.name,
            message.hops,
        )

    def __repr__(self) -> str:
        return (
            f"V2XNode(id={self.node_id!r}, state={self.state.name}, "
            f"peers={len(self._peers)}, queue={self.get_queue_size()})"
        )
