"""
DSRC Radio Interface Module

This module implements the DSRCInterface class which provides low-level
control of a 5.9 GHz DSRC (Dedicated Short-Range Communications) radio.
It manages DSRC channels (172-184), transmission power, WAVE (Wireless
Access in Vehicular Environments) protocol operations, and channel
scheduling in compliance with IEEE 802.11p and IEEE 1609.4.

References:
    - IEEE 802.11p-2010: Wireless Access in Vehicular Environments
    - IEEE 1609.4-2016: Multi-Channel Operations
    - FCC Report and Order 06-110: DSRC Service Rules
    - ETSI EN 302 663: ITS-G5 Access Layer
"""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# DSRC Constants
# ---------------------------------------------------------------------------

DSRC_FREQUENCY_BASE: float = 5.855e9   # 5855 MHz – start of DSRC band
DSRC_CHANNEL_BANDWIDTH: float = 10e6   # 10 MHz per channel
DSRC_MAX_TX_POWER_DBM: float = 33.0    # FCC max: 33 dBm EIRP for public safety
DSRC_MIN_TX_POWER_DBM: float = -10.0   # Practical minimum


class DSRCChannel(IntEnum):
    """DSRC service channels as defined in IEEE 1609.4.

    The DSRC band (5.855–5.925 GHz) is divided into seven 10 MHz channels.
    Channel 178 is the control channel (CCH); the rest are service channels (SCH).
    """
    CH172 = 172   # Service Channel – safety / V2V BSM
    CH174 = 174   # Service Channel
    CH176 = 176   # Service Channel
    CH178 = 178   # Control Channel (CCH) – management, WSA
    CH180 = 180   # Service Channel
    CH182 = 182   # Service Channel
    CH184 = 184   # Service Channel – high-power public safety

    @classmethod
    def is_control_channel(cls, channel: int) -> bool:
        """Return True if the channel is the DSRC Control Channel (CCH)."""
        return channel == cls.CH178

    @classmethod
    def is_valid(cls, channel: int) -> bool:
        """Return True if the channel number is a valid DSRC channel."""
        return channel in (c.value for c in cls)

    def frequency_hz(self) -> float:
        """Return the centre frequency in Hz for this channel.

        The centre frequency is calculated as:
            f = 5860 MHz + (channel_number - 175) * 10 MHz
        """
        return 5.860e9 + (self.value - 175) * 10e6


class ChannelAccessMode(Enum):
    """Channel access mode per IEEE 1609.4."""
    CONTINUOUS = "continuous"   # Always on – single radio, single channel
    ALTERNATING = "alternating" # Time-division between CCH and SCH intervals
    IMMEDIATE = "immediate"     # Switch immediately on demand (single radio)


class WaveServicePriority(IntEnum):
    """WAVE service priority for WSA (WAVE Service Advertisement) ordering."""
    BEST_EFFORT = 0
    PRIORITY_1 = 1
    PRIORITY_2 = 2
    PRIORITY_3 = 3
    PRIORITY_4 = 4
    PRIORITY_5 = 5
    EMERGENCY = 6


@dataclass
class ChannelConfig:
    """Configuration for a single DSRC channel.

    Attributes:
        channel: The DSRC channel number.
        tx_power_dbm: Transmission power in dBm.
        access_mode: Channel access mode.
        data_rate_mbps: PHY data rate in Mbps (3, 4.5, 6, 9, 12, 18, 24, 27).
        service_priority: WAVE service priority.
        enabled: Whether the channel is currently active.
    """
    channel: DSRCChannel
    tx_power_dbm: float = 20.0
    access_mode: ChannelAccessMode = ChannelAccessMode.ALTERNATING
    data_rate_mbps: float = 6.0
    service_priority: WaveServicePriority = WaveServicePriority.BEST_EFFORT
    enabled: bool = True


@dataclass
class WaveTiming:
    """WAVE sync interval timing parameters (IEEE 1609.4).

    Attributes:
        sync_interval_ms: Total sync interval in milliseconds (default 100 ms).
        cch_interval_ms: Control Channel interval in milliseconds.
        sch_interval_ms: Service Channel interval in milliseconds.
        guard_interval_ms: Guard interval at each switch (default 4 ms).
    """
    sync_interval_ms: float = 100.0
    cch_interval_ms: float = 46.0
    sch_interval_ms: float = 46.0
    guard_interval_ms: float = 4.0

    def validate(self) -> bool:
        """Validate that the timing parameters are self-consistent."""
        total = self.cch_interval_ms + self.sch_interval_ms + 2 * self.guard_interval_ms
        return abs(total - self.sync_interval_ms) < 1.0


@dataclass
class WaveServiceAdvertisement:
    """WAVE Service Advertisement (WSA) per IEEE 1609.3.

    Attributes:
        psid: Provider Service Identifier.
        channel: The channel on which the service is available.
        priority: Service priority.
        service_mac_addr: MAC address of the advertising device.
        timestamp: Advertisement creation time.
        repeat_rate: Advertisement repeat rate in transmissions per second.
    """
    psid: int
    channel: DSRCChannel
    priority: WaveServicePriority = WaveServicePriority.BEST_EFFORT
    service_mac_addr: str = "00:00:00:00:00:00"
    timestamp: float = field(default_factory=time.time)
    repeat_rate: float = 5.0


@dataclass
class ChannelStatistics:
    """Statistics for a single DSRC channel.

    Attributes:
        channel: The channel number.
        tx_count: Number of transmitted frames.
        rx_count: Number of received frames.
        tx_error_count: Transmission errors.
        rx_error_count: Reception errors.
        avg_rssi_dbm: Average received signal strength.
        channel_busy_ratio: Fraction of time the channel was busy (0..1).
        last_tx_time: Timestamp of last transmission.
        last_rx_time: Timestamp of last reception.
    """
    channel: DSRCChannel
    tx_count: int = 0
    rx_count: int = 0
    tx_error_count: int = 0
    rx_error_count: int = 0
    avg_rssi_dbm: float = -80.0
    channel_busy_ratio: float = 0.0
    last_tx_time: float = 0.0
    last_rx_time: float = 0.0


# Type alias for receive callbacks
ReceiveCallback = Callable[[DSRCChannel, bytes], None]


class DSRCInterface:
    """DSRC radio interface for 5.9 GHz V2X communication.

    Provides low-level management of DSRC channels, transmission power,
    WAVE multi-channel scheduling, and frame-level transmit/receive
    operations. This class wraps the hardware radio driver and exposes
    a Pythonic API for higher-level protocol stacks.

    Thread Safety:
        All public methods are thread-safe. Transmit and receive paths
        are independently synchronized.

    Usage Example::

        dsrc = DSRCInterface(device_index=0)
        dsrc.initialize()
        dsrc.configure_channel(DSRCChannel.CH172, tx_power_dbm=20.0)
        dsrc.transmit(DSRCChannel.CH172, b"\\x00\\x14...BSM data...")
        dsrc.shutdown()
    """

    def __init__(
        self,
        device_index: int = 0,
        default_tx_power_dbm: float = 20.0,
        default_data_rate_mbps: float = 6.0,
        wave_timing: Optional[WaveTiming] = None,
        mac_address: str = "02:00:00:00:00:01",
    ) -> None:
        """Initialize the DSRC interface.

        Args:
            device_index: Radio device index (for multi-radio platforms).
            default_tx_power_dbm: Default transmission power in dBm.
            default_data_rate_mbps: Default PHY data rate in Mbps.
            wave_timing: WAVE sync interval timing; uses defaults if None.
            mac_address: MAC address assigned to this radio interface.
        """
        self.device_index: int = device_index
        self.default_tx_power_dbm: float = default_tx_power_dbm
        self.default_data_rate_mbps: float = default_data_rate_mbps
        self.wave_timing: WaveTiming = wave_timing or WaveTiming()
        self.mac_address: str = mac_address

        self._lock: threading.RLock = threading.RLock()
        self._initialized: bool = False
        self._channel_configs: Dict[DSRCChannel, ChannelConfig] = {}
        self._channel_stats: Dict[DSRCChannel, ChannelStatistics] = {}
        self._active_channels: Set[DSRCChannel] = set()
        self._receive_callbacks: List[ReceiveCallback] = []
        self._services: Dict[int, WaveServiceAdvertisement] = {}  # psid -> WSA
        self._wave_scheduler_thread: Optional[threading.Thread] = None
        self._running: threading.Event = threading.Event()
        self._current_interval_is_cch: bool = True

        # Initialize configs for all DSRC channels
        for ch in DSRCChannel:
            self._channel_configs[ch] = ChannelConfig(
                channel=ch,
                tx_power_dbm=default_tx_power_dbm,
                data_rate_mbps=default_data_rate_mbps,
                enabled=(ch == DSRCChannel.CH178),  # Only CCH enabled by default
            )
            self._channel_stats[ch] = ChannelStatistics(channel=ch)

    # -------------------------------------------------------------------------
    # Lifecycle
    # -------------------------------------------------------------------------

    def initialize(self) -> bool:
        """Initialize the DSRC radio hardware and WAVE protocol stack.

        Configures the radio with default parameters, enables the control
        channel (CH178), and starts the WAVE multi-channel scheduler.

        Returns:
            True if initialization succeeded, False otherwise.
        """
        with self._lock:
            if self._initialized:
                logger.warning("DSRCInterface already initialized")
                return True

            logger.info(
                "Initializing DSRC interface device=%d, MAC=%s",
                self.device_index,
                self.mac_address,
            )

            # Validate WAVE timing
            if not self.wave_timing.validate():
                logger.error("Invalid WAVE timing parameters")
                return False

            # Hardware initialization simulation
            try:
                self._hw_init()
            except Exception as exc:
                logger.error("Hardware initialization failed: %s", exc)
                return False

            self._initialized = True
            self._running.set()

            # Start WAVE scheduler
            self._wave_scheduler_thread = threading.Thread(
                target=self._wave_scheduler_loop,
                name=f"dsrc-wave-scheduler-{self.device_index}",
                daemon=True,
            )
            self._wave_scheduler_thread.start()

            logger.info("DSRC interface initialized successfully")
            return True

    def shutdown(self) -> None:
        """Shut down the DSRC interface and release hardware resources.

        Stops the WAVE scheduler, disables all channels, and powers
        off the radio.
        """
        logger.info("Shutting down DSRC interface device=%d", self.device_index)
        self._running.clear()

        if self._wave_scheduler_thread is not None:
            self._wave_scheduler_thread.join(timeout=3.0)

        with self._lock:
            self._hw_deinit()
            self._active_channels.clear()
            self._initialized = False

        logger.info("DSRC interface shut down")

    @property
    def is_initialized(self) -> bool:
        """Return True if the interface has been initialized."""
        return self._initialized

    # -------------------------------------------------------------------------
    # Channel Management
    # -------------------------------------------------------------------------

    def configure_channel(
        self,
        channel: DSRCChannel,
        tx_power_dbm: Optional[float] = None,
        data_rate_mbps: Optional[float] = None,
        access_mode: Optional[ChannelAccessMode] = None,
        service_priority: Optional[WaveServicePriority] = None,
    ) -> bool:
        """Configure parameters for a DSRC channel.

        Args:
            channel: The channel to configure.
            tx_power_dbm: Transmission power in dBm (None to keep current).
            data_rate_mbps: PHY data rate in Mbps.
            access_mode: Channel access mode.
            service_priority: WAVE service priority.

        Returns:
            True if configuration was applied, False on error.
        """
        with self._lock:
            if not self._initialized:
                logger.error("Cannot configure channel: interface not initialized")
                return False

            config = self._channel_configs.get(channel)
            if config is None:
                logger.error("Unknown channel: %s", channel)
                return False

            if tx_power_dbm is not None:
                if not (DSRC_MIN_TX_POWER_DBM <= tx_power_dbm <= DSRC_MAX_TX_POWER_DBM):
                    logger.error(
                        "TX power %.1f dBm out of range [%.1f, %.1f]",
                        tx_power_dbm, DSRC_MIN_TX_POWER_DBM, DSRC_MAX_TX_POWER_DBM,
                    )
                    return False
                config.tx_power_dbm = tx_power_dbm

            if data_rate_mbps is not None:
                valid_rates = {3, 4.5, 6, 9, 12, 18, 24, 27}
                if data_rate_mbps not in valid_rates:
                    logger.error("Invalid data rate: %.1f Mbps", data_rate_mbps)
                    return False
                config.data_rate_mbps = data_rate_mbps

            if access_mode is not None:
                config.access_mode = access_mode

            if service_priority is not None:
                config.service_priority = service_priority

            logger.info(
                "Configured CH%d: power=%.1f dBm, rate=%.1f Mbps, mode=%s",
                channel.value, config.tx_power_dbm, config.data_rate_mbps,
                config.access_mode.value,
            )
            return True

    def enable_channel(self, channel: DSRCChannel) -> bool:
        """Enable a DSRC channel for transmission and reception.

        Args:
            channel: The channel to enable.

        Returns:
            True if the channel was enabled successfully.
        """
        with self._lock:
            config = self._channel_configs.get(channel)
            if config is None:
                return False
            config.enabled = True
            self._active_channels.add(channel)
            logger.info("Enabled channel CH%d", channel.value)
            return True

    def disable_channel(self, channel: DSRCChannel) -> bool:
        """Disable a DSRC channel.

        Args:
            channel: The channel to disable.

        Returns:
            True if the channel was disabled successfully.
        """
        with self._lock:
            config = self._channel_configs.get(channel)
            if config is None:
                return False
            if DSRCChannel.is_control_channel(channel.value):
                logger.warning("Cannot disable Control Channel CH178")
                return False
            config.enabled = False
            self._active_channels.discard(channel)
            logger.info("Disabled channel CH%d", channel.value)
            return True

    def get_active_channels(self) -> List[DSRCChannel]:
        """Return a list of currently active DSRC channels."""
        with self._lock:
            return sorted(self._active_channels)

    def get_channel_config(self, channel: DSRCChannel) -> Optional[ChannelConfig]:
        """Return the configuration for a specific channel."""
        with self._lock:
            return self._channel_configs.get(channel)

    # -------------------------------------------------------------------------
    # Transmit / Receive
    # -------------------------------------------------------------------------

    def transmit(self, channel: DSRCChannel, data: bytes, priority: Optional[WaveServicePriority] = None) -> bool:
        """Transmit data on the specified DSRC channel.

        Args:
            channel: The channel to transmit on.
            data: Raw frame data to transmit.
            priority: Optional override for service priority.

        Returns:
            True if the frame was accepted for transmission.
        """
        with self._lock:
            if not self._initialized:
                logger.error("Cannot transmit: interface not initialized")
                return False

            if channel not in self._active_channels and channel != DSRCChannel.CH178:
                logger.error("Cannot transmit on inactive channel CH%d", channel.value)
                return False

            config = self._channel_configs[channel]
            if not config.enabled:
                logger.error("Channel CH%d is disabled", channel.value)
                return False

            # Check WAVE scheduling: can we TX on this channel now?
            if config.access_mode == ChannelAccessMode.ALTERNATING:
                is_cch = DSRCChannel.is_control_channel(channel.value)
                if is_cch != self._current_interval_is_cch:
                    logger.debug("Cannot TX on CH%d: wrong interval", channel.value)
                    return False

            stats = self._channel_stats[channel]
            try:
                self._hw_transmit(channel, data, config.tx_power_dbm, config.data_rate_mbps)
                stats.tx_count += 1
                stats.last_tx_time = time.time()
                logger.debug(
                    "TX on CH%d: %d bytes, power=%.1f dBm",
                    channel.value, len(data), config.tx_power_dbm,
                )
                return True
            except Exception as exc:
                stats.tx_error_count += 1
                logger.error("TX error on CH%d: %s", channel.value, exc)
                return False

    def register_receive_callback(self, callback: ReceiveCallback) -> None:
        """Register a callback to be invoked when a frame is received.

        The callback receives the channel number and raw frame data.

        Args:
            callback: Function with signature (channel: DSRCChannel, data: bytes) -> None.
        """
        with self._lock:
            self._receive_callbacks.append(callback)

    def unregister_receive_callback(self, callback: ReceiveCallback) -> None:
        """Remove a previously registered receive callback."""
        with self._lock:
            if callback in self._receive_callbacks:
                self._receive_callbacks.remove(callback)

    def _on_frame_received(self, channel: DSRCChannel, data: bytes) -> None:
        """Internal handler invoked when the hardware receives a frame.

        Updates channel statistics and dispatches to registered callbacks.
        """
        with self._lock:
            stats = self._channel_stats.get(channel)
            if stats is not None:
                stats.rx_count += 1
                stats.last_rx_time = time.time()
            callbacks = list(self._receive_callbacks)

        for cb in callbacks:
            try:
                cb(channel, data)
            except Exception:
                logger.exception("Receive callback raised an exception on CH%d", channel.value)

    # -------------------------------------------------------------------------
    # Transmission Power Control
    # -------------------------------------------------------------------------

    def set_tx_power(self, channel: DSRCChannel, power_dbm: float) -> bool:
        """Set the transmission power for a specific channel.

        Args:
            channel: The target channel.
            power_dbm: Desired power in dBm.

        Returns:
            True if the power was set successfully.
        """
        return self.configure_channel(channel, tx_power_dbm=power_dbm)

    def set_tx_power_all(self, power_dbm: float) -> bool:
        """Set the transmission power for all active channels.

        Args:
            power_dbm: Desired power in dBm.

        Returns:
            True if the power was set on all channels.
        """
        success = True
        for ch in self._active_channels:
            if not self.set_tx_power(ch, power_dbm):
                success = False
        return success

    def get_tx_power(self, channel: DSRCChannel) -> Optional[float]:
        """Return the current TX power in dBm for the given channel."""
        with self._lock:
            config = self._channel_configs.get(channel)
            return config.tx_power_dbm if config else None

    def auto_power_control(self, channel: DSRCChannel, target_rssi_dbm: float = -60.0) -> float:
        """Adjust TX power based on a target received signal strength.

        Uses a simple proportional control law to increase or decrease
        transmission power so that the expected RSSI at a typical
        communication distance matches the target.

        Args:
            channel: The channel to adjust.
            target_rssi_dbm: Desired RSSI at the receiver in dBm.

        Returns:
            The new TX power in dBm after adjustment.
        """
        with self._lock:
            config = self._channel_configs.get(channel)
            stats = self._channel_stats.get(channel)
            if config is None or stats is None:
                return self.default_tx_power_dbm

            # Proportional adjustment based on observed vs. target RSSI
            error = target_rssi_dbm - stats.avg_rssi_dbm
            adjustment = 0.5 * error  # gain factor of 0.5
            new_power = max(
                DSRC_MIN_TX_POWER_DBM,
                min(DSRC_MAX_TX_POWER_DBM, config.tx_power_dbm + adjustment),
            )
            config.tx_power_dbm = new_power
            logger.info(
                "Auto power control CH%d: %.1f -> %.1f dBm (error=%.1f dB)",
                channel.value, config.tx_power_dbm, new_power, error,
            )
            return new_power

    # -------------------------------------------------------------------------
    # WAVE Protocol Support
    # -------------------------------------------------------------------------

    def advertise_service(self, wsa: WaveServiceAdvertisement) -> bool:
        """Register a WAVE Service Advertisement for periodic broadcasting.

        Args:
            wsa: The WSA to advertise.

        Returns:
            True if the service was registered.
        """
        with self._lock:
            self._services[wsa.psid] = wsa
            logger.info("Advertising service PSID=%d on CH%d", wsa.psid, wsa.channel.value)
            return True

    def remove_service(self, psid: int) -> bool:
        """Remove a previously advertised service.

        Args:
            psid: Provider Service Identifier of the service to remove.

        Returns:
            True if the service was found and removed.
        """
        with self._lock:
            if psid in self._services:
                del self._services[psid]
                logger.info("Removed service PSID=%d", psid)
                return True
            return False

    def get_advertised_services(self) -> List[WaveServiceAdvertisement]:
        """Return all currently advertised WAVE services."""
        with self._lock:
            return list(self._services.values())

    def get_wave_timing(self) -> WaveTiming:
        """Return the current WAVE sync interval timing configuration."""
        return self.wave_timing

    # -------------------------------------------------------------------------
    # Statistics
    # -------------------------------------------------------------------------

    def get_channel_stats(self, channel: DSRCChannel) -> Optional[ChannelStatistics]:
        """Return statistics for a specific channel."""
        with self._lock:
            stats = self._channel_stats.get(channel)
            return ChannelStatistics(**stats.__dict__) if stats else None

    def get_all_stats(self) -> Dict[DSRCChannel, ChannelStatistics]:
        """Return statistics for all DSRC channels."""
        with self._lock:
            return {ch: ChannelStatistics(**s.__dict__) for ch, s in self._channel_stats.items()}

    def reset_stats(self, channel: Optional[DSRCChannel] = None) -> None:
        """Reset statistics for a specific channel or all channels.

        Args:
            channel: Channel to reset, or None to reset all.
        """
        with self._lock:
            channels = [channel] if channel else list(DSRCChannel)
            for ch in channels:
                self._channel_stats[ch] = ChannelStatistics(channel=ch)

    # -------------------------------------------------------------------------
    # Internal: WAVE scheduler loop
    # -------------------------------------------------------------------------

    def _wave_scheduler_loop(self) -> None:
        """Background loop implementing IEEE 1609.4 multi-channel scheduling.

        Alternates between CCH and SCH intervals according to the
        configured WaveTiming parameters.
        """
        logger.debug("WAVE scheduler started")
        while self._running.is_set():
            # CCH interval
            self._current_interval_is_cch = True
            self._guard_interval()
            time.sleep(self.wave_timing.cch_interval_ms / 1000.0)

            # SCH interval
            self._current_interval_is_cch = False
            self._guard_interval()
            time.sleep(self.wave_timing.sch_interval_ms / 1000.0)

        logger.debug("WAVE scheduler stopped")

    def _guard_interval(self) -> None:
        """Wait for the guard interval during channel switching.

        During the guard interval no transmissions are scheduled to
        allow the radio to settle on the new channel.
        """
        time.sleep(self.wave_timing.guard_interval_ms / 1000.0)

    # -------------------------------------------------------------------------
    # Internal: hardware abstraction (simulation)
    # -------------------------------------------------------------------------

    def _hw_init(self) -> None:
        """Simulated hardware initialization.

        In production this would call into the radio driver (e.g., Cohda
        Wireless MK5, Unex DPXA-1001) to power on the radio, set the
        MAC address, and configure initial channel parameters.
        """
        logger.info("HW init: powering on DSRC radio device %d", self.device_index)
        # Simulate power-on delay
        time.sleep(0.001)
        self._active_channels.add(DSRCChannel.CH178)

    def _hw_deinit(self) -> None:
        """Simulated hardware deinitialization / power-off."""
        logger.info("HW deinit: powering off DSRC radio device %d", self.device_index)

    def _hw_transmit(self, channel: DSRCChannel, data: bytes, power_dbm: float, data_rate: float) -> None:
        """Simulated hardware frame transmission.

        In production this would call the radio driver's transmit API
        with the appropriate channel, power, and data rate settings.
        """
        logger.debug(
            "HW TX: CH%d, %d bytes, %.1f dBm, %.1f Mbps",
            channel.value, len(data), power_dbm, data_rate,
        )

    def __repr__(self) -> str:
        return (
            f"DSRCInterface(device={self.device_index}, init={self._initialized}, "
            f"channels={len(self._active_channels)}, MAC={self.mac_address})"
        )
