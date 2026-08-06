# =============================================================================
#  wifi.py  —  Autonomous Vehicle Control System / ESP32 / MicroPython
# =============================================================================
#  WiFi helpers: connect with retries, set hostname, run mDNS, status API.
#  Uses `network.WLAN` (STA + AP) and `asyncio` for non-blocking connect.
#
#  Author: AVC team   |  License: MIT   |  Date: 2024
# =============================================================================

import network
import socket
import time
import uasyncio as asyncio
import sys

# Module-level state
_wlan_sta = None
_wlan_ap  = None
_hostname = "avc-esp32-mpy"
_reconnects = 0

# -----------------------------------------------------------------------------
#  Public helpers
# -----------------------------------------------------------------------------
def is_connected() -> bool:
    """@brief True if STA is up and has an IP."""
    return _wlan_sta is not None and _wlan_sta.isconnected()

def rssi() -> int:
    """@brief Current RSSI in dBm, -127 if unknown."""
    if not is_connected():
        return -127
    try:
        return _wlan_sta.status("rssi")
    except Exception:
        return -127

def ip() -> str:
    """@brief Current STA IP address as a string."""
    if not is_connected():
        return "0.0.0.0"
    return _wlan_sta.ifconfig()[0]

def hostname() -> str:
    """@brief The mDNS hostname (avc-esp32-mpy.local)."""
    return _hostname

def reconnects() -> int:
    """@brief Count of reconnect attempts since boot."""
    return _reconnects

# -----------------------------------------------------------------------------
#  STA + AP setup
# -----------------------------------------------------------------------------
def _init_sta():
    global _wlan_sta
    if _wlan_sta is None:
        _wlan_sta = network.WLAN(network.STA_IF)
    _wlan_sta.active(False)
    _wlan_sta.active(True)
    try:
        _wlan_sta.config(hostname=_hostname)
    except Exception as e:
        print(f"[wifi] hostname set failed: {e}")

def set_hostname(name: str):
    """@brief Set the DHCP / mDNS hostname."""
    global _hostname
    _hostname = name
    if _wlan_sta:
        try:
            _wlan_sta.config(hostname=name)
        except Exception:
            pass

# -----------------------------------------------------------------------------
#  Synchronous connect — blocks until connected or timeout.
# -----------------------------------------------------------------------------
def connect_sync(ssid: str, password: str, timeout_s: int = 10) -> bool:
    """@brief Blocking STA connect. Returns True on success."""
    _init_sta()
    if not ssid:
        print("[wifi] no SSID — skipping connect")
        return False
    print(f"[wifi] connecting to '{ssid}' ...")
    _wlan_sta.connect(ssid, password)
    t0 = time.ticks_ms()
    while not _wlan_sta.isconnected():
        if time.ticks_diff(time.ticks_ms(), t0) > timeout_s * 1000:
            print(f"[wifi] timeout after {timeout_s}s status={_wlan_sta.status()}")
            return False
        time.sleep_ms(100)
    print(f"[wifi] connected ip={_wlan_sta.ifconfig()[0]} rssi={rssi()} dBm")
    return True

# -----------------------------------------------------------------------------
#  Async connect — yields to the event loop while waiting.
# -----------------------------------------------------------------------------
async def connect_async(ssid: str, password: str, timeout_s: int = 15) -> bool:
    """@brief Non-blocking STA connect using uasyncio."""
    global _reconnects
    _init_sta()
    if not ssid:
        print("[wifi] no SSID — skipping connect")
        return False
    print(f"[wifi] async connect to '{ssid}' ...")
    _wlan_sta.connect(ssid, password)
    deadline = time.ticks_add(time.ticks_ms(), timeout_s * 1000)
    while not _wlan_sta.isconnected():
        if time.ticks_diff(time.ticks_ms(), deadline) > 0:
            print(f"[wifi] timeout status={_wlan_sta.status()}")
            return False
        await asyncio.sleep_ms(100)
    print(f"[wifi] connected ip={_wlan_sta.ifconfig()[0]} rssi={rssi()} dBm")
    _reconnects = 0
    return True

# -----------------------------------------------------------------------------
#  Async auto-reconnect loop — call as a long-running task.
# -----------------------------------------------------------------------------
async def reconnect_loop(ssid: str, password: str, interval_s: int = 5):
    """@brief Watchdog task that reconnects whenever STA drops."""
    global _reconnects
    while True:
        if not _wlan_sta or not _wlan_sta.isconnected():
            _reconnects += 1
            print(f"[wifi] dropped — reconnect #{_reconnects}")
            try:
                await connect_async(ssid, password, timeout_s=10)
            except Exception as e:
                sys.print_exception(e)
        await asyncio.sleep(interval_s)

# -----------------------------------------------------------------------------
#  AP fallback (captive portal mode)
# -----------------------------------------------------------------------------
def start_ap(ssid: str = None, password: str = None, channel: int = 6):
    """@brief Bring up an open AP for provisioning."""
    global _wlan_ap
    if ssid is None:
        # Use last 3 bytes of MAC for unique SSID.
        mac = _wlan_sta.config("mac") if _wlan_sta else b"\x00\x00\x00\x00\x00\x00"
        ssid = "AVC-Setup-" + "".join("%02X" % b for b in mac[-3:])
    _wlan_ap = network.WLAN(network.AP_IF)
    _wlan_ap.active(True)
    if password and len(password) >= 8:
        _wlan_ap.config(essid=ssid, authmode=network.AUTH_WPA_WPA2_PSK,
                        password=password, channel=channel)
    else:
        _wlan_ap.config(essid=ssid, authmode=network.AUTH_OPEN, channel=channel)
    print(f"[wifi] AP up '{ssid}' ip={_wlan_ap.ifconfig()[0]}")

# -----------------------------------------------------------------------------
#  mDNS announce (MicroPython port-specific; some ports provide `mdns` module)
# -----------------------------------------------------------------------------
def announce_mdns():
    """@brief Announce _http._tcp and _mqtt._tcp services if mdns is available."""
    try:
        # ESP32 port exposes `network.mdns` in some builds; otherwise no-op.
        from network import mdns
        mdns.start(_hostname, "MicroPython AVC node")
        mdns.addService("_http", "_tcp", 80, {"fw": "1.4.2", "node": _hostname})
        mdns.addService("_mqtt", "_tcp", 1883, {})
        print(f"[wifi] mDNS up as { _hostname }.local")
    except ImportError:
        print("[wifi] mdns module not available — skipping")
    except Exception as e:
        print(f"[wifi] mdns failed: {e}")

# -----------------------------------------------------------------------------
#  Diagnostic — print full status
# -----------------------------------------------------------------------------
def status() -> dict:
    """@brief Return a JSON-serialisable status dict."""
    if not _wlan_sta:
        return {"connected": False, "mode": "off"}
    return {
        "connected": _wlan_sta.isconnected(),
        "rssi":      rssi(),
        "ip":        ip(),
        "hostname":  _hostname,
        "reconnects": _reconnects,
        "status":    _wlan_sta.status(),
    }
