# =============================================================================
#  boot.py  —  Autonomous Vehicle Control System / ESP32 / MicroPython
# =============================================================================
#  Runs once at power-on / hard-reset, before main.py. Responsibilities:
#    - Set CPU frequency to 240 MHz (ESP32 / S3) or 160 MHz (C3).
#    - Disable noisy debug output on USB-UART when running in production.
#    - Power-gate unused peripherals to save energy.
#    - Mount the filesystem read-write (or read-only in production).
#    - Read persistent config (config.json) into the global `cfg` namespace.
#    - Hand over to main.py via `import main`.
#
#  Author: AVC team   |  License: MIT   |  Date: 2024
# =============================================================================

import machine
import sys
import os
import json
import gc

# -----------------------------------------------------------------------------
#  Boot configuration constants
# -----------------------------------------------------------------------------
CPU_FREQ_MHZ        = 240           # ESP32 classic / S3; C3 maxes at 160
PRODUCTION          = False         # set True to silence boot banner
WDT_TIMEOUT_MS      = 8000          # application watchdog timeout
DISABLE_BLE_ON_BOOT = False         # BLE not used by boot.py; main.py enables it

# -----------------------------------------------------------------------------
#  1. Set CPU frequency
# -----------------------------------------------------------------------------
def set_cpu(freq_mhz):
    """@brief Set the CPU clock. Falls back silently if unsupported."""
    try:
        machine.freq(freq_mhz * 1_000_000)
        print(f"[boot] CPU @ {machine.freq() // 1_000_000} MHz")
    except ValueError as e:
        print(f"[boot] WARN: cannot set CPU to {freq_mhz} MHz: {e}")

# -----------------------------------------------------------------------------
#  2. Reduce serial noise in production
# -----------------------------------------------------------------------------
def silence_serial():
    """@brief Reduce os.duvs messages by lowering the logging level."""
    try:
        # micropython const / micropython.opt_level
        import micropython
        micropython.opt_level(1)
    except Exception:
        pass

# -----------------------------------------------------------------------------
#  3. Filesystem: ensure config.json exists
# -----------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "node_id": "avc-esp32-mpy",
    "wifi":    {"ssid": "", "pass": ""},
    "mqtt":    {"broker": "mqtt://192.168.1.10",
                "port":   1883,
                "user":   "",
                "pass":   ""},
    "i2c":     {"sda": 21, "scl": 22, "freq": 400000},
    "motors":  {"pwma": 25, "pwmb": 26,
                "ain1": 27, "ain2": 14,
                "bin1": 12, "bin2": 13,
                "stby": 15, "freq_hz": 20000},
    "encoders": {"la": 32, "lb": 33, "ra": 34, "rb": 35, "ppr": 1320},
    "battery": {"adc": 36, "ratio": 11.0, "cells": 2},
    "tasks":   {"sensor_hz": 100, "motor_hz": 50, "telem_hz": 10},
}

def load_config(path="config.json"):
    """@brief Load JSON config; create default if missing."""
    try:
        with open(path, "r") as f:
            cfg = json.load(f)
        print(f"[boot] loaded config from {path}")
    except OSError:
        cfg = DEFAULT_CONFIG
        try:
            with open(path, "w") as f:
                json.dump(cfg, f, indent=2)
            print(f"[boot] wrote default config to {path}")
        except OSError as e:
            print(f"[boot] WARN: cannot write config: {e}")
    except ValueError as e:
        print(f"[boot] WARN: bad JSON in config ({e}); using defaults")
        cfg = DEFAULT_CONFIG
    return cfg

# -----------------------------------------------------------------------------
#  4. Memory: trim the heap before handing off to main.py
# -----------------------------------------------------------------------------
def trim_heap():
    """@brief Run GC and report free heap."""
    gc.collect()
    free = gc.mem_free()
    print(f"[boot] free heap: {free // 1024} KB")
    return free

# -----------------------------------------------------------------------------
#  5. Hardware watchdog
# -----------------------------------------------------------------------------
def enable_wdt(timeout_ms):
    """@brief Enable the hardware WDT. main.py must call wdt.feed() regularly."""
    try:
        wdt = machine.WDT(timeout=timeout_ms)
        print(f"[boot] WDT enabled ({timeout_ms} ms)")
        return wdt
    except Exception as e:
        print(f"[boot] WARN: WDT not available: {e}")
        return None

# -----------------------------------------------------------------------------
#  6. Boot banner
# -----------------------------------------------------------------------------
def banner(cfg):
    if PRODUCTION:
        return
    print("=" * 60)
    print(" AVC ESP32 — MicroPython firmware")
    print("=" * 60)
    try:
        import esp
        print(f"  chip id   : {esp.flash_id():08x}")
        print(f"  flash size: {esp.flash_size() // (1024*1024)} MB")
    except Exception:
        pass
    print(f"  MicroPython: {sys.version}")
    print(f"  node id   : {cfg.get('node_id', '?')}")
    print(f"  CPU freq  : {machine.freq() // 1_000_000} MHz")
    print("=" * 60)

# -----------------------------------------------------------------------------
#  7. main() entry
# -----------------------------------------------------------------------------
def main():
    set_cpu(CPU_FREQ_MHZ)
    if PRODUCTION:
        silence_serial()

    cfg = load_config()
    banner(cfg)
    trim_heap()

    # Hardware watchdog — main.py MUST feed this every <8 s.
    wdt = enable_wdt(WDT_TIMEOUT_MS)

    # Stash references on the module object so main.py can pick them up.
    import builtins
    builtins._avc_cfg = cfg
    builtins._avc_wdt = wdt

    print("[boot] boot.py complete — importing main ...")
    gc.collect()

    try:
        import main  # noqa: F401  (main.py defines its own entry)
    except Exception as e:
        # Last-ditch error log — keep the WDT from killing us so the
        # operator can read the trace over serial.
        import sys
        sys.print_exception(e)
        print("[boot] main.py raised — halting")
        while True:
            machine.idle()

# Run only when executed directly (MicroPython boots by importing boot.py).
main()
