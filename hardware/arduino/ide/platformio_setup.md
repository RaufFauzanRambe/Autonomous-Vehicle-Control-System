# PlatformIO for ESP32 — Setup Guide

> Build, flash and OTA-update the ESP32 platform firmware with
> [PlatformIO Core](https://docs.platformio.org/) or the PlatformIO IDE
> (VS Code extension).

---

## 1. Install PlatformIO

### 1.1 VS Code extension (recommended)
1. Install **Visual Studio Code** (≥ 1.85).
2. Open the Extensions tab (`Ctrl+Shift+X`).
3. Search for **PlatformIO IDE** and click **Install**.
4. After ~30 s the PlatformIO icon appears in the sidebar; the PIO Core and
   Python toolchain are installed automatically under `~/.platformio/`.

### 1.2 Standalone CLI
```bash
python3 -m pip install --user platformio
# Add ~/.local/bin to PATH on Linux/macOS
export PATH=$PATH:~/.local/bin
pio --version
```

---

## 2. Open the project

```bash
cd esp32
code .                # opens VS Code in this folder
```

PlatformIO auto-detects `platformio.ini` and lists the environments in the
bottom status bar. Click the environment name (or use `PlatformIO: Switch
Environment` in the Command Palette) to pick:

| Environment | Target | Framework | Notes |
|-------------|--------|-----------|-------|
| `esp32dev` | ESP-WROOM-32 | Arduino | Default, 4 MB flash |
| `esp32-s3-devkitc` | S3 DevKitC-1 | Arduino | 8 MB flash, PSRAM |
| `esp32-c3-devkitm-1` | C3 DevKitM-1 | Arduino | RISC-V, 4 MB flash |
| `esp32-s3-idf` | S3 DevKitC-1 | ESP-IDF | Native IDF build |
| `esp32dev_ota` | OTA target | Arduino | Network upload |

---

## 3. Framework choice — Arduino vs ESP-IDF

| Aspect | Arduino | ESP-IDF |
|--------|---------|---------|
| Library ecosystem | Massive (Adafruit, etc.) | Smaller, but raw APIs |
| Setup effort | Trivial | Higher (config + build system) |
| FreeRTOS control | Indirect (lazy create) | Full control |
| OTA primitives | `ESPhttpUpdate` wrapper | `esp_https_ota` low-level |
| Best for | Quick dev, sensor drivers | Production, tight memory |

The default environments use Arduino because Adafruit's BNO055/BMP280/VL53L0X
drivers are Arduino-only. The `esp32-s3-idf` env is provided for users who want
to drop the Arduino layer.

---

## 4. Library manager

`lib_deps` in `platformio.ini` lists every dependency. On first build, PIO
downloads them into `.pio/libdeps/<env>/`. To add a new library:

```bash
pio pkg install -l "owner/LibraryName@^1.0" -e esp32-s3-devkitc
```

Or edit `platformio.ini` `lib_deps` and re-build.

---

## 5. Build, flash, monitor

### 5.1 CLI
```bash
pio run -e esp32-s3-devkitc              # build
pio run -e esp32-s3-devkitc -t upload    # flash via USB
pio device monitor -e esp32-s3-devkitc   # serial monitor
# combined:
pio run -e esp32-s3-devkitc -t upload -t monitor
```

### 5.2 GUI
Use the check-mark (✓) for build, the right-arrow (→) for upload, and the
plug icon for the serial monitor.

`monitor_filters = esp32_exception_decoder` auto-decodes panic backtraces.

---

## 6. Upload via OTA

1. Make sure the device is running firmware that exposes the ESP-IDF OTA
   service (the `OtaUpdate` class subscribes to MQTT topic
   `avc/<node_id>/cmd/ota`).
2. Or use PlatformIO's built-in `espota`:
   ```bash
   pio run -e esp32-s3-devkitc -t upload \
       --upload-port 192.168.1.42 \
       --upload-flag --auth=avc-ota-secret
   ```
3. The `esp32dev_ota` environment already specifies the OTA auth password;
   override `upload_port` with `--upload-port`.

---

## 7. Custom partition table

Each environment points to its own `partitions/partitions_<variant>.csv`.
To inspect partition usage:

```bash
pio run -e esp32-s3-devkitc -t dumbsize   # or 'size'
```

---

## 8. Debugging (JTAG)

* **ESP32-S3 DevKitC-1**: built-in USB-JTAG, no extra hardware.
  In `platformio.ini`:
  ```ini
  debug_tool    = esp-builtin
  debug_init_break = tbreak app_main
  debug_load_mode = modified
  ```
* **ESP32 classic**: needs an external JTAG adapter (e.g. FT2232H) — see
  <https://docs.platformio.org/en/latest/plus/debug-tools/esp-prog.html>.

---

## 9. Common issues

| Symptom | Fix |
|---------|-----|
| `Error: Could not find the package with 'espressif32'` | `pio pkg install -g -p "platformio/platform-espressif32@^6.5.0"` |
| `A fatal error occurred: Failed to connect to ESP32` | Hold **BOOT**, press **EN**, release **BOOT**; lower `upload_speed` to 460800. |
| Out-of-memory on S3 (MQTT disconnects) | Enable PSRAM (`BOARD_HAS_PSRAM=1`, already on). |
| `ModuleNotFoundError: charset_normalizer` | `python -m pip install --upgrade pip` then `pio pkg update -g -p tool-pio`. |
| Slow builds | Install `ccache`, then add `build_type = release` or set `platform_packages = toolchain-xtensa-esp32s3 @ ~3.x` in `platformio.ini`. |

---

## 10. Useful CLI cheatsheet

```bash
pio boards esp32s3               # list S3 boards
pio pkg list -g                  # global packages
pio pkg show -g "framework-arduinoespressif32"
pio device list                  # list serial ports
pio settings set strict_ssl true # enable cert verification
pio run -t menuconfig            # only with ESP-IDF framework envs
pio run -t size                  # flash/RAM usage
pio run -t nobuild -t upload     # upload without rebuild
```

---

License: MIT — see source file headers.
