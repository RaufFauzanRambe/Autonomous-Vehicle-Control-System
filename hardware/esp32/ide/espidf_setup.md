# ESP-IDF v5.x Setup Guide

> Complete step-by-step instructions to build, flash and debug the ESP32 platform
> firmware using the **Espressif IoT Development Framework (ESP-IDF)** v5.2+.

---

## 1. Prerequisites

| OS | Requirements |
|----|--------------|
| **Linux (Ubuntu 22.04+)** | `sudo apt install git wget flex bison gperf python3 python3-pip python3-venv cmake ninja-build ccache libffi-dev libssl-dev dfu-util libusb-1.0-0` |
| **macOS** | Xcode Command Line Tools, Homebrew packages: `cmake ninja dfu-util python3` |
| **Windows 10/11** | Use the official **ESP-IDF Tools Installer** from <https://dl.espressif.com/dl/esp-idf/> |

You also need:
* **Python 3.9+** (system or pyenv).
* **Git** ≥ 2.30.
* ~1.5 GB free disk space for the toolchain.
* A USB cable (data-capable!) to the ESP32 board's UART.

---

## 2. Clone & install ESP-IDF

```bash
mkdir -p ~/esp && cd ~/esp
git clone --recursive https://github.com/espressif/esp-idf.git
cd esp-idf
git checkout v5.2.2           # pin a release; v5.3+ also works
git submodule update --init --recursive

# Install the toolchain + Python packages
./install.sh esp32s3          # or: esp32 | esp32c3 | all
```

On Windows run `install.bat` from a "ESP-IDF PowerShell" prompt instead.

---

## 3. Set up environment variables

Every new terminal must source the export script (or add it to your `~/.bashrc` /
`~/.zshrc`):

```bash
. ~/esp/esp-idf/export.sh
```

Verify:

```bash
idf.py --version
# Should print: ESP-IDF v5.2.2
```

---

## 4. Set the target chip

From the `esp32/` project directory:

```bash
cd esp32
idf.py set-target esp32s3      # or: esp32 / esp32c3
```

This populates `sdkconfig` (you may already have one — `set-target` regenerates
it from `sdkconfig.defaults` and the snippet in `sdkconfig` shipped here).

---

## 5. Configure (optional)

```bash
idf.py menuconfig
```

Key menus used by this project:

* `Component config → FreeRTOS → Tick rate` → **1000 Hz** (mandatory).
* `Component config → ESP32S3 (or target) → CPU frequency` → **240 MHz**.
* `Partition Table → Custom partition CSV` → `partitions/partitions_s3.csv`.
* `Component config → Wi-Fi` → STA, dynamic buffers, WPA3 SAE.
* `Component config → Bluetooth → NimBLE` → enable, peripheral role.
* `Application manager → OTA` → enable rollback.
* `Serial flasher config → Flash size` → 8 MB (S3) / 4 MB (others).

The shipped `sdkconfig` snippet already applies all of these — feel free to
tweak per-vehicle.

---

## 6. Build, flash, monitor

```bash
idf.py build                    # compile & link
idf.py -p /dev/ttyUSB0 flash    # program flash
idf.py -p /dev/ttyUSB0 monitor  # serial monitor (Ctrl+] to exit)
# Or all in one:
idf.py -p /dev/ttyUSB0 build flash monitor
```

On Windows the port is `COM3`, `COM5`, etc.

The monitor decodes panic backtraces automatically (see `tools/esp32_exception_decoder`).

---

## 7. VS Code integration

1. Install the **Espressif IDF** extension from the Marketplace.
2. `View → Command Palette → ESP-IDF: Configure ESP-IDF Extension`.
3. Pick the ESP-IDF directory (`~/esp/esp-idf`) and tools path.
4. Open the `esp32/` folder. The extension provides buttons for build, flash,
   monitor, menuconfig, debug (JTAG) and size analysis.

For JTAG debugging on the S3 (built-in USB-JTAG) add to `.vscode/launch.json`:

```json
{
  "type": "espidf",
  "name": "Debug (ESP32-S3 USB-JTAG)",
  "request": "launch",
  "target": "esp32s3",
  "initGdbCommands": ["target remote :3333", "monitor reset halt", "thb app_main"]
}
```

---

## 8. Hardware wiring

| Function | ESP32 (classic) | ESP32-S3 | ESP32-C3 |
|----------|-----------------|----------|----------|
| I²C SDA | GPIO 21 | GPIO 8 | GPIO 4 |
| I²C SCL | GPIO 22 | GPIO 9 | GPIO 5 |
| Motor PWM A (LEDC) | GPIO 25 | GPIO 4 | GPIO 1 |
| Motor PWM B (LEDC) | GPIO 26 | GPIO 5 | GPIO 2 |
| Motor DIR AIN1 / AIN2 | GPIO 27 / GPIO 14 | GPIO 6 / GPIO 7 | GPIO 3 / GPIO 8 |
| Motor DIR BIN1 / BIN2 | GPIO 12 / GPIO 13 | GPIO 17 / GPIO 18 | GPIO 9 / GPIO 10 |
| Encoder LA / LB | GPIO 32 / GPIO 33 | GPIO 36 / GPIO 37 | GPIO 18 / GPIO 19 |
| Encoder RA / RB | GPIO 34 / GPIO 35 | GPIO 38 / GPIO 39 | GPIO 20 / GPIO 21 |
| Battery ADC | GPIO 36 (ADC1_CH0) | GPIO 1 (ADC1_CH0) | GPIO 0 (ADC1_CH0) |
| Status LED | GPIO 2 | GPIO 48 | GPIO 8 |
| UART0 (log) | TX 1 / RX 3 | TX 43 / RX 44 | TX 21 / RX 20 |

> Note: ESP32-C3 has no Classic Bluetooth — BLE only.

---

## 9. OTA via ESP-IDF

The `OtaUpdate` C++ class wraps `esp_https_ota`:

```cpp
OtaUpdate ota;
ota.setCallback([](int pct, size_t rcv, size_t tot){
    log_i("OTA %d%% (%u/%u)", pct, rcv, tot);
});
bool ok = ota.start("https://fleet.example.com/fw/esp32s3-v1.4.2.bin");
// On reboot, app_main() must call ota.confirm() within 30 s
```

A failed boot (no `confirm()`) triggers automatic rollback to the previous
slot thanks to `CONFIG_APP_ROLLBACK_ENABLE=y` in `sdkconfig`.

---

## 10. Troubleshooting

| Problem | Fix |
|---------|-----|
| `idf.py: command not found` | You forgot to source `export.sh` — run `. ~/esp/esp-idf/export.sh`. |
| `x3100: error: cannot find -lstdc++` | Toolchain mismatch — re-run `./install.sh <target>`. |
| `A fatal error occurred: Failed to connect to ESP32` | Boot mode wrong — hold **BOOT**, press **EN/RST**, release **BOOT**. Or add a 10 µF cap between EN and GND. |
| Brownout detector triggered | Power rail sag — use a powered USB hub or external 5 V supply. |
| `partition table not found` | Run `idf.py set-target` again after editing `partitions_s3.csv`. |
| Monitor shows garbage at first | Different baud — set with `idf.py monitor -b 115200`. |

---

## 11. Useful resources

* Official docs: <https://docs.espressif.com/projects/esp-idf/en/v5.2.2/>
* NimBLE: <https://mynewt.apache.org/latest/network/index.html>
* OTA: <https://docs.espressif.com/projects/esp-idf/en/v5.2.2/esp32s3/api-reference/system/ota.html>

---

License: MIT — see source file headers.
