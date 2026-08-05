# Arduino IDE 2.x — Setup Guide

This guide walks you through installing and configuring the **Arduino IDE 2.x**
for building the Autonomous Vehicle Control System (AVCS) firmware.

---

## 1. Download & Install

### Windows

1. Visit <https://www.arduino.cc/en/software>.
2. Download **"Windows Win 10 and newer, 64 bits"** — the MSI installer.
3. Run `arduino-ide_2.x.x_Windows_64bit.exe`.
4. Accept the license, choose the install path (default `C:\Program Files\Arduino IDE`).
5. Allow the installer to install the **USB driver pack** — this is essential
   for the CH340/FT232 clones used by most cheap Arduino clones.
6. Click **Finish** and launch Arduino IDE.

### macOS

1. Download the **"macOS, 64 bits"** ZIP file.
2. Unzip and drag the `Arduino IDE.app` bundle into `/Applications`.
3. On first launch, macOS will warn that the app is "from an unidentified
   developer".  Right-click → **Open** → **Open anyway**.

### Linux

```bash
# Option A: tarball
wget https://downloads.arduino.cc/arduino-ide/arduino-ide_2.3.2_Linux_64bit.AppImage
chmod +x arduino-ide_2.3.2_Linux_64bit.AppImage
./arduino-ide_2.3.2_Linux_64bit.AppImage

# Option B: Flatpak
flatpak install flathub cc.arduino.IDE2
flatpak run cc.arduino.IDE2
```

> **udev rules**: to access the serial port as a non-root user, install the
> udev rules from the official repository:
> ```bash
> curl -fsSL https://raw.githubusercontent.com/arduino/arduino-ide/main/build/linux/installer/linux/install.sh | sh
> ```
> Then log out and back in.

---

## 2. Configure Board Manager

1. Open Arduino IDE 2.x.
2. Go to **File → Preferences** (or `Ctrl+,`).
3. In **Additional boards manager URLs**, paste:
   ```
   https://downloads.arduino.cc/packages/package_index.json
   ```
   (already there by default).  For third-party boards (e.g. Teensy) add the
   appropriate URL.
4. Click **OK**.
5. Open **Tools → Board → Boards Manager…**
6. Install the cores you need:

   | Core name                       | Boards                       |
   |---------------------------------|------------------------------|
   | `Arduino AVR Boards`            | Uno, Mega 2560, Nano         |
   | `Arduino SAM Boards (32-bits ARM Cortex-M3)` | Arduino Due |
   | `Arduino SAMD Boards`           | Nano 33 IoT, MKR series      |

7. After installation, the boards appear under **Tools → Board**.

---

## 3. Install Libraries

The AVCS firmware depends on several libraries. Install them via the
**Library Manager** (**Tools → Manage Libraries…** or `Ctrl+Shift+I`):

| Library name                | Author              | Required for            |
|-----------------------------|---------------------|-------------------------|
| `Servo`                     | Michael Margolis    | Steering servo          |
| `ArduinoJson`               | Benoît Blanchon     | Telemetry parsing       |
| `PID` (PID_v1)              | Brett Beauregard    | PID controllers         |
| `Adafruit BNO055`           | Adafruit            | 9-DOF IMU               |
| `Adafruit Unified Sensor`   | Adafruit            | Required by BNO055      |
| `TinyGPSPlus`               | Mikal Hart          | GPS NMEA parsing        |

Click each result, select the latest version, and click **Install**.

> **Tip:** `Wire` and `SPI` are built into every Arduino core — no install
> required.

---

## 4. Open the Firmware

1. In Arduino IDE, **File → Open**.
2. Navigate to `arduino/languages/main.cpp`.
3. Arduino IDE will create a sketch folder; rename `main.cpp` to `main.ino`
   if prompted (Arduino IDE requires the .ino extension).

> **Project layout alternative:** copy the `languages/` directory contents
> into a new sketch folder `avcs_firmware/avcs_firmware.ino` for cleaner IDE
> handling.

---

## 5. Board & Port Selection

1. Plug your Arduino into the computer via USB.
2. Go to **Tools → Board** and pick your model:
   - Arduino Mega 2560 → `Arduino Mega or Mega 2560` → Processor `ATmega2560 (Mega 2560)`.
   - Arduino Due (Programming Port) → `Arduino Due (Programming Port)`.
   - Arduino Nano 33 IoT → `Arduino Nano 33 IoT`.
3. Go to **Tools → Port** and select the COM port (Windows) or `/dev/ttyUSBx`
   / `/dev/ttyACMx` (Linux/macOS).
   - On Windows, find the port in **Device Manager → Ports (COM & LPT)**.
   - On Linux, run `ls /dev/ttyUSB* /dev/ttyACM*` before and after plugging in.

---

## 6. Build & Upload

1. Click the **✓ Verify** button (top-left) to compile.
2. Look for `Binary sketch size: X bytes (of a Y byte maximum)` — confirm
   the binary fits.
3. Click the **→ Upload** button.  The TX/RX LEDs on the board will flicker
   during flashing.
4. Open the **Serial Monitor** (`Ctrl+Shift+M`), set baud to **115200**, and
   you should see:
   ```
   [AVCS] boot — board: Arduino Mega 2560
   [IMU ] BNO055 detected, calibrating…
   [GPS ] waiting for fix…
   ```

---

## 7. Troubleshooting

### Upload fails — "avrdude: stk500_recv(): programmer is not responding"

- Verify the correct **Port** is selected.
- On Mega, ensure the **Processor** menu item is set to `ATmega2560` —
  `ATmega1280` is a common wrong choice.
- Close any other program holding the COM port (e.g. Serial Monitor in
  another instance of the IDE, PlatformIO Monitor, Cura, etc.).
- Try a different USB cable — many "charging-only" cables lack the data pair.

### Upload fails — "No device found on COMx"

- Press the **RESET** button on the board immediately before clicking Upload.
- For Arduino Due, use the **Programming Port** (the one closer to the power
  jack), not the **Native USB Port**.

### `Servo.h: No such file or directory`

The `Servo` library is missing.  Install it via Library Manager (step 3).

### `BNO055 not detected` at boot

- Run an I²C scanner sketch — confirm the IMU appears at `0x28`.
- Check SDA/SCL are not swapped.
- On Nano 33 IoT, set `IMU_I2C_ADDRESS` to `0x29` if you tied ADR high.

### Library version conflicts

Arduino IDE 2.x lets you install **multiple versions** side-by-side. To
select the active version, click the library in Library Manager and pick
the version from the drop-down.

---

## 8. Useful Keyboard Shortcuts

| Shortcut            | Action                          |
|---------------------|---------------------------------|
| `Ctrl+R`            | Verify / Compile                |
| `Ctrl+U`            | Upload                          |
| `Ctrl+Shift+M`      | Open Serial Monitor             |
| `Ctrl+Shift+N`      | New tab in sketch               |
| `Ctrl+,`            | Preferences                     |
| `Ctrl+/`            | Toggle line comment             |
| `Ctrl+F`            | Find in current file            |
| `Ctrl+Shift+F`      | Find across all sketch files    |

---

## 9. Next Steps

Once you can upload the `blink_example.ino` sketch and see the LED blink,
move on to:

1. `examples/motor_test.ino` — verify motor & encoder wiring.
2. `examples/gps_test.ino` — confirm GPS fix outdoors.
3. `examples/imu_test.ino` — calibrate the BNO055.
4. The full firmware in `languages/main.cpp`.
