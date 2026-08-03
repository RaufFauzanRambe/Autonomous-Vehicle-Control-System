# Arduino `Wire` (I²C) Library — Reference

`Wire` is Arduino's standard I²C / TWI library.  It is used by virtually
every digital sensor in the AVCS stack — most notably the **Adafruit BNO055
IMU** (default address `0x28`).

---

## 1. Installation

`Wire` ships with every Arduino core — no install required.  Just include
the header:

```cpp
#include <Wire.h>
```

On PlatformIO it is part of the framework automatically.

---

## 2. I²C Bus Fundamentals

- **2-wire bus**: SDA (data) + SCL (clock)
- **7-bit addresses** (0x03–0x77); the LSB is the R/W bit added by the lib.
- **Speeds**:
  - Standard mode: 100 kbps (default)
  - Fast mode: 400 kbps (set via `Wire.setClock(400000)`)
  - Fast mode+: 1 Mbps (SAMD only)
- **Pull-ups**: 4.7 kΩ to VCC required on both SDA and SCL.  Many breakout
  boards already include them.

> On Arduino Mega 2560, the dedicated I²C pins are **20 (SDA)** and
> **21 (SCL)**.  On Uno/Nano, A4/A5 double as SDA/SCL.

---

## 3. API Reference

### `Wire.begin(address)`

```cpp
void begin();                  // Master mode
void begin(uint8_t address);   // Slave mode
```

Call once in `setup()`.  No argument → master mode (most common).

### `Wire.setClock(frequencyHz)`

```cpp
void setClock(uint32_t frequency);
```

Defaults to 100 kHz.  Bump to 400 kHz for high-throughput sensors (BNO055
supports it).

### Master write transaction

```cpp
Wire.beginTransmission(addr);
Wire.write(byte);
Wire.write(buf, len);
uint8_t status = Wire.endTransmission();
```

`endTransmission()` return values:

| Code | Meaning                                       |
|------|-----------------------------------------------|
| 0    | Success                                       |
| 1    | Data too long for transmit buffer             |
| 2    | NACK on address (no device at `addr`)         |
| 3    | NACK on data                                  |
| 4    | Other error (bus stuck / arbitration lost)    |

### Master read transaction

```cpp
Wire.requestFrom(addr, quantity);
// or, with stop flag:
Wire.requestFrom(addr, quantity, true);   // release bus after read
int n = Wire.available();
while (Wire.available()) { byte b = Wire.read(); ... }
```

### Register read pattern (most common)

```cpp
uint8_t readRegister(uint8_t addr, uint8_t reg) {
    Wire.beginTransmission(addr);
    Wire.write(reg);
    Wire.endTransmission(false);            // repeated start
    Wire.requestFrom(addr, (uint8_t)1);
    return Wire.read();
}
```

### `Wire.onReceive(handler)` / `Wire.onRequest(handler)` (slave mode)

```cpp
void onReceive(void (*handler)(int));
void onRequest(void (*handler)(void));
```

Used when the Arduino itself acts as an I²C slave (e.g. to expose data to a
Raspberry Pi master).

---

## 4. Common I²C Sensor Addresses

| Device                                | Address(es)                | Notes                          |
|---------------------------------------|----------------------------|--------------------------------|
| Adafruit BNO055 (IMU)                 | 0x28 (default), 0x29 (ADR=high) | 9-DOF fusion sensor        |
| MPU-6050 / MPU-9250 (IMU)             | 0x68 (AD0=low), 0x69 (AD0=high) | Legacy 6/9-DOF             |
| HMC5883L (magnetometer)               | 0x1E                       | End-of-life, prefer QMC5883L   |
| BMP280 / BME280 (pressure/humidity)   | 0x76 (default), 0x77       | Widely cloned                  |
| MS5611 (barometer)                    | 0x77                       |                                |
| ADS1115 (4-channel 16-bit ADC)        | 0x48–0x4B                  | ADDR pin selects              |
| PCA9685 (16-channel PWM driver)       | 0x40–0x7F                  | Used for multi-servo rigs      |
| PCF8574 (8-bit I/O expander)          | 0x20–0x27                  |                                |
| DS1307 / DS3231 (RTC)                 | 0x68                       |                                |
| AT24C32 (EEPROM)                      | 0x50–0x57                  | Often on RTC breakout boards   |

### I²C Scanner Sketch

```cpp
#include <Wire.h>
void setup() {
    Wire.begin();
    Serial.begin(115200);
    Serial.println("Scanning...");
    for (uint8_t addr = 1; addr < 127; addr++) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            Serial.print("Found 0x"); Serial.println(addr, HEX);
        }
    }
    Serial.println("Done.");
}
void loop() {}
```

Run this whenever a sensor isn't detected — it's the fastest way to find
wiring faults or wrong addresses.

---

## 5. Complete Example — Read BNO055 Chip ID

```cpp
#include <Wire.h>

constexpr uint8_t BNO055_ADDR     = 0x28;
constexpr uint8_t BNO055_CHIP_ID  = 0x00;  // register address
constexpr uint8_t BNO055_ID_VALUE = 0xA0;  // expected value

void setup() {
    Serial.begin(115200);
    Wire.begin();
    Wire.setClock(400000);

    Wire.beginTransmission(BNO055_ADDR);
    Wire.write(BNO055_CHIP_ID);
    if (Wire.endTransmission(false) != 0) {
        Serial.println("Sensor not found");
        return;
    }
    Wire.requestFrom(BNO055_ADDR, (uint8_t)1);
    uint8_t id = Wire.read();
    Serial.print("Chip ID = 0x"); Serial.println(id, HEX);
    Serial.println(id == BNO055_ID_VALUE ? "OK" : "MISMATCH");
}

void loop() {}
```

---

## 6. Buffer Sizes & Performance

- Default AVR `Wire` buffer: 32 bytes (defined as `BUFFER_LENGTH` in
  `Wire.h`).  Increase via `#define BUFFER_LENGTH 64` before `#include <Wire.h>`.
- Throughput @ 400 kHz: ~40 kB/s practical.
- The `Wire` library is **interrupt-driven** on AVR; `requestFrom()` blocks
  until the read completes but does not disable global interrupts.

---

## 7. Troubleshooting

| Symptom                                | Likely Cause / Fix                            |
|----------------------------------------|-----------------------------------------------|
| `endTransmission()` returns 2          | Device not at expected address; run scanner.  |
| Random `0xFF` reads                    | Missing pull-ups (4.7 kΩ to 3.3/5 V).         |
| Works at 100 kHz, fails at 400 kHz     | Pull-ups too weak — try 2.2 kΩ.               |
| Bus stuck after brown-out              | Power-cycle; SCL pulses 9× to release.        |
| Multiple devices, only one reads       | Address clash — check duplicate devices.      |
| Different addresses on Due vs Mega     | Due is 3.3 V; level-shift if you mix 5 V sensors.|

---

## 8. Mixing 3.3 V and 5 V Devices

| Master   | Slave    | Action                                          |
|---------|----------|--------------------------------------------------|
| 5 V (Mega)| 3.3 V | Pull up to **3.3 V** only; signal will be read as HIGH by both. |
| 3.3 V (Due/Nano 33 IoT) | 5 V | Use a level shifter (BSS138-based) on SDA & SCL. |

Never pull the bus to 5 V when any 3.3 V device is connected.

---

## 9. See Also

- Official reference: <https://www.arduino.cc/reference/en/language/functions/communication/wire/>
- Source: <https://github.com/arduino/ArduinoCore-avr/tree/master/libraries/Wire>
- I²C spec summary: <https://www.i2c-bus.org/>
