# Arduino `SPI` Library — Reference

`SPI` (Serial Peripheral Interface) is a synchronous, full-duplex, 4-wire
serial bus used by high-throughput peripherals such as SD cards, displays,
ADCs, and some IMUs.  Arduino's `SPI` library provides a uniform master-mode
API across all supported cores.

---

## 1. Installation

`SPI` ships with every Arduino core — just include the header:

```cpp
#include <SPI.h>
```

---

## 2. SPI Bus Fundamentals

| Line   | Full name              | Direction (master POV) | Purpose                       |
|--------|------------------------|------------------------|-------------------------------|
| SCK    | Serial Clock           | Master → Slave         | Clock line                    |
| MOSI   | Master Out, Slave In   | Master → Slave         | Data from master              |
| MISO   | Master In, Slave Out   | Slave → Master         | Data to master                |
| SS     | Slave Select           | Master → Slave         | Active-low chip-select         |

- **Synchronous**: master generates the clock, so speed is bounded by the
  slave's maximum (typically 1–50 MHz).
- **Full-duplex**: master and slave transmit simultaneously (1 bit each per
  clock cycle).
- **Multi-slave**: each slave needs its own SS line.

### Default pin assignment per board

| Board             | SCK | MOSI | MISO | SS (default) |
|-------------------|-----|------|------|---------------|
| Arduino Uno       | 13  | 11   | 12   | 10            |
| Arduino Mega 2560 | 52  | 51   | 50   | 53            |
| Arduino Due       | 76 (ICSP) | 75 | 74 | 10          |
| Arduino Nano 33 IoT | 13 | 11   | 12   | 10           |

> On AVR, pins 11/12/13 double as SPI on the Uno but are dedicated on the
> Mega's ICSP header (50/51/52).  Always check your board's pinout.

---

## 3. SPI Modes (CPOL / CPHA)

Four combinations of clock polarity (CPOL) and clock phase (CPHA) define the
"SPI mode".  Check the slave's datasheet for the correct mode.

| Mode | CPOL | CPHA | Clock idle | Sample edge |
|------|------|------|------------|-------------|
| 0    | 0    | 0    | Low        | Rising      |
| 1    | 0    | 1    | Low        | Falling     |
| 2    | 1    | 0    | High       | Falling     |
| 3    | 1    | 1    | High       | Rising      |

> Most sensors use **Mode 0** or **Mode 3**.  The BNO055 supports SPI Mode 0
  or Mode 3 at up to 1 MHz.

### Bit order

```cpp
SPI.setBitOrder(MSBFIRST);   // Most-significant bit first (most common)
SPI.setBitOrder(LSBFIRST);   // Some ADCs use this
```

### Clock divider (speed)

```cpp
SPI.setClockDivider(SPI_CLOCK_DIV4);   // F_CPU / 4
// On AVR Uno: DIV2=8 MHz, DIV4=4 MHz, DIV8=2 MHz, DIV16=1 MHz, DIV32=500 kHz ...
// On SAMD/Due: use SPI.beginTransaction() with explicit Hz value instead.
```

---

## 4. API Reference (modern style — recommended)

The Arduino team now recommends using `SPI.beginTransaction()` /
`endTransaction()` instead of the legacy `setBitOrder/setClockDivider/setDataMode`
calls.  This is **thread-safe** on multi-core boards and respects other
devices' settings.

### `SPISettings(speedHz, bitOrder, dataMode)`

```cpp
SPISettings mySettings(1000000, MSBFIRST, SPI_MODE0);
```

### `SPI.beginTransaction(settings)` / `endTransaction()`

```cpp
SPI.beginTransaction(SPISettings(1000000, MSBFIRST, SPI_MODE0));
digitalWrite(SS_PIN, LOW);
SPI.transfer(0xXX);
digitalWrite(SS_PIN, HIGH);
SPI.endTransaction();
```

### `SPI.transfer(byte)` / `transfer(buf, count)`

```cpp
uint8_t rx = SPI.transfer(tx);
// 16-bit transfer:
uint16_t val = SPI.transfer16(0xABCD);
// Block transfer:
SPI.transfer(buffer, length);   // in-place: TX overwritten by RX
```

`transfer()` is **blocking** and **full-duplex**: the byte received is what
was clocked in *while* the byte being transmitted was shifted out.

### `SPI.begin()` / `SPI.end()`

```cpp
SPI.begin();   // configures SCK/MOSI/SS as outputs, MISO as input
SPI.end();     // releases the pins
```

### Using `SPI.usingInterrupt(interruptNum)`

If you call `SPI.transfer()` from inside an ISR, register it:

```cpp
SPI.usingInterrupt(digitalPinToInterrupt(2));
```

This prevents `beginTransaction()` from deadlocking when the ISR fires
mid-transaction.

---

## 5. Complete Example — Read an SPI ADC (MCP3008-style)

```cpp
#include <SPI.h>

const uint8_t SS_PIN = 10;
SPISettings adcSettings(2000000, MSBFIRST, SPI_MODE0);

uint16_t readMcp3008(uint8_t channel) {
    SPI.beginTransaction(adcSettings);
    digitalWrite(SS_PIN, LOW);

    SPI.transfer(0x01);                              // start bit
    uint8_t b1 = SPI.transfer(0x80 | (channel << 4)); // config byte
    uint8_t b2 = SPI.transfer(0x00);                  // dummy, reads back MSB
    digitalWrite(SS_PIN, HIGH);
    SPI.endTransaction();

    return ((b1 & 0x03) << 8) | b2;
}

void setup() {
    Serial.begin(115200);
    pinMode(SS_PIN, OUTPUT);
    digitalWrite(SS_PIN, HIGH);
    SPI.begin();
}

void loop() {
    uint16_t v = readMcp3008(0);
    Serial.print("CH0 = "); Serial.println(v);
    delay(100);
}
```

---

## 6. Common SPI Peripherals in AVCS

| Device                       | Max SPI clock | Mode  | Bit order | Notes                       |
|------------------------------|---------------|-------|-----------|-----------------------------|
| BNO055 (IMU, SPI mode)       | 1 MHz         | 0 or 3 | MSB     | Set BOOT pin low for SPI.   |
| LSM6DS3 (Nano 33 IoT IMU)    | 10 MHz        | 0 or 3 | MSB     | Onboard SPI bus             |
| SD card (SPI mode)           | 25 MHz        | 0     | MSB       | Use `SD` library.           |
| MCP3008 / MCP3208 (ADC)      | 3.2 MHz       | 0     | MSB       | 8-channel 10/12-bit         |
| W25Q32 (Flash)               | 104 MHz       | 0 or 3 | MSB     | JEDEC ID at 0x9F            |
| ILI9341 (TFT display)        | 40 MHz        | 0     | MSB       | Use `Adafruit_ILI9341`      |

---

## 7. Multi-Slave Wiring

Each slave gets its own SS line; SCK/MOSI/MISO are shared.

```
Arduino      ┌─────────┬─────────┬─────────┐
   SCK ──────┤ SCK     │ SCK     │ SCK     │
   MOSI ─────┤ MOSI    │ MOSI    │ MOSI    │
   MISO ─────┤ MISO    │ MISO    │ MISO    │
   SS1 ──────┤ CS      │         │         │  ← device 1
   SS2 ──────┤         │ CS      │         │  ← device 2
   SS3 ──────┤         │         │ CS      │  ← device 3
              └─────────┴─────────┴─────────┘
```

> Each `digitalWrite(SS, LOW)` selects exactly one device.  Always pull SS
> lines **HIGH** in `setup()` so no slave is accidentally selected at boot.

---

## 8. Troubleshooting

| Symptom                                | Likely Cause / Fix                            |
|----------------------------------------|-----------------------------------------------|
| `0xFF` or `0x00` always returned       | SS not asserted LOW, or wiring fault.         |
| Works at 1 MHz, fails at 10 MHz        | Long jumper wires — keep ≤10 cm.              |
| MISO reads garbage                     | MISO not wired or device held in reset.       |
| Multiple devices interfere             | Forgot `endTransaction()` or didn't pull SS HIGH. |
| Data shifted by one bit                | Wrong bit order (MSB vs LSB).                 |
| Wrong byte read                        | Wrong SPI mode (try MODE3 instead of MODE0).  |
| Hangs when SPI used in ISR             | Forgot `SPI.usingInterrupt(n)`.               |

---

## 9. Performance Tips

1. Use **block transfers** (`SPI.transfer(buf, n)`) instead of byte loops —
   the library uses DMA on SAMD and Due.
2. Avoid `digitalWrite(SS)` for performance-critical loops — use direct port
   manipulation (`PORTB &= ~_BV(2);` on Mega) to halve CS toggle time.
3. Pre-compute `SPISettings` as a `static const` to avoid re-construction.
4. Run SCK no faster than the slave datasheet allows; 1 MHz is a safe default.

---

## 10. See Also

- Official reference: <https://www.arduino.cc/reference/en/language/functions/communication/spi/>
- Source: <https://github.com/arduino/ArduinoCore-avr/tree/master/libraries/SPI>
- SparkFun SPI tutorial: <https://learn.sparkfun.com/tutorials/serial-peripheral-interface-spi>
