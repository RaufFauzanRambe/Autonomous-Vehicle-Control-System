# Arduino CAN-Bus Libraries — Reference

The Controller Area Network (CAN) bus is the de-facto automotive network
standard.  In the AVCS stack, the **Arduino Due** (with its built-in two
CAN controllers) is the recommended target for talking to vehicle CAN
peripherals such as ECUs, motor controllers, and BMS units.

This page covers the two most popular Arduino CAN libraries:

1. **`due_can`** — native CAN on the Arduino Due (ATSAM3X8E).
2. **`mcp_can`** (aka `Adafruit-MCP2515`) — external MCP2515 controller over
   SPI, usable on any Arduino.

---

## 1. Installation

| Library              | Hardware              | Install command                                                  |
|----------------------|-----------------------|------------------------------------------------------------------|
| `due_can`            | Arduino Due (built-in CAN0/CAN1) | Library Manager → `due_can` by Cristian Maglie        |
| `mcp_can`            | MCP2515 + TJA1050 over SPI | Library Manager → `mcp_can` by coryjpoi / Adafruit           |

PlatformIO `lib_deps`:

```ini
lib_deps = collin80/due_can @ ^1.3.2     # Due
# or
lib_deps = coryjfowler/MCP_CAN_lib @ ^1.5.0   # MCP2515
```

---

## 2. CAN Frame Format

A standard CAN 2.0A frame contains:

| Field           | Bits | Description                              |
|-----------------|------|------------------------------------------|
| SOF             | 1    | Start of frame (dominant)                |
| Identifier      | 11   | Message ID (standard frame)              |
| RTR             | 1    | Remote Transmission Request (0=data)     |
| IDE             | 1    | Identifier Extension (0=standard)        |
| r0              | 1    | Reserved                                 |
| DLC             | 4    | Data Length Code (0–8 bytes)             |
| Data            | 0–64 | Up to 8 bytes of payload                 |
| CRC             | 15   | Cyclic redundancy check                  |
| ACK             | 2    | Acknowledge slot + delimiter             |
| EOF             | 7    | End of frame                             |

CAN 2.0B extends the identifier to 29 bits ("extended frame").

---

## 3. Bit-Rate Selection

Common automotive baud rates:

| Bus                | Baud rate | Notes                                  |
|--------------------|-----------|----------------------------------------|
| OBD-II (high-speed)| 500 kbps  | Most modern vehicles                   |
| OBD-II (legacy)    | 250 kbps  | Heavy-duty / older vehicles            |
| OBD-II (very old)  | 125 kbps  | Rare                                   |
| CANopen            | 125–1000 kbps | Industrial / robotics              |

---

## 4. `due_can` API Reference

### `CAN.begin(speed)`

```cpp
CAN.begin(CAN_BPS_500K);   // or CAN_BPS_250K, CAN_BPS_1000K, etc.
```

### `CAN.sendFrame(frame)`

```cpp
CAN_FRAME frame;
frame.id = 0x123;
frame.extended = false;     // 11-bit ID
frame.rtr = 0;
frame.priority = 4;
frame.length = 8;
frame.data.byte[0] = 0xAA;
frame.data.byte[1] = 0xBB;
// ...
CAN.sendFrame(frame);
```

### Receive via callback

```cpp
void onCanReceive(CAN_FRAME *f) {
    Serial.print("ID 0x"); Serial.print(f->id, HEX);
    Serial.print(" DLC=");  Serial.print(f->length);
    for (int i = 0; i < f->length; i++) {
        Serial.print(" "); Serial.print(f->data.byte[i], HEX);
    }
    Serial.println();
}

void setup() {
    CAN.begin(CAN_BPS_500K);
    CAN.attachInterrupt(onCanReceive);
    CAN.watchFor();         // accept all frames
}
```

### Filters / masks

```cpp
// Accept only IDs 0x100–0x1FF
CAN.watchFor(0x100, 0xF00);
// Or full filter+mask:
CAN.setRXFilter(0, 0x123, 0x7FF, false);
```

---

## 5. `mcp_can` (MCP2515) API Reference

```cpp
#include <mcp_can.h>
#include <SPI.h>

MCP_CAN CAN(10);   // CS pin = 10

void setup() {
    Serial.begin(115200);
    while (CAN_OK != CAN.begin(MCP_ANY, CAN_500KBPS, MCP_8MHZ)) {
        Serial.println("CAN init fail — retry");
        delay(100);
    }
    Serial.println("CAN init success");
}

void loop() {
    byte data[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    CAN.sendMsgBuf(0x123, 0, 8, data);   // standard ID, 8 bytes
    delay(100);

    // Receive
    unsigned long id;
    byte ext, len, rtr;
    byte buf[8];
    if (CAN.checkReceive() == CAN_MSGAVAIL) {
        CAN.readMsgBuf(&id, &len, buf);
        Serial.print("ID 0x"); Serial.print(id, HEX);
        Serial.print(" DLC="); Serial.println(len);
    }
}
```

---

## 6. Common Automotive PGNs (J1939)

For heavy-vehicle / industrial CAN buses the parameter group number (PGN)
identifies the data type.  Some examples:

| PGN    | SPNs inside             | Description                          |
|--------|-------------------------|--------------------------------------|
| 0xF004 | 91 (accelerator pos)    | Electronic Engine Controller 1 (EEC1)|
| 0xF003 | 92 (engine load)        | EEC2                                  |
| 0xFEF1 | 110 (engine coolant T)  | Engine Temperature 1 (ET1)           |
| 0xFEF2 | 84 (wheel-based speed)  | Vehicle Speed                        |
| 0xFEC1 | 513 (start-up message)  | Engine Hours                         |
| 0xDA00 | 5884 (VIN)              | Vehicle Identification               |

For passenger cars (OBD-II), the standard 11-bit request ID is `0x7DF`
(broadcast) or `0x7E0–0x7E7` (specific ECU); responses come back on
`0x7E8–0x7EF`.

---

## 7. Complete Example — Request Engine RPM (OBD-II)

```cpp
#include <due_can.h>

void setup() {
    Serial.begin(115200);
    CAN.begin(CAN_BPS_500K);

    // Build a single-frame OBD-II request for PID 0x0C (engine RPM).
    CAN_FRAME f;
    f.id = 0x7DF;            // broadcast request
    f.extended = false;
    f.length = 8;
    f.data.byte[0] = 0x02;   // number of data bytes
    f.data.byte[1] = 0x01;   // service 01 (current data)
    f.data.byte[2] = 0x0C;   // PID: engine RPM
    f.data.byte[3..7] = 0x00;  // padding
    CAN.sendFrame(f);

    CAN.attachInterrupt([](CAN_FRAME *r) {
        if (r->id == 0x7E8 && r->data.byte[2] == 0x0C) {
            uint16_t raw = (r->data.byte[3] << 8) | r->data.byte[4];
            float rpm = raw / 4.0f;
            Serial.print("RPM = "); Serial.println(rpm);
        }
    });
    CAN.watchFor(0x7E8, 0x7F8);   // accept responses
}

void loop() {}
```

---

## 8. Hardware Wiring

### Arduino Due (built-in CAN)

| Due pin | CAN transceiver (e.g. SN65HVD230) |
|---------|------------------------------------|
| CAN0_TX (pin 1, CAN0 port) | D (TX input) |
| CAN0_RX (pin 2, CAN0 port) | R (RX output) |
| 3V3      | VCC                              |
| GND      | GND                              |
| –        | CANH, CANL → vehicle bus         |
| –        | 120 Ω termination at each bus end |

> The Due exposes CAN0 on the lower row of the CAN0 header (between the
> power jack and the USB connector).  A transceiver chip is required — the
> SAM3X only contains the controller.

### MCP2515 breakout (any Arduino)

| MCP2515 pin | Arduino pin (Uno) |
|-------------|-------------------|
| VCC         | 5V                |
| GND         | GND               |
| CS          | 10 (configurable) |
| SO          | 12 (MISO)         |
| SI          | 11 (MOSI)         |
| SCK          | 13 (SCK)          |
| INT         | 2 (optional, for IRQ) |

> Set the crystal frequency in `CAN.begin()` to match the breakout — most
> use 8 MHz or 16 MHz.

---

## 9. Troubleshooting

| Symptom                                | Likely Cause / Fix                            |
|----------------------------------------|-----------------------------------------------|
| `CAN.begin()` returns error            | Wrong baud rate or crystal freq set.          |
| Frames sent but no reply               | Missing 120 Ω termination resistor.           |
| Receive works only sporadically         | Common ground missing between nodes.          |
| `Bus-Off` error                        | Excessive traffic / wrong baud → check wiring. |
| MCP2515 not detected                   | Wrong SPI mode (must be 0,0). Check CS pin.   |
| `Message RAM is full` on Due           | Call `CAN.available()` faster / increase buffers. |

---

## 10. See Also

- `due_can` source: <https://github.com/collin80/due_can>
- `MCP_CAN_lib`: <https://github.com/coryjfowler/MCP_CAN_lib>
- J1939 digital annex: <https://www.sae.org/standards/content/j1939da_202401/>
- CAN bus basics: <https://www.csselectronics.com/pages/can-bus-simple-intro-tutorial>
