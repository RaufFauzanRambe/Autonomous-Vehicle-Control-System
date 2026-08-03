# Arduino `Servo` Library — Reference

The `Servo` library (originally by Michael Margolis) is the standard Arduino
interface to **RC hobby servos** and **electronic speed controllers** (ESCs)
that accept a 50 Hz PWM signal with 1000–2000 µs pulse widths.

---

## 1. Installation

| Tool              | Action                                                         |
|-------------------|----------------------------------------------------------------|
| Arduino IDE       | Tools → Manage Libraries → search `Servo` → Install            |
| PlatformIO        | Already built into the Arduino core; or add `arduino-libraries/Servo` to `lib_deps` |

```ini
# platformio.ini
lib_deps = arduino-libraries/Servo @ ^1.2.2
```

---

## 2. Signal Fundamentals

- **Frequency**: 50 Hz (20 ms period)
- **Pulse width**: 1.0 ms (full reverse / 0°) to 2.0 ms (full forward / 180°)
- **Centre**: 1.5 ms
- **Voltage**: signal pin is 5 V TTL; the servo's power comes from a separate
  5 V rail (typically a BEC).

> Most servos accept 4.8–6.0 V power. Drawing servo power from the Arduino's
> 5 V pin is **not** recommended — servos can draw >1 A stall current and
> will brown-out the MCU.

---

## 3. API Reference

### `Servo::attach(pin)` / `attach(pin, min, max)`

```cpp
uint8_t attach(int pin);
uint8_t attach(int pin, int min, int max);
```

| Parameter | Description                                                      |
|-----------|------------------------------------------------------------------|
| `pin`     | Any digital pin (AVR Mega: 2–13, 44–46; on SAMD any digital).    |
| `min`     | Minimum pulse width in µs (default 544).                         |
| `max`     | Maximum pulse width in µs (default 2400).                        |
| @return   | Channel index (0–11 on AVR), or 0 on failure.                    |

**Example**

```cpp
Servo steering;
void setup() {
    steering.attach(STEERING_PWM_PIN, 1000, 2000);  // hobby-servo range
}
```

> The AVR `Servo` library uses Timer 1 (and Timer 3/4/5 on the Mega) to
> generate pulses on up to 12 pins.  **Pin 9 and 10 lose PWM capability
> via `analogWrite()`** when `Servo` is used on an Uno.

### `Servo::write(angle)`

```cpp
void write(int angle);
```

Sets the servo to the given angle (0–180°).  Internally mapped to the
`min`–`max` µs range passed to `attach()`.

### `Servo::writeMicroseconds(us)`

```cpp
void writeMicroseconds(int us);
```

Sets the pulse width directly (typical range 1000–2000 µs).  **This is the
preferred call** for vehicles where you need sub-degree precision and a
well-defined centre (1500 µs).

### `Servo::read()` / `readMicroseconds()`

```cpp
int read();              // last angle written (0–180)
int readMicroseconds();  // last pulse width written (µs)
```

Returns the last commanded value — does **not** read the servo's actual
position (RC servos have no feedback line).

### `Servo::attached()` / `detach()`

```cpp
bool attached();
void detach();
```

`detach()` releases the timer and stops the PWM pulses; the servo will go
limp.  Useful for power-saving between movements.

---

## 4. Supported Pins by Board

| Board             | Supported pins                          | Notes                        |
|-------------------|------------------------------------------|------------------------------|
| Arduino Uno       | 2, 3, 4, 5, 6, 7, 8, 9, 10, 11           | 12 servos max; pins 9/10 lose analogWrite |
| Arduino Mega 2560 | 2–13, 44, 45, 46                         | 12 servos max                |
| Arduino Due       | All digital pins (software-driven)       | Up to 48                     |
| Arduino Nano 33 IoT | All digital pins                       | Hardware TCC peripheral      |

---

## 5. Complete Example — Steering Sweep

```cpp
#include <Servo.h>

Servo steering;
const int STEERING_PIN = 9;

void setup() {
    Serial.begin(115200);
    steering.attach(STEERING_PIN, 1000, 2000);
    Serial.println("Sweeping -30° to +30°");
}

void loop() {
    for (int us = 1200; us <= 1800; us += 10) {
        steering.writeMicroseconds(us);
        Serial.print("us="); Serial.println(us);
        delay(20);
    }
    for (int us = 1800; us >= 1200; us -= 10) {
        steering.writeMicroseconds(us);
        delay(20);
    }
}
```

---

## 6. Controlling an ESC (Brushless Motor)

An ESC behaves exactly like a servo: write 1000 µs = stopped, 2000 µs = full
throttle.  Most ESCs require an **arming sequence** on power-up:

```cpp
esc.attach(ESC_PIN, 1000, 2000);
esc.writeMicroseconds(1000);   // arm
delay(3000);                    // wait for ESC beep confirmation
esc.writeMicroseconds(1500);   // idle (for bidirectional ESCs)
```

---

## 7. Troubleshooting

| Symptom                                | Likely Cause / Fix                            |
|----------------------------------------|-----------------------------------------------|
| Servo doesn't move, no error           | 5 V not connected; check BEC & ground.        |
| Servo jitters / motor reset            | Add a 1000 µF electrolytic across 5 V.        |
| `attach()` returns 0                   | Pin not supported — check the table above.    |
| `analogWrite()` on pin 9/10 stops working (Uno) | Normal — Servo library takes over Timer 1. Use pin 5/6 instead. |
| Servo moves only one direction         | Wrong `min`/`max` — most servos need 1000/2000.|
| Continuous-rotation servo won't stop   | Use `writeMicroseconds(1500)` and trim the on-board pot. |
| Servo hums at rest                     | Slightly off-centre command; send exactly 1500 µs. |

---

## 8. Power Considerations

| Servo class         | Stall current | Recommended supply         |
|---------------------|---------------|----------------------------|
| Sub-micro (9 g)     | ~0.6 A        | 5 V BEC, 1 A               |
| Standard (Plastic)  | ~1.2 A        | 5 V BEC, 2 A               |
| High-torque (metal) | 2.5–4 A       | Dedicated 5 V/5 A UBEC     |

Always fuse the servo rail (1–3 A polyfuse) and place a 0.1 µF ceramic +
470 µF electrolytic decoupling capacitor close to the servo connector.

---

## 9. See Also

- Official reference: <https://www.arduino.cc/reference/en/libraries/servo/>
- Source: <https://github.com/arduino-libraries/Servo>
- ESC calibration procedure: `examples/motor_test.ino`
