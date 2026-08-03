# Arduino `PID` Library (PID_v1) — Reference

The `PID` library by Brett Beauregard (<https://github.com/br3ttb/Arduino-PID-Library>)
is the most widely used PID controller implementation for Arduino.  It is
used by the AVCS firmware for both the drive-motor and steering control loops.

---

## 1. Installation

| Tool            | Action                                                       |
|-----------------|--------------------------------------------------------------|
| Arduino IDE     | Library Manager → search `PID` by Brett Beauregard → Install |
| PlatformIO      | Add `br3ttb/PID @ ^1.2.1` to `lib_deps`                      |

```ini
lib_deps = br3ttb/PID @ ^1.2.1
```

```cpp
#include <PID_v1.h>
```

---

## 2. PID Theory Recap

A PID controller computes the actuator command `u(t)` from the error
`e(t) = setpoint − input` as:

```
              Kp · e(t)  +  Ki · ∫e(t) dt  +  Kd · de(t)/dt
    u(t)   =  ─────────────────────────────────────────────────
                                    1
```

- **Kp (proportional)** — reaction to the *current* error.  Too high →
  oscillation.
- **Ki (integral)** — reaction to *accumulated* error.  Eliminates steady-
  state offset.  Too high → windup / overshoot.
- **Kd (derivative)** — reaction to the *rate of change* of error.  Damps
  oscillation.  Sensitive to noise.

> The `PID_v1` library uses the **velocity algorithm** with derivative-on-
  measurement (avoids "derivative kick" on set-point changes).

---

## 3. API Reference

### Constructor

```cpp
PID(double* Input, double* Output, double* Setpoint,
    double Kp, double Ki, double Kd,
    int ControllerDirection = DIRECT);
```

| Parameter           | Description                                       |
|---------------------|---------------------------------------------------|
| `Input`             | Pointer to the **process variable** (measured).   |
| `Output`            | Pointer to where the controller writes its result. |
| `Setpoint`          | Pointer to the desired value.                     |
| `Kp, Ki, Kd`        | Tuning gains.                                     |
| `ControllerDirection` | `DIRECT` (output increases with error) or `REVERSE`. |

> The pointers must point to **`double` variables that persist for the
> lifetime of the PID object** (e.g. member variables of your class).
> `PID_v1` reads/writes them on every `Compute()` call.

### `SetMode(mode)`

```cpp
void SetMode(int Mode);   // AUTOMATIC or MANUAL
```

- `AUTOMATIC` — the loop is active; `Compute()` updates `*Output`.
- `MANUAL` — the loop is bypassed; the user writes `*Output` directly.

Switching from MANUAL → AUTOMATIC performs an automatic bumpless transfer
(integrator is re-initialised to the current output).

### `Compute()`

```cpp
bool Compute();
```

Returns `true` if a new output was written.  Internally rate-limits to the
sample time set by `SetSampleTime()`.  If called faster than the sample
time, `Compute()` returns `false` without doing any work.

### `SetOutputLimits(min, max)`

```cpp
void SetOutputLimits(double Min, double Max);
```

Clamps the controller output and also prevents integrator wind-up by
back-calculating the integrator term when the limit is hit.

### `SetTunings(Kp, Ki, Kd)` / `SetTunings(Kp, Ki, Kd, POn)`

```cpp
void SetTunings(double Kp, double Ki, double Kd);
void SetTunings(double Kp, double Ki, double Kd, int POn);
```

`POn` selects proportional-on-error (default `P_ON_E`) vs
proportional-on-measurement (`P_ON_M`) — the latter reduces overshoot on
set-point changes.

### `SetSampleTime(ms)`

```cpp
void SetSampleTime(int NewSampleTimeInMs);
```

How often `Compute()` is allowed to update `*Output`.  The library
**automatically rescales Ki and Kd** when the sample time changes so the
closed-loop behaviour is preserved.

### `SetControllerDirection(direction)`

```cpp
void SetControllerDirection(int Direction);   // DIRECT or REVERSE
```

### `GetKp()` / `GetKi()` / `GetKd()` / `GetMode()` / `GetDirection()`

Read-only accessors for the current settings.

---

## 4. Complete Example — Motor Speed PID

```cpp
#include <PID_v1.h>

const uint8_t PWM_PIN = 4;

double setpoint = 1.0;   // target speed, m/s
double input    = 0.0;   // measured speed, m/s
double output   = 0.0;   // PWM command, -255..+255

PID motorPid(&input, &output, &setpoint,
             2.0, 0.4, 0.1, DIRECT);

void setup() {
    pinMode(PWM_PIN, OUTPUT);
    motorPid.SetOutputLimits(-255.0, 255.0);
    motorPid.SetSampleTime(5);        // 200 Hz loop
    motorPid.SetMode(AUTOMATIC);
}

void loop() {
    input = readEncoderSpeed();       // user-supplied
    if (motorPid.Compute()) {
        double pwm = fabs(output);
        bool forward = (output >= 0);
        analogWrite(PWM_PIN, (int)pwm);
        digitalWrite(5, forward ? HIGH : LOW);
        digitalWrite(6, forward ? LOW  : HIGH);
    }
}
```

---

## 5. Tuning Guide

### Manual Ziegler–Nichols Method

1. Set `Ki = Kd = 0`.
2. Increase `Kp` until the output oscillates with constant amplitude — call
   this `Ku` and the period `Tu`.
3. Apply classical ZN settings:

| Controller | Kp        | Ki              | Kd            |
|------------|-----------|-----------------|---------------|
| P only     | 0.5·Ku    | 0               | 0             |
| PI         | 0.45·Ku   | 0.54·Ku / Tu    | 0             |
| PID (classic) | 0.6·Ku | 1.2·Ku / Tu    | 0.075·Ku·Tu   |
| PID (some overshoot) | 0.33·Ku | 0.66·Ku/Tu | 0.11·Ku·Tu    |
| PID (no overshoot)   | 0.2·Ku  | 0.4·Ku/Tu  | 0.066·Ku·Tu   |

> ZN gives aggressive gains — reduce by 30–50% as a starting point.

### Practical Tips

- **Always set output limits** before enabling the PID.
- **Use the lowest sample time** your hardware supports — faster is better
  up to ~1 kHz.
- **Filter noisy inputs** (encoders, ADCs) before they reach the PID — a
  one-pole low-pass with α ≈ 0.1 usually suffices.
- **If overshoot is unacceptable**, set `P_ON_M` (proportional-on-
  measurement) — this is a variant of "I-PD" control.

---

## 6. Anti-Windup

`PID_v1` includes **conditional integration anti-windup**: when the output
hits a limit, the integrator is frozen (and back-calculated).  You get this
automatically — just remember to call `SetOutputLimits()`.

If you need more aggressive anti-windup (e.g. tracking-mode), consider
upgrading to **`PID_AutoTune`** (also by Brett Beauregard) or the newer
`PID_v2` library which exposes more internals.

---

## 7. AutoTuning (`PID_AutoTune`)

The companion library `PID_AutoTune_v0` (<https://github.com/br3ttb/Arduino-AutoTune>)
runs a relay-feedback experiment to estimate Ku and Tu, then computes PID
gains automatically.  Typical flow:

```cpp
PID_AutoTune aTune(&input, &output);
aTune.SetLookbackSec(2);
aTune.SetControlType(1);   // PID
aTune.SetOutputStep(50);   // relay amplitude

while (!aTune.Runtime()) { /* keep loop alive */ }

double Kp = aTune.GetKp();
double Ki = aTune.GetKi();
double Kd = aTune.GetKd();
motorPid.SetTunings(Kp, Ki, Kd);
```

> AutoTune requires the system to oscillate during the experiment — only
  run it with the wheels off the ground.

---

## 8. Common Pitfalls

| Symptom                                | Likely Cause / Fix                            |
|----------------------------------------|-----------------------------------------------|
| Output stuck at limit                  | Integrator wound up — check limits & input sign. |
| Oscillation grows                      | Kp too high → reduce 25%.                     |
| Slow response                          | Kp too low → increase 25%.                    |
| Constant offset from set-point         | Ki = 0 → add a small integral term.            |
| Noisy, jerky output                    | Kd amplifying noise → filter input or reduce Kd. |
| `Compute()` never returns true         | Sample time too long; check `SetSampleTime()`. |
| Direction reversed (motor runs away)   | Wrong `DIRECT` vs `REVERSE` — swap.            |

---

## 9. See Also

- Official blog series (highly recommended):
  <http://brettbeauregard.com/blog/2011/04/improving-the-beginners-pid-introduction/>
- Source: <https://github.com/br3ttb/Arduino-PID-Library>
- `PID_v2` (modernised fork): <https://github.com/br3ttb/Arduino-PID-Library/tree/master/PID_v2>
- Ziegler-Nichols: <https://en.wikipedia.org/wiki/Ziegler–Nichols_method>
