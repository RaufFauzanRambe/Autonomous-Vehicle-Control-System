/**
 * @file    motor_test.ino
 * @brief   Drive-motor & encoder test sketch.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 *
 * @details
 * Cycles the drive motor through a controlled sequence:
 *
 *     1. Ramp forward 0 → 100 % PWM over 3 s.
 *     2. Hold 100 % for 1 s.
 *     3. Ramp down 100 → 0 % over 3 s.
 *     4. Brake for 0.5 s.
 *     5. Reverse the sequence (0 → -100 → 0).
 *     6. Repeat.
 *
 * The sketch also reads the quadrature encoder via an interrupt and
 * prints the cumulative tick count, instantaneous RPM and inferred
 * linear speed to the serial monitor at 10 Hz.
 *
 * Wiring (Arduino Mega 2560):
 *
 *     MOTOR_PWM_PIN        = 4   (PWM to H-bridge PWMA)
 *     MOTOR_IN1_PIN        = 5   (direction bit 1)
 *     MOTOR_IN2_PIN        = 6   (direction bit 2)
 *     MOTOR_BRAKE_PIN      = 7   (active-high brake)
 *     MOTOR_ENCODER_A_PIN  = 18  (interrupt-capable, INT3)
 *     MOTOR_ENCODER_B_PIN  = 19  (direction sense)
 */

#include <Arduino.h>

// ---- Pin assignments (override via -D flags if needed) ----------------
#ifndef MOTOR_PWM_PIN
#  define MOTOR_PWM_PIN       4
#endif
#ifndef MOTOR_IN1_PIN
#  define MOTOR_IN1_PIN       5
#endif
#ifndef MOTOR_IN2_PIN
#  define MOTOR_IN2_PIN       6
#endif
#ifndef MOTOR_BRAKE_PIN
#  define MOTOR_BRAKE_PIN     7
#endif
#ifndef MOTOR_ENCODER_A_PIN
#  define MOTOR_ENCODER_A_PIN 18
#endif
#ifndef MOTOR_ENCODER_B_PIN
#  define MOTOR_ENCODER_B_PIN 19
#endif

// ---- Motor / encoder constants ----------------------------------------
static constexpr uint16_t ENCODER_PPR        = 220 * 4;  // quadrature x4
static constexpr float    WHEEL_DIAMETER_MM  = 65.0f;
static constexpr uint16_t PWM_MAX            = 255;
static constexpr uint32_t SERIAL_BAUD        = 115200;

// ---- Volatile state mutated by the ISR --------------------------------
volatile int32_t g_encoderTicks = 0;

// ---- Static instance pointer so the ISR can dispatch ------------------
// (For a single-motor test this is fine; in production use the
// MotorController class which manages this internally.)

/**
 * @brief Quadrature-encoder ISR — fires on rising edge of phase A.
 */
static void encoderIsr() {
    int b = digitalRead(MOTOR_ENCODER_B_PIN);
    if (b == HIGH) g_encoderTicks++;
    else           g_encoderTicks--;
}

/**
 * @brief Set the H-bridge direction & PWM signals.
 * @param dir  +1 forward, -1 reverse, 0 brake.
 * @param pwm  0–255.
 */
static void writeDriver(int dir, uint8_t pwm) {
    switch (dir) {
        case  1:
            digitalWrite(MOTOR_IN1_PIN, HIGH);
            digitalWrite(MOTOR_IN2_PIN, LOW);
            digitalWrite(MOTOR_BRAKE_PIN, LOW);
            analogWrite(MOTOR_PWM_PIN, pwm);
            break;
        case -1:
            digitalWrite(MOTOR_IN1_PIN, LOW);
            digitalWrite(MOTOR_IN2_PIN, HIGH);
            digitalWrite(MOTOR_BRAKE_PIN, LOW);
            analogWrite(MOTOR_PWM_PIN, pwm);
            break;
        default:  // brake
            digitalWrite(MOTOR_IN1_PIN, HIGH);
            digitalWrite(MOTOR_IN2_PIN, HIGH);
            digitalWrite(MOTOR_BRAKE_PIN, HIGH);
            analogWrite(MOTOR_PWM_PIN, 0);
            break;
    }
}

/**
 * @brief Print the current encoder count, RPM and inferred linear speed.
 */
static void printStatus(uint8_t pwm, int dir) {
    static int32_t   lastTicks   = 0;
    static uint32_t  lastMicros  = 0;

    noInterrupts();
    int32_t ticksNow = g_encoderTicks;
    interrupts();

    uint32_t now = micros();
    int32_t  delta = ticksNow - lastTicks;
    uint32_t dtUs  = now - lastMicros;
    lastTicks  = ticksNow;
    lastMicros = now;

    float rpm = 0.0f, mps = 0.0f;
    if (dtUs > 0) {
        float dtSec = dtUs / 1.0e6f;
        rpm = (static_cast<float>(delta) / ENCODER_PPR) / dtSec * 60.0f;
        float circM = (PI * WHEEL_DIAMETER_MM) / 1000.0f;
        mps = (rpm / 60.0f) * circM;
    }

    Serial.print(F("PWM="));     Serial.print(pwm);
    Serial.print(F(" dir="));    Serial.print(dir > 0 ? F("FWD ") : (dir < 0 ? F("REV ") : F("BRK ")));
    Serial.print(F(" ticks="));  Serial.print(ticksNow);
    Serial.print(F(" rpm="));    Serial.print(rpm, 1);
    Serial.print(F(" mps="));    Serial.println(mps, 3);
}

/**
 * @brief Drive the motor through one full ramp-up / ramp-down cycle in the
 *        given direction (+1 or -1).
 */
static void runCycle(int dir) {
    // Ramp up over 3 s.
    for (uint16_t pwm = 0; pwm <= PWM_MAX; pwm += 5) {
        writeDriver(dir, pwm);
        printStatus(pwm, dir);
        delay(30);   // → 60 ms × 51 steps ≈ 3 s
    }
    // Hold 1 s.
    delay(1000);
    // Ramp down.
    for (int16_t pwm = PWM_MAX; pwm >= 0; pwm -= 5) {
        writeDriver(dir, (uint8_t)pwm);
        printStatus((uint8_t)pwm, dir);
        delay(30);
    }
    // Brake briefly.
    writeDriver(0, 0);
    printStatus(0, 0);
    delay(500);
}

void setup() {
    Serial.begin(SERIAL_BAUD);
    while (!Serial) { ; }

    pinMode(MOTOR_PWM_PIN,   OUTPUT);
    pinMode(MOTOR_IN1_PIN,   OUTPUT);
    pinMode(MOTOR_IN2_PIN,   OUTPUT);
    pinMode(MOTOR_BRAKE_PIN, OUTPUT);
    pinMode(MOTOR_ENCODER_A_PIN, INPUT_PULLUP);
    pinMode(MOTOR_ENCODER_B_PIN, INPUT_PULLUP);

    analogWrite(MOTOR_PWM_PIN, 0);
    writeDriver(0, 0);   // brake at boot for safety

    attachInterrupt(digitalPinToInterrupt(MOTOR_ENCODER_A_PIN),
                    encoderIsr, RISING);

    Serial.println(F("\n[motor_test] starting — ensure wheels are OFF the ground!"));
    delay(2000);   // 2 s grace period
}

void loop() {
    Serial.println(F("\n=== FORWARD CYCLE ==="));
    runCycle(+1);

    Serial.println(F("\n=== REVERSE CYCLE ==="));
    runCycle(-1);
}
