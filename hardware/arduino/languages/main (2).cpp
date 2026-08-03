/**
 * @file    main.cpp
 * @brief   Entry point for the AVCS Arduino firmware.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 *
 * @details
 * This file wires together all subsystems (sensors, motor, steering, GPS,
 * IMU) into a single, time-triggered control loop.  The loop executes at
 * CONTROL_LOOP_HZ (default 200 Hz) and dispatches to each subsystem based
 * on its own rate limit.
 *
 * The vehicle state machine has three top-level modes:
 *
 *   - MANUAL         : actuator commands come from the companion computer.
 *   - AUTONOMOUS     : firmware drives to the next GPS waypoint.
 *   - EMERGENCY_STOP : all actuators off, motor brake engaged.
 */
#include <Arduino.h>

#include "arduino_config.h"
#include "board_config.h"
#include "wiring_config.h"
#include "sensors.h"
#include "motor_controller.h"
#include "steering_controller.h"
#include "gps_module.h"
#include "imu_module.h"
#include "utilities.h"

using namespace avcs;

/* ========================================================================= *
 *  Globals (singletons — we have exactly one of each subsystem per vehicle)
 * ========================================================================= */
static SensorManager        g_sensors;
static MotorController      g_motor(MOTOR_PWM_PIN, MOTOR_IN1_PIN, MOTOR_IN2_PIN,
                                    MOTOR_BRAKE_PIN, MOTOR_ENCODER_A_PIN,
                                    MOTOR_ENCODER_B_PIN);
static SteeringController   g_steering(STEERING_PWM_PIN, STEERING_FEEDBACK_PIN);
static GpsModule            g_gps;
static ImuModule            g_imu;

static VehicleMode g_mode = VehicleMode::MANUAL;
static uint32_t    g_lastControlUs = 0;
static uint32_t    g_lastTelemetryMs = 0;

/* ========================================================================= *
 *  Forward declarations
 * ========================================================================= */
static void initStatusLed();
static void blinkHeartbeat();
static void checkEStopInput();
static void runAutonomousStep();
static void transmitTelemetry();
static void handleCompanionLink();

/* ========================================================================= *
 *  Arduino setup()
 * ========================================================================= */
void setup() {
    // ---- 1. Debug UART ------------------------------------------------
    DBG_BEGIN(DEBUG_BAUD);
    delay(100);
    DBG_PRINTLN(F("\n[AVCS] boot -------------------------------------------"));
    DBG_PRINT(F("[AVCS] board  : "));
    DBG_PRINTLN(F(BOARD_NAME));

    // ---- 2. Status LED & e-stop pin ----------------------------------
    initStatusLed();
    pinMode(E_STOP_PIN, INPUT_PULLUP);

    // ---- 3. Subsystem init -------------------------------------------
    g_sensors.begin();
    g_motor.begin();
    g_steering.begin();
    g_gps.begin();
    if (!g_imu.begin()) {
        DBG_PRINTLN(F("[AVCS] WARNING: continuing without IMU"));
    }

    // ---- 4. Default mode ---------------------------------------------
    g_mode = VehicleMode::MANUAL;

    // Prime the time-stamps so the first loop() iteration isn't a no-op.
    g_lastControlUs   = micros();
    g_lastTelemetryMs = millis();

    DBG_PRINTLN(F("[AVCS] setup complete"));
}

/* ========================================================================= *
 *  Arduino loop()
 * ========================================================================= */
void loop() {
    // ---- A. E-stop is the highest-priority check ----------------------
    checkEStopInput();

    // ---- B. Pump the GPS parser every iteration (cheap) --------------
    g_gps.update();

    // ---- C. Time-triggered control loop (CONTROL_LOOP_HZ) ------------
    uint32_t now = micros();
    if ((now - g_lastControlUs) >= CONTROL_LOOP_US) {
        g_lastControlUs = now;

        // 1. Update every sensor.
        g_sensors.update();
        g_imu.update();

        // 2. Run the PID loops.
        g_motor.update();
        g_steering.update();

        // 3. Mode-specific behaviour.
        switch (g_mode) {
            case VehicleMode::MANUAL:
                // Actuator commands come from the companion computer; the
                // PID set-points are written in handleCompanionLink().
                break;
            case VehicleMode::AUTONOMOUS:
                runAutonomousStep();
                break;
            case VehicleMode::EMERGENCY_STOP:
            case VehicleMode::FAULT:
            default:
                g_motor.emergencyStop();
                g_steering.setAngle(0.0f);
                break;
        }

        // 4. Safety layer — proximity check (overrides mode).
        const FusedSensorState& s = g_sensors.state();
        if (!isnan(s.obstacleMinForwardM)
            && s.obstacleMinForwardM < OBSTACLE_STOP_M) {
            g_motor.brake();
            DBG_PRINT(F("[SAFE] obstacle stop @ "));
            DBG_PRINT(s.obstacleMinForwardM);
            DBG_PRINTLN(F(" m"));
        }
    }

    // ---- D. Telemetry transmission (TELEMETRY_HZ) --------------------
    if ((millis() - g_lastTelemetryMs) >= (1000UL / TELEMETRY_HZ)) {
        g_lastTelemetryMs = millis();
        transmitTelemetry();
    }

    // ---- E. Companion-computer link (non-blocking) -------------------
    handleCompanionLink();

    // ---- F. Heartbeat blink ------------------------------------------
    blinkHeartbeat();
}

/* ========================================================================= *
 *  Helper implementations
 * ========================================================================= */

static void initStatusLed() {
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);
}

/// Blink the status LED at HEARTBEAT_HZ — indicates firmware is alive.
static void blinkHeartbeat() {
    static uint32_t lastToggle = 0;
    static bool     state      = false;
    uint32_t period = 500UL / HEARTBEAT_HZ;
    if ((millis() - lastToggle) >= period) {
        lastToggle = millis();
        state = !state;
        digitalWrite(STATUS_LED_PIN, state ? HIGH : LOW);
    }
}

/// Monitor the hardware e-stop button (interrupt-capable pin).
static void checkEStopInput() {
    uint8_t level = digitalRead(E_STOP_PIN);
    bool stopAsserted = (level == E_STOP_LEVEL);

    if (stopAsserted && g_mode != VehicleMode::EMERGENCY_STOP) {
        g_mode = VehicleMode::EMERGENCY_STOP;
        g_motor.emergencyStop();
        DBG_PRINTLN(F("[E-ST] hardware e-stop asserted"));
    }
}

/**
 * @brief One autonomous-control step.
 *
 * Strategy:
 *   - If GPS fix is not navigable, brake and wait.
 *   - Otherwise compute the bearing & distance to the next waypoint
 *     (hard-coded for now; replace with a queue received over the
 *     companion link), then command the steering angle proportional
 *     to the heading error.
 */
static void runAutonomousStep() {
    if (!g_gps.isNavigable()) {
        g_motor.brake();
        return;
    }

    // ----- Example: head toward a fixed target waypoint ----------------
    // In a production system these would come from a queue populated by
    // the companion computer via handleCompanionLink().
    constexpr double TARGET_LAT = 37.7749;
    constexpr double TARGET_LON = -122.4194;

    float bearing = g_gps.bearingTo(TARGET_LAT, TARGET_LON);  // degrees true
    float distance = g_gps.distanceToM(TARGET_LAT, TARGET_LON);

    if (distance < 1.0f) {
        // Arrived.
        g_motor.setSpeed(0.0f);
        g_steering.setAngle(0.0f);
        return;
    }

    // Heading error: difference between the desired bearing and the IMU yaw.
    // Both are in degrees; we use wrapAngle180 to take the short way around.
    float headingErr = wrapAngle180(bearing - g_imu.state().yawDeg);

    // Proportional steering: 1° of heading error → 0.5° of steering.
    float steerCmd = clamp(headingErr * 0.5f,
                           -STEER_MAX_ANGLE, STEER_MAX_ANGLE);
    g_steering.setAngle(steerCmd);

    // Speed scales inversely with obstacle distance and heading error.
    const FusedSensorState& s = g_sensors.state();
    float obstacle = s.obstacleMinForwardM;
    float speedCmd = 1.0f;  // nominal cruising speed (m/s)

    if (!isnan(obstacle) && obstacle < OBSTACLE_SLOW_M) {
        speedCmd *= clamp(obstacle / OBSTACLE_SLOW_M, 0.0f, 1.0f);
    }
    // Slow down for large heading errors (sharp turn ahead).
    speedCmd *= clamp(1.0f - (fabsf(headingErr) / 60.0f), 0.2f, 1.0f);

    g_motor.setSpeed(speedCmd);
}

/// Emit a compact JSON telemetry packet on the companion link.
static void transmitTelemetry() {
    const FusedSensorState& s = g_sensors.state();
    const MotorState&       m = g_motor.state();
    const SteeringState&    t = g_steering.state();
    const GpsState&         g = g_gps.state();
    const ImuState&         i = g_imu.state();

    // Format: comma-separated key=value pairs (parseable by the companion
    // computer without a JSON parser to keep things light on the 8-bit AVR).
    COMPANION_SERIAL.print(F("T,"));
    COMPANION_SERIAL.print(millis());                 COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(static_cast<int>(g_mode)); COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(m.currentSpeedMps, 2);     COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(m.targetSpeedMps,  2);     COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(t.targetAngleDeg,  2);     COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(i.yawDeg,          2);     COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(g.latitudeDeg,     6);     COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(g.longitudeDeg,    6);     COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(g.speedMps,        2);     COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(g.courseDeg,       2);     COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(g.satellites);             COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(s.obstacleMinForwardM, 2); COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(s.batteryV,        2);     COMPANION_SERIAL.print(',');
    COMPANION_SERIAL.print(s.motorCurrentA,   2);
    COMPANION_SERIAL.println();
}

/**
 * @brief Parse incoming commands from the companion computer.
 *
 * Supported commands (single ASCII char prefix + payload):
 *   M  — set mode              (0=manual, 1=auto, 2=estop, 3=fault)
 *   V  — set target speed      (m/s, float)
 *   S  — set steering angle    (deg, float)
 *   P  — set motor PID gains   (Kp, Ki, Kd, comma-separated floats)
 *
 * Commands are terminated with '\n'.
 */
static void handleCompanionLink() {
    static char buf[64];
    static uint8_t idx = 0;

    while (COMPANION_SERIAL.available() > 0) {
        char c = COMPANION_SERIAL.read();
        if (c == '\n') {
            buf[idx] = '\0';
            idx = 0;

            switch (buf[0]) {
                case 'M': {
                    int m = atoi(&buf[1]);
                    g_mode = static_cast<VehicleMode>(m);
                    if (g_mode == VehicleMode::EMERGENCY_STOP)
                        g_motor.emergencyStop();
                    else if (g_mode == VehicleMode::MANUAL)
                        g_motor.resetEmergency();
                    DBG_PRINT(F("[CMD] mode=")); DBG_PRINTLN(m);
                    break;
                }
                case 'V': {
                    float v = strtof(&buf[1], nullptr);
                    if (g_mode == VehicleMode::MANUAL) g_motor.setSpeed(v);
                    break;
                }
                case 'S': {
                    float a = strtof(&buf[1], nullptr);
                    if (g_mode == VehicleMode::MANUAL) g_steering.setAngle(a);
                    break;
                }
                case 'P': {
                    float kp, ki, kd;
                    if (sscanf(&buf[1], "%f,%f,%f", &kp, &ki, &kd) == 3) {
                        g_motor.setTunings(kp, ki, kd);
                    }
                    break;
                }
                default:
                    DBG_PRINT(F("[CMD] unknown: ")); DBG_PRINTLN(buf);
                    break;
            }
        } else if (idx < (sizeof(buf) - 1)) {
            buf[idx++] = c;
        }
    }
}
