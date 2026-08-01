/**
 * @file    arduino_config.h
 * @brief   Global configuration constants for the AVCS Arduino firmware.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 *
 * @details
 * This header is the **single source of truth** for all tunable parameters of
 * the firmware: control-loop timing, PID gains, sensor sample rates, debug
 * flags, and the operating-mode enumeration.  Adjust values here and rebuild;
 * no other source file should hard-code these constants.
 *
 * All constants use `constexpr` so they enjoy type safety and are visible in
 * the debugger while still being evaluated at compile time.
 */
#ifndef ARDUINO_CONFIG_H
#define ARDUINO_CONFIG_H
#pragma once

/* ------------------------------------------------------------------------- *
 *  Operating mode enumeration
 * ------------------------------------------------------------------------- */
/**
 * @brief Operating state of the vehicle state machine.
 */
enum class VehicleMode : uint8_t {
    MANUAL          = 0, ///< Direct radio/teleop control — no autonomous logic.
    AUTONOMOUS      = 1, ///< Full self-driving — PID + GPS waypoint following.
    EMERGENCY_STOP  = 2, ///< All actuators off; motor brake engaged.
    FAULT            = 3  ///< Sensor fault detected — await operator reset.
};

/* ------------------------------------------------------------------------- *
 *  Control-loop timing
 * ------------------------------------------------------------------------- */
/// Main control loop frequency in Hz.  200 Hz = 5 ms period.
static constexpr uint32_t CONTROL_LOOP_HZ      = 200UL;
static constexpr uint32_t CONTROL_LOOP_US      = 1'000'000UL / CONTROL_LOOP_HZ;

/// Telemetry / state-print frequency in Hz.
static constexpr uint32_t TELEMETRY_HZ         = 10UL;

/// Heart-beat LED blink frequency in Hz.
static constexpr uint32_t HEARTBEAT_HZ         = 2UL;

/* ------------------------------------------------------------------------- *
 *  PID tuning defaults  (overridable at runtime via setTunings())
 * ------------------------------------------------------------------------- */
/// Drive-motor speed PID gains.  Tuned for a 12 V brushed motor with a 220 PPR
/// encoder, wheel diameter 65 mm.
static constexpr float MOTOR_KP    = 2.0f;
static constexpr float MOTOR_KI    = 0.4f;
static constexpr float MOTOR_KD    = 0.10f;
static constexpr float MOTOR_OUT_MIN = -255.0f; ///< PWM lower bound (reverse).
static constexpr float MOTOR_OUT_MAX =  255.0f; ///< PWM upper bound (forward).

/// Steering PID gains.  Output range maps to ±30° from centre.
static constexpr float STEER_KP    = 1.5f;
static constexpr float STEER_KI    = 0.0f;
static constexpr float STEER_KD    = 0.05f;
static constexpr float STEER_OUT_MIN = -30.0f;   ///< Degrees of centre.
static constexpr float STEER_OUT_MAX =  30.0f;
static constexpr float STEER_CENTER_US = 1500.0f; ///< Neutral servo pulse (µs).
static constexpr float STEER_RANGE_US  =  500.0f; ///< ±range from centre (µs).
static constexpr float STEER_MAX_ANGLE =  30.0f;  ///< Mechanical travel limit (°).

/* ------------------------------------------------------------------------- *
 *  Sensor sample rates & timeouts
 * ------------------------------------------------------------------------- */
static constexpr uint32_t GPS_BAUD            = 9600UL;   ///< NEO-6M default.
static constexpr uint32_t GPS_FIX_TIMEOUT_MS  = 5000UL;   ///< Failsafe trigger.
static constexpr uint32_t TELEMETRY_BAUD      = 115200UL; ///< Companion link.
static constexpr uint32_t DEBUG_BAUD          = 115200UL;

static constexpr uint32_t IMU_SAMPLE_HZ       = 100UL;
static constexpr uint32_t IMU_TIMEOUT_MS      = 200UL;

/// HC-SR04 echo timeout (out-of-range) in µs (~5 m round trip).
static constexpr uint32_t ULTRASONIC_TIMEOUT_US = 30000UL;
static constexpr uint32_t ULTRASONIC_SAMPLE_HZ  = 20UL;
/// Speed of sound at 20 °C, dry air.  Use for distance calculation.
static constexpr float    SPEED_OF_SOUND_MM_PER_US = 0.343f;

/// Encoder: pulses per revolution of the motor shaft (after quadrature x4).
static constexpr uint16_t ENCODER_PPR = 220 * 4;

/// Wheel diameter (mm) — used to convert RPM → linear speed.
static constexpr float    WHEEL_DIAMETER_MM = 65.0f;

/* ------------------------------------------------------------------------- *
 *  Battery & power monitoring
 * ------------------------------------------------------------------------- */
static constexpr float    BATTERY_LOW_V       = 10.5f;  ///< 3S LiPo cut-off.
static constexpr float    BATTERY_FULL_V      = 12.6f;
/// Voltage-divider ratio on the battery sense pin (R2 / (R1+R2)).
static constexpr float    BATTERY_DIV_RATIO   = 0.25f;

/* ------------------------------------------------------------------------- *
 *  Obstacle detection / safety
 * ------------------------------------------------------------------------- */
static constexpr float    OBSTACLE_STOP_M     = 0.30f; ///< Hard stop distance.
static constexpr float    OBSTACLE_SLOW_M     = 1.00f; ///< Begin slowing down.

/* ------------------------------------------------------------------------- *
 *  Debug & feature flags
 * ------------------------------------------------------------------------- */
#ifndef DEBUG
#  define DEBUG 1
#endif

#if DEBUG
#  define DBG_BEGIN(...)   do { Serial.begin(__VA_ARGS__); } while (0)
#  define DBG_PRINT(...)   do { Serial.print(__VA_ARGS__); } while (0)
#  define DBG_PRINTLN(...) do { Serial.println(__VA_ARGS__); } while (0)
#else
#  define DBG_BEGIN(...)   do {} while (0)
#  define DBG_PRINT(...)   do {} while (0)
#  define DBG_PRINTLN(...) do {} while (0)
#endif

/// E-stop input logic level (1 = active-high NC button, 0 = active-low NO).
static constexpr uint8_t  E_STOP_LEVEL        = 0;

/// Compile-time assertion helper.
#define STATIC_ASSERT(cond, name) typedef char static_assert_##name[(cond) ? 1 : -1]

#endif // ARDUINO_CONFIG_H
