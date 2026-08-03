/**
 * @file    motor_controller.h
 * @brief   Closed-loop motor-speed controller using PID + quadrature encoder.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#ifndef MOTOR_CONTROLLER_H
#define MOTOR_CONTROLLER_H
#pragma once

#include <Arduino.h>
#include <PID_v1.h>
#include "arduino_config.h"
#include "wiring_config.h"

namespace avcs {

/**
 * @brief Motor direction enumeration.
 */
enum class MotorDirection : uint8_t {
    FORWARD = 0,
    REVERSE = 1,
    BRAKE   = 2,
    COAST   = 3
};

/**
 * @brief Snapshot of the motor controller state.
 *
 * Exposed for telemetry & debugging.
 */
struct MotorState {
    float    currentSpeedMps;   ///< Measured linear speed (m/s).
    float    targetSpeedMps;    ///< Commanded set-point (m/s).
    float    pwmOutput;         ///< Raw PID output (−255 … +255).
    MotorDirection direction;   ///< Current direction.
    int32_t  encoderTicks;      ///< Cumulative encoder count.
    float    rpm;               ///< Measured motor-shaft RPM.
    bool     pidEnabled;        ///< True if PID loop is active.
    bool     emergencyStop;     ///< True if e-stop is asserted.
};

/**
 * @brief Closed-loop motor controller.
 *
 * Hardware assumptions:
 *   - H-bridge motor driver (TB6612FNG or L298N) with separate PWM, IN1, IN2
 *     and (optional) BRAKE inputs.
 *   - Quadrature encoder with phase A on an interrupt-capable pin.
 *
 * The controller maintains a PID loop over the linear vehicle speed
 * (m/s).  Encoder ticks are converted to RPM using ENCODER_PPR and then to
 * m/s using WHEEL_DIAMETER_MM.  The PID output is mapped directly to the
 * 8-bit PWM register (0–255).
 */
class MotorController {
public:
    /**
     * @brief Construct a MotorController.
     * @param pwmPin    PWM output pin to the H-bridge (must be PWM-capable).
     * @param in1Pin    Direction bit 1 pin.
     * @param in2Pin    Direction bit 2 pin.
     * @param brakePin  Active-high brake pin (use 0xFF to disable).
     * @param encAPin   Encoder phase A pin (must be interrupt-capable).
     * @param encBPin   Encoder phase B pin.
     */
    MotorController(uint8_t pwmPin, uint8_t in1Pin, uint8_t in2Pin,
                    uint8_t brakePin, uint8_t encAPin, uint8_t encBPin);

    ~MotorController();

    /// Initialise pins, attach encoder ISR, and prime the PID.
    void begin();

    /// Set new PID gains.  Safe to call at runtime.
    void setTunings(float kp, float ki, float kd);

    /**
     * @brief  Set the target linear speed (m/s).
     * @param  speedMps  Target speed; negative = reverse.
     */
    void setSpeed(float speedMps);

    /// Engage / disengage the PID loop (false = open-loop PWM passthrough).
    void enablePid(bool enabled);

    /// Apply the brake (motor shorted, PID suspended).
    void brake();

    /// Release the brake (returns to PID control with the existing set-point).
    void release();

    /**
     * @brief  Assert emergency stop: PWM = 0, brake ON, PID disabled.
     *         Stays asserted until resetEmergency() is called.
     */
    void emergencyStop();

    /// Clear the e-stop latch and resume PID control.
    void resetEmergency();

    /**
     * @brief  Run one iteration of the controller.  Call at CONTROL_LOOP_HZ.
     *
     * Steps:
     *   1. Compute current RPM from the encoder-tick delta since last call.
     *   2. Convert RPM → m/s.
     *   3. Run the PID Compute() step.
     *   4. Write direction & PWM pins.
     */
    void update();

    /// Read-only access to the current state snapshot.
    inline const MotorState& state() const { return _state; }

    /* ----------------------------------------------------------------- *
     * The encoder ISR must be static so it can be installed by
     * attachInterrupt().  We keep a single instance pointer so the ISR
     * can dispatch to the active object.
     * ----------------------------------------------------------------- */
    static void encoderIsr();

private:
    /// Apply the direction-select and PWM signals to the H-bridge.
    void writeDriver(MotorDirection dir, uint8_t pwm);

    // Pin numbers
    uint8_t _pwmPin, _in1Pin, _in2Pin, _brakePin, _encAPin, _encBPin;

    // PID
    PID     _pid;
    double  _pidInput;     ///< process variable (current speed m/s)
    double  _pidSetpoint;  ///< set-point (target speed m/s)
    double  _pidOutput;    ///< controller output (−255 … +255)

    // State
    MotorState    _state;
    volatile int32_t _encoderTicks;   ///< Cumulative encoder count (ISR-mutated).
    int32_t        _lastEncoderTicks; ///< Value at previous update().
    uint32_t       _lastUpdateUs;     ///< Time-stamp of previous update().

    static MotorController* _active;  ///< Pointer used by the ISR trampoline.
};

}  // namespace avcs

#endif // MOTOR_CONTROLLER_H
