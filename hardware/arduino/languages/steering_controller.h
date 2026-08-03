/**
 * @file    steering_controller.h
 * @brief   Ackermann-aware steering-servo controller with PID.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#ifndef STEERING_CONTROLLER_H
#define STEERING_CONTROLLER_H
#pragma once

#include <Arduino.h>
#include <Servo.h>
#include <PID_v1.h>
#include "arduino_config.h"
#include "wiring_config.h"

namespace avcs {

/**
 * @brief Steering controller state snapshot.
 */
struct SteeringState {
    float  currentAngleDeg;  ///< Measured angle (from feedback pot) — deg from centre.
    float  targetAngleDeg;   ///< Commanded angle — deg from centre (negative = left).
    float  servoUs;          ///< Last commanded servo pulse width (µs).
    bool   calibrated;       ///< True once calibration has been performed.
    bool   pidEnabled;
};

/**
 * @brief Steering controller wrapping a hobby servo.
 *
 * The controller accepts angle commands in degrees, with 0° = straight ahead,
 * positive = right, negative = left.  The angle is clamped to
 * ±STEER_MAX_ANGLE.  Internally the angle is mapped to a 1000–2000 µs servo
 * pulse (1500 µs = centre).
 *
 * An optional PID loop can be enabled when an analog feedback pot is wired
 * to STEERING_FEEDBACK_PIN; otherwise the servo is driven open-loop.
 *
 * Ackermann geometry note: for a vehicle with wheelbase L and track T, the
 * inside wheel must turn through a sharper angle than the outside wheel.
 * For a single-servo steering linkage (typical of RC cars) this is handled
 * mechanically by the steering rods; the controller therefore commands a
 * single virtual "steer angle" that approximates the average of both wheels.
 */
class SteeringController {
public:
    /**
     * @brief Construct a SteeringController.
     * @param servoPin     Digital pin driving the servo signal line.
     * @param feedbackPin  Analog pin reading servo-horn pot (0xFF if absent).
     */
    SteeringController(uint8_t servoPin, uint8_t feedbackPin = 0xFF);

    ~SteeringController();

    /// Attach servo, set default angle, prime PID.  Call from setup().
    void begin();

    /**
     * @brief  Command the steering to a target angle.
     * @param  angleDeg  Angle in degrees, clamped to ±STEER_MAX_ANGLE.
     * @param  positive = right, negative = left.
     */
    void setAngle(float angleDeg);

    /// Set PID gains at runtime.
    void setTunings(float kp, float ki, float kd);

    /**
     * @brief  Calibrate the centre pulse and feedback voltage.
     *
     * With the wheels pointing dead-ahead, call this method to record the
     * current servo pulse as the new STEER_CENTER_US reference (stored in
     * EEPROM on AVR boards if persistent flag is set).
     */
    void calibrateCentre();

    /// Run one control iteration.  Call at CONTROL_LOOP_HZ.
    void update();

    /// Read-only access to the current state.
    inline const SteeringState& state() const { return _state; }

    /// Convert a target angle (deg) into a servo pulse-width (µs).
    uint16_t angleToUs(float angleDeg) const;

private:
    uint8_t   _servoPin;
    uint8_t   _feedbackPin;
    Servo     _servo;
    PID       _pid;
    double    _pidInput;     // measured angle (deg)
    double    _pidSetpoint;  // target angle (deg)
    double    _pidOutput;    // correction offset (µs)

    SteeringState _state;

    /// Pulse-width stored from last calibrateCentre() call.
    float    _centerUs;
};

}  // namespace avcs

#endif // STEERING_CONTROLLER_H
