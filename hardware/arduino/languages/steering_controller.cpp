/**
 * @file    steering_controller.cpp
 * @brief   Implementation of the steering-servo controller.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#include "steering_controller.h"
#include "utilities.h"

namespace avcs {

/* ========================================================================= *
 *  Constructor / destructor
 * ========================================================================= */
SteeringController::SteeringController(uint8_t servoPin, uint8_t feedbackPin)
    : _servoPin(servoPin)
    , _feedbackPin(feedbackPin)
    , _pid(&_pidInput, &_pidOutput, &_pidSetpoint,
           STEER_KP, STEER_KI, STEER_KD, DIRECT)
    , _centerUs(STEER_CENTER_US)
{
    _state.currentAngleDeg = 0.0f;
    _state.targetAngleDeg  = 0.0f;
    _state.servoUs         = STEER_CENTER_US;
    _state.calibrated      = false;
    _state.pidEnabled      = false;
}

SteeringController::~SteeringController() {
    _servo.detach();
}

/* ========================================================================= *
 *  begin()
 * ========================================================================= */
void SteeringController::begin() {
    _servo.attach(_servoPin);
    // Move to centre to prevent jerks at boot.
    _servo.writeMicroseconds(static_cast<int>(STEER_CENTER_US));

    if (_feedbackPin != 0xFF) {
        pinMode(_feedbackPin, INPUT);
        _pid.SetOutputLimits(-STEER_RANGE_US, STEER_RANGE_US);
        _pid.SetSampleTime(1000UL / CONTROL_LOOP_HZ);
        _pid.SetMode(AUTOMATIC);
        _state.pidEnabled = true;
    } else {
        _state.pidEnabled = false;
        _pid.SetMode(MANUAL);
    }

    DBG_PRINTLN(F("[STR ] SteeringController initialised"));
}

/* ========================================================================= *
 *  setAngle / setTunings
 * ========================================================================= */
void SteeringController::setAngle(float angleDeg) {
    // Software clamp to mechanical limits.
    _state.targetAngleDeg = clamp(angleDeg, -STEER_MAX_ANGLE, STEER_MAX_ANGLE);
    _pidSetpoint = static_cast<double>(_state.targetAngleDeg);

    // Open-loop path: write directly when no feedback is connected.
    if (!_state.pidEnabled) {
        uint16_t us = angleToUs(_state.targetAngleDeg);
        _servo.writeMicroseconds(us);
        _state.servoUs = static_cast<float>(us);
    }
}

void SteeringController::setTunings(float kp, float ki, float kd) {
    _pid.SetTunings(kp, ki, kd);
}

/* ========================================================================= *
 *  calibrateCentre()
 * ========================================================================= */
void SteeringController::calibrateCentre() {
    // Average 8 ADC readings to reduce noise.
    uint32_t sum = 0;
    if (_feedbackPin != 0xFF) {
        for (uint8_t i = 0; i < 8; i++) sum += analogRead(_feedbackPin);
        // Convert ADC → voltage → angle (placeholder: depends on pot mounting).
        // For demo we just store the current centre pulse as the reference.
    }
    _centerUs       = static_cast<float>(_servo.readMicroseconds());
    _state.calibrated = true;
    DBG_PRINT(F("[STR ] Calibrated centre = "));
    DBG_PRINTLN(_centerUs, 1);
}

/* ========================================================================= *
 *  update()
 * ========================================================================= */
void SteeringController::update() {
    if (!_state.pidEnabled) return; // open-loop: nothing to do per-cycle

    // ---- Read feedback pot & convert to degrees ------------------------
    // Assumption: pot sweeps 0 V → 5 V as the servo sweeps -30° → +30°.
    uint16_t adc   = analogRead(_feedbackPin);
    float volts    = (static_cast<float>(adc) / 1023.0f) * 5.0f;
    _state.currentAngleDeg =
        mapFloat(volts, 0.0f, 5.0f, -STEER_MAX_ANGLE, STEER_MAX_ANGLE);
    _pidInput = static_cast<double>(_state.currentAngleDeg);

    // ---- PID step ------------------------------------------------------
    if (_pid.Compute()) {
        // Output is an additive correction (µs) around the open-loop set-point.
        float openUs  = angleToUs(_state.targetAngleDeg);
        float cmdUs   = clamp(openUs + static_cast<float>(_pidOutput),
                              STEER_CENTER_US - STEER_RANGE_US,
                              STEER_CENTER_US + STEER_RANGE_US);
        _servo.writeMicroseconds(static_cast<int>(cmdUs));
        _state.servoUs = cmdUs;
    }
}

/* ========================================================================= *
 *  angleToUs()
 * ========================================================================= */
uint16_t SteeringController::angleToUs(float angleDeg) const {
    // Linear mapping:  -STEER_MAX_ANGLE → STEER_CENTER_US - STEER_RANGE_US
    //                  +STEER_MAX_ANGLE → STEER_CENTER_US + STEER_RANGE_US
    float us = mapFloat(angleDeg,
                        -STEER_MAX_ANGLE, STEER_MAX_ANGLE,
                        _centerUs - STEER_RANGE_US,
                        _centerUs + STEER_RANGE_US);
    return static_cast<uint16_t>(us);
}

}  // namespace avcs
