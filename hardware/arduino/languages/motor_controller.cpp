/**
 * @file    motor_controller.cpp
 * @brief   Implementation of the closed-loop motor controller.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#include "motor_controller.h"
#include "utilities.h"

namespace avcs {

// Static pointer used by the ISR trampoline.
MotorController* MotorController::_active = nullptr;

/* ========================================================================= *
 *  Constructor / destructor
 * ========================================================================= */

MotorController::MotorController(uint8_t pwmPin, uint8_t in1Pin, uint8_t in2Pin,
                                 uint8_t brakePin, uint8_t encAPin, uint8_t encBPin)
    : _pwmPin(pwmPin)
    , _in1Pin(in1Pin)
    , _in2Pin(in2Pin)
    , _brakePin(brakePin)
    , _encAPin(encAPin)
    , _encBPin(encBPin)
    , _pid(& _pidInput, &_pidOutput, &_pidSetpoint,
           MOTOR_KP, MOTOR_KI, MOTOR_KD, DIRECT)
    , _encoderTicks(0)
    , _lastEncoderTicks(0)
    , _lastUpdateUs(0)
{
    _state.currentSpeedMps = 0.0f;
    _state.targetSpeedMps  = 0.0f;
    _state.pwmOutput       = 0.0f;
    _state.direction       = MotorDirection::COAST;
    _state.encoderTicks    = 0;
    _state.rpm             = 0.0f;
    _state.pidEnabled      = false;
    _state.emergencyStop   = false;
}

MotorController::~MotorController() {
    if (_active == this) _active = nullptr;
}

/* ========================================================================= *
 *  begin()
 * ========================================================================= */
void MotorController::begin() {
    pinMode(_pwmPin,   OUTPUT);
    pinMode(_in1Pin,   OUTPUT);
    pinMode(_in2Pin,   OUTPUT);
    if (_brakePin != 0xFF) pinMode(_brakePin, OUTPUT);
    pinMode(_encAPin,  INPUT_PULLUP);
    pinMode(_encBPin,  INPUT_PULLUP);

    // Configure PWM frequency.  On AVR, timer 0 (pins 5/6) defaults to ~980 Hz
    // while the others are ~490 Hz.  For motor control 490 Hz is fine.
    analogWrite(_pwmPin, 0);

    // Default: motor idle.
    writeDriver(MotorDirection::COAST, 0);

    // Configure PID library.
    _pid.SetOutputLimits(MOTOR_OUT_MIN, MOTOR_OUT_MAX);
    _pid.SetSampleTime(1000UL / CONTROL_LOOP_HZ); // ms
    _pid.SetMode(AUTOMATIC);

    // Install the encoder ISR — only one MotorController object per vehicle.
    _active = this;
    attachInterrupt(digitalPinToInterrupt(_encAPin), MotorController::encoderIsr, RISING);

    _lastUpdateUs = micros();
    DBG_PRINTLN(F("[MOTR] MotorController initialised"));
}

/* ========================================================================= *
 *  Tuning & set-point
 * ========================================================================= */
void MotorController::setTunings(float kp, float ki, float kd) {
    _pid.SetTunings(kp, ki, kd);
}

void MotorController::setSpeed(float speedMps) {
    if (_state.emergencyStop) return;
    _pidSetpoint      = static_cast<double>(speedMps);
    _state.targetSpeedMps = speedMps;
    _state.pidEnabled = true;
    _pid.SetMode(AUTOMATIC);
}

void MotorController::enablePid(bool enabled) {
    _state.pidEnabled = enabled;
    _pid.SetMode(enabled ? AUTOMATIC : MANUAL);
}

void MotorController::brake() {
    if (_state.emergencyStop) return;
    _state.pidEnabled = false;
    _pid.SetMode(MANUAL);
    writeDriver(MotorDirection::BRAKE, 0);
}

void MotorController::release() {
    if (_state.emergencyStop) return;
    writeDriver(MotorDirection::FORWARD, 0);
    _state.pidEnabled = true;
    _pid.SetMode(AUTOMATIC);
}

void MotorController::emergencyStop() {
    _state.emergencyStop = true;
    _state.pidEnabled    = false;
    _pid.SetMode(MANUAL);
    _pidSetpoint = 0.0;
    writeDriver(MotorDirection::BRAKE, 0);
    DBG_PRINTLN(F("[MOTR] EMERGENCY STOP"));
}

void MotorController::resetEmergency() {
    _state.emergencyStop = false;
    _state.pidEnabled    = true;
    _pid.SetMode(AUTOMATIC);
}

/* ========================================================================= *
 *  update()
 * ========================================================================= */
void MotorController::update() {
    // ---- 1. Snapshot ISR-mutated encoder count ---------------------------
    noInterrupts();
    int32_t ticksNow = _encoderTicks;
    interrupts();

    int32_t deltaTicks = ticksNow - _lastEncoderTicks;
    _lastEncoderTicks  = ticksNow;
    _state.encoderTicks = ticksNow;

    uint32_t nowUs = micros();
    uint32_t dtUs  = microsDelta(_lastUpdateUs, nowUs);
    _lastUpdateUs  = nowUs;

    // ---- 2. Convert ticks → RPM → m/s -----------------------------------
    // ticks_per_rev = ENCODER_PPR (already ×4 for quadrature).
    // rpm = (deltaTicks / PPR) * (60 / dtSeconds)
    if (dtUs > 0) {
        float dtSec   = dtUs / 1.0e6f;
        float revs    = static_cast<float>(deltaTicks) / static_cast<float>(ENCODER_PPR);
        _state.rpm    = (revs / dtSec) * 60.0f;
        _state.currentSpeedMps = rpmToMps(_state.rpm, WHEEL_DIAMETER_MM);
    }

    // ---- 3. Feed PID -----------------------------------------------------
    _pidInput = static_cast<double>(_state.currentSpeedMps);

    if (_state.pidEnabled && _pid.Compute()) {
        // PID output is in [-255, +255]; sign indicates direction.
        double out  = _pidOutput;
        float  pwm  = static_cast<float>(fabs(out));
        uint8_t pwm8 = static_cast<uint8_t>(clamp(pwm, 0.0f, 255.0f));
        MotorDirection dir = (out >= 0.0) ? MotorDirection::FORWARD : MotorDirection::REVERSE;
        writeDriver(dir, pwm8);
        _state.pwmOutput = static_cast<float>(out);
        _state.direction = dir;
    }
}

/* ========================================================================= *
 *  Encoder ISR
 * ========================================================================= */
void MotorController::encoderIsr() {
    if (_active == nullptr) return;
    // Read phase B to determine direction: A leads B → forward.
    int b = digitalRead(_active->_encBPin);
    if (b == HIGH) _active->_encoderTicks++;
    else           _active->_encoderTicks--;
}

/* ========================================================================= *
 *  Driver pin writer
 * ========================================================================= */
void MotorController::writeDriver(MotorDirection dir, uint8_t pwm) {
    switch (dir) {
        case MotorDirection::FORWARD:
            digitalWrite(_in1Pin, HIGH);
            digitalWrite(_in2Pin, LOW);
            analogWrite (_pwmPin, pwm);
            if (_brakePin != 0xFF) digitalWrite(_brakePin, LOW);
            break;
        case MotorDirection::REVERSE:
            digitalWrite(_in1Pin, LOW);
            digitalWrite(_in2Pin, HIGH);
            analogWrite (_pwmPin, pwm);
            if (_brakePin != 0xFF) digitalWrite(_brakePin, LOW);
            break;
        case MotorDirection::BRAKE:
            // Short motor windings by driving both inputs HIGH.
            digitalWrite(_in1Pin, HIGH);
            digitalWrite(_in2Pin, HIGH);
            analogWrite (_pwmPin, 0);
            if (_brakePin != 0xFF) digitalWrite(_brakePin, HIGH);
            break;
        case MotorDirection::COAST:
        default:
            digitalWrite(_in1Pin, LOW);
            digitalWrite(_in2Pin, LOW);
            analogWrite (_pwmPin, 0);
            if (_brakePin != 0xFF) digitalWrite(_brakePin, LOW);
            break;
    }
    _state.direction = dir;
}

}  // namespace avcs
