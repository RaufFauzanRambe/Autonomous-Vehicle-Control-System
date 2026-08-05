/**
 * @file motor_driver.cpp
 * @brief MotorDriver implementation — LEDC PWM + dual PID + ramp limit.
 * @author AVC team
 * @date 2024
 * @license MIT
 */

#include "motor_driver.h"
#include <driver/ledc.h>
#include <math.h>

// LEDC channel assignment (ESP32 has 16 channels; classic uses 0-7 for high-speed).
static constexpr ledc_channel_t kChanA = LEDC_CHANNEL_0;
static constexpr ledc_channel_t kChanB = LEDC_CHANNEL_1;
static constexpr ledc_timer_t   kTimer = LEDC_TIMER_0;

// Maximum wheel linear speed (m/s) at full PWM (for ramp limiting & saturation).
static constexpr float kMaxSpeedMs =
    (WHEEL_DIAMETER_M * PI) *
    (MOTOR_PWM_FREQ / 1.0f) /  // dummy; replaced by physical max below
    1.0f;  // ignored — see kPhysicalMaxSpeedMs
static constexpr float kPhysicalMaxSpeedMs = 1.5f;   // 1.5 m/s for AVC small vehicle

// ---------------------------------------------------------------------------
//  Construction / begin()
// ---------------------------------------------------------------------------
MotorDriver::MotorDriver() {
    // Default PID gains — tuned for small differential-drive robots
    // with 65 mm wheels and TB6612FNG @ 7.4 V.
    _gainsL = { 0.8f, 4.0f, 0.02f, -2.0f, 2.0f };
    _gainsR = { 0.8f, 4.0f, 0.02f, -2.0f, 2.0f };
}

bool MotorDriver::begin() {
    // Configure the LEDC timer for 20 kHz, 10-bit resolution.
    ledc_timer_config_t t = {};
    t.speed_mode          = LEDC_HIGH_SPEED_MODE;
    t.duty_resolution     = (ledc_timer_bit_t)MOTOR_PWM_BITS;
    t.timer_num           = kTimer;
    t.freq_hz             = MOTOR_PWM_FREQ;
    t.clk_cfg             = LEDC_AUTO_CLK;
    if (ledc_timer_config(&t) != ESP_OK) {
        AVC_LOGE("motor", "ledc_timer_config failed");
        return false;
    }

    // Configure each PWM channel.
    auto cfgChan = [&](int pin, ledc_channel_t ch) {
        ledc_channel_config_t c = {};
        c.gpio_num       = pin;
        c.speed_mode     = LEDC_HIGH_SPEED_MODE;
        c.channel        = ch;
        c.intr_type      = LEDC_INTR_DISABLE;
        c.timer_sel      = kTimer;
        c.duty           = 0;
        c.hpoint         = 0;
        ledc_channel_config(&c);
    };
    cfgChan(MOTOR_PWMA_PIN, kChanA);
    cfgChan(MOTOR_PWMB_PIN, kChanB);

    // Direction pins as outputs, default coast (both low).
    pinMode(MOTOR_AIN1_PIN, OUTPUT); digitalWrite(MOTOR_AIN1_PIN, LOW);
    pinMode(MOTOR_AIN2_PIN, OUTPUT); digitalWrite(MOTOR_AIN2_PIN, LOW);
    pinMode(MOTOR_BIN1_PIN, OUTPUT); digitalWrite(MOTOR_BIN1_PIN, LOW);
    pinMode(MOTOR_BIN2_PIN, OUTPUT); digitalWrite(MOTOR_BIN2_PIN, LOW);

    // Standby pin — high = bridge enabled.
    pinMode(MOTOR_STBY_PIN, OUTPUT);
    digitalWrite(MOTOR_STBY_PIN, HIGH);

    _initialised = true;
    _estopped    = false;
    AVC_LOGI("motor", "TB6612FNG initialised @ %u Hz / %u-bit",
             MOTOR_PWM_FREQ, MOTOR_PWM_BITS);
    return true;
}

// ---------------------------------------------------------------------------
//  Setpoint API
// ---------------------------------------------------------------------------
void MotorDriver::setSpeed(float left, float right) {
    // Clamp to physical limits.
    if (left  >  kPhysicalMaxSpeedMs) left  =  kPhysicalMaxSpeedMs;
    if (left  < -kPhysicalMaxSpeedMs) left  = -kPhysicalMaxSpeedMs;
    if (right >  kPhysicalMaxSpeedMs) right =  kPhysicalMaxSpeedMs;
    if (right < -kPhysicalMaxSpeedMs) right = -kPhysicalMaxSpeedMs;
    _cmdL = left;
    _cmdR = right;
}

void MotorDriver::setDiffDrive(float v, float w) {
    // Unicycle → differential-drive: vL = v - (w * L/2), vR = v + (w * L/2)
    float half = w * (WHEEL_BASE_M * 0.5f);
    setSpeed(v - half, v + half);
}

void MotorDriver::brake() {
    _cmdL = 0; _cmdR = 0;
    _dutyL = 0; _dutyR = 0;
    // Both bridge inputs HIGH = dynamic brake.
    digitalWrite(MOTOR_AIN1_PIN, HIGH); digitalWrite(MOTOR_AIN2_PIN, HIGH);
    digitalWrite(MOTOR_BIN1_PIN, HIGH); digitalWrite(MOTOR_BIN2_PIN, HIGH);
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, kChanA, 0);
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, kChanB, 0);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, kChanA);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, kChanB);
    _stateL.dir = MotorDir::kBrake;
    _stateR.dir = MotorDir::kBrake;
}

void MotorDriver::coast() {
    _cmdL = 0; _cmdR = 0;
    _dutyL = 0; _dutyR = 0;
    digitalWrite(MOTOR_AIN1_PIN, LOW); digitalWrite(MOTOR_AIN2_PIN, LOW);
    digitalWrite(MOTOR_BIN1_PIN, LOW); digitalWrite(MOTOR_BIN2_PIN, LOW);
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, kChanA, 0);
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, kChanB, 0);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, kChanA);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, kChanB);
    _stateL.dir = MotorDir::kCoast;
    _stateR.dir = MotorDir::kCoast;
}

void MotorDriver::estop() {
    _estopped = true;
    brake();
    digitalWrite(MOTOR_STBY_PIN, LOW);   // bridge into standby
    AVC_LOGW("motor", "E-STOP engaged");
}

void MotorDriver::clearEstop() {
    _estopped = false;
    digitalWrite(MOTOR_STBY_PIN, HIGH);
    AVC_LOGI("motor", "E-STOP cleared");
}

// ---------------------------------------------------------------------------
//  Per-cycle control step
// ---------------------------------------------------------------------------
void MotorDriver::step(float v_left_actual, float v_right_actual,
                       WheelState* state_out) {
    if (!_initialised || _estopped) return;
    uint32_t now = millis();
    float dt = (now - _lastStepMs) / 1000.0f;
    if (dt <= 0.0f || dt > 0.5f) dt = 1.0f / TASK_MOTOR_HZ;
    _lastStepMs = now;

    // Run PID on each wheel.
    float dutyL = pidStep(_gainsL, _integL, _prevErrL, _cmdL, v_left_actual);
    float dutyR = pidStep(_gainsR, _integR, _prevErrR, _cmdR, v_right_actual);

    // Apply acceleration ramp.
    _dutyL = ramp(_dutyL, dutyL);
    _dutyR = ramp(_dutyR, dutyR);

    writeMotor((uint8_t)kChanA, MOTOR_AIN1_PIN, MOTOR_AIN2_PIN, _dutyL);
    writeMotor((uint8_t)kChanB, MOTOR_BIN1_PIN, MOTOR_BIN2_PIN, _dutyR);

    _stateL.v_ms = v_left_actual;  _stateL.pwm = _dutyL;
    _stateL.dir = (_dutyL > 0.01f) ? MotorDir::kForward :
                  (_dutyL < -0.01f) ? MotorDir::kReverse : MotorDir::kCoast;
    _stateR.v_ms = v_right_actual; _stateR.pwm = _dutyR;
    _stateR.dir = (_dutyR > 0.01f) ? MotorDir::kForward :
                  (_dutyR < -0.01f) ? MotorDir::kReverse : MotorDir::kCoast;

    if (state_out) *state_out = _stateL;  // caller can also read right()
}

// ---------------------------------------------------------------------------
//  Internal helpers
// ---------------------------------------------------------------------------
void MotorDriver::writeMotor(uint8_t pwm_chan, uint8_t in1, uint8_t in2,
                             float duty_signed) {
    // Clamp.
    if (duty_signed >  1.0f) duty_signed =  1.0f;
    if (duty_signed < -1.0f) duty_signed = -1.0f;

    bool fwd = duty_signed >= 0.0f;
    uint32_t duty = (uint32_t)(fabsf(duty_signed) * MOTOR_PWM_MAX);

    digitalWrite(in1, fwd ? HIGH : LOW);
    digitalWrite(in2, fwd ? LOW  : HIGH);
    ledc_set_duty(LEDC_HIGH_SPEED_MODE, (ledc_channel_t)pwm_chan, duty);
    ledc_update_duty(LEDC_HIGH_SPEED_MODE, (ledc_channel_t)pwm_chan);
}

float MotorDriver::ramp(float current, float target) const {
    // Limit duty change to (1 / MOTOR_RAMP_MS) per millisecond ≈ per step (20 ms).
    float maxStep = 1.0f / (MOTOR_RAMP_MS / 20.0f);  // per cycle @ 50 Hz
    if (target - current >  maxStep) target = current + maxStep;
    if (target - current < -maxStep) target = current - maxStep;
    return target;
}

float MotorDriver::pidStep(PidGains& g, float& integ, float& prevErr,
                           float setpoint, float measured) {
    float err = setpoint - measured;
    integ += err * (1.0f / TASK_MOTOR_HZ);
    if (integ > g.i_max) integ = g.i_max;
    if (integ < g.i_min) integ = g.i_min;
    float deriv = (err - prevErr) * TASK_MOTOR_HZ;
    prevErr = err;
    float out = g.kp * err + g.ki * integ + g.kd * deriv;
    // Saturation.
    if (out >  1.0f) { out =  1.0f; if (err > 0) integ = g.i_max; }
    if (out < -1.0f) { out = -1.0f; if (err < 0) integ = g.i_min; }
    return out;
}

void MotorDriver::setGains(const PidGains& left, const PidGains& right) {
    _gainsL = left;
    _gainsR = right;
    _integL = 0; _integR = 0;
    _prevErrL = 0; _prevErrR = 0;
    AVC_LOGI("motor", "PID gains updated kpL=%.2f kiL=%.2f kpR=%.2f kiR=%.2f",
             left.kp, left.ki, right.kp, right.ki);
}
