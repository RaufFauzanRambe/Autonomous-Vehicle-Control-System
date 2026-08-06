/**
 * @file motor_driver.h
 * @brief MotorDriver — TB6612FNG dual H-bridge with LEDC PWM + PID.
 * @author AVC team
 * @date 2024
 * @license MIT
 */
#ifndef AVC_ESP32_MOTOR_DRIVER_H
#define AVC_ESP32_MOTOR_DRIVER_H

#include <Arduino.h>
#include "config.h"

/// Direction of rotation for a single motor.
enum class MotorDir : int8_t {
    kCoast  = 0,   ///< Both bridge inputs low — motor coasts.
    kBrake  = 1,   ///< Both inputs high — dynamic braking.
    kForward= 2,
    kReverse= -2,
};

/// Per-wheel setpoint the high-level controller pushes down.
struct WheelSetpoint {
    float v_ms;     ///< Target linear wheel speed (m/s). Sign = direction.
    bool  brake;    ///< If true, dynamic brake regardless of v_ms.
};

/// Per-wheel state reported back up.
struct WheelState {
    float v_ms;        ///< Actual wheel linear speed (m/s), from encoder.
    float pwm;         ///< Last applied PWM duty (signed, -1..1).
    MotorDir dir;      ///< Last applied direction.
};

/// Per-wheel discrete PID gains.
struct PidGains {
    float kp;
    float ki;
    float kd;
    float i_min;
    float i_max;
};

/**
 * @class MotorDriver
 * @brief Drives two DC motors via a TB6612FNG H-bridge:
 *        - LEDC PWM at 20 kHz (inaudible),
 *        - Per-wheel PI controller with derivative filter,
 *        - Acceleration / jerk limiting ("ramp"),
 *        - Emergency-stop input,
 *        - Standby pin handling.
 *
 * Designed to be called from a 50 Hz FreeRTOS task.
 */
class MotorDriver {
public:
    /**
     * @brief Constructor — sets default gains.
     */
    MotorDriver();

    /**
     * @brief Attach pins, configure LEDC channels, raise STBY.
     * @return true on success.
     */
    bool begin();

    /**
     * @brief Set per-wheel target speeds (m/s).
     * @param[in] left  Linear speed (m/s), sign = direction.
     * @param[in] right Linear speed (m/s), sign = direction.
     * @note Respects ramp limits — actual command tracks the setpoint.
     */
    void setSpeed(float left, float right);

    /**
     * @brief Differential (unicycle) command: linear + angular velocity.
     * @param[in] v Linear velocity (m/s).
     * @param[in] w Angular velocity (rad/s).
     */
    void setDiffDrive(float v, float w);

    /**
     * @brief Engage dynamic braking (short-circuit the motor).
     */
    void brake();

    /**
     * @brief Coast (high-impedance) — motor free-wheels.
     */
    void coast();

    /**
     * @brief Emergency stop — same as brake() + STBY low.
     */
    void estop();

    /**
     * @brief Clear an E-stop condition and re-enable the bridge.
     */
    void clearEstop();

    /**
     * @brief Periodic control step — call from the 50 Hz task.
     * @param[in] v_left_actual  Measured left wheel speed (m/s).
     * @param[in] v_right_actual Measured right wheel speed (m/s).
     * @param[out] state_out     Optional pointer to receive the latest state.
     */
    void step(float v_left_actual, float v_right_actual, WheelState* state_out = nullptr);

    /**
     * @brief Tune PID gains at runtime (e.g. via MQTT cmd/config).
     */
    void setGains(const PidGains& left, const PidGains& right);

    /**
     * @brief Current per-wheel state snapshot.
     */
    inline WheelState left()  const { return _stateL; }
    inline WheelState right() const { return _stateR; }

    /**
     * @brief Is the bridge in standby (E-stop)?
     */
    inline bool estopped() const { return _estopped; }

private:
    /// Apply a single motor command (raw PWM + direction).
    void writeMotor(uint8_t pwm_chan, uint8_t in1, uint8_t in2,
                    float duty_signed);

    /// Saturate and ramp-limit a desired duty.
    float ramp(float current, float target) const;

    /// Run a single PID iteration.
    float pidStep(PidGains& g, float& integ, float& prevErr, float setpoint, float measured);

    PidGains  _gainsL;
    PidGains  _gainsR;
    float     _integL = 0, _integR = 0;
    float     _prevErrL = 0, _prevErrR = 0;
    uint32_t  _lastStepMs = 0;

    float     _cmdL = 0;   ///< ramp-limited target speed (m/s)
    float     _cmdR = 0;
    float     _dutyL = 0;  ///< last applied duty (-1..1)
    float     _dutyR = 0;

    WheelState _stateL;
    WheelState _stateR;

    bool      _estopped = false;
    bool      _initialised = false;
};

#endif // AVC_ESP32_MOTOR_DRIVER_H
