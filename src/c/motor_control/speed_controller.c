/**
 * @file speed_controller.c
 * @brief PID speed controller implementation with anti-windup and feed-forward
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "speed_controller.h"
#include "pwm_control.h"
#include "motor_driver.h"
#include <string.h>
#include <math.h>

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static PidState_t         s_pid_state[SPEED_CTRL_MAX_MOTORS];
static SpeedCtrlConfig_t  s_config[SPEED_CTRL_MAX_MOTORS];
static PidDebug_t         s_debug[SPEED_CTRL_MAX_MOTORS];
static bool               s_initialized[SPEED_CTRL_MAX_MOTORS];

/* ========================================================================
 * Public Functions
 * ======================================================================== */

int SpeedController_Init(const SpeedCtrlConfig_t *config)
{
    if (config == NULL || config->motor_channel >= SPEED_CTRL_MAX_MOTORS) {
        return -1;
    }

    uint8_t ch = config->motor_channel;
    memcpy(&s_config[ch], config, sizeof(SpeedCtrlConfig_t));
    memset(&s_pid_state[ch], 0, sizeof(PidState_t));
    memset(&s_debug[ch], 0, sizeof(PidDebug_t));

    s_pid_state[ch].dt = config->pid_config.dt;
    s_pid_state[ch].enabled = false;

    s_initialized[ch] = true;
    return 0;
}

void SpeedController_SetTarget(uint8_t motor_channel, float target_rpm)
{
    if (motor_channel >= SPEED_CTRL_MAX_MOTORS || !s_initialized[motor_channel]) {
        return;
    }

    s_pid_state[motor_channel].setpoint_rpm = target_rpm;
}

float SpeedController_Update(uint8_t motor_channel, float measured_rpm)
{
    if (motor_channel >= SPEED_CTRL_MAX_MOTORS || !s_initialized[motor_channel]) {
        return 0.0f;
    }

    PidState_t   *pid = &s_pid_state[motor_channel];
    PidConfig_t  *cfg = &s_config[motor_channel].pid_config;
    PidDebug_t   *dbg = &s_debug[motor_channel];

    if (!pid->enabled) {
        return 0.0f;
    }

    pid->measured_rpm = measured_rpm;

    /* Calculate error */
    pid->error = pid->setpoint_rpm - measured_rpm;

    /* Proportional term */
    dbg->p_term = cfg->kp * pid->error;

    /* Integral term with anti-windup (clamping) */
    pid->integral += pid->error * pid->dt;

    /* Clamp integral to prevent windup */
    if (pid->integral > cfg->integral_limit) {
        pid->integral = cfg->integral_limit;
    } else if (pid->integral < -cfg->integral_limit) {
        pid->integral = -cfg->integral_limit;
    }

    dbg->i_term = cfg->ki * pid->integral;

    /* Derivative term with low-pass filter */
    float raw_derivative = (pid->error - pid->prev_error) / pid->dt;
    if (cfg->d_filter_coeff > 0.0f && cfg->d_filter_coeff < 1.0f) {
        pid->derivative = (cfg->d_filter_coeff * raw_derivative) +
                          ((1.0f - cfg->d_filter_coeff) * pid->prev_derivative);
    } else {
        pid->derivative = raw_derivative;
    }
    dbg->d_term = cfg->kd * pid->derivative;

    /* Feed-forward term */
    dbg->ff_term = cfg->feed_forward_gain * pid->setpoint_rpm;

    /* Sum all terms */
    pid->output = dbg->p_term + dbg->i_term + dbg->d_term + dbg->ff_term;

    /* Apply rate limiting (acceleration/deceleration) */
    float max_delta = s_config[motor_channel].max_accel_rpm_s * pid->dt;
    if (pid->error < 0) {
        max_delta = s_config[motor_channel].max_decel_rpm_s * pid->dt;
    }

    /* Clamp output */
    if (pid->output > cfg->output_max) {
        pid->output = cfg->output_max;
    } else if (pid->output < cfg->output_min) {
        pid->output = cfg->output_min;
    }

    /* Store previous values */
    pid->prev_error = pid->error;
    pid->prev_derivative = pid->derivative;

    /* Apply output to PWM */
    PWM_SetDutyCycle(motor_channel, pid->output);

    return pid->output;
}

void SpeedController_Enable(uint8_t motor_channel, bool enable)
{
    if (motor_channel >= SPEED_CTRL_MAX_MOTORS || !s_initialized[motor_channel]) {
        return;
    }

    s_pid_state[motor_channel].enabled = enable;
    if (enable) {
        SpeedController_Reset(motor_channel);
        PWM_Start(motor_channel);
        MotorDriver_SetEnable(motor_channel, true);
    } else {
        PWM_Stop(motor_channel);
        MotorDriver_SetEnable(motor_channel, false);
    }
}

void SpeedController_Reset(uint8_t motor_channel)
{
    if (motor_channel >= SPEED_CTRL_MAX_MOTORS) return;

    PidState_t *pid = &s_pid_state[motor_channel];
    pid->error = 0.0f;
    pid->integral = 0.0f;
    pid->derivative = 0.0f;
    pid->output = 0.0f;
    pid->prev_error = 0.0f;
    pid->prev_derivative = 0.0f;
}

const PidState_t* SpeedController_GetState(uint8_t motor_channel)
{
    if (motor_channel >= SPEED_CTRL_MAX_MOTORS) return NULL;
    return &s_pid_state[motor_channel];
}

void SpeedController_GetDebugTerms(uint8_t motor_channel, PidDebug_t *debug)
{
    if (motor_channel >= SPEED_CTRL_MAX_MOTORS || debug == NULL) return;
    *debug = s_debug[motor_channel];
}

void SpeedController_SetGains(uint8_t motor_channel, float kp, float ki, float kd)
{
    if (motor_channel >= SPEED_CTRL_MAX_MOTORS || !s_initialized[motor_channel]) return;

    s_config[motor_channel].pid_config.kp = kp;
    s_config[motor_channel].pid_config.ki = ki;
    s_config[motor_channel].pid_config.kd = kd;
}