/**
 * @file steering_controller.c
 * @brief Steering controller with Ackermann geometry
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "steering_controller.h"
#include "pwm_control.h"
#include <math.h>
#include <string.h>

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static SteeringConfig_t  s_config;
static SteeringStatus_t  s_status;
static float             s_prev_angle = 0.0f;

/* ========================================================================
 * Public Functions
 * ======================================================================== */

int SteeringController_Init(const SteeringConfig_t *config)
{
    if (config == NULL) return -1;

    memcpy(&s_config, config, sizeof(SteeringConfig_t));
    memset(&s_status, 0, sizeof(SteeringStatus_t));

    s_status.calibrated = false;
    s_status.enabled = false;
    s_status.current_angle_deg = STEER_NEUTRAL_DEG;

    return 0;
}

void SteeringController_SetAngle(float angle_deg)
{
    /* Clamp to limits */
    if (angle_deg < s_config.min_angle_deg) {
        angle_deg = s_config.min_angle_deg;
    } else if (angle_deg > s_config.max_angle_deg) {
        angle_deg = s_config.max_angle_deg;
    }

    s_status.target_angle_deg = angle_deg;

    /* Calculate Ackermann geometry for front wheels */
    SteeringController_CalcAckermann(angle_deg,
                                      &s_status.inner_wheel_deg,
                                      &s_status.outer_wheel_deg);
}

void SteeringController_CalcAckermann(float steer_angle, float *inner_deg, float *outer_deg)
{
    if (steer_angle == 0.0f || inner_deg == NULL || outer_deg == NULL) {
        if (inner_deg) *inner_deg = 0.0f;
        if (outer_deg) *outer_deg = 0.0f;
        return;
    }

    /* Convert to radians */
    float steer_rad = steer_angle * (float)M_PI / 180.0f;
    float wheelbase = s_config.geometry.wheelbase_m;
    float track     = s_config.geometry.track_width_m;

    /* Ackermann formula:
     * tan(inner) = L / (R - track/2)
     * tan(outer) = L / (R + track/2)
     * where R = L / tan(steer_angle) is the turning radius
     */
    float R = wheelbase / tanf(steer_rad);

    float tan_inner = wheelbase / (R - track / 2.0f);
    float tan_outer = wheelbase / (R + track / 2.0f);

    *inner_deg = atanf(tan_inner) * 180.0f / (float)M_PI;
    *outer_deg = atanf(tan_outer) * 180.0f / (float)M_PI;

    /* Determine which side is inner/outer based on turn direction */
    if (steer_angle < 0.0f) {
        /* Turning left: left is inner, right is outer */
        s_status.left_angle_deg  = *inner_deg;
        s_status.right_angle_deg = *outer_deg;
    } else {
        /* Turning right: right is inner, left is outer */
        s_status.left_angle_deg  = *outer_deg;
        s_status.right_angle_deg = *inner_deg;
    }
}

void SteeringController_Update(float measured_angle)
{
    s_status.current_angle_deg = measured_angle;

    if (!s_status.enabled) return;

    /* Rate limiting */
    float max_delta = s_config.max_rate_deg_s * 0.001f; /* assuming 1ms loop */
    float delta = s_status.target_angle_deg - measured_angle;

    if (delta > max_delta) {
        delta = max_delta;
    } else if (delta < -max_delta) {
        delta = -max_delta;
    }

    float command = measured_angle + delta;

    /* Convert angle to PWM duty cycle (centered at 50%) */
    float duty = 0.5f + (command / s_config.max_angle_deg) * 0.4f;
    if (duty < 0.1f) duty = 0.1f;
    if (duty > 0.9f) duty = 0.9f;

    PWM_SetDutyCycle(s_config.pwm_channel, duty);

    s_prev_angle = command;
}

int SteeringController_Calibrate(void)
{
    /* Drive steering to full left, measure, then full right, measure */
    /* Find center point */
    /* Store calibration values */

    s_status.calibrated = true;
    return 0;
}

void SteeringController_Center(void)
{
    SteeringController_SetAngle(STEER_NEUTRAL_DEG);
}

const SteeringStatus_t* SteeringController_GetStatus(void)
{
    return &s_status;
}

void SteeringController_Enable(bool enable)
{
    s_status.enabled = enable;
    if (!enable) {
        PWM_Stop(s_config.pwm_channel);
    } else {
        PWM_Start(s_config.pwm_channel);
    }
}
