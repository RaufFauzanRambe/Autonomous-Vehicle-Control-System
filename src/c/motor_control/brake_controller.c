/**
 * @file brake_controller.c
 * @brief Brake controller with regen/friction blending and ABS
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "brake_controller.h"
#include "pwm_control.h"
#include <string.h>
#include <math.h>

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static BrakeConfig_t  s_config;
static BrakeStatus_t  s_status;
static float          s_abs_pressure_mod = 0.0f;

/* ========================================================================
 * Private Functions
 * ======================================================================== */

static void ApplyFrictionBrake(float pressure)
{
    /* Convert pressure to PWM duty cycle (0-100% maps to 0-max pressure) */
    float duty = (pressure / BRAKE_MAX_PRESSURE_BAR) * 0.9f + 0.05f;
    if (duty > 0.95f) duty = 0.95f;
    if (duty < 0.05f) duty = 0.05f;

    PWM_SetDutyCycle(s_config.pwm_channel, duty);
    s_status.friction_pressure = pressure;
}

static float CalculateWheelSlip(float wheel_speed, float vehicle_speed)
{
    if (fabsf(vehicle_speed) < 0.1f) {
        return 0.0f;
    }
    return (vehicle_speed - wheel_speed) / fabsf(vehicle_speed);
}

/* ========================================================================
 * Public Functions
 * ======================================================================== */

int BrakeController_Init(const BrakeConfig_t *config)
{
    if (config == NULL) return -1;

    memcpy(&s_config, config, sizeof(BraceConfig_t));
    memset(&s_status, 0, sizeof(BraceStatus_t));

    s_status.mode = BRAKE_MODE_NONE;
    s_status.regen_available = true;

    return 0;
}

void BrakeController_RequestBrake(float pressure_bar, float vehicle_speed_ms)
{
    if (pressure_bar < 0.0f) pressure_bar = 0.0f;
    if (pressure_bar > BRAKE_MAX_PRESSURE_BAR) pressure_bar = BRAKE_MAX_PRESSURE_BAR;

    s_status.requested_pressure = pressure_bar;

    /* Determine brake mode based on pressure and regen availability */
    if (pressure_bar < 0.01f) {
        s_status.mode = BRAKE_MODE_NONE;
        s_status.regen_torque = 0.0f;
        ApplyFrictionBrake(0.0f);
        return;
    }

    if (s_status.emergency_brake) {
        s_status.mode = BRAKE_MODE_EMERGENCY;
        ApplyFrictionBrake(BRAKE_MAX_PRESSURE_BAR);
        s_status.regen_torque = s_config.max_regen_kw;
        return;
    }

    /* Blended braking strategy */
    if (s_status.regen_available && vehicle_speed_ms > 0.5f) {
        s_status.mode = BRAKE_MODE_BLENDED;

        /* Regen handles light to medium braking */
        float regen_fraction = pressure_bar / BRAKE_MAX_PRESSURE_BAR;
        float regen_torque = regen_fraction * s_config.max_regen_kw * 1000.0f;

        if (regen_torque > s_config.max_regen_kw * 1000.0f) {
            regen_torque = s_config.max_regen_kw * 1000.0f;
        }

        s_status.regen_torque = regen_torque;

        /* Friction brake fills the gap above regen capability */
        float friction_pressure = pressure_bar - (regen_fraction * s_config.regen_start_pressure);
        if (friction_pressure < 0.0f) friction_pressure = 0.0f;

        ApplyFrictionBrake(friction_pressure);
    } else {
        /* Pure friction braking */
        s_status.mode = BRAKE_MODE_FRICTION;
        s_status.regen_torque = 0.0f;
        ApplyFrictionBrake(pressure_bar);
    }
}

void BrakeController_EmergencyBrake(void)
{
    s_status.emergency_brake = true;
    s_status.mode = BRAKE_MODE_EMERGENCY;
    s_status.requested_pressure = BRAKE_MAX_PRESSURE_BAR;
    ApplyFrictionBrake(BRAKE_MAX_PRESSURE_BAR);
    s_status.regen_torque = s_config.max_regen_kw;
}

void BrakeController_Update(const float wheel_speeds[WHEEL_COUNT], float vehicle_speed_ms)
{
    /* Copy wheel speeds and calculate slip */
    s_status.abs_active = false;

    for (uint8_t i = 0; i < WHEEL_COUNT; i++) {
        s_status.wheel_speed[i] = wheel_speeds[i];
        s_status.wheel_slip[i] = CalculateWheelSlip(wheel_speeds[i], vehicle_speed_ms);

        /* Check for ABS intervention */
        if (fabsf(s_status.wheel_slip[i]) > s_config.abs_slip_threshold &&
            s_status.requested_pressure > 0.0f) {
            s_status.abs_active = true;
        }
    }

    if (s_status.abs_active) {
        s_status.mode = BRAKE_MODE_ABS;
        /* Reduce brake pressure to prevent wheel lock
         * Simple ABS: modulate pressure based on max slip */
        float max_slip = 0.0f;
        for (uint8_t i = 0; i < WHEEL_COUNT; i++) {
            float abs_slip = fabsf(s_status.wheel_slip[i]);
            if (abs_slip > max_slip) {
                max_slip = abs_slip;
            }
        }

        /* Reduce pressure proportionally to excess slip */
        float slip_ratio = s_config.abs_slip_threshold / max_slip;
        float modulated_pressure = s_status.requested_pressure * slip_ratio;
        ApplyFrictionBrake(modulated_pressure);
    }
}

const BrakeStatus_t* BrakeController_GetStatus(void)
{
    return &s_status;
}

void BrakeController_SetRegenAvailable(bool available)
{
    s_status.regen_available = available;
}
