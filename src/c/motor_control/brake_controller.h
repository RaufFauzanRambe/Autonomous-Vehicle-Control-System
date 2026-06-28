/**
 * @file brake_controller.h
 * @brief Brake controller header for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#ifndef BRAKE_CONTROLLER_H
#define BRAKE_CONTROLLER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* ========================================================================
 * Brake Types and Configuration
 * ======================================================================== */
#define BRAKE_MAX_PRESSURE_BAR  150.0f
#define BRAKE_REGEN_MAX_KW      30.0f
#define BRAKE_ABS_THRESHOLD     0.15f  /* 15% wheel slip */
#define BRAKE_BIAS_FRONT        0.65f  /* 65% front bias */
#define BRAKE_BIAS_REAR         0.35f

#define WHEEL_COUNT             4

typedef enum {
    BRAKE_MODE_NONE        = 0,  /**< No braking */
    BRAKE_MODE_REGEN       = 1,  /**< Regenerative only */
    BRAKE_MODE_FRICTION    = 2,  /**< Friction only */
    BRAKE_MODE_BLENDED     = 3,  /**< Blended regen + friction */
    BRAKE_MODE_ABS         = 4,  /**< ABS modulating */
    BRAKE_MODE_EMERGENCY   = 5   /**< Emergency full brake */
} BrakeMode_t;

typedef struct {
    float max_regen_kw;          /**< Maximum regenerative braking power */
    float regen_start_pressure;  /**< Pressure where regen transitions to friction */
    float abs_slip_threshold;    /**< Wheel slip threshold for ABS */
    float front_bias;            /**< Front/rear brake bias */
    float rear_bias;             /**< Rear brake bias */
    uint8_t pwm_channel;         /**< Brake actuator PWM channel */
} BrakeConfig_t;

typedef struct {
    BrakeMode_t mode;            /**< Current brake mode */
    float requested_pressure;    /**< Requested brake pressure (bar) */
    float regen_torque;          /**< Current regen torque (Nm) */
    float friction_pressure;     /**< Current friction brake pressure (bar) */
    float wheel_speed[WHEEL_COUNT]; /**< Individual wheel speeds (m/s) */
    float wheel_slip[WHEEL_COUNT];  /**< Calculated wheel slip ratios */
    bool  abs_active;            /**< ABS is modulating */
    bool  regen_available;       /**< Regen braking available */
    bool  emergency_brake;       /**< Emergency brake active */
} BrakeStatus_t;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize brake controller
 * @param config Brake configuration
 * @return 0 on success
 */
int BrakeController_Init(const BrakeConfig_t *config);

/**
 * @brief Request braking
 * @param pressure_bar Requested brake pressure (0 - BRAKE_MAX_PRESSURE_BAR)
 * @param vehicle_speed_ms Current vehicle speed (m/s)
 */
void BrakeController_RequestBrake(float pressure_bar, float vehicle_speed_ms);

/**
 * @brief Emergency brake (maximum braking force)
 */
void BrakeController_EmergencyBrake(void);

/**
 * @brief Update brake controller (call periodically)
 * @param wheel_speeds Array of 4 wheel speeds (m/s)
 * @param vehicle_speed_ms Reference vehicle speed (m/s)
 */
void BrakeController_Update(const float wheel_speeds[WHEEL_COUNT], float vehicle_speed_ms);

/**
 * @brief Get brake status
 * @return Pointer to brake status
 */
const BrakeStatus_t* BrakeController_GetStatus(void);

/**
 * @brief Set regen availability
 * @param available true if regen braking is available
 */
void BrakeController_SetRegenAvailable(bool available);

#ifdef __cplusplus
}
#endif

#endif /* BRAKE_CONTROLLER_H */
