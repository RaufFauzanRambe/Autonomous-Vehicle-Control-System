/**
 * @file speed_controller.h
 * @brief PID speed controller header for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#ifndef SPEED_CONTROLLER_H
#define SPEED_CONTROLLER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* ========================================================================
 * PID Controller Configuration
 * ======================================================================== */
typedef struct {
    float kp;               /**< Proportional gain */
    float ki;               /**< Integral gain */
    float kd;               /**< Derivative gain */
    float integral_limit;   /**< Anti-windup: max integral term */
    float output_min;       /**< Minimum output limit */
    float output_max;       /**< Maximum output limit */
    float d_filter_coeff;   /**< Derivative filter coefficient (0-1) */
    float feed_forward_gain; /**< Feed-forward gain */
    float dt;               /**< Sample time in seconds */
} PidConfig_t;

typedef struct {
    float setpoint_rpm;     /**< Target speed (RPM) */
    float measured_rpm;     /**< Measured speed (RPM) */
    float error;            /**< Current error */
    float integral;         /**< Accumulated integral term */
    float derivative;       /**< Filtered derivative term */
    float output;           /**< Controller output (duty cycle 0-1) */
    float prev_error;       /**< Previous error for derivative */
    float prev_derivative;  /**< Previous filtered derivative */
    bool  enabled;          /**< Controller enabled */
} PidState_t;

typedef struct {
    float p_term;           /**< Proportional term */
    float i_term;           /**< Integral term */
    float d_term;           /**< Derivative term */
    float ff_term;          /**< Feed-forward term */
} PidDebug_t;

/* ========================================================================
 * Speed Controller Configuration
 * ======================================================================== */
#define SPEED_CTRL_MAX_MOTORS   4

typedef struct {
    uint8_t     motor_channel;  /**< Associated motor channel */
    PidConfig_t pid_config;     /**< PID parameters */
    float       max_accel_rpm_s; /**< Maximum acceleration (RPM/s) */
    float       max_decel_rpm_s; /**< Maximum deceleration (RPM/s) */
} SpeedCtrlConfig_t;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize speed controller for a motor
 * @param config Speed controller configuration
 * @return 0 on success
 */
int SpeedController_Init(const SpeedCtrlConfig_t *config);

/**
 * @brief Set target speed
 * @param motor_channel Motor channel
 * @param target_rpm Target speed in RPM
 */
void SpeedController_SetTarget(uint8_t motor_channel, float target_rpm);

/**
 * @brief Run one PID control cycle
 * @param motor_channel Motor channel
 * @param measured_rpm Current measured RPM
 * @return Controller output (duty cycle 0-1)
 */
float SpeedController_Update(uint8_t motor_channel, float measured_rpm);

/**
 * @brief Enable/disable speed controller
 * @param motor_channel Motor channel
 * @param enable Enable state
 */
void SpeedController_Enable(uint8_t motor_channel, bool enable);

/**
 * @brief Reset controller state (clear integral, etc.)
 * @param motor_channel Motor channel
 */
void SpeedController_Reset(uint8_t motor_channel);

/**
 * @brief Get current PID state
 * @param motor_channel Motor channel
 * @return Pointer to PID state
 */
const PidState_t* SpeedController_GetState(uint8_t motor_channel);

/**
 * @brief Get PID debug terms
 * @param motor_channel Motor channel
 * @param debug Output debug data
 */
void SpeedController_GetDebugTerms(uint8_t motor_channel, PidDebug_t *debug);

/**
 * @brief Set PID gains
 * @param motor_channel Motor channel
 * @param kp Proportional gain
 * @param ki Integral gain
 * @param kd Derivative gain
 */
void SpeedController_SetGains(uint8_t motor_channel, float kp, float ki, float kd);

#ifdef __cplusplus
}
#endif

#endif /* SPEED_CONTROLLER_H */