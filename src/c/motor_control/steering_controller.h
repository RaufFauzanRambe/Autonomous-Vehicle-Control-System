/**
 * @file steering_controller.h
 * @brief Steering controller header for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#ifndef STEERING_CONTROLLER_H
#define STEERING_CONTROLLER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* ========================================================================
 * Steering Configuration
 * ======================================================================== */
#define STEER_MIN_ANGLE_DEG     -45.0f   /* Maximum left turn */
#define STEER_MAX_ANGLE_DEG      45.0f   /* Maximum right turn */
#define STEER_NEUTRAL_DEG        0.0f    /* Center/straight */
#define WHEELBASE_M              2.6f    /* Wheelbase in meters */
#define TRACK_WIDTH_M            1.6f    /* Track width in meters */
#define MAX_STEER_RATE_DEG_S     120.0f  /* Maximum steering rate */

/* Vehicle geometry */
typedef struct {
    float wheelbase_m;         /**< Distance between front and rear axles */
    float track_width_m;       /**< Distance between left and right wheels */
    float max_inner_angle_deg; /**< Maximum inner wheel steering angle */
    float max_outer_angle_deg; /**< Maximum outer wheel steering angle */
    float steering_ratio;      /**< Steering wheel to road wheel ratio */
} VehicleGeometry_t;

typedef struct {
    float target_angle_deg;    /**< Commanded steering angle */
    float current_angle_deg;   /**< Measured steering angle */
    float inner_wheel_deg;     /**< Ackermann inner wheel angle */
    float outer_wheel_deg;     /**< Ackermann outer wheel angle */
    float left_angle_deg;      /**< Actual left wheel angle */
    float right_angle_deg;     /**< Actual right wheel angle */
    float rate_limit_deg_s;    /**< Current rate limit */
    bool   calibrated;         /**< Endpoint calibration done */
    bool   enabled;            /**< Steering enabled */
} SteeringStatus_t;

typedef struct {
    VehicleGeometry_t geometry; /**< Vehicle dimensions */
    float             min_angle_deg; /**< Minimum steering angle */
    float             max_angle_deg; /**< Maximum steering angle */
    float             max_rate_deg_s; /**< Maximum rate of change */
    uint8_t           pwm_channel;    /**< PWM channel for steering motor */
} SteeringConfig_t;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize steering controller
 * @param config Steering configuration
 * @return 0 on success
 */
int SteeringController_Init(const SteeringConfig_t *config);

/**
 * @brief Set target steering angle
 * @param angle_deg Desired angle in degrees
 */
void SteeringController_SetAngle(float angle_deg);

/**
 * @brief Calculate Ackermann steering geometry
 * @param steer_angle Input steering angle (degrees)
 * @param inner_deg Output inner wheel angle
 * @param outer_deg Output outer wheel angle
 */
void SteeringController_CalcAckermann(float steer_angle, float *inner_deg, float *outer_deg);

/**
 * @brief Update steering controller (call periodically)
 * @param measured_angle Current measured angle
 */
void SteeringController_Update(float measured_angle);

/**
 * @brief Run endpoint calibration routine
 * @return 0 on success
 */
int SteeringController_Calibrate(void);

/**
 * @brief Center steering to neutral position
 */
void SteeringController_Center(void);

/**
 * @brief Get steering status
 * @return Pointer to steering status
 */
const SteeringStatus_t* SteeringController_GetStatus(void);

/**
 * @brief Enable/disable steering
 * @param enable Enable state
 */
void SteeringController_Enable(bool enable);

#ifdef __cplusplus
}
#endif

#endif /* STEERING_CONTROLLER_H */
