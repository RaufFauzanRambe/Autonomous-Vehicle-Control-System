/**
 * @file motor_driver.h
 * @brief Motor driver interface for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#ifndef MOTOR_DRIVER_H
#define MOTOR_DRIVER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* ========================================================================
 * Motor Configuration
 * ======================================================================== */
#define MOTOR_MAX_CHANNELS      4

typedef enum {
    MOTOR_DIR_FORWARD  = 0,
    MOTOR_DIR_REVERSE  = 1,
    MOTOR_DIR_BRAKE    = 2,  /**< Short brake (both low-side) */
    MOTOR_DIR_COAST    = 3   /**< Free running (both high-side off) */
} MotorDirection_t;

typedef enum {
    MOTOR_TYPE_DC_BRUSHED     = 0,
    MOTOR_TYPE_DC_BRUSHLESS   = 1,
    MOTOR_TYPE_STEPPER        = 2
} MotorType_t;

typedef struct {
    uint8_t  channel;            /**< Motor channel (0-3) */
    MotorType_t type;            /**< Motor type */
    uint32_t max_rpm;            /**< Maximum RPM */
    uint32_t max_current_ma;     /**< Maximum current limit (mA) */
    float    max_duty_cycle;     /**< Maximum duty cycle (0.0 - 1.0) */
    bool     inverted;           /**< Invert direction polarity */
    bool     enable;             /**< Motor enabled */
} MotorConfig_t;

typedef struct {
    MotorDirection_t direction;  /**< Current direction */
    float           duty_cycle;  /**< Current duty cycle (0.0 - 1.0) */
    int32_t         current_ma;  /**< Measured current (mA) */
    int16_t         temperature; /**< Driver temperature (°C x10) */
    uint32_t        rpm;         /**< Current speed (RPM) */
    bool            enabled;     /**< Driver enabled */
    bool            fault;       /**< Driver fault flag */
} MotorStatus_t;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize motor driver
 * @param config Motor configuration
 * @return 0 on success, negative on error
 */
int MotorDriver_Init(const MotorConfig_t *config);

/**
 * @brief Set motor direction
 * @param channel Motor channel
 * @param dir Direction
 * @return 0 on success
 */
int MotorDriver_SetDirection(uint8_t channel, MotorDirection_t dir);

/**
 * @brief Set motor enable state
 * @param channel Motor channel
 * @param enable Enable (true) or disable (false)
 * @return 0 on success
 */
int MotorDriver_SetEnable(uint8_t channel, bool enable);

/**
 * @brief Emergency stop all motors
 */
void MotorDriver_EmergencyStop(void);

/**
 * @brief Get motor status
 * @param channel Motor channel
 * @return Pointer to motor status
 */
const MotorStatus_t* MotorDriver_GetStatus(uint8_t channel);

/**
 * @brief Check motor current against limit
 * @param channel Motor channel
 * @return true if overcurrent detected
 */
bool MotorDriver_IsOverCurrent(uint8_t channel);

#ifdef __cplusplus
}
#endif

#endif /* MOTOR_DRIVER_H */