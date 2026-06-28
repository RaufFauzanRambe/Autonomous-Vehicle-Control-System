/**
 * @file pwm_control.h
 * @brief PWM generation and control header for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#ifndef PWM_CONTROL_H
#define PWM_CONTROL_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* ========================================================================
 * PWM Configuration
 * ======================================================================== */
#define PWM_MAX_CHANNELS       4
#define PWM_DEFAULT_FREQ_HZ    20000U  /* 20 kHz for motor control */
#define PWM_MIN_FREQ_HZ        1000U
#define PWM_MAX_FREQ_HZ        100000U
#define PWM_DEAD_TIME_NS       500U    /* 500ns dead time */

typedef enum {
    PWM_MODE_EDGE_ALIGNED   = 0,
    PWM_MODE_CENTER_ALIGNED = 1
} PwmMode_t;

typedef struct {
    uint8_t    channel;          /**< PWM channel */
    uint32_t   frequency_hz;     /**< PWM frequency */
    float      duty_cycle;       /**< Initial duty cycle (0.0 - 1.0) */
    PwmMode_t  mode;             /**< PWM alignment mode */
    bool       complementary;    /**< Enable complementary output */
    uint32_t   dead_time_ns;     /**< Dead time in nanoseconds */
    bool       active_high;      /**< Active high polarity */
} PwmConfig_t;

typedef struct {
    uint32_t   frequency_hz;     /**< Actual frequency */
    float      duty_cycle;       /**< Current duty cycle */
    bool       running;          /**< PWM output active */
} PwmStatus_t;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize PWM channel
 * @param config PWM configuration
 * @return 0 on success
 */
int PWM_Init(const PwmConfig_t *config);

/**
 * @brief Set duty cycle for a channel
 * @param channel PWM channel
 * @param duty Duty cycle (0.0 - 1.0)
 * @return 0 on success
 */
int PWM_SetDutyCycle(uint8_t channel, float duty);

/**
 * @brief Set PWM frequency
 * @param channel PWM channel
 * @param freq_hz Frequency in Hz
 * @return 0 on success
 */
int PWM_SetFrequency(uint8_t channel, uint32_t freq_hz);

/**
 * @brief Start PWM output
 * @param channel PWM channel
 */
void PWM_Start(uint8_t channel);

/**
 * @brief Stop PWM output
 * @param channel PWM channel
 */
void PWM_Stop(uint8_t channel);

/**
 * @brief Get PWM status
 * @param channel PWM channel
 * @return Pointer to status, NULL if invalid
 */
const PwmStatus_t* PWM_GetStatus(uint8_t channel);

/**
 * @brief Set dead time for complementary output
 * @param channel PWM channel
 * @param dead_time_ns Dead time in nanoseconds
 */
void PWM_SetDeadTime(uint8_t channel, uint32_t dead_time_ns);

#ifdef __cplusplus
}
#endif

#endif /* PWM_CONTROL_H */