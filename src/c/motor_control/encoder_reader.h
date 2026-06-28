/**
 * @file encoder_reader.h
 * @brief Quadrature encoder reader header for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#ifndef ENCODER_READER_H
#define ENCODER_READER_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>
#include <stdbool.h>

/* ========================================================================
 * Encoder Configuration
 * ======================================================================== */
#define ENCODER_MAX_CHANNELS   4
#define ENCODER_CPR_DEFAULT    1024    /**< Counts per revolution */
#define ENCODER_MAX_CPR        8192

/* ========================================================================
 * Encoder Types
 * ======================================================================== */
typedef enum {
    ENCODER_TYPE_QUADRATURE  = 0,  /**< A/B phase quadrature encoder */
    ENCODER_TYPE_ABSOLUTE    = 1,  /**< Absolute position encoder */
    ENCODER_TYPE_SINGLE      = 2   /**< Single pulse (tachometer) */
} EncoderType_t;

typedef struct {
    uint8_t       channel;         /**< Encoder channel (0-3) */
    EncoderType_t type;            /**< Encoder type */
    uint32_t      cpr;             /**< Counts per revolution */
    float         wheel_diameter_m;/**< Wheel diameter for speed calc */
    float         gear_ratio;      /**< Motor to wheel gear ratio */
    bool          index_pulse;     /**< Has index (Z) pulse */
    bool          reverse_direction; /**< Reverse counting direction */
} EncoderConfig_t;

typedef struct {
    int32_t  count;               /**< Current pulse count */
    int32_t  velocity_count;      /**< Velocity measurement count */
    float    angle_deg;           /**< Calculated angle (0-360) */
    float    angle_rad;           /**< Calculated angle (radians) */
    float    rpm;                 /**< Calculated speed (RPM) */
    float    speed_mps;           /**< Linear speed (m/s) */
    int32_t  total_count;         /**< Total cumulative count */
    bool     direction;           /**< true=CW, false=CCW */
    bool     index_detected;      /**< Index pulse detected */
    bool     error;               /**< Encoder error flag */
} EncoderData_t;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize encoder reader
 * @param config Encoder configuration
 * @return 0 on success
 */
int EncoderReader_Init(const EncoderConfig_t *config);

/**
 * @brief Get encoder data
 * @param channel Encoder channel
 * @return Pointer to encoder data, NULL if invalid
 */
const EncoderData_t* EncoderReader_GetData(uint8_t channel);

/**
 * @brief Get current count
 * @param channel Encoder channel
 * @return Current count value
 */
int32_t EncoderReader_GetCount(uint8_t channel);

/**
 * @brief Get calculated RPM
 * @param channel Encoder channel
 * @return Speed in RPM
 */
float EncoderReader_GetRPM(uint8_t channel);

/**
 * @brief Get linear speed
 * @param channel Encoder channel
 * @return Speed in meters per second
 */
float EncoderReader_GetSpeedMps(uint8_t channel);

/**
 * @brief Reset encoder count
 * @param channel Encoder channel
 */
void EncoderReader_ResetCount(uint8_t channel);

/**
 * @brief Set count value (e.g., from timer capture interrupt)
 * @param channel Encoder channel
 * @param count Count value
 * @param direction Rotation direction
 */
void EncoderReader_SetCount(uint8_t channel, int32_t count, bool direction);

/**
 * @brief Index pulse interrupt handler
 * @param channel Encoder channel
 */
void EncoderReader_IndexHandler(uint8_t channel);

/**
 * @brief Update speed calculation (call at fixed interval)
 * @param channel Encoder channel
 */
void EncoderReader_UpdateSpeed(uint8_t channel);

#ifdef __cplusplus
}
#endif

#endif /* ENCODER_READER_H */
