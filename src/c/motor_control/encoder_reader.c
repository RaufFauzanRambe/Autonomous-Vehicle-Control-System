/**
 * @file encoder_reader.c
 * @brief Quadrature encoder reader with speed calculation
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "encoder_reader.h"
#include <string.h>
#include <math.h>

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static EncoderConfig_t  s_config[ENCODER_MAX_CHANNELS];
static EncoderData_t    s_data[ENCODER_MAX_CHANNELS];
static int32_t          s_prev_count[ENCODER_MAX_CHANNELS];
static bool             s_initialized[ENCODER_MAX_CHANNELS];

/* ========================================================================
 * Public Functions
 * ======================================================================== */

int EncoderReader_Init(const EncoderConfig_t *config)
{
    if (config == NULL || config->channel >= ENCODER_MAX_CHANNELS) {
        return -1;
    }

    uint8_t ch = config->channel;
    memcpy(&s_config[ch], config, sizeof(EncoderConfig_t));
    memset(&s_data[ch], 0, sizeof(EncoderData_t));
    s_prev_count[ch] = 0;

    /* Configure timer for encoder mode */
    /* TIM encoder mode: SMS=001, CCMR1 CC1S=01, CC2S=01 */

    s_initialized[ch] = true;
    return 0;
}

const EncoderData_t* EncoderReader_GetData(uint8_t channel)
{
    if (channel >= ENCODER_MAX_CHANNELS || !s_initialized[channel]) {
        return NULL;
    }
    return &s_data[channel];
}

int32_t EncoderReader_GetCount(uint8_t channel)
{
    if (channel >= ENCODER_MAX_CHANNELS || !s_initialized[channel]) {
        return 0;
    }
    return s_data[channel].count;
}

float EncoderReader_GetRPM(uint8_t channel)
{
    if (channel >= ENCODER_MAX_CHANNELS || !s_initialized[channel]) {
        return 0.0f;
    }
    return s_data[channel].rpm;
}

float EncoderReader_GetSpeedMps(uint8_t channel)
{
    if (channel >= ENCODER_MAX_CHANNELS || !s_initialized[channel]) {
        return 0.0f;
    }
    return s_data[channel].speed_mps;
}

void EncoderReader_ResetCount(uint8_t channel)
{
    if (channel >= ENCODER_MAX_CHANNELS) return;

    s_data[channel].count = 0;
    s_data[channel].total_count = 0;
    s_data[channel].angle_deg = 0.0f;
    s_data[channel].angle_rad = 0.0f;
    s_prev_count[channel] = 0;
}

void EncoderReader_SetCount(uint8_t channel, int32_t count, bool direction)
{
    if (channel >= ENCODER_MAX_CHANNELS || !s_initialized[channel]) return;

    int32_t delta = count - s_data[channel].count;

    if (s_config[channel].reverse_direction) {
        delta = -delta;
    }

    s_data[channel].count = count;
    s_data[channel].total_count += delta;
    s_data[channel].direction = direction;
    s_data[channel].velocity_count = delta;

    /* Calculate angle */
    uint32_t cpr = s_config[channel].cpr;
    if (cpr > 0) {
        float revolutions = (float)(s_data[channel].count % (int32_t)cpr) / (float)cpr;
        if (revolutions < 0.0f) revolutions += 1.0f;
        s_data[channel].angle_deg = revolutions * 360.0f;
        s_data[channel].angle_rad = revolutions * 2.0f * (float)M_PI;
    }
}

void EncoderReader_IndexHandler(uint8_t channel)
{
    if (channel >= ENCODER_MAX_CHANNELS || !s_initialized[channel]) return;

    s_data[channel].index_detected = true;

    /* Reset count on index pulse for absolute reference */
    if (s_config[channel].index_pulse) {
        s_data[channel].count = 0;
    }
}

void EncoderReader_UpdateSpeed(uint8_t channel)
{
    if (channel >= ENCODER_MAX_CHANNELS || !s_initialized[channel]) return;

    EncoderConfig_t *cfg = &s_config[channel];
    EncoderData_t   *data = &s_data[channel];

    /* Speed calculation assumes this is called at a fixed interval (e.g., 1ms) */
    float sample_period_s = 0.001f;  /* 1ms default */

    int32_t delta = data->velocity_count;
    s_prev_count[channel] = data->count;

    /* Calculate RPM: (counts/interval) * (60 / CPR) */
    float counts_per_sec = (float)delta / sample_period_s;
    float rps = counts_per_sec / (float)cfg->cpr;  /* Revolutions per second */

    /* Apply gear ratio to get output shaft speed */
    data->rpm = rps * 60.0f / cfg->gear_ratio;

    /* Calculate linear speed: v = RPM * pi * D / (60 * gear_ratio) */
    if (cfg->wheel_diameter_m > 0.0f) {
        data->speed_mps = data->rpm * (float)M_PI * cfg->wheel_diameter_m / 60.0f;
    }

    /* Clear velocity count for next interval */
    data->velocity_count = 0;
}
