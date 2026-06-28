/**
 * @file motor_driver.c
 * @brief Motor driver implementation - H-bridge control
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "motor_driver.h"
#include "pwm_control.h"
#include <string.h>

/* ========================================================================
 * Private Constants
 * ======================================================================== */
/* H-bridge GPIO pin assignments (abstract) */
#define HBRIDGE_EN_PORT     0
#define HBRIDGE_IN1_PORT    0
#define HBRIDGE_IN2_PORT    0
#define HBRIDGE_EN_PIN(ch)  (ch)
#define HBRIDGE_IN1_PIN(ch) (ch * 2)
#define HBRIDGE_IN2_PIN(ch) (ch * 2 + 1)

/* Abstract GPIO functions (provided by low_level_drivers) */
extern void GPIO_WritePin(uint8_t port, uint16_t pin, bool state);
extern bool GPIO_ReadPin(uint8_t port, uint16_t pin);

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static MotorConfig_t s_config[MOTOR_MAX_CHANNELS];
static MotorStatus_t s_status[MOTOR_MAX_CHANNELS];
static bool s_initialized = false;

/* ========================================================================
 * Private Functions
 * ======================================================================== */

static void SetHBridgePins(uint8_t ch, MotorDirection_t dir)
{
    uint8_t in1_state = 0;
    uint8_t in2_state = 0;

    switch (dir) {
        case MOTOR_DIR_FORWARD:
            in1_state = 1;
            in2_state = 0;
            break;
        case MOTOR_DIR_REVERSE:
            in1_state = 0;
            in2_state = 1;
            break;
        case MOTOR_DIR_BRAKE:
            in1_state = 0;
            in2_state = 0;
            break;
        case MOTOR_DIR_COAST:
            in1_state = 1;
            in2_state = 1;
            break;
    }

    /* Apply inversion if configured */
    if (s_config[ch].inverted) {
        uint8_t tmp = in1_state;
        in1_state = in2_state;
        in2_state = tmp;
    }

    GPIO_WritePin(HBRIDGE_IN1_PORT, HBRIDGE_IN1_PIN(ch), (bool)in1_state);
    GPIO_WritePin(HBRIDGE_IN2_PORT, HBRIDGE_IN2_PIN(ch), (bool)in2_state);
}

/* ========================================================================
 * Public Functions
 * ======================================================================== */

int MotorDriver_Init(const MotorConfig_t *config)
{
    if (config == NULL) {
        return -1;
    }

    uint8_t ch = config->channel;
    if (ch >= MOTOR_MAX_CHANNELS) {
        return -2;
    }

    memcpy(&s_config[ch], config, sizeof(MotorConfig_t));
    memset(&s_status[ch], 0, sizeof(MotorStatus_t));
    s_status[ch].direction = MOTOR_DIR_BRAKE;
    s_status[ch].enabled = false;
    s_status[ch].fault = false;

    /* Configure H-bridge GPIO pins */
    GPIO_WritePin(HBRIDGE_EN_PORT, HBRIDGE_EN_PIN(ch), false);
    SetHBridgePins(ch, MOTOR_DIR_BRAKE);

    s_initialized = true;
    return 0;
}

int MotorDriver_SetDirection(uint8_t channel, MotorDirection_t dir)
{
    if (channel >= MOTOR_MAX_CHANNELS || !s_initialized) {
        return -1;
    }

    s_status[channel].direction = dir;
    SetHBridgePins(channel, dir);
    return 0;
}

int MotorDriver_SetEnable(uint8_t channel, bool enable)
{
    if (channel >= MOTOR_MAX_CHANNELS || !s_initialized) {
        return -1;
    }

    s_status[channel].enabled = enable;
    GPIO_WritePin(HBRIDGE_EN_PORT, HBRIDGE_EN_PIN(channel), enable);
    return 0;
}

void MotorDriver_EmergencyStop(void)
{
    for (uint8_t ch = 0; ch < MOTOR_MAX_CHANNELS; ch++) {
        s_status[ch].enabled = false;
        s_status[ch].duty_cycle = 0.0f;
        PWM_SetDutyCycle(ch, 0.0f);
        SetHBridgePins(ch, MOTOR_DIR_BRAKE);
        GPIO_WritePin(HBRIDGE_EN_PORT, HBRIDGE_EN_PIN(ch), false);
    }
}

const MotorStatus_t* MotorDriver_GetStatus(uint8_t channel)
{
    if (channel >= MOTOR_MAX_CHANNELS) {
        return NULL;
    }
    return &s_status[channel];
}

bool MotorDriver_IsOverCurrent(uint8_t channel)
{
    if (channel >= MOTOR_MAX_CHANNELS) {
        return true;
    }
    return (s_status[channel].current_ma > (int32_t)s_config[channel].max_current_ma);
}