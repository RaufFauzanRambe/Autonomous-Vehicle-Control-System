/**
 * @file pwm_control.c
 * @brief PWM generation using hardware timers
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "pwm_control.h"
#include <string.h>

/* ========================================================================
 * Register Definitions (STM32F4 TIM1 style)
 * ======================================================================== */
#define TIM1_BASE                0x40010000UL
#define TIM1_CR1     (*(volatile uint32_t *)(TIM1_BASE + 0x00U))
#define TIM1_CR2     (*(volatile uint32_t *)(TIM1_BASE + 0x04U))
#define TIM1_DIER    (*(volatile uint32_t *)(TIM1_BASE + 0x0CU))
#define TIM1_SR      (*(volatile uint32_t *)(TIM1_BASE + 0x10U))
#define TIM1_CCMR1   (*(volatile uint32_t *)(TIM1_BASE + 0x18U))
#define TIM1_CCMR2   (*(volatile uint32_t *)(TIM1_BASE + 0x1CU))
#define TIM1_CCER    (*(volatile uint32_t *)(TIM1_BASE + 0x20U))
#define TIM1_PSC     (*(volatile uint32_t *)(TIM1_BASE + 0x28U))
#define TIM1_ARR     (*(volatile uint32_t *)(TIM1_BASE + 0x2CU))
#define TIM1_CCR1    (*(volatile uint32_t *)(TIM1_BASE + 0x34U))
#define TIM1_CCR2    (*(volatile uint32_t *)(TIM1_BASE + 0x38U))
#define TIM1_CCR3    (*(volatile uint32_t *)(TIM1_BASE + 0x3CU))
#define TIM1_CCR4    (*(volatile uint32_t *)(TIM1_BASE + 0x40U))
#define TIM1_BDTR    (*(volatile uint32_t *)(TIM1_BASE + 0x44U))

#define TIM_CR1_CEN  (1U << 0)
#define TIM_CR1_ARPE (1U << 7)

#define APB2_TIMER_CLK_HZ        168000000UL

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static PwmConfig_t s_config[PWM_MAX_CHANNELS];
static PwmStatus_t s_status[PWM_MAX_CHANNELS];
static uint32_t    s_timer_clock_hz = APB2_TIMER_CLK_HZ;

/* ========================================================================
 * Private Functions
 * ======================================================================== */

static volatile uint32_t* GetCCR(uint8_t channel)
{
    switch (channel) {
        case 0: return &TIM1_CCR1;
        case 1: return &TIM1_CCR2;
        case 2: return &TIM1_CCR3;
        case 3: return &TIM1_CCR4;
        default: return NULL;
    }
}

/* ========================================================================
 * Public Functions
 * ======================================================================== */

int PWM_Init(const PwmConfig_t *config)
{
    if (config == NULL || config->channel >= PWM_MAX_CHANNELS) {
        return -1;
    }

    uint8_t ch = config->channel;
    memcpy(&s_config[ch], config, sizeof(PwmConfig_t));

    uint32_t arr = s_timer_clock_hz / config->frequency_hz - 1U;
    uint32_t ccr = (uint32_t)((float)arr * config->duty_cycle);

    /* Configure timer base */
    TIM1_PSC = 0;  /* No prescaler, run at full timer clock */
    TIM1_ARR = arr;

    /* Set capture/compare for this channel */
    volatile uint32_t *ccr = GetCCR(ch);
    if (ccr) {
        *ccr = ccr;
    }

    /* Configure PWM mode 1 for channel */
    if (ch < 2) {
        TIM1_CCMR1 |= (6U << (ch * 8));  /* OCxM = PWM mode 1 */
        TIM1_CCMR1 |= (1U << (ch * 8 + 3)); /* OCxPE = preload enable */
    } else {
        uint8_t shift = (ch - 2) * 8;
        TIM1_CCMR2 |= (6U << shift);
        TIM1_CCMR2 |= (1U << (shift + 3));
    }

    /* Enable output and complementary output with dead time */
    uint32_t ccen_bit  = (1U << (ch * 4));
    uint32_t ccne_bit  = (1U << (ch * 4 + 2));
    TIM1_CCER |= ccen_bit;
    if (config->complementary) {
        TIM1_CCER |= ccne_bit;

        /* Configure dead time */
        uint32_t dtg = 0;
        if (config->dead_time_ns <= 127) {
            dtg = config->dead_time_ns;
        } else {
            dtg = 127;
        }
        TIM1_BDTR = (TIM1_BDTR & ~0xFFU) | dtg;
        TIM1_BDTR |= (1U << 15);  /* MOE = main output enable */
    }

    s_status[ch].frequency_hz = config->frequency_hz;
    s_status[ch].duty_cycle   = config->duty_cycle;
    s_status[ch].running      = false;

    return 0;
}

int PWM_SetDutyCycle(uint8_t channel, float duty)
{
    if (channel >= PWM_MAX_CHANNELS) {
        return -1;
    }

    /* Clamp duty cycle */
    if (duty < 0.0f) duty = 0.0f;
    if (duty > 1.0f) duty = 1.0f;

    uint32_t arr = TIM1_ARR;
    uint32_t ccr = (uint32_t)((float)arr * duty);

    volatile uint32_t *ccr_reg = GetCCR(channel);
    if (ccr_reg) {
        *ccr_reg = ccr;
    }

    s_status[channel].duty_cycle = duty;
    return 0;
}

int PWM_SetFrequency(uint8_t channel, uint32_t freq_hz)
{
    if (channel >= PWM_MAX_CHANNELS) {
        return -1;
    }
    if (freq_hz < PWM_MIN_FREQ_HZ || freq_hz > PWM_MAX_FREQ_HZ) {
        return -2;
    }

    uint32_t arr = s_timer_clock_hz / freq_hz - 1U;
    TIM1_ARR = arr;

    /* Re-apply current duty cycle */
    PWM_SetDutyCycle(channel, s_status[channel].duty_cycle);

    s_status[channel].frequency_hz = freq_hz;
    return 0;
}

void PWM_Start(uint8_t channel)
{
    if (channel >= PWM_MAX_CHANNELS) return;

    /* Enable capture/compare output */
    TIM1_CCER |= (1U << (channel * 4));
    TIM1_CR1 |= TIM_CR1_CEN;

    s_status[channel].running = true;
}

void PWM_Stop(uint8_t channel)
{
    if (channel >= PWM_MAX_CHANNELS) return;

    TIM1_CCER &= ~(1U << (channel * 4));

    volatile uint32_t *ccr_reg = GetCCR(channel);
    if (ccr_reg) {
        *ccr_reg = 0;
    }

    s_status[channel].running = false;
    s_status[channel].duty_cycle = 0.0f;
}

const PwmStatus_t* PWM_GetStatus(uint8_t channel)
{
    if (channel >= PWM_MAX_CHANNELS) return NULL;
    return &s_status[channel];
}

void PWM_SetDeadTime(uint8_t channel, uint32_t dead_time_ns)
{
    if (channel >= PWM_MAX_CHANNELS) return;

    uint32_t dtg = (dead_time_ns <= 127) ? dead_time_ns : 127;
    TIM1_BDTR = (TIM1_BDTR & ~0xFFU) | dtg;
    s_config[channel].dead_time_ns = dead_time_ns;
}