/**
 * @file watchdog.c
 * @brief Watchdog timer management implementation for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "watchdog.h"

/* ========================================================================
 * Register Definitions
 * ======================================================================== */

/* IWDG registers (STM32F4) */
#define IWDG_BASE                0x40003000UL
#define IWDG_KR                  (*(volatile uint32_t *)(IWDG_BASE + 0x00U))
#define IWDG_PR                  (*(volatile uint32_t *)(IWDG_BASE + 0x04U))
#define IWDG_RLR                 (*(volatile uint32_t *)(IWDG_BASE + 0x08U))
#define IWDG_SR                  (*(volatile uint32_t *)(IWDG_BASE + 0x0CU))

/* WWDG registers */
#define WWDG_BASE                0x40002C00UL
#define WWDG_CR                  (*(volatile uint32_t *)(WWDG_BASE + 0x00U))
#define WWDG_CFR                 (*(volatile uint32_t *)(WWDG_BASE + 0x04U))
#define WWDG_SR                  (*(volatile uint32_t *)(WWDG_BASE + 0x08U))

/* RCC register for IWDG/WWDG clock enable */
#define RCC_APB1ENR_IWDG_POS     11U
#define RCC_APB1ENR_WWDG_POS     11U

/* IWDG Key values */
#define IWDG_KEY_ENABLE          0xCCCCU
#define IWDG_KEY_WRITE_ACCESS    0x5555U
#define IWDG_KEY_REFRESH         0xAAAAU

/* WWDG Configuration */
#define WWDG_COUNTER_MAX         0x7FU
#define WWDG_ENABLE_BIT          0x80U
#define WWDG_EWI_BIT             0x01U

/* LSI oscillator frequency (typical 32 kHz) */
#define LSI_FREQ_HZ              32000UL

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static WatchdogStatus_t s_status;
static WatchdogConfig_t s_config;
static uint8_t          s_iwdg_prescaler;
static uint16_t         s_iwdg_reload;
static uint8_t          s_wwdg_prescaler;
static uint8_t          s_wwdg_window;

/* ========================================================================
 * Private Functions
 * ======================================================================== */

/**
 * @brief Calculate IWDG prescaler and reload for desired timeout
 * @param timeout_ms Desired timeout
 * @param prescaler Output prescaler register value
 * @param reload Output reload register value
 * @return true if valid configuration found
 */
static bool CalculateIWDGParams(uint32_t timeout_ms, uint8_t *prescaler, uint16_t *reload)
{
    /* IWDG timeout = (prescaler * reload) / LSI_freq
       Prescaler values: 4, 8, 16, 32, 64, 128, 256 */
    static const uint16_t prescaler_values[] = {4, 8, 16, 32, 64, 128, 256};
    static const uint8_t  prescaler_regs[]   = {0, 1, 2, 3, 4, 5, 6};

    uint32_t target_ticks = (timeout_ms * LSI_FREQ_HZ) / 1000UL;

    for (uint8_t i = 0; i < 7; i++) {
        uint32_t presc = prescaler_values[i];
        uint32_t rl = target_ticks / presc;

        if (rl > 0U && rl <= 0x0FFFU) {
            *prescaler = prescaler_regs[i];
            *reload = (uint16_t)rl;
            return true;
        }
    }

    return false;
}

/**
 * @brief Calculate WWDG prescaler and window for desired timeout
 * @param timeout_ms Desired timeout
 * @param window_pct Window percentage
 * @param prescaler Output prescaler register value
 * @param window Output window register value
 * @param counter Output counter value
 * @return true if valid configuration found
 */
static bool CalculateWWDGParams(uint32_t timeout_ms, uint8_t window_pct,
                                 uint8_t *prescaler, uint8_t *window, uint8_t *counter)
{
    /* WWDG counter clock = PCLK1 / (4096 * prescaler) */
    uint32_t pclk1 = 42000000UL;  /* APB1 clock */
    static const uint32_t wwdg_presc[] = {1, 2, 4, 8};

    for (uint8_t i = 0; i < 4; i++) {
        uint32_t cnt_clk = pclk1 / (4096UL * wwdg_presc[i]);
        uint32_t cnt_val = (timeout_ms * cnt_clk) / 1000UL;

        if (cnt_val > 0 && cnt_val <= WWDG_COUNTER_MAX) {
            *prescaler = i;
            *counter = (uint8_t)cnt_val;
            *window = (uint8_t)((uint32_t)cnt_val * window_pct / 100U);
            if (*window < 0x40) *window = 0x40;
            return true;
        }
    }

    return false;
}

/* ========================================================================
 * Public Functions
 * ======================================================================== */

AvcsErrorCode_t Watchdog_Init(uint32_t timeout_ms)
{
    memset(&s_status, 0, sizeof(WatchdogStatus_t));

    /* Initialize IWDG as primary watchdog */
    AvcsErrorCode_t ret = Watchdog_InitIWDG(timeout_ms);
    if (ret != AVCS_OK) {
        return ret;
    }

    s_config.type = WDG_TYPE_INDEPENDENT;
    s_config.timeout_ms = timeout_ms;
    s_config.enable_early_warning = false;
    s_config.window_percent = 0;

    return AVCS_OK;
}

AvcsErrorCode_t Watchdog_InitIWDG(uint32_t timeout_ms)
{
    if (!CalculateIWDGParams(timeout_ms, &s_iwdg_prescaler, &s_iwdg_reload)) {
        return AVCS_ERR_INVALID;
    }

    /* Enable write access to IWDG registers */
    IWDG_KR = IWDG_KEY_WRITE_ACCESS;

    /* Set prescaler */
    IWDG_PR = s_iwdg_prescaler;

    /* Set reload value */
    IWDG_RLR = s_iwdg_reload;

    /* Wait for registers to be updated */
    while (IWDG_SR != 0U) {
        /* Spin */
    }

    /* Start the IWDG */
    IWDG_KR = IWDG_KEY_ENABLE;

    s_status.iwdg_active = true;
    s_status.iwdg_timeout_ms = timeout_ms;

    return AVCS_OK;
}

AvcsErrorCode_t Watchdog_InitWWDG(uint32_t timeout_ms, uint8_t window_percent)
{
    uint8_t counter_val;

    if (!CalculateWWDGParams(timeout_ms, window_percent,
                             &s_wwdg_prescaler, &s_wwdg_window, &counter_val)) {
        return AVCS_ERR_INVALID;
    }

    /* Configure WWDG */
    WWDG_CFR = (s_wwdg_prescaler << 7) | s_wwdg_window;

    if (s_config.enable_early_warning) {
        WWDG_CFR |= WWDG_EWI_BIT;
    }

    /* Activate WWDG with counter */
    WWDG_CR = WWDG_ENABLE_BIT | counter_val;

    s_status.wwdg_active = true;
    s_status.wwdg_timeout_ms = timeout_ms;

    return AVCS_OK;
}

void Watchdog_Refresh(void)
{
    Watchdog_RefreshIWDG();
    Watchdog_RefreshWWDG();
}

void Watchdog_RefreshIWDG(void)
{
    if (s_status.iwdg_active) {
        IWDG_KR = IWDG_KEY_REFRESH;
        s_status.last_refresh_ms = System_GetUptimeMs();
        s_status.refresh_count++;
    }
}

void Watchdog_RefreshWWDG(void)
{
    if (s_status.wwdg_active) {
        /* Reload the WWDG counter */
        WWDG_CR = WWDG_ENABLE_BIT | WWDG_COUNTER_MAX;
        s_status.last_refresh_ms = System_GetUptimeMs();
        s_status.refresh_count++;
    }
}

const WatchdogStatus_t* Watchdog_GetStatus(void)
{
    return &s_status;
}

bool Watchdog_WasResetByWatchdog(void)
{
    /* Check RCC CSR register for IWDG/WWDG reset flags */
    /* In production: read RCC->CSR and check IWDGRSTF/WWDGRSTF */
    return false;
}

void Watchdog_EarlyWarningHandler(void)
{
    s_status.early_warning = true;

    /* Log the early warning event */
    /* Attempt to complete critical operations before timeout */

    /* Clear early warning interrupt flag */
    WWDG_SR = 0U;
}

AvcsErrorCode_t Watchdog_Disable(void)
{
    /* IWDG cannot be disabled once started (hardware restriction) */
    /* WWDG can be disabled by clearing the WDGA bit */

    if (s_status.wwdg_active) {
        WWDG_CR = WWDG_COUNTER_MAX;
        s_status.wwdg_active = false;
    }

    return AVCS_OK;
}