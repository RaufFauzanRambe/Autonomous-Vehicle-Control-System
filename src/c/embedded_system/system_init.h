/**
 * @file system_init.h
 * @brief System initialization header for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#ifndef SYSTEM_INIT_H
#define SYSTEM_INIT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

/* ========================================================================
 * Clock Configuration
 * ======================================================================== */
typedef enum {
    CLOCK_SOURCE_HSI   = 0,  /**< Internal RC oscillator (16 MHz) */
    CLOCK_SOURCE_HSE   = 1,  /**< External crystal oscillator */
    CLOCK_SOURCE_PLL   = 2   /**< PLL output */
} ClockSource_t;

typedef struct {
    ClockSource_t source;         /**< Main clock source */
    uint32_t      hse_freq_hz;    /**< HSE crystal frequency */
    uint32_t      target_freq_hz; /**< Target system clock */
    uint8_t       ahb_prescaler;  /**< AHB bus prescaler */
    uint8_t       apb1_prescaler; /**< APB1 bus prescaler */
    uint8_t       apb2_prescaler; /**< APB2 bus prescaler */
    bool          enable_overdrive; /**< Enable overdrive for >168 MHz */
} ClockConfig_t;

/* ========================================================================
 * Peripheral Init Order
 * ======================================================================== */
typedef enum {
    PERIPH_INIT_CLOCK    = 0,
    PERIPH_INIT_GPIO     = 1,
    PERIPH_INIT_UART     = 2,
    PERIPH_INIT_SPI      = 3,
    PERIPH_INIT_I2C      = 4,
    PERIPH_INIT_TIMERS   = 5,
    PERIPH_INIT_ADC      = 6,
    PERIPH_INIT_CAN      = 7,
    PERIPH_INIT_DMA      = 8,
    PERIPH_INIT_COUNT    = 9
} PeripheralInitOrder_t;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize hardware clocks and PLL
 * @return AVCS_OK on success
 */
AvcsErrorCode_t SystemInit_Clock(void);

/**
 * @brief Configure SysTick timer
 * @return AVCS_OK on success
 */
AvcsErrorCode_t SystemInit_SysTick(void);

/**
 * @brief Perform hardware-level initialization (called before main init)
 * @note Configures clocks, enables FPU, sets vector table
 */
void SystemInit_Hardware(void);

/**
 * @brief Initialize all peripheral drivers in order
 * @return AVCS_OK if all peripherals initialized successfully
 */
AvcsErrorCode_t SystemInit_Peripherals(void);

/**
 * @brief Initialize memory regions (BSS clear, data init)
 */
void SystemInit_Memory(void);

/**
 * @brief Get current clock configuration
 * @return Pointer to active clock config
 */
const ClockConfig_t* SystemInit_GetClockConfig(void);

/**
 * @brief Get bus clock frequencies
 * @param ahb_hz  AHB bus frequency output
 * @param apb1_hz APB1 bus frequency output
 * @param apb2_hz APB2 bus frequency output
 */
void SystemInit_GetBusClocks(uint32_t *ahb_hz, uint32_t *apb1_hz, uint32_t *apb2_hz);

#ifdef __cplusplus
}
#endif

#endif /* SYSTEM_INIT_H */