/**
 * @file system_init.c
 * @brief System initialization implementation for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "system_init.h"

/* ========================================================================
 * Private Constants
 * ======================================================================== */
#define HSI_FREQ_HZ              16000000UL
#define HSE_FREQ_HZ              8000000UL
#define PLL_M_DEFAULT            8U
#define PLL_N_DEFAULT            336U
#define PLL_P_DEFAULT            2U
#define PLL_Q_DEFAULT            7U

/* Register base addresses (STM32F4 style) */
#define RCC_BASE                 0x40023800UL
#define RCC_CR                   (*(volatile uint32_t *)(RCC_BASE + 0x00U))
#define RCC_PLLCFGR              (*(volatile uint32_t *)(RCC_BASE + 0x04U))
#define RCC_CFGR                 (*(volatile uint32_t *)(RCC_BASE + 0x08U))
#define RCC_CIR                  (*(volatile uint32_t *)(RCC_BASE + 0x0CU))
#define RCC_AHB1ENR              (*(volatile uint32_t *)(RCC_BASE + 0x30U))
#define RCC_APB1ENR              (*(volatile uint32_t *)(RCC_BASE + 0x40U))
#define RCC_APB2ENR              (*(volatile uint32_t *)(RCC_BASE + 0x44U))

#define SCB_BASE                 0xE000ED00UL
#define SCB_CPACR                (*(volatile uint32_t *)(SCB_BASE + 0x88U))

#define FLASH_BASE               0x40023C00UL
#define FLASH_ACR                (*(volatile uint32_t *)(FLASH_BASE + 0x00U))

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static ClockConfig_t s_clock_config;
static uint32_t s_ahb_freq_hz  = MCU_CLOCK_HZ;
static uint32_t s_apb1_freq_hz = 42000000UL;
static uint32_t s_apb2_freq_hz = 84000000UL;

/* ========================================================================
 * Private Functions
 * ======================================================================== */

static void ConfigurePLL(void)
{
    /* Disable PLL first */
    RCC_CR &= ~((uint32_t)1U << 24);

    /* Configure PLL: HSE/PLL_M * PLL_N / PLL_P */
    uint32_t pllcfgr = 0;
    pllcfgr |= (PLL_M_DEFAULT        <<  0);  /* PLLM */
    pllcfgr |= (PLL_N_DEFAULT        <<  6);  /* PLLN */
    pllcfgr |= (((PLL_P_DEFAULT / 2) - 1) << 16); /* PLLP */
    pllcfgr |= (PLL_Q_DEFAULT        << 24);  /* PLLQ */
    pllcfgr |= (1U << 22);                   /* PLLSRC = HSE */

    RCC_PLLCFGR = pllcfgr;

    /* Enable PLL */
    RCC_CR |= ((uint32_t)1U << 24);

    /* Wait for PLL ready */
    while ((RCC_CR & ((uint32_t)1U << 25)) == 0U) {
        /* Spin */
    }
}

static void SwitchToPLL(void)
{
    /* Configure flash wait states for 168 MHz (5 wait states) */
    FLASH_ACR = (FLASH_ACR & ~0x07U) | 5U;
    FLASH_ACR |= (1U << 8);  /* Enable prefetch */
    FLASH_ACR |= (1U << 9);  /* Enable instruction cache */
    FLASH_ACR |= (1U << 10); /* Enable data cache */

    /* Select PLL as system clock */
    uint32_t cfgr = RCC_CFGR;
    cfgr &= ~0x03U;
    cfgr |= 0x02U;  /* SW = PLL */
    RCC_CFGR = cfgr;

    /* Wait for PLL selected as system clock */
    while ((RCC_CFGR & 0x0CU) != 0x08U) {
        /* Spin */
    }
}

static void EnableFPU(void)
{
    /* Enable CP10 and CP11 for FPU (Cortex-M4F) */
    SCB_CPACR = (SCB_CPACR & ~((3U << 20) | (3U << 22)))
                | ((3U << 20) | (3U << 22));
}

/* ========================================================================
 * Public Functions
 * ======================================================================== */

void SystemInit_Hardware(void)
{
    /* Enable FPU before any floating-point operations */
    EnableFPU();

    /* Clear BSS section (done by startup code, but ensure) */
    SystemInit_Memory();

    /* Configure PLL and switch system clock */
    ConfigurePLL();
    SwitchToPLL();

    /* Store clock config */
    s_clock_config.source         = CLOCK_SOURCE_PLL;
    s_clock_config.hse_freq_hz    = HSE_FREQ_HZ;
    s_clock_config.target_freq_hz = MCU_CLOCK_HZ;
    s_clock_config.ahb_prescaler  = 1;
    s_clock_config.apb1_prescaler = 4;
    s_clock_config.apb2_prescaler = 2;
    s_clock_config.enable_overdrive = false;

    /* Calculate bus clocks */
    s_ahb_freq_hz  = MCU_CLOCK_HZ / s_clock_config.ahb_prescaler;
    s_apb1_freq_hz = s_ahb_freq_hz / s_clock_config.apb1_prescaler;
    s_apb2_freq_hz = s_ahb_freq_hz / s_clock_config.apb2_prescaler;
}

AvcsErrorCode_t SystemInit_Clock(void)
{
    /* Clock already configured in SystemInit_Hardware */
    /* This function verifies and returns status */
    if (s_ahb_freq_hz != MCU_CLOCK_HZ) {
        return AVCS_ERR_INIT;
    }
    return AVCS_OK;
}

AvcsErrorCode_t SystemInit_SysTick(void)
{
    uint32_t ticks = s_ahb_freq_hz / SYSTICK_FREQ_HZ;

    /* SysTick configuration using CMSIS-style approach */
    /* Reload value = ticks - 1 */
    *(volatile uint32_t *)0xE000E014U = (ticks - 1U);

    /* Enable SysTick, use processor clock, enable interrupt */
    *(volatile uint32_t *)0xE000E010U = 0x07U;

    /* Set SysTick priority */
    *(volatile uint32_t *)0xE000ED1CUL = (0xFFU << 16) | (0x0FU << 24);

    return AVCS_OK;
}

void SystemInit_Memory(void)
{
    /* BSS zero-init and data copy handled by startup assembly code */
    /* This function provides additional initialization if needed */
}

AvcsErrorCode_t SystemInit_Peripherals(void)
{
    AvcsErrorCode_t ret = AVCS_OK;

    /* Enable AHB1 bus clocks for GPIOA-GPIOI */
    RCC_AHB1ENR |= 0x000001FFU;

    /* Enable APB1 bus clocks for UART2, UART3, SPI2, SPI3, I2C1, I2C2, I2C3,
       TIM2-TIM7, CAN1, CAN2, etc. */
    RCC_APB1ENR |= (1U << 17)  /* UART2 */
                 | (1U << 18)  /* UART3 */
                 | (1U << 14)  /* SPI2 */
                 | (1U << 15)  /* SPI3 */
                 | (1U << 21)  /* I2C1 */
                 | (1U << 22)  /* I2C2 */
                 | (1U << 0)   /* TIM2 */
                 | (1U << 1)   /* TIM3 */
                 | (1U << 2)   /* TIM4 */
                 | (1U << 3)   /* TIM5 */
                 | (1U << 25)  /* CAN1 */
                 | (1U << 26); /* CAN2 */

    /* Enable APB2 bus clocks for USART1, SPI1, TIM1, TIM8, ADC1-ADC3 */
    RCC_APB2ENR |= (1U << 4)   /* SPI1 */
                 | (1U << 12)  /* USART1 */
                 | (1U << 0)   /* TIM1 */
                 | (1U << 1)   /* TIM8 */
                 | (1U << 8)   /* ADC1 */
                 | (1U << 9)   /* ADC2 */
                 | (1U << 10); /* ADC3 */

    return ret;
}

const ClockConfig_t* SystemInit_GetClockConfig(void)
{
    return &s_clock_config;
}

void SystemInit_GetBusClocks(uint32_t *ahb_hz, uint32_t *apb1_hz, uint32_t *apb2_hz)
{
    if (ahb_hz)  *ahb_hz  = s_ahb_freq_hz;
    if (apb1_hz) *apb1_hz = s_apb1_freq_hz;
    if (apb2_hz) *apb2_hz = s_apb2_freq_hz;
}