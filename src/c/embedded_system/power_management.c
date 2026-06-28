/**
 * @file power_management.c
 * @brief Power management implementation for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "power_management.h"
#include "fault_handler.h"

/* ========================================================================
 * Private Constants
 * ======================================================================== */
#define RAIL_CORE_NOMINAL_V     3.3f
#define RAIL_IO_NOMINAL_V       3.3f
#define RAIL_MOTOR_NOMINAL_V    12.0f
#define RAIL_CAN_NOMINAL_V      5.0f
#define RAIL_SENSOR_NOMINAL_V   3.3f
#define RAIL_COMMS_NOMINAL_V    3.3f

#define UV_THRESHOLD_PCT        90.0f
#define OV_THRESHOLD_PCT        110.0f
#define VOLTAGE_ADC_CHANNEL     1U

/* PWR register base (STM32F4) */
#define PWR_BASE                0x40007000UL
#define PWR_CR                  (*(volatile uint32_t *)(PWR_BASE + 0x00U))
#define PWR_CSR                 (*(volatile uint32_t *)(PWR_BASE + 0x04U))

/* SCB register for sleep/wakeup */
#define SCB_SCR                 (*(volatile uint32_t *)0xE000ED10U)

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static PowerMode_t         s_current_mode   = POWER_MODE_RUN;
static RailStatus_t        s_rail_status[RAIL_COUNT];
static PowerMgmtConfig_t   s_config;
static PowerStateCallback_t s_callback      = NULL;
static uint32_t            s_wakeup_sources = WAKEUP_CAN | WAKEUP_GPIO;

/* ========================================================================
 * Private Functions
 * ======================================================================== */

static void ConfigureDefaultRails(void)
{
    s_config.rail_voltages[RAIL_CORE]   = RAIL_CORE_NOMINAL_V;
    s_config.rail_voltages[RAIL_IO]     = RAIL_IO_NOMINAL_V;
    s_config.rail_voltages[RAIL_MOTOR]  = RAIL_MOTOR_NOMINAL_V;
    s_config.rail_voltages[RAIL_CAN]    = RAIL_CAN_NOMINAL_V;
    s_config.rail_voltages[RAIL_SENSOR] = RAIL_SENSOR_NOMINAL_V;
    s_config.rail_voltages[RAIL_COMMS]  = RAIL_COMMS_NOMINAL_V;

    s_config.uv_threshold_percent = UV_THRESHOLD_PCT;
    s_config.ov_threshold_percent = OV_THRESHOLD_PCT;
    s_config.voltage_monitor_interval_ms = 10U;
    s_config.wakeup_timeout_ms = 0xFFFFFFFFU;

    /* Default over-current thresholds per rail (mA) */
    s_config.oc_threshold_ma[RAIL_CORE]   = 500.0f;
    s_config.oc_threshold_ma[RAIL_IO]     = 300.0f;
    s_config.oc_threshold_ma[RAIL_MOTOR]  = 30000.0f;
    s_config.oc_threshold_ma[RAIL_CAN]    = 200.0f;
    s_config.oc_threshold_ma[RAIL_SENSOR] = 500.0f;
    s_config.oc_threshold_ma[RAIL_COMMS]  = 400.0f;

    for (uint8_t i = 0; i < RAIL_COUNT; i++) {
        s_rail_status[i].voltage_v = s_config.rail_voltages[i];
        s_rail_status[i].current_ma = 0.0f;
        s_rail_status[i].temperature_c = 25.0f;
        s_rail_status[i].over_voltage = false;
        s_rail_status[i].under_voltage = false;
        s_rail_status[i].over_current = false;
        s_rail_status[i].enabled = (i != RAIL_MOTOR);  /* Motor rail off by default */
    }
}

static void UpdateRailMeasurements(void)
{
    /* In production, read from ADC channels via DMA */
    /* Simulated measurements for structure */
    for (uint8_t i = 0; i < RAIL_COUNT; i++) {
        if (s_rail_status[i].enabled) {
            float nominal = s_config.rail_voltages[i];
            float uv_limit = nominal * (s_config.uv_threshold_percent / 100.0f);
            float ov_limit = nominal * (s_config.ov_threshold_percent / 100.0f);

            s_rail_status[i].under_voltage = (s_rail_status[i].voltage_v < uv_limit);
            s_rail_status[i].over_voltage  = (s_rail_status[i].voltage_v > ov_limit);
            s_rail_status[i].over_current  = (s_rail_status[i].current_ma >
                                              s_config.oc_threshold_ma[i]);
        }
    }
}

/* ========================================================================
 * Public Functions
 * ======================================================================== */

AvcsErrorCode_t PowerManagement_Init(void)
{
    ConfigureDefaultRails();

    /* Enable power clock */
    /* Enable access to backup domain */
    /* Configure voltage regulator */

    s_current_mode = POWER_MODE_RUN;

    return AVCS_OK;
}

AvcsErrorCode_t PowerManagement_EnterLowPower(PowerMode_t mode)
{
    PowerMode_t old_mode = s_current_mode;

    switch (mode) {
        case POWER_MODE_SLEEP:
            /* Clear SLEEPDEEP bit -> sleep mode */
            SCB_SCR &= ~(1U << 2);
            __asm volatile("WFI");
            s_current_mode = mode;
            break;

        case POWER_MODE_STOP:
            /* Set SLEEPDEEP bit */
            SCB_SCR |= (1U << 2);

            /* Set LPDS bit for low-power regulator in STOP mode */
            PWR_CR |= (1U << 0);

            /* Clear wakeup flag */
            PWR_CR |= (1U << 2);

            __asm volatile("WFI");

            /* After wakeup, reconfigure clocks */
            s_current_mode = POWER_MODE_RUN;  /* Back to run after wakeup */
            break;

        case POWER_MODE_STANDBY:
            /* Set SLEEPDEEP + PDDS bits */
            SCB_SCR |= (1U << 2);
            PWR_CR |= (1U << 1);

            /* Clear wakeup flag */
            PWR_CR |= (1U << 2);

            __asm volatile("WFI");

            /* STANDBY causes reset - this code won't execute */
            break;

        case POWER_MODE_SHUTDOWN:
            /* Disable all rails except core */
            for (uint8_t i = RAIL_COUNT; i > 0; i--) {
                PowerManagement_DisableRail((PowerRail_t)(i - 1));
            }
            s_current_mode = mode;
            break;

        default:
            return AVCS_ERR_INVALID;
    }

    if (s_callback) {
        s_callback(old_mode, s_current_mode);
    }

    return AVCS_OK;
}

AvcsErrorCode_t PowerManagement_Wakeup(void)
{
    s_current_mode = POWER_MODE_RUN;
    return AVCS_OK;
}

AvcsErrorCode_t PowerManagement_EnableRail(PowerRail_t rail)
{
    if (rail >= RAIL_COUNT) {
        return AVCS_ERR_INVALID;
    }

    s_rail_status[rail].enabled = true;
    /* In production: set GPIO to enable power rail switch */
    return AVCS_OK;
}

AvcsErrorCode_t PowerManagement_DisableRail(PowerRail_t rail)
{
    if (rail >= RAIL_COUNT) {
        return AVCS_ERR_INVALID;
    }

    /* Never disable core rail while running */
    if (rail == RAIL_CORE && s_current_mode != POWER_MODE_SHUTDOWN) {
        return AVCS_ERR_INVALID;
    }

    s_rail_status[rail].enabled = false;
    /* In production: clear GPIO to disable power rail switch */
    return AVCS_OK;
}

const RailStatus_t* PowerManagement_GetRailStatus(PowerRail_t rail)
{
    if (rail >= RAIL_COUNT) {
        return NULL;
    }
    return &s_rail_status[rail];
}

void PowerManagement_MonitorRails(void)
{
    UpdateRailMeasurements();

    /* Report faults for any rail issues */
    for (uint8_t i = 0; i < RAIL_COUNT; i++) {
        if (!s_rail_status[i].enabled) continue;

        if (s_rail_status[i].under_voltage) {
            FaultHandler_Report(FAULT_SOURCE_POWER, FAULT_SEVERITY_ERROR,
                                FAULT_CODE_POWER_UNDERVOLTAGE);
        }
        if (s_rail_status[i].over_voltage) {
            FaultHandler_Report(FAULT_SOURCE_POWER, FAULT_SEVERITY_CRITICAL,
                                FAULT_CODE_POWER_OVERVOLTAGE);
        }
        if (s_rail_status[i].over_current) {
            FaultHandler_Report(FAULT_SOURCE_POWER, FAULT_SEVERITY_CRITICAL,
                                FAULT_CODE_POWER_OVERCURRENT);
        }
    }
}

void PowerManagement_SetWakeupSources(uint32_t sources)
{
    s_wakeup_sources = sources;
}

PowerMode_t PowerManagement_GetMode(void)
{
    return s_current_mode;
}

void PowerManagement_RegisterCallback(PowerStateCallback_t callback)
{
    s_callback = callback;
}