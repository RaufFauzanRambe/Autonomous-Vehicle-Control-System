/**
 * @file power_management.h
 * @brief Power management header for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#ifndef POWER_MANAGEMENT_H
#define POWER_MANAGEMENT_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

/* ========================================================================
 * Power States
 * ======================================================================== */
typedef enum {
    POWER_MODE_RUN       = 0,  /**< Full power, all peripherals active */
    POWER_MODE_SLEEP     = 1,  /**< CPU stopped, peripherals running (WFI) */
    POWER_MODE_STOP      = 2,  /**< All clocks stopped, SRAM retained */
    POWER_MODE_STANDBY   = 3,  /**< Minimum power, wakeup on external event */
    POWER_MODE_SHUTDOWN  = 4   /**< Complete power off sequence */
} PowerMode_t;

/* ========================================================================
 * Voltage Rail Definitions
 * ======================================================================== */
typedef enum {
    RAIL_CORE    = 0,  /**< MCU core voltage rail */
    RAIL_IO      = 1,  /**< I/O voltage rail */
    RAIL_MOTOR   = 2,  /**< Motor driver power rail */
    RAIL_CAN     = 3,  /**< CAN transceiver power rail */
    RAIL_SENSOR  = 4,  /**< Sensor power rail */
    RAIL_COMMS   = 5,  /**< Communication module rail */
    RAIL_COUNT   = 6
} PowerRail_t;

/* ========================================================================
 * Voltage Monitoring
 * ======================================================================== */
typedef struct {
    float    voltage_v;         /**< Measured voltage */
    float    current_ma;        /**< Measured current draw */
    float    temperature_c;     /**< Rail temperature */
    bool     over_voltage;      /**< OV fault flag */
    bool     under_voltage;     /**< UV fault flag */
    bool     over_current;      /**< OC fault flag */
    bool     enabled;           /**< Rail enabled state */
} RailStatus_t;

/* ========================================================================
 * Power Management Configuration
 * ======================================================================== */
typedef struct {
    float     rail_voltages[RAIL_COUNT];    /**< Nominal voltage per rail */
    float     uv_threshold_percent;         /**< Under-voltage threshold (%) */
    float     ov_threshold_percent;         /**< Over-voltage threshold (%) */
    float     oc_threshold_ma[RAIL_COUNT];  /**< Over-current threshold per rail */
    uint32_t  voltage_monitor_interval_ms;  /**< ADC sampling interval */
    uint32_t  wakeup_timeout_ms;            /**< Max time in low-power mode */
} PowerMgmtConfig_t;

/* ========================================================================
 * Wakeup Source
 * ======================================================================== */
typedef enum {
    WAKEUP_CAN       = 0x01,  /**< Wake on CAN activity */
    WAKEUP_UART      = 0x02,  /**< Wake on UART RX */
    WAKEUP_GPIO      = 0x04,  /**< Wake on GPIO interrupt */
    WAKEUP_TIMER     = 0x08,  /**< Wake on timer timeout */
    WAKEUP_WATCHDOG  = 0x10   /**< Wake on watchdog */
} WakeupSource_t;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize power management subsystem
 * @return AVCS_OK on success
 */
AvcsErrorCode_t PowerManagement_Init(void);

/**
 * @brief Enter specified low-power mode
 * @param mode Target power mode
 * @return AVCS_OK on success
 */
AvcsErrorCode_t PowerManagement_EnterLowPower(PowerMode_t mode);

/**
 * @brief Wake up from low-power mode
 * @return AVCS_OK on success
 */
AvcsErrorCode_t PowerManagement_Wakeup(void);

/**
 * @brief Enable a specific power rail
 * @param rail Rail to enable
 * @return AVCS_OK on success
 */
AvcsErrorCode_t PowerManagement_EnableRail(PowerRail_t rail);

/**
 * @brief Disable a specific power rail
 * @param rail Rail to disable
 * @return AVCS_OK on success
 */
AvcsErrorCode_t PowerManagement_DisableRail(PowerRail_t rail);

/**
 * @brief Get status of a power rail
 * @param rail Rail to query
 * @return Pointer to rail status (valid until next call)
 */
const RailStatus_t* PowerManagement_GetRailStatus(PowerRail_t rail);

/**
 * @brief Monitor all voltage rails (called periodically)
 */
void PowerManagement_MonitorRails(void);

/**
 * @brief Configure wakeup sources
 * @param sources Bitmask of wakeup sources
 */
void PowerManagement_SetWakeupSources(uint32_t sources);

/**
 * @brief Get current power mode
 * @return Current power mode
 */
PowerMode_t PowerManagement_GetMode(void);

/**
 * @brief Callback type for power state changes
 * @param old_mode Previous power mode
 * @param new_mode New power mode
 */
typedef void (*PowerStateCallback_t)(PowerMode_t old_mode, PowerMode_t new_mode);

/**
 * @brief Register callback for power state transitions
 * @param callback Function to call on state change
 */
void PowerManagement_RegisterCallback(PowerStateCallback_t callback);

#ifdef __cplusplus
}
#endif

#endif /* POWER_MANAGEMENT_H */