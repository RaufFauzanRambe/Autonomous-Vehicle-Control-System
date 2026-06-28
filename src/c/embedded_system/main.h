/**
 * @file main.h
 * @brief Main application header for Autonomous Vehicle Control System
 * @version 1.0.0
 * @date 2026-06-27
 *
 * This file contains system-wide includes, type definitions, macros,
 * and extern declarations used across all modules of the AVCS.
 */

#ifndef MAIN_H
#define MAIN_H

#ifdef __cplusplus
extern "C" {
#endif

/* ========================================================================
 * Includes
 * ======================================================================== */
#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>
#include <string.h>

/* ========================================================================
 * MCU Definitions (ARM Cortex-M4F assumed)
 * ======================================================================== */
#define MCU_CLOCK_HZ             168000000UL
#define MCU_CLOCK_MHZ            168UL
#define SYSTICK_FREQ_HZ          1000UL
#define SYSTICK_PERIOD_MS        (1000UL / SYSTICK_FREQ_HZ)

/* ========================================================================
 * System Version
 * ======================================================================== */
#define AVCS_VERSION_MAJOR       2
#define AVCS_VERSION_MINOR       1
#define AVCS_VERSION_PATCH       0
#define AVCS_VERSION_STRING      "2.1.0"

/* ========================================================================
 * Error Code Definitions
 * ======================================================================== */
typedef enum {
    AVCS_OK              =  0,  /**< Operation successful */
    AVCS_ERR_INIT        = -1,  /**< Initialization failed */
    AVCS_ERR_TIMEOUT     = -2,  /**< Operation timed out */
    AVCS_ERR_INVALID     = -3,  /**< Invalid parameter */
    AVCS_ERR_BUSY        = -4,  /**< Resource busy */
    AVCS_ERR_FAULT       = -5,  /**< Hardware fault detected */
    AVCS_ERR_COMM        = -6,  /**< Communication error */
    AVCS_ERR_FLASH       = -7,  /**< Flash operation error */
    AVCS_ERR_WATCHDOG    = -8,  /**< Watchdog triggered */
    AVCS_ERR_POWER       = -9,  /**< Power domain error */
    AVCS_ERR_CRC         = -10, /**< CRC mismatch */
    AVCS_ERR_UNKNOWN     = -99  /**< Unknown error */
} AvcsErrorCode_t;

/* ========================================================================
 * System State Machine
 * ======================================================================== */
typedef enum {
    SYS_STATE_BOOT        = 0,  /**< Boot/initialization phase */
    SYS_STATE_SELFTEST    = 1,  /**< Power-on self-test */
    SYS_STATE_STANDBY     = 2,  /**< System idle, awaiting commands */
    SYS_STATE_INIT_DRIVERS = 3, /**< Peripheral driver initialization */
    SYS_STATE_CALIBRATION = 4,  /**< Sensor/actuator calibration */
    SYS_STATE_READY       = 5,  /**< System ready for operation */
    SYS_STATE_DRIVING     = 6,  /**< Active driving mode */
    SYS_STATE_PARKING     = 7,  /**< Parking mode */
    SYS_STATE_EMERGENCY   = 8,  /**< Emergency stop / fault state */
    SYS_STATE_SHUTDOWN    = 9,  /**< Controlled shutdown */
    SYS_STATE_FAULT       = 10  /**< Unrecoverable fault */
} SystemState_t;

/* ========================================================================
 * System Configuration
 * ======================================================================== */
typedef struct {
    uint32_t    system_clock_hz;      /**< Configured system clock */
    uint32_t    systick_freq_hz;      /**< SysTick interrupt frequency */
    uint32_t    watchdog_timeout_ms;  /**< Watchdog refresh period */
    uint8_t     can_bus_count;        /**< Number of CAN interfaces */
    uint8_t     motor_count;          /**< Number of motor channels */
    bool        enable_diagnostics;   /**< Diagnostic subsystem enabled */
    bool        enable_ota;           /**< Over-the-air update enabled */
    uint32_t    boot_timeout_ms;      /**< Max boot time before fault */
} SystemConfig_t;

/* ========================================================================
 * System Status
 * ======================================================================== */
typedef struct {
    SystemState_t   current_state;     /**< Current system state */
    SystemState_t   previous_state;    /**< Previous system state */
    uint32_t        uptime_ms;         /**< System uptime in milliseconds */
    uint32_t        loop_count;        /**< Main loop iteration counter */
    uint16_t        cpu_usage_percent; /**< Estimated CPU usage */
    uint32_t        free_heap_bytes;   /**< Free heap memory */
    int16_t         board_temp_celsius; /**< Board temperature */
    uint32_t        last_state_change_ms; /**< Timestamp of last state change */
    bool            fault_active;      /**< Active fault flag */
    uint8_t         fault_severity;    /**< Current fault severity */
} SystemStatus_t;

/* ========================================================================
 * Global Variables (extern)
 * ======================================================================== */
extern volatile uint32_t g_system_tick_ms;
extern SystemConfig_t    g_system_config;
extern SystemStatus_t    g_system_status;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize all system subsystems
 * @return AVCS_OK on success, error code on failure
 */
AvcsErrorCode_t System_InitAll(void);

/**
 * @brief Main system state machine handler
 * @note Called every main loop iteration
 */
void System_StateMachine(void);

/**
 * @brief Transition to a new system state
 * @param new_state Target state
 * @return AVCS_OK on success, error if transition invalid
 */
AvcsErrorCode_t System_TransitionState(SystemState_t new_state);

/**
 * @brief Get current system uptime in milliseconds
 * @return Uptime in ms
 */
static inline uint32_t System_GetUptimeMs(void)
{
    extern volatile uint32_t g_system_tick_ms;
    return g_system_tick_ms;
}

/**
 * @brief Check if a timeout has elapsed
 * @param start_ms Starting timestamp
 * @param timeout_ms Timeout duration
 * @return true if timeout has elapsed
 */
static inline bool System_TimeoutElapsed(uint32_t start_ms, uint32_t timeout_ms)
{
    return (System_GetUptimeMs() - start_ms) >= timeout_ms;
}

/**
 * @brief Delay for specified milliseconds (blocking)
 * @param ms Delay duration
 */
void System_DelayMs(uint32_t ms);

/**
 * @brief Enter emergency stop state
 */
void System_EmergencyStop(void);

#ifdef __cplusplus
}
#endif

#endif /* MAIN_H */