/**
 * @file fault_handler.h
 * @brief Fault handler header for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 *
 * Central fault management system with severity classification,
 * fault injection, recovery actions, and safe-state management.
 */

#ifndef FAULT_HANDLER_H
#define FAULT_HANDLER_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

/* ========================================================================
 * Fault Severity Levels
 * ======================================================================== */
typedef enum {
    FAULT_SEVERITY_INFO     = 0,  /**< Informational, no action needed */
    FAULT_SEVERITY_WARNING  = 1,  /**< Warning, log and continue */
    FAULT_SEVERITY_ERROR    = 2,  /**< Error, attempt recovery */
    FAULT_SEVERITY_CRITICAL = 3   /**< Critical, enter safe state immediately */
} FaultSeverity_t;

/* ========================================================================
 * Fault Sources
 * ======================================================================== */
typedef enum {
    FAULT_SOURCE_SYSTEM    = 0x01,
    FAULT_SOURCE_CAN       = 0x02,
    FAULT_SOURCE_MOTOR     = 0x03,
    FAULT_SOURCE_POWER     = 0x04,
    FAULT_SOURCE_SENSOR    = 0x05,
    FAULT_SOURCE_FLASH     = 0x06,
    FAULT_SOURCE_COMMS     = 0x07,
    FAULT_SOURCE_WATCHDOG  = 0x08,
    FAULT_SOURCE_FIRMWARE  = 0x09,
    FAULT_SOURCE_STEERING  = 0x0A,
    FAULT_SOURCE_BRAKE     = 0x0B
} FaultSource_t;

/* ========================================================================
 * Fault Codes (per source)
 * ======================================================================== */
/* System faults */
#define FAULT_CODE_SELFTEST_MEM        0x0101
#define FAULT_CODE_SELFTEST_PERIPH     0x0102
#define FAULT_CODE_SELFTEST_CAN        0x0103
#define FAULT_CODE_SELFTEST_MOTOR      0x0104
#define FAULT_CODE_SELFTEST_TIMEOUT    0x0105
#define FAULT_CODE_STACK_OVERFLOW      0x0106
#define FAULT_CODE_HEAP_CORRUPT        0x0107

/* CAN faults */
#define FAULT_CODE_CAN_BUS_OFF         0x0201
#define FAULT_CODE_CAN_ERROR_PASSIVE   0x0202
#define FAULT_CODE_CAN_ERROR_WARNING   0x0203
#define FAULT_CODE_CAN_FIFO_OVERRUN    0x0204
#define FAULT_CODE_CAN_ARB_LOST        0x0205

/* Motor faults */
#define FAULT_CODE_MOTOR_OVERCURRENT   0x0301
#define FAULT_CODE_MOTOR_OVERTEMP      0x0302
#define FAULT_CODE_MOTOR_STALL         0x0303
#define FAULT_CODE_MOTOR_ENCODER       0x0304
#define FAULT_CODE_MOTOR_DRIVER_FAULT  0x0305

/* Power faults */
#define FAULT_CODE_POWER_UNDERVOLTAGE  0x0401
#define FAULT_CODE_POWER_OVERVOLTAGE   0x0402
#define FAULT_CODE_POWER_OVERCURRENT   0x0403

/* Sensor faults */
#define FAULT_CODE_SENSOR_TIMEOUT      0x0501
#define FAULT_CODE_SENSOR_INVALID_DATA 0x0502
#define FAULT_CODE_SENSOR_DISCONNECTION 0x0503

/* Flash faults */
#define FAULT_CODE_FLASH_WRITE_ERR     0x0601
#define FAULT_CODE_FLASH_READ_ERR      0x0602
#define FAULT_CODE_FLASH_ERASE_ERR     0x0603
#define FAULT_CODE_FLASH_CRC_ERR       0x0604

/* ========================================================================
 * Fault Record
 * ======================================================================== */
#define FAULT_LOG_MAX              256
#define FAULT_DEBOUNCE_COUNT       3

typedef struct {
    uint32_t        code;            /**< Fault code */
    FaultSource_t   source;          /**< Fault source */
    FaultSeverity_t severity;        /**< Fault severity */
    uint32_t        timestamp_ms;    /**< When fault occurred */
    uint32_t        occurrence_count;/**< Number of occurrences */
    bool            active;          /**< Currently active */
    bool            recovered;       /**< Successfully recovered */
} FaultRecord_t;

/* ========================================================================
 * Fault Handler Configuration
 * ======================================================================== */
typedef struct {
    uint32_t max_fault_records;      /**< Max stored fault records */
    uint32_t critical_fault_timeout_ms; /**< Time to enter safe state */
    bool     auto_recovery_enabled;  /**< Enable auto-recovery attempts */
    uint8_t  max_recovery_attempts;  /**< Max auto-recovery tries */
} FaultHandlerConfig_t;

/* ========================================================================
 * Recovery Action Types
 * ======================================================================== */
typedef enum {
    RECOVERY_NONE              = 0,  /**< No recovery possible */
    RECOVERY_RETRY             = 1,  /**< Retry the failed operation */
    RECOVERY_RESET_PERIPHERAL  = 2,  /**< Reset the faulty peripheral */
    RECOVERY_SWITCH_FAILOVER   = 3,  /**< Switch to redundant unit */
    RECOVERY_DEGRADE_MODE      = 4,  /**< Enter degraded operation mode */
    RECOVERY_SAFE_STOP         = 5,  /**< Safe controlled stop */
    RECOVERY_FULL_RESET        = 6   /**< Full system reset */
} RecoveryAction_t;

/* ========================================================================
 * Callback Types
 * ======================================================================== */

/**
 * @brief Callback for fault notification
 * @param record Fault record
 */
typedef void (*FaultCallback_t)(const FaultRecord_t *record);

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize fault handler
 * @return AVCS_OK on success
 */
AvcsErrorCode_t FaultHandler_Init(void);

/**
 * @brief Report a fault
 * @param source Fault source
 * @param severity Fault severity
 * @param code Fault code
 */
void FaultHandler_Report(FaultSource_t source, FaultSeverity_t severity, uint32_t code);

/**
 * @brief Clear a fault
 * @param code Fault code to clear
 */
void FaultHandler_ClearFault(uint32_t code);

/**
 * @brief Check if any fault of given severity is active
 * @param min_severity Minimum severity to check
 * @return true if active fault at or above severity
 */
bool FaultHandler_HasActiveFault(FaultSeverity_t min_severity);

/**
 * @brief Get fault record by index
 * @param index Record index
 * @return Pointer to fault record, NULL if invalid
 */
const FaultRecord_t* FaultHandler_GetRecord(uint16_t index);

/**
 * @brief Get total number of recorded faults
 * @return Fault record count
 */
uint16_t FaultHandler_GetRecordCount(void);

/**
 * @brief Attempt recovery for a fault
 * @param code Fault code
 * @return Recovery action taken
 */
RecoveryAction_t FaultHandler_AttemptRecovery(uint32_t code);

/**
 * @brief Enter safe state (called for critical faults)
 */
void FaultHandler_EnterSafeState(void);

/**
 * @brief Register fault notification callback
 * @param callback Callback function
 */
void FaultHandler_RegisterCallback(FaultCallback_t callback);

/**
 * @brief Clear all fault records
 */
void FaultHandler_ClearAllRecords(void);

/**
 * @brief Get recovery action for a fault code
 * @param source Fault source
 * @param code Fault code
 * @param severity Fault severity
 * @return Recommended recovery action
 */
RecoveryAction_t FaultHandler_GetRecoveryAction(FaultSource_t source,
                                                  uint32_t code,
                                                  FaultSeverity_t severity);

#ifdef __cplusplus
}
#endif

#endif /* FAULT_HANDLER_H */