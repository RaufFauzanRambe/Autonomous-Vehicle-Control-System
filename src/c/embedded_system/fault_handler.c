/**
 * @file fault_handler.c
 * @brief Fault handler implementation for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "fault_handler.h"
#include "diagnostics.h"
#include "power_management.h"
#include <string.h>

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static FaultRecord_t        s_fault_log[FAULT_LOG_MAX];
static uint16_t             s_fault_count = 0;
static FaultHandlerConfig_t s_config;
static FaultCallback_t      s_callback = NULL;

/* ========================================================================
 * Private Functions
 * ======================================================================== */

static int16_t FindFaultRecord(uint32_t code)
{
    for (uint16_t i = 0; i < s_fault_count; i++) {
        if (s_fault_log[i].code == code && s_fault_log[i].active) {
            return (int16_t)i;
        }
    }
    return -1;
}

static void NotifyCallback(const FaultRecord_t *record)
{
    if (s_callback) {
        s_callback(record);
    }
}

static const char* GetFaultDescription(FaultSource_t source, uint32_t code)
{
    switch (source) {
        case FAULT_SOURCE_SYSTEM:
            switch (code) {
                case FAULT_CODE_SELFTEST_MEM:    return "Memory integrity check failed";
                case FAULT_CODE_SELFTEST_PERIPH: return "Peripheral self-test failed";
                case FAULT_CODE_SELFTEST_CAN:    return "CAN bus self-test failed";
                case FAULT_CODE_SELFTEST_MOTOR:  return "Motor driver self-test failed";
                case FAULT_CODE_SELFTEST_TIMEOUT: return "Self-test timeout";
                case FAULT_CODE_STACK_OVERFLOW:  return "Stack overflow detected";
                case FAULT_CODE_HEAP_CORRUPT:    return "Heap corruption detected";
                default: return "Unknown system fault";
            }
        case FAULT_SOURCE_CAN:
            switch (code) {
                case FAULT_CODE_CAN_BUS_OFF:       return "CAN bus-off condition";
                case FAULT_CODE_CAN_ERROR_PASSIVE: return "CAN error passive";
                case FAULT_CODE_CAN_ERROR_WARNING: return "CAN error warning";
                case FAULT_CODE_CAN_FIFO_OVERRUN:  return "CAN FIFO overrun";
                case FAULT_CODE_CAN_ARB_LOST:      return "CAN arbitration lost";
                default: return "Unknown CAN fault";
            }
        case FAULT_SOURCE_MOTOR:
            switch (code) {
                case FAULT_CODE_MOTOR_OVERCURRENT:  return "Motor overcurrent";
                case FAULT_CODE_MOTOR_OVERTEMP:     return "Motor overtemperature";
                case FAULT_CODE_MOTOR_STALL:        return "Motor stall detected";
                case FAULT_CODE_MOTOR_ENCODER:      return "Motor encoder error";
                case FAULT_CODE_MOTOR_DRIVER_FAULT: return "Motor driver fault";
                default: return "Unknown motor fault";
            }
        case FAULT_SOURCE_POWER:
            switch (code) {
                case FAULT_CODE_POWER_UNDERVOLTAGE: return "Under-voltage detected";
                case FAULT_CODE_POWER_OVERVOLTAGE:  return "Over-voltage detected";
                case FAULT_CODE_POWER_OVERCURRENT:  return "Over-current detected";
                default: return "Unknown power fault";
            }
        default:
            return "Unknown fault";
    }
}

/* ========================================================================
 * Public Functions
 * ======================================================================== */

AvcsErrorCode_t FaultHandler_Init(void)
{
    memset(s_fault_log, 0, sizeof(s_fault_log));
    s_fault_count = 0;

    s_config.max_fault_records = FAULT_LOG_MAX;
    s_config.critical_fault_timeout_ms = 100;
    s_config.auto_recovery_enabled = true;
    s_config.max_recovery_attempts = 3;

    return AVCS_OK;
}

void FaultHandler_Report(FaultSource_t source, FaultSeverity_t severity, uint32_t code)
{
    int16_t existing = FindFaultRecord(code);

    if (existing >= 0) {
        /* Update existing fault */
        s_fault_log[existing].occurrence_count++;
        s_fault_log[existing].timestamp_ms = System_GetUptimeMs();
    } else if (s_fault_count < FAULT_LOG_MAX) {
        /* Create new fault record */
        FaultRecord_t *rec = &s_fault_log[s_fault_count];
        rec->code = code;
        rec->source = source;
        rec->severity = severity;
        rec->timestamp_ms = System_GetUptimeMs();
        rec->occurrence_count = 1;
        rec->active = true;
        rec->recovered = false;
        s_fault_count++;
        existing = (int16_t)(s_fault_count - 1);
    } else {
        return;
    }

    /* Store in diagnostics DTC system */
    Diagnostics_StoreDTC(code, (uint8_t)severity, GetFaultDescription(source, code));

    NotifyCallback(&s_fault_log[existing]);

    /* Immediate actions based on severity */
    switch (severity) {
        case FAULT_SEVERITY_CRITICAL:
            FaultHandler_EnterSafeState();
            break;
        case FAULT_SEVERITY_ERROR:
            FaultHandler_AttemptRecovery(code);
            break;
        case FAULT_SEVERITY_WARNING:
            /* Log only, continue operation */
            break;
        case FAULT_SEVERITY_INFO:
            /* Informational only */
            break;
        default:
            break;
    }
}

void FaultHandler_ClearFault(uint32_t code)
{
    int16_t idx = FindFaultRecord(code);
    if (idx >= 0) {
        s_fault_log[idx].active = false;
        s_fault_log[idx].recovered = true;
    }
}

bool FaultHandler_HasActiveFault(FaultSeverity_t min_severity)
{
    for (uint16_t i = 0; i < s_fault_count; i++) {
        if (s_fault_log[i].active && s_fault_log[i].severity >= min_severity) {
            return true;
        }
    }
    return false;
}

const FaultRecord_t* FaultHandler_GetRecord(uint16_t index)
{
    if (index >= s_fault_count) {
        return NULL;
    }
    return &s_fault_log[index];
}

uint16_t FaultHandler_GetRecordCount(void)
{
    return s_fault_count;
}

RecoveryAction_t FaultHandler_AttemptRecovery(uint32_t code)
{
    RecoveryAction_t action = FaultHandler_GetRecoveryAction(
        FAULT_SOURCE_SYSTEM, code, FAULT_SEVERITY_ERROR);

    switch (action) {
        case RECOVERY_RESET_PERIPHERAL:
            /* Attempt peripheral reset */
            break;
        case RECOVERY_SWITCH_FAILOVER:
            /* Switch to redundant hardware */
            break;
        case RECOVERY_DEGRADE_MODE:
            /* Enter degraded operation */
            break;
        case RECOVERY_RETRY:
            /* Simple retry */
            break;
        default:
            break;
    }

    return action;
}

void FaultHandler_EnterSafeState(void)
{
    /* Disable all motor outputs */
    /* Apply brakes */
    /* Set steering to center/neutral */
    /* Disable throttle */
    /* Activate hazard lights */
    /* Notify via CAN */

    g_system_status.fault_active = true;
    g_system_status.fault_severity = FAULT_SEVERITY_CRITICAL;

    System_EmergencyStop();
}

void FaultHandler_RegisterCallback(FaultCallback_t callback)
{
    s_callback = callback;
}

void FaultHandler_ClearAllRecords(void)
{
    memset(s_fault_log, 0, sizeof(s_fault_log));
    s_fault_count = 0;
}

RecoveryAction_t FaultHandler_GetRecoveryAction(FaultSource_t source,
                                                  uint32_t code,
                                                  FaultSeverity_t severity)
{
    (void)code;

    if (severity >= FAULT_SEVERITY_CRITICAL) {
        return RECOVERY_SAFE_STOP;
    }

    switch (source) {
        case FAULT_SOURCE_CAN:
            return RECOVERY_RESET_PERIPHERAL;
        case FAULT_SOURCE_MOTOR:
            return RECOVERY_DEGRADE_MODE;
        case FAULT_SOURCE_SENSOR:
            return RECOVERY_SWITCH_FAILOVER;
        case FAULT_SOURCE_POWER:
            return RECOVERY_SAFE_STOP;
        default:
            return RECOVERY_RETRY;
    }
}