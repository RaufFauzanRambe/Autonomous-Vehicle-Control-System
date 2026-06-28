/**
 * @file diagnostics.h
 * @brief Diagnostics subsystem header for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 *
 * Provides OBD-II / UDS-like diagnostic capabilities including
 * DTC (Diagnostic Trouble Code) management, diagnostic sessions,
 * and standardized service handlers.
 */

#ifndef DIAGNOSTICS_H
#define DIAGNOSTICS_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

/* ========================================================================
 * Diagnostic Session Types
 * ======================================================================== */
typedef enum {
    DIAG_SESSION_DEFAULT         = 0x01,  /**< Default session */
    DIAG_SESSION_PROGRAMMING     = 0x02,  /**< Programming/flash session */
    DIAG_SESSION_EXTENDED        = 0x03,  /**< Extended diagnostic session */
    DIAG_SESSION_SAFETY_DISABLED = 0x04   /**< Safety systems disabled */
} DiagSession_t;

/* ========================================================================
 * UDS Service IDs
 * ======================================================================== */
typedef enum {
    UDS_SID_DIAGNOSTIC_SESSION   = 0x10,
    UDS_SID_ECU_RESET            = 0x11,
    UDS_SID_CLEAR_DTC            = 0x14,
    UDS_SID_READ_DTC             = 0x19,
    UDS_SID_READ_DATA_BY_ID      = 0x22,
    UDS_SID_READ_MEMORY_BY_ADDR  = 0x23,
    UDS_SID_SECURITY_ACCESS      = 0x27,
    UDS_SID_COMMUNICATION_CTRL   = 0x28,
    UDS_SID_TESTER_PRESENT       = 0x3E,
    UDS_SID_WRITE_DATA_BY_ID     = 0x2E,
    UDS_SID_INPUT_OUTPUT_CONTROL = 0x2F,
    UDS_SID_ROUTINE_CONTROL      = 0x31
} UdsServiceId_t;

/* ========================================================================
 * Diagnostic Trouble Code (DTC)
 * ======================================================================== */
#define DTC_MAX_COUNT            128

typedef struct {
    uint32_t code;               /**< DTC code (UUDT format) */
    uint8_t  status;             /**< DTC status byte */
    uint32_t occurrence_count;   /**< Number of occurrences */
    uint32_t first_occurrence_ms;/**< First occurrence timestamp */
    uint32_t last_occurrence_ms; /**< Last occurrence timestamp */
    uint8_t  severity;           /**< 0=info, 1=warning, 2=error, 3=critical */
    uint8_t  operation_cycle;    /**< Operation cycle when occurred */
    char     description[48];    /**< Human-readable description */
} DiagnosticTroubleCode_t;

/* DTC Status bit definitions */
#define DTC_STATUS_TEST_FAILED        (1U << 0)
#define DTC_STATUS_TEST_FAILED_THIS_OP (1U << 1)
#define DTC_STATUS_PENDING            (1U << 2)
#define DTC_STATUS_CONFIRMED          (1U << 3)
#define DTC_STATUS_TEST_NOT_COMPLETED (1U << 4)
#define DTC_STATUS_TEST_FAILED_SINCE_CLR (1U << 5)
#define DTC_STATUS_WARNING_INDICATOR (1U << 6)
#define DTC_STATUS_CONFIRMED_THIS_OP (1U << 7)

/* ========================================================================
 * Diagnostic Data Identifier (DID)
 * ======================================================================== */
typedef enum {
    DID_SW_VERSION        = 0xF190,
    DID_HW_VERSION        = 0xF191,
    DID_SERIAL_NUMBER     = 0xF18C,
    DID_ECUNAME           = 0xF190,
    DID_VIN               = 0xF190,
    DID_SYSTEM_UPTIME     = 0xF191,
    DID_CPU_LOAD          = 0xF192,
    DID_MEMORY_USAGE      = 0xF193,
    DID_TEMPERATURE       = 0xF194,
    DID_VOLTAGE           = 0xF195,
    DID_FREEZE_FRAME      = 0xF196
} DataIdentifier_t;

/* ========================================================================
 * Diagnostic Response
 * ======================================================================== */
typedef struct {
    uint8_t  sid;                /**< Service ID + 0x40 (positive response) */
    uint8_t  data[256];          /**< Response data */
    uint16_t length;             /**< Response data length */
    bool     negative;           /**< True if negative response */
    uint8_t  nrc;                /**< Negative Response Code */
} DiagResponse_t;

/* Negative Response Codes */
#define NRC_SUCCESS              0x00
#define NRC_SUB_FUNC_NOT_SUPP    0x11
#define NRC_INCORRECT_MSG_LEN    0x13
#define NRC_INCORRECT_SEQ        0x22
#define NRC_REQ_OUT_OF_RANGE     0x31
#define NRC_SECURITY_ACCESS_DENIED 0x33
#define NRC_GENERAL_REJECT       0x10
#define NRC_SERVICE_NOT_SUPP     0x11
#define NRCConditions_NOT_MET    0x22
#define NRC_RESPONSE_PENDING     0x78

/* ========================================================================
 * Diagnostic Statistics
 * ======================================================================== */
typedef struct {
    uint32_t total_dtcs_stored;
    uint32_t confirmed_dtcs;
    uint32_t pending_dtcs;
    uint32_t total_requests;
    uint32_t total_errors;
    DiagSession_t current_session;
} DiagStats_t;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize diagnostics subsystem
 * @return AVCS_OK on success
 */
AvcsErrorCode_t Diagnostics_Init(void);

/**
 * @brief Store a new DTC
 * @param code DTC code
 * @param severity Severity level
 * @param description Human-readable description
 */
void Diagnostics_StoreDTC(uint32_t code, uint8_t severity, const char *description);

/**
 * @brief Clear all DTCs
 * @return AVCS_OK on success
 */
AvcsErrorCode_t Diagnostics_ClearDTCs(void);

/**
 * @brief Get DTC by index
 * @param index DTC index (0-based)
 * @return Pointer to DTC, NULL if invalid index
 */
const DiagnosticTroubleCode_t* Diagnostics_GetDTC(uint16_t index);

/**
 * @brief Get number of stored DTCs
 * @return DTC count
 */
uint16_t Diagnostics_GetDTCCount(void);

/**
 * @brief Get confirmed (active) DTC count
 * @return Confirmed DTC count
 */
uint16_t Diagnostics_GetConfirmedDTCCount(void);

/**
 * @brief Process a UDS diagnostic request
 * @param request Request data
 * @param request_len Request length
 * @param response Response output
 */
void Diagnostics_ProcessRequest(const uint8_t *request, uint16_t request_len,
                                 DiagResponse_t *response);

/**
 * @brief Set diagnostic session
 * @param session Target session
 * @return AVCS_OK on success
 */
AvcsErrorCode_t Diagnostics_SetSession(DiagSession_t session);

/**
 * @brief Get current diagnostic session
 * @return Current session type
 */
DiagSession_t Diagnostics_GetSession(void);

/**
 * @brief Read data by identifier
 * @param did Data identifier
 * @param data Output data buffer
 * @param max_len Maximum output length
 * @return Bytes written, or 0 on error
 */
uint16_t Diagnostics_ReadDataById(uint16_t did, uint8_t *data, uint16_t max_len);

/**
 * @brief Get diagnostic statistics
 * @return Pointer to statistics
 */
const DiagStats_t* Diagnostics_GetStats(void);

/**
 * @brief Power-on self-test: memory integrity
 * @return true if passed
 */
bool Diagnostics_CheckMemoryIntegrity(void);

/**
 * @brief Power-on self-test: peripheral check
 * @return true if passed
 */
bool Diagnostics_CheckPeripherals(void);

/**
 * @brief Power-on self-test: CAN bus check
 * @return true if passed
 */
bool Diagnostics_CheckCANBus(void);

/**
 * @brief Power-on self-test: motor driver check
 * @return true if passed
 */
bool Diagnostics_CheckMotorDrivers(void);

#ifdef __cplusplus
}
#endif

#endif /* DIAGNOSTICS_H */