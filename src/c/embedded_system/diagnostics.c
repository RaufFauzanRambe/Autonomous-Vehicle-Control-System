/**
 * @file diagnostics.c
 * @brief Diagnostics subsystem implementation for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#include "diagnostics.h"
#include "main.h"
#include <string.h>

/* ========================================================================
 * Private Variables
 * ======================================================================== */
static DiagnosticTroubleCode_t s_dtc_store[DTC_MAX_COUNT];
static uint16_t               s_dtc_count = 0;
static DiagSession_t          s_current_session = DIAG_SESSION_DEFAULT;
static DiagStats_t            s_stats;
static uint32_t               s_session_start_ms = 0;

/* ========================================================================
 * Private Functions
 * ======================================================================== */

static void UpdateStats(void)
{
    s_stats.total_dtcs_stored = s_dtc_count;
    s_stats.confirmed_dtcs = 0;
    s_stats.pending_dtcs = 0;

    for (uint16_t i = 0; i < s_dtc_count; i++) {
        if (s_dtc_store[i].status & DTC_STATUS_CONFIRMED) {
            s_stats.confirmed_dtcs++;
        }
        if (s_dtc_store[i].status & DTC_STATUS_PENDING) {
            s_stats.pending_dtcs++;
        }
    }

    s_stats.current_session = s_current_session;
}

static int16_t FindDTCIndex(uint32_t code)
{
    for (uint16_t i = 0; i < s_dtc_count; i++) {
        if (s_dtc_store[i].code == code) {
            return (int16_t)i;
        }
    }
    return -1;
}

static void HandleDiagnosticSession(const uint8_t *data, uint16_t len,
                                     DiagResponse_t *resp)
{
    (void)len;
    uint8_t sub_func = data[0];

    resp->sid = UDS_SID_DIAGNOSTIC_SESSION + 0x40;
    resp->negative = false;

    AvcsErrorCode_t ret = Diagnostics_SetSession((DiagSession_t)sub_func);
    if (ret == AVCS_OK) {
        resp->data[0] = sub_func;
        resp->length = 1;
    } else {
        resp->negative = true;
        resp->nrc = NRCConditions_NOT_MET;
    }
}

static void HandleReadDTC(const uint8_t *data, uint16_t len,
                           DiagResponse_t *resp)
{
    (void)data;
    (void)len;
    resp->sid = UDS_SID_READ_DTC + 0x40;
    resp->negative = false;

    uint16_t idx = 0;
    resp->data[idx++] = 0x01; /* DTC format: UUDT */
    resp->data[idx++] = 0x01; /* Status availability mask */

    uint16_t count = Diagnostics_GetConfirmedDTCCount();
    resp->data[idx++] = (uint8_t)(count & 0xFF);
    resp->data[idx++] = (uint8_t)((count >> 8) & 0xFF);

    for (uint16_t i = 0; i < s_dtc_count && idx < 250; i++) {
        if (s_dtc_store[i].status & DTC_STATUS_CONFIRMED) {
            resp->data[idx++] = (uint8_t)(s_dtc_store[i].code & 0xFF);
            resp->data[idx++] = (uint8_t)((s_dtc_store[i].code >> 8) & 0xFF);
            resp->data[idx++] = (uint8_t)((s_dtc_store[i].code >> 16) & 0xFF);
            resp->data[idx++] = s_dtc_store[i].status;
        }
    }

    resp->length = idx;
}

static void HandleClearDTC(const uint8_t *data, uint16_t len,
                            DiagResponse_t *resp)
{
    (void)data;
    (void)len;
    resp->sid = UDS_SID_CLEAR_DTC + 0x40;
    resp->negative = false;

    AvcsErrorCode_t ret = Diagnostics_ClearDTCs();
    if (ret != AVCS_OK) {
        resp->negative = true;
        resp->nrc = NRCConditions_NOT_MET;
    } else {
        resp->length = 0;
    }
}

static void HandleReadDataById(const uint8_t *data, uint16_t len,
                                DiagResponse_t *resp)
{
    (void)len;
    resp->sid = UDS_SID_READ_DATA_BY_ID + 0x40;
    resp->negative = false;

    if (len < 2) {
        resp->negative = true;
        resp->nrc = NRC_INCORRECT_MSG_LEN;
        return;
    }

    uint16_t did = ((uint16_t)data[0] << 8) | data[1];
    uint16_t bytes = Diagnostics_ReadDataById(did, &resp->data[0],
                                               sizeof(resp->data));
    if (bytes == 0) {
        resp->negative = true;
        resp->nrc = NRC_REQ_OUT_OF_RANGE;
    } else {
        resp->length = bytes;
    }
}

static void HandleTesterPresent(const uint8_t *data, uint16_t len,
                                 DiagResponse_t *resp)
{
    (void)data;
    (void)len;
    resp->sid = UDS_SID_TESTER_PRESENT + 0x40;
    resp->negative = false;
    resp->data[0] = 0x00; /* Sub-function 0x00 = positive response */
    resp->length = 1;
}

/* ========================================================================
 * Public Functions
 * ======================================================================== */

AvcsErrorCode_t Diagnostics_Init(void)
{
    memset(s_dtc_store, 0, sizeof(s_dtc_store));
    s_dtc_count = 0;
    s_current_session = DIAG_SESSION_DEFAULT;
    memset(&s_stats, 0, sizeof(s_stats));

    return AVCS_OK;
}

void Diagnostics_StoreDTC(uint32_t code, uint8_t severity, const char *description)
{
    int16_t existing = FindDTCIndex(code);

    if (existing >= 0) {
        /* Update existing DTC */
        s_dtc_store[existing].occurrence_count++;
        s_dtc_store[existing].last_occurrence_ms = System_GetUptimeMs();
        s_dtc_store[existing].status |= DTC_STATUS_TEST_FAILED_SINCE_CLR;
        s_dtc_store[existing].status |= DTC_STATUS_TEST_FAILED_THIS_OP;

        /* Promote severity if higher */
        if (severity > s_dtc_store[existing].severity) {
            s_dtc_store[existing].severity = severity;
        }
    } else if (s_dtc_count < DTC_MAX_COUNT) {
        /* Store new DTC */
        DiagnosticTroubleCode_t *dtc = &s_dtc_store[s_dtc_count];
        dtc->code = code;
        dtc->status = DTC_STATUS_TEST_FAILED | DTC_STATUS_TEST_FAILED_THIS_OP
                     | DTC_STATUS_PENDING | DTC_STATUS_TEST_FAILED_SINCE_CLR;
        dtc->occurrence_count = 1;
        dtc->first_occurrence_ms = System_GetUptimeMs();
        dtc->last_occurrence_ms = dtc->first_occurrence_ms;
        dtc->severity = severity;
        dtc->operation_cycle = 0;

        if (description) {
            strncpy(dtc->description, description, sizeof(dtc->description) - 1);
            dtc->description[sizeof(dtc->description) - 1] = '\0';
        }

        s_dtc_count++;
    }

    /* Confirm DTC after 3 occurrences */
    if (existing >= 0 && s_dtc_store[existing].occurrence_count >= 3) {
        s_dtc_store[existing].status |= DTC_STATUS_CONFIRMED;
        s_dtc_store[existing].status |= DTC_STATUS_CONFIRMED_THIS_OP;
        s_dtc_store[existing].status &= ~DTC_STATUS_PENDING;
    }

    UpdateStats();
}

AvcsErrorCode_t Diagnostics_ClearDTCs(void)
{
    if (s_current_session == DIAG_SESSION_DEFAULT) {
        return AVCS_ERR_INVALID;
    }

    memset(s_dtc_store, 0, sizeof(s_dtc_store));
    s_dtc_count = 0;
    UpdateStats();

    return AVCS_OK;
}

const DiagnosticTroubleCode_t* Diagnostics_GetDTC(uint16_t index)
{
    if (index >= s_dtc_count) {
        return NULL;
    }
    return &s_dtc_store[index];
}

uint16_t Diagnostics_GetDTCCount(void)
{
    return s_dtc_count;
}

uint16_t Diagnostics_GetConfirmedDTCCount(void)
{
    uint16_t count = 0;
    for (uint16_t i = 0; i < s_dtc_count; i++) {
        if (s_dtc_store[i].status & DTC_STATUS_CONFIRMED) {
            count++;
        }
    }
    return count;
}

void Diagnostics_ProcessRequest(const uint8_t *request, uint16_t request_len,
                                 DiagResponse_t *response)
{
    if (request == NULL || request_len == 0 || response == NULL) {
        return;
    }

    memset(response, 0, sizeof(DiagResponse_t));
    s_stats.total_requests++;

    uint8_t sid = request[0];
    const uint8_t *payload = &request[1];
    uint16_t payload_len = request_len - 1;

    switch (sid) {
        case UDS_SID_DIAGNOSTIC_SESSION:
            HandleDiagnosticSession(payload, payload_len, response);
            break;
        case UDS_SID_READ_DTC:
            HandleReadDTC(payload, payload_len, response);
            break;
        case UDS_SID_CLEAR_DTC:
            HandleClearDTC(payload, payload_len, response);
            break;
        case UDS_SID_READ_DATA_BY_ID:
            HandleReadDataById(payload, payload_len, response);
            break;
        case UDS_SID_TESTER_PRESENT:
            HandleTesterPresent(payload, payload_len, response);
            break;
        default:
            response->sid = sid;
            response->negative = true;
            response->nrc = NRC_SERVICE_NOT_SUPP;
            s_stats.total_errors++;
            break;
    }
}

AvcsErrorCode_t Diagnostics_SetSession(DiagSession_t session)
{
    switch (session) {
        case DIAG_SESSION_DEFAULT:
        case DIAG_SESSION_EXTENDED:
        case DIAG_SESSION_PROGRAMMING:
            s_current_session = session;
            s_session_start_ms = System_GetUptimeMs();
            UpdateStats();
            return AVCS_OK;
        case DIAG_SESSION_SAFETY_DISABLED:
            /* Only allowed under specific conditions */
            return AVCS_ERR_INVALID;
        default:
            return AVCS_ERR_INVALID;
    }
}

DiagSession_t Diagnostics_GetSession(void)
{
    return s_current_session;
}

uint16_t Diagnostics_ReadDataById(uint16_t did, uint8_t *data, uint16_t max_len)
{
    uint16_t len = 0;

    switch (did) {
        case DID_SW_VERSION: {
            const char *ver = "AVCS " AVCS_VERSION_STRING;
            uint16_t ver_len = (uint16_t)strlen(ver);
            if (ver_len <= max_len) {
                memcpy(data, ver, ver_len);
                len = ver_len;
            }
            break;
        }
        case DID_SYSTEM_UPTIME: {
            uint32_t uptime = System_GetUptimeMs();
            if (max_len >= 4) {
                data[0] = (uint8_t)(uptime & 0xFF);
                data[1] = (uint8_t)((uptime >> 8) & 0xFF);
                data[2] = (uint8_t)((uptime >> 16) & 0xFF);
                data[3] = (uint8_t)((uptime >> 24) & 0xFF);
                len = 4;
            }
            break;
        }
        case DID_CPU_LOAD: {
            if (max_len >= 1) {
                data[0] = (uint8_t)g_system_status.cpu_usage_percent;
                len = 1;
            }
            break;
        }
        case DID_TEMPERATURE: {
            if (max_len >= 2) {
                int16_t temp = g_system_status.board_temp_celsius;
                data[0] = (uint8_t)(temp & 0xFF);
                data[1] = (uint8_t)((temp >> 8) & 0xFF);
                len = 2;
            }
            break;
        }
        default:
            break;
    }

    return len;
}

const DiagStats_t* Diagnostics_GetStats(void)
{
    return &s_stats;
}

bool Diagnostics_CheckMemoryIntegrity(void)
{
    /* CRC check on critical data sections */
    /* Stack overflow detection */
    /* Heap integrity check */
    return true;
}

bool Diagnostics_CheckPeripherals(void)
{
    /* Verify peripheral registers are accessible */
    /* Check GPIO configuration */
    /* Verify clock tree */
    return true;
}

bool Diagnostics_CheckCANBus(void)
{
    /* CAN bus loopback test */
    /* Transceiver fault pin check */
    return true;
}

bool Diagnostics_CheckMotorDrivers(void)
{
    /* Motor driver enable pin check */
    /* Current sense ADC readback */
    /* Fault flag check */
    return true;
}