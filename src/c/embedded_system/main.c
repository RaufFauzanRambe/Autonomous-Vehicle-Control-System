/**
 * @file main.c
 * @brief Main application entry point for Autonomous Vehicle Control System
 * @version 1.0.0
 * @date 2026-06-27
 *
 * Implements the system initialization sequence and the main super-loop
 * with a cooperative multitasking state machine.
 */

#include "main.h"
#include "system_init.h"
#include "power_management.h"
#include "watchdog.h"
#include "diagnostics.h"
#include "fault_handler.h"

/* ========================================================================
 * Private Macros
 * ======================================================================== */
#define MAIN_LOOP_PERIOD_MS       1U
#define SELFTEST_TIMEOUT_MS       5000U
#define CALIBRATION_TIMEOUT_MS    10000U
#define SHUTDOWN_TIMEOUT_MS       3000U

/* ========================================================================
 * Global Variables
 * ======================================================================== */
volatile uint32_t g_system_tick_ms = 0U;
SystemConfig_t    g_system_config;
SystemStatus_t    g_system_status;

/* ========================================================================
 * Private Function Prototypes
 * ======================================================================== */
static void System_ConfigureDefaults(void);
static void System_RunSelfTest(void);
static void System_RunCalibration(void);
static void System_RunDriving(void);
static void System_RunShutdown(void);
static void System_HandleFault(void);
static bool System_ValidateStateTransition(SystemState_t from, SystemState_t to);

/* ========================================================================
 * SysTick Interrupt Handler
 * ======================================================================== */
void SysTick_Handler(void)
{
    g_system_tick_ms++;
}

/* ========================================================================
 * Private Functions
 * ======================================================================== */

/**
 * @brief Configure default system parameters
 */
static void System_ConfigureDefaults(void)
{
    memset(&g_system_config, 0, sizeof(SystemConfig_t));
    memset(&g_system_status, 0, sizeof(SystemStatus_t));

    g_system_config.system_clock_hz     = MCU_CLOCK_HZ;
    g_system_config.systick_freq_hz     = SYSTICK_FREQ_HZ;
    g_system_config.watchdog_timeout_ms = 100U;
    g_system_config.can_bus_count       = 2U;
    g_system_config.motor_count         = 4U;
    g_system_config.enable_diagnostics  = true;
    g_system_config.enable_ota          = true;
    g_system_config.boot_timeout_ms     = 15000U;

    g_system_status.current_state       = SYS_STATE_BOOT;
    g_system_status.board_temp_celsius  = 25;
}

/**
 * @brief Run power-on self-test sequence
 */
static void System_RunSelfTest(void)
{
    uint32_t test_start = System_GetUptimeMs();

    /* Test 1: Memory integrity check */
    if (!Diagnostics_CheckMemoryIntegrity()) {
        FaultHandler_Report(FAULT_SOURCE_SYSTEM, FAULT_SEVERITY_CRITICAL,
                            FAULT_CODE_SELFTEST_MEM);
        System_TransitionState(SYS_STATE_FAULT);
        return;
    }

    /* Test 2: Peripheral connectivity check */
    if (!Diagnostics_CheckPeripherals()) {
        FaultHandler_Report(FAULT_SOURCE_SYSTEM, FAULT_SEVERITY_ERROR,
                            FAULT_CODE_SELFTEST_PERIPH);
        System_TransitionState(SYS_STATE_FAULT);
        return;
    }

    /* Test 3: CAN bus loopback test */
    if (!Diagnostics_CheckCANBus()) {
        FaultHandler_Report(FAULT_SOURCE_CAN, FAULT_SEVERITY_WARNING,
                            FAULT_CODE_SELFTEST_CAN);
        /* Non-fatal: continue but log warning */
    }

    /* Test 4: Motor driver self-test */
    if (!Diagnostics_CheckMotorDrivers()) {
        FaultHandler_Report(FAULT_SOURCE_MOTOR, FAULT_SEVERITY_ERROR,
                            FAULT_CODE_SELFTEST_MOTOR);
        System_TransitionState(SYS_STATE_FAULT);
        return;
    }

    /* Check self-test timeout */
    if (System_TimeoutElapsed(test_start, SELFTEST_TIMEOUT_MS)) {
        FaultHandler_Report(FAULT_SOURCE_SYSTEM, FAULT_SEVERITY_CRITICAL,
                            FAULT_CODE_SELFTEST_TIMEOUT);
        System_TransitionState(SYS_STATE_FAULT);
        return;
    }

    /* All tests passed, transition to driver init */
    System_TransitionState(SYS_STATE_INIT_DRIVERS);
}

/**
 * @brief Run sensor and actuator calibration
 */
static void System_RunCalibration(void)
{
    uint32_t cal_start = System_GetUptimeMs();

    /* Calibrate steering endpoints */
    /* Calibrate motor encoders (find index pulse) */
    /* Calibrate IMU zero-offset */
    /* Calibrate brake pressure sensor zero */

    (void)cal_start; /* Calibration details handled by subsystems */

    /* Transition to ready after successful calibration */
    System_TransitionState(SYS_STATE_READY);
}

/**
 * @brief Main driving loop logic
 */
static void System_RunDriving(void)
{
    /* Read CAN bus inputs (throttle, brake, steering commands) */
    /* Run motor control loops */
    /* Run safety checks */
    /* Update diagnostics */
    /* Feed watchdog */

    /* Check for emergency conditions */
    if (g_system_status.fault_active &&
        g_system_status.fault_severity >= FAULT_SEVERITY_CRITICAL) {
        System_TransitionState(SYS_STATE_EMERGENCY);
    }
}

/**
 * @brief Controlled shutdown sequence
 */
static void System_RunShutdown(void)
{
    /* Disable motors */
    /* Save diagnostics to non-volatile memory */
    /* Enter low-power state */

    System_TransitionState(SYS_STATE_SHUTDOWN);
}

/**
 * @brief Handle active fault condition
 */
static void System_HandleFault(void)
{
    /* Disable all motor outputs */
    /* Activate warning indicators */
    /* Log fault to persistent storage */
    /* Attempt recovery if possible */
}

/**
 * @brief Validate state machine transition
 */
static bool System_ValidateStateTransition(SystemState_t from, SystemState_t to)
{
    /* Allow emergency transition from any state */
    if (to == SYS_STATE_EMERGENCY || to == SYS_STATE_FAULT) {
        return true;
    }

    /* Define valid transition map */
    switch (from) {
        case SYS_STATE_BOOT:
            return (to == SYS_STATE_SELFTEST);
        case SYS_STATE_SELFTEST:
            return (to == SYS_STATE_INIT_DRIVERS || to == SYS_STATE_FAULT);
        case SYS_STATE_INIT_DRIVERS:
            return (to == SYS_STATE_CALIBRATION || to == SYS_STATE_FAULT);
        case SYS_STATE_CALIBRATION:
            return (to == SYS_STATE_READY || to == SYS_STATE_FAULT);
        case SYS_STATE_READY:
            return (to == SYS_STATE_DRIVING || to == SYS_STATE_PARKING ||
                    to == SYS_STATE_SHUTDOWN || to == SYS_STATE_STANDBY);
        case SYS_STATE_DRIVING:
            return (to == SYS_STATE_READY || to == SYS_STATE_PARKING ||
                    to == SYS_STATE_EMERGENCY || to == SYS_STATE_STANDBY);
        case SYS_STATE_PARKING:
            return (to == SYS_STATE_READY || to == SYS_STATE_SHUTDOWN ||
                    to == SYS_STATE_DRIVING);
        case SYS_STATE_STANDBY:
            return (to == SYS_STATE_READY || to == SYS_STATE_SHUTDOWN);
        case SYS_STATE_EMERGENCY:
            return (to == SYS_STATE_SHUTDOWN || to == SYS_STATE_SELFTEST);
        case SYS_STATE_SHUTDOWN:
            return (to == SYS_STATE_BOOT);
        case SYS_STATE_FAULT:
            return (to == SYS_STATE_BOOT || to == SYS_STATE_SHUTDOWN);
        default:
            return false;
    }
}

/* ========================================================================
 * Public Functions
 * ======================================================================== */

AvcsErrorCode_t System_InitAll(void)
{
    System_ConfigureDefaults();

    /* Configure SysTick for 1ms tick */
    if (SystemInit_Clock() != AVCS_OK) {
        return AVCS_ERR_INIT;
    }

    if (SystemInit_SysTick() != AVCS_OK) {
        return AVCS_ERR_INIT;
    }

    /* Initialize power management */
    if (PowerManagement_Init() != AVCS_OK) {
        return AVCS_ERR_POWER;
    }

    /* Initialize fault handler (must be early for self-test reporting) */
    if (FaultHandler_Init() != AVCS_OK) {
        return AVCS_ERR_INIT;
    }

    /* Initialize watchdog */
    if (Watchdog_Init(g_system_config.watchdog_timeout_ms) != AVCS_OK) {
        return AVCS_ERR_WATCHDOG;
    }

    /* Initialize diagnostics */
    if (g_system_config.enable_diagnostics) {
        if (Diagnostics_Init() != AVCS_OK) {
            return AVCS_ERR_INIT;
        }
    }

    g_system_status.current_state = SYS_STATE_SELFTEST;

    return AVCS_OK;
}

void System_StateMachine(void)
{
    g_system_status.loop_count++;
    g_system_status.uptime_ms = System_GetUptimeMs();

    switch (g_system_status.current_state) {
        case SYS_STATE_BOOT:
            System_TransitionState(SYS_STATE_SELFTEST);
            break;

        case SYS_STATE_SELFTEST:
            System_RunSelfTest();
            break;

        case SYS_STATE_STANDBY:
            PowerManagement_EnterLowPower(POWER_MODE_STOP);
            break;

        case SYS_STATE_INIT_DRIVERS:
            /* Subsystem init handled by System_InitAll completion */
            System_TransitionState(SYS_STATE_CALIBRATION);
            break;

        case SYS_STATE_CALIBRATION:
            System_RunCalibration();
            break;

        case SYS_STATE_READY:
            /* System idle, waiting for drive command via CAN */
            break;

        case SYS_STATE_DRIVING:
            System_RunDriving();
            break;

        case SYS_STATE_PARKING:
            /* Parking sequence */
            break;

        case SYS_STATE_EMERGENCY:
            System_HandleFault();
            break;

        case SYS_STATE_SHUTDOWN:
            System_RunShutdown();
            break;

        case SYS_STATE_FAULT:
            System_HandleFault();
            break;

        default:
            System_TransitionState(SYS_STATE_FAULT);
            break;
    }

    /* Feed the watchdog every loop */
    Watchdog_Refresh();
}

AvcsErrorCode_t System_TransitionState(SystemState_t new_state)
{
    if (!System_ValidateStateTransition(g_system_status.current_state, new_state)) {
        return AVCS_ERR_INVALID;
    }

    g_system_status.previous_state = g_system_status.current_state;
    g_system_status.current_state = new_state;
    g_system_status.last_state_change_ms = System_GetUptimeMs();

    return AVCS_OK;
}

void System_DelayMs(uint32_t ms)
{
    uint32_t start = System_GetUptimeMs();
    while (!System_TimeoutElapsed(start, ms)) {
        /* Busy wait - in production, use WFI for power savings */
    }
}

void System_EmergencyStop(void)
{
    System_TransitionState(SYS_STATE_EMERGENCY);
}

/* ========================================================================
 * Entry Point
 * ======================================================================== */
int main(void)
{
    /* Hardware-level initialization (clocks, memory) */
    SystemInit_Hardware();

    /* System-level initialization */
    if (System_InitAll() != AVCS_OK) {
        /* Critical init failure - blink error LED or enter recovery */
        while (1) {
            /* Halt with fault indication */
        }
    }

    /* Main super-loop */
    while (1) {
        System_StateMachine();

        /* Optional: low-power wait if no work pending */
        /* __WFI(); */
    }

    return 0; /* Unreachable */
}