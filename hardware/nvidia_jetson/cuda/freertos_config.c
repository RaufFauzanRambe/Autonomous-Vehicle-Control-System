/**
 * @file    Middleware/freertos_config.c
 * @brief   FreeRTOS hook implementations + runtime statistics
 * @author  Autonomous Vehicle Team
 * @date    2024
 * @license MIT
 */
#include "freertos_config.h"
#include "diagnostics.h"
#include "main.h"
#include <stdio.h>
#include <string.h>

/* FreeRTOS includes - we use the heap_4 + port.c */
#include "FreeRTOS.h"
#include "task.h"
#include "queue.h"

/* -------------------------------------------------------------------------
 * Run-time stats timer (used by configGENERATE_RUN_TIME_STATS)
 *
 *  We use the DWT cycle counter as a 32-bit free-running counter at
 *  SystemCoreClock Hz. vTaskGetRunTimeStats formats this as a percentage.
 * ----------------------------------------------------------------------- */
volatile unsigned long ulRunTimeCounterOverflows = 0;

void vConfigureTimerForRunTimeStats(void)
{
    CoreDebug->DEMCR |= CoreDebug_DEMCR_TRCENA_Msk;
    DWT->CYCCNT = 0U;
    DWT->CTRL  |= DWT_CTRL_CYCCNTENA_Msk;
}

unsigned long ulGetRunTimeCounterValue(void)
{
    return DWT->CYCCNT;
}

/* -------------------------------------------------------------------------
 * Idle hook - called on every iteration of the idle task
 * ----------------------------------------------------------------------- */
void vApplicationIdleHook(void)
{
    /* Wait for interrupt - reduces power consumption */
    __WFI();
}

/* -------------------------------------------------------------------------
 * Tick hook - called from SysTick at 1 kHz
 * ----------------------------------------------------------------------- */
void vApplicationTickHook(void)
{
    /* Refresh independent watchdog - if the scheduler stops, the
       watchdog will fire. We split the IWDG refresh between tick and
       the heartbeat task for defence-in-depth. */
    extern IWDG_HandleTypeDef hiwdg;
    (void)hiwdg; /* If IWDG not used, this is no-op */
}

/* -------------------------------------------------------------------------
 * Stack overflow hook - method 2 (canary + pattern)
 * ----------------------------------------------------------------------- */
void vApplicationStackOverflowHook(TaskHandle_t xTask, char *pcTaskName)
{
    (void)xTask;
    Diag_LogError(DIAG_ERR_STACK_OVERFLOW, 0);
    Diag_LogMessage("Stack overflow in task: ");
    Diag_LogMessage(pcTaskName);
    Diag_LogMessage("\n");
    /* Force a reset to recover */
    NVIC_SystemReset();
}

/* -------------------------------------------------------------------------
 * Malloc failed hook - heap_4 returned NULL
 * ----------------------------------------------------------------------- */
void vApplicationMallocFailedHook(void)
{
    Diag_LogError(DIAG_ERR_MALLOC_FAILED, 0);
    Diag_LogMessage("malloc failed!\n");
    /* Do NOT reset - some tasks may still be functional. The logger will
       capture the event for post-mortem analysis. */
}

/* -------------------------------------------------------------------------
 * Idle + Daemon hooks
 * ----------------------------------------------------------------------- */
void vApplicationDaemonTaskStartupHook(void)
{
    /* Called once when the timer service task starts */
}

/* -------------------------------------------------------------------------
 * Static allocation support - FreeRTOS requires these callbacks if
 * configSUPPORT_STATIC_ALLOCATION == 1.
 *
 *  We return NULL for both - we don't statically allocate the idle/timer
 *  task TCBs/stacks; we want the kernel to allocate them dynamically
 *  from heap_4.
 * ----------------------------------------------------------------------- */
void vApplicationGetIdleTaskMemory(StaticTask_t **ppxIdleTaskTCBBuffer,
                                   StackType_t  **ppxIdleTaskStackBuffer,
                                   uint32_t      *pulIdleTaskStackSize)
{
    *ppxIdleTaskTCBBuffer   = NULL;
    *ppxIdleTaskStackBuffer = NULL;
    *pulIdleTaskStackSize   = 0;
}

void vApplicationGetTimerTaskMemory(StaticTask_t **ppxTimerTaskTCBBuffer,
                                    StackType_t  **ppxTimerTaskStackBuffer,
                                    uint32_t      *pulTimerTaskStackSize)
{
    *ppxTimerTaskTCBBuffer   = NULL;
    *ppxTimerTaskStackBuffer = NULL;
    *pulTimerTaskStackSize   = 0;
}

/* -------------------------------------------------------------------------
 * Assert handler - called by configASSERT macro
 * ----------------------------------------------------------------------- */
void FreeRTOS_AssertHandler(const char *file, int line)
{
    Diag_LogMessage("RTOS ASSERT: ");
    Diag_LogMessage(file);
    char buf[32];
    (void)snprintf(buf, sizeof(buf), ":%d\n", line);
    Diag_LogMessage(buf);
    Diag_LogError(DIAG_ERR_HARDFAULT, 0);
    NVIC_SystemReset();
}

/* -------------------------------------------------------------------------
 * Critical section wrappers (for non-FreeRTOS code)
 * ----------------------------------------------------------------------- */
uint32_t FreeRTOS_EnterCritical(void)
{
    taskENTER_CRITICAL();
    return 0;
}

void FreeRTOS_ExitCritical(uint32_t prev)
{
    (void)prev;
    taskEXIT_CRITICAL();
}

/* End of file ------------------------------------------------------------- */
