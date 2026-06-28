/**
 * @file watchdog.h
 * @brief Watchdog timer management header for AVCS
 * @version 1.0.0
 * @date 2026-06-27
 */

#ifndef WATCHDOG_H
#define WATCHDOG_H

#ifdef __cplusplus
extern "C" {
#endif

#include "main.h"

/* ========================================================================
 * Watchdog Types
 * ======================================================================== */
typedef enum {
    WDG_TYPE_INDEPENDENT = 0,  /**< Independent Watchdog (IWDG) - LS oscillator */
    WDG_TYPE_WINDOW      = 1   /**< Window Watchdog (WWDG) - APB1 clock */
} WatchdogType_t;

/* ========================================================================
 * Watchdog Configuration
 * ======================================================================== */
typedef struct {
    WatchdogType_t type;           /**< Watchdog type */
    uint32_t       timeout_ms;     /**< Timeout in milliseconds */
    bool           enable_early_warning; /**< Enable early warning interrupt */
    uint8_t        window_percent; /**< Window percentage (WWDG only, 0-100) */
} WatchdogConfig_t;

/* ========================================================================
 * Watchdog Status
 * ======================================================================== */
typedef struct {
    bool     iwdg_active;       /**< IWDG is running */
    bool     wwdg_active;       /**< WWDG is running */
    uint32_t iwdg_timeout_ms;   /**< IWDG configured timeout */
    uint32_t wwdg_timeout_ms;   /**< WWDG configured timeout */
    uint32_t last_refresh_ms;   /**< Last watchdog refresh timestamp */
    uint32_t refresh_count;     /**< Total number of refreshes */
    bool     early_warning;     /**< Early warning flag */
} WatchdogStatus_t;

/* ========================================================================
 * Function Prototypes
 * ======================================================================== */

/**
 * @brief Initialize watchdog timer(s)
 * @param timeout_ms Watchdog timeout in milliseconds
 * @return AVCS_OK on success
 */
AvcsErrorCode_t Watchdog_Init(uint32_t timeout_ms);

/**
 * @brief Initialize Independent Watchdog (IWDG)
 * @param timeout_ms Timeout in milliseconds
 * @return AVCS_OK on success
 */
AvcsErrorCode_t Watchdog_InitIWDG(uint32_t timeout_ms);

/**
 * @brief Initialize Window Watchdog (WWDG)
 * @param timeout_ms Timeout in milliseconds
 * @param window_percent Window open percentage (0-100)
 * @return AVCS_OK on success
 */
AvcsErrorCode_t Watchdog_InitWWDG(uint32_t timeout_ms, uint8_t window_percent);

/**
 * @brief Refresh (kick) the watchdog timer
 * @note Must be called periodically before timeout expires
 */
void Watchdog_Refresh(void);

/**
 * @brief Refresh only the IWDG
 */
void Watchdog_RefreshIWDG(void);

/**
 * @brief Refresh only the WWDG
 */
void Watchdog_RefreshWWDG(void);

/**
 * @brief Get watchdog status
 * @return Pointer to watchdog status
 */
const WatchdogStatus_t* Watchdog_GetStatus(void);

/**
 * @brief Check if watchdog reset was the cause of last reset
 * @return true if reset was caused by watchdog
 */
bool Watchdog_WasResetByWatchdog(void);

/**
 * @brief Early warning interrupt handler
 * @note Called from WWDG interrupt context
 */
void Watchdog_EarlyWarningHandler(void);

/**
 * @brief Disable watchdog (for debug/recovery only)
 * @warning Cannot disable IWDG once started
 * @return AVCS_OK on success
 */
AvcsErrorCode_t Watchdog_Disable(void);

#ifdef __cplusplus
}
#endif

#endif /* WATCHDOG_H */