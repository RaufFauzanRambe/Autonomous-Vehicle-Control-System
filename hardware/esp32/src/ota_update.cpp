/**
 * @file ota_update.cpp
 * @brief OtaUpdate — HTTP(S) OTA via esp_https_ota with rollback support.
 * @author AVC team
 * @date 2024
 * @license MIT
 *
 * Exposes C-callable entry points (ota_init, ota_check_pending, ota_apply)
 * consumed by main.cpp.
 *
 * Partition scheme (factory/app0/app1):
 *   1. New firmware is streamed into the *inactive* OTA slot by the
 *      `esp_https_ota` API.
 *   2. Boot partition is switched and the device reboots.
 *   3. On next boot, `ota_init()` detects we're in a pending-verify state.
 *      It waits 30 s for the application to call `ota_confirm()`. If the
 *      application fails to do so (e.g. it crashed or failed WiFi), the
 *      bootloader auto-rolls back to the previous slot.
 */

#include <Arduino.h>
#include <esp_https_ota.h>
#include <esp_ota_ops.h>
#include <esp_partition.h>
#include <Preferences.h>
#include "config.h"

// Forward declaration from main.cpp — set when a new OTA is in flight.
extern bool g_ota_in_progress;  // defined weakly below if main doesn't define it
__attribute__((weak)) bool g_ota_in_progress = false;

static Preferences g_nvs;
static bool        g_pendingConfirm = false;
static uint32_t    g_bootIntoOtaSlot = 0;
static std::function<void(int,size_t,size_t)> g_progressCb;

// ---------------------------------------------------------------------------
//  Internal helpers
// ---------------------------------------------------------------------------
static bool isOtaBootPending() {
    const esp_partition_t* running = esp_ota_get_running_partition();
    esp_ota_img_states_t state;
    if (esp_ota_get_state_partition(running, &state) != ESP_OK) return false;
    return state == ESP_OTA_IMG_PENDING_VERIFY;
}

static void logPartitionInfo() {
    const esp_partition_t* r = esp_ota_get_running_partition();
    AVC_LOGI("ota", "running partition '%s' @ 0x%08X size=%u",
             r->label, (unsigned)r->address, (unsigned)r->size);
    const esp_partition_t* next = esp_ota_get_next_update_partition(nullptr);
    if (next) {
        AVC_LOGI("ota", "next update partition '%s' @ 0x%08X",
                 next->label, (unsigned)next->address);
    }
}

// ---------------------------------------------------------------------------
//  Public C-style API
// ---------------------------------------------------------------------------
extern "C" void ota_init() {
    g_nvs.begin(NVS_NAMESPACE, false);
    logPartitionInfo();

    if (isOtaBootPending()) {
        g_pendingConfirm = true;
        AVC_LOGI("ota", "booted into PENDING_VERIFY slot — awaiting confirm()");

        // Schedule a 30 s watchdog: if confirm() is not called, roll back.
        // We use a FreeRTOS one-shot timer.
        // (esp_https_ota also relies on CONFIG_APP_ROLLBACK_ENABLE=y in sdkconfig.)
        xTaskCreatePinnedToCore(
            [](void* arg) {
                vTaskDelay(pdMS_TO_TICKS(30000));
                if (g_pendingConfirm) {
                    AVC_LOGE("ota", "confirm() not received in 30 s — rolling back");
                    esp_ota_mark_app_invalid_rollback_and_reboot();
                }
                vTaskDelete(nullptr);
            },
            "ota_wdt", 4096, nullptr, 1, nullptr, 0);
    } else {
        AVC_LOGI("ota", "normal boot — no pending OTA verification");
    }
}

extern "C" bool ota_check_pending() {
    return g_pendingConfirm;
}

/// Call this once WiFi + MQTT are up and the device is healthy, to mark the
/// newly-flashed image as good and prevent rollback.
extern "C" void ota_confirm() {
    if (g_pendingConfirm) {
        if (esp_ota_mark_app_valid_cancel_rollback() == ESP_OK) {
            AVC_LOGI("ota", "firmware marked valid — rollback cancelled");
            g_pendingConfirm = false;
        } else {
            AVC_LOGE("ota", "failed to mark firmware valid");
        }
    }
}

extern "C" bool ota_apply(const String& url) {
    if (url.length() == 0) {
        AVC_LOGE("ota", "ota_apply called with empty URL");
        return false;
    }
    AVC_LOGI("ota", "starting OTA from %s", url.c_str());

    esp_http_client_config_t cfg = {};
    cfg.url = url.c_str();
    cfg.timeout_ms = 30000;
    cfg.keep_alive_enable = true;
    // For HTTPS without a known root CA, allow insecure in dev.
    // Production: ship a CA bundle and set cfg.crt_bundle_attach.
    cfg.skip_cert_common_name_check = true;

    esp_https_ota_config_t ota_cfg = {};
    ota_cfg.http_config = &cfg;
    ota_cfg.partial_http_download = true;
    ota_cfg.max_http_request_size = 4096;

    // Progress callback (called from the esp_https_ota task).
    // We can't easily get byte counts from esp_https_ota; instead, we publish
    // periodic state via the global flag.
    g_ota_in_progress = true;

    esp_err_t ret = esp_https_ota(&ota_cfg);
    g_ota_in_progress = false;

    if (ret == ESP_OK) {
        AVC_LOGI("ota", "downloaded OK — rebooting in 2 s");
        if (g_progressCb) g_progressCb(100, 0, 0);
        delay(2000);
        esp_restart();
        return true;   // unreachable
    } else {
        AVC_LOGE("ota", "OTA failed: %s", esp_err_to_name(ret));
        return false;
    }
}

// Optional C++ class wrapper for users who prefer OO style.
class OtaUpdate {
public:
    void setCallback(std::function<void(int,size_t,size_t)> cb) {
        g_progressCb = std::move(cb);
    }
    bool start(const String& url) { return ota_apply(url); }
    bool pending() const          { return ota_check_pending(); }
    void confirm() const          { ota_confirm(); }
};
