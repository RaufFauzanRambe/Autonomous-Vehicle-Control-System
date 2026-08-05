/**
 * @file battery_monitor.cpp
 * @brief BatteryMonitor — ADC read, voltage divider, LiPo SoC, cycles.
 * @author AVC team
 * @date 2024
 * @license MIT
 *
 * Exposes C-callable entry points (battery_init, battery_read_voltage,
 * battery_read_pct) consumed by main.cpp.
 */

#include <Arduino.h>
#include <esp_adc_cal.h>
#include <Preferences.h>
#include "config.h"

// ---------------------------------------------------------------------------
//  LiPo SoC lookup table (open-circuit voltage vs capacity %).
//  Values per cell — applied to BAT_LIPO_FULL_V..BAT_LIPO_EMPTY_V.
// ---------------------------------------------------------------------------
struct SocPoint { float v_cell; uint8_t pct; };
static constexpr SocPoint kSocCurve[] = {
    { 4.20f, 100 },
    { 4.10f,  95 },
    { 4.00f,  85 },
    { 3.92f,  75 },
    { 3.85f,  65 },
    { 3.79f,  55 },
    { 3.74f,  45 },
    { 3.70f,  35 },
    { 3.66f,  25 },
    { 3.60f,  15 },
    { 3.50f,   8 },
    { 3.30f,   3 },
    { 3.20f,   0 },
};
static constexpr size_t kSocLen = sizeof(kSocCurve) / sizeof(kSocCurve[0]);

// ---------------------------------------------------------------------------
//  Internal state
// ---------------------------------------------------------------------------
static esp_adc_cal_characteristics_t g_adcChars;
static Preferences       g_nvs;
static uint32_t          g_cycles      = 0;
static uint32_t          g_lastLowWarn = 0;
static float             g_lastPct     = 100.0f;
static float             g_tempC       = 25.0f;       // No NTC by default.
static constexpr const char* kNvsCycles = "bat_cycles";
static constexpr const char* kNvsLastPct = "bat_last_pct";

// Oversampling — average N reads to reduce ADC noise.
static constexpr int kOversample = 16;

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------
static uint32_t readAdcRaw() {
    uint32_t sum = 0;
    for (int i = 0; i < kOversample; ++i) {
        sum += adc1_get_raw(ADC1_CHANNEL_0);
    }
    return sum / kOversample;
}

static float readVoltageV() {
    uint32_t raw = readAdcRaw();
    uint32_t mv = esp_adc_cal_raw_to_voltage(raw, &g_adcChars);
    // Apply divider ratio: actual = measured * (R1+R2)/R2.
    return (mv / 1000.0f) * BAT_DIVIDER_RATIO;
}

static uint8_t voltageToPct(float pack_v) {
    float cell = pack_v / BAT_CELLS;
    if (cell >= BAT_LIPO_FULL_V)  return 100;
    if (cell <= BAT_LIPO_EMPTY_V) return 0;
    for (size_t i = 0; i + 1 < kSocLen; ++i) {
        const SocPoint& a = kSocCurve[i];
        const SocPoint& b = kSocCurve[i + 1];
        if (cell <= a.v_cell && cell >= b.v_cell) {
            // Linear interp.
            float t = (a.v_cell - cell) / (a.v_cell - b.v_cell);
            return (uint8_t)(a.pct + t * (b.pct - a.pct));
        }
    }
    return 0;
}

// ---------------------------------------------------------------------------
//  Cycle counting — detect charge→discharge→charge transitions.
//  A "cycle" is registered when SoC drops below 20 % and recovers above 95 %.
// ---------------------------------------------------------------------------
static void updateCycles(float pct) {
    static enum { kNormal, kDischarged } state = kNormal;
    if (state == kNormal && pct < BAT_LOW_PCT) {
        state = kDischarged;
    } else if (state == kDischarged && pct > 95.0f) {
        state = kNormal;
        g_cycles++;
        g_nvs.putUInt(kNvsCycles, g_cycles);
        AVC_LOGI("bat", "charge cycle #%u recorded", g_cycles);
    }
    g_lastPct = pct;
}

// ---------------------------------------------------------------------------
//  Public C-style API (called from main.cpp)
// ---------------------------------------------------------------------------
extern "C" void battery_init() {
    adc1_config_width(ADC_WIDTH_BIT_12);
    adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_DB_11);
    esp_adc_cal_characterize(ADC_UNIT_1, ADC_ATTEN_DB_11,
                             ADC_WIDTH_BIT_12, 1100, &g_adcChars);

    g_nvs.begin(NVS_NAMESPACE, false);
    g_cycles  = g_nvs.getUInt(kNvsCycles, 0);
    g_lastPct = g_nvs.getFloat(kNvsLastPct, 100.0f);

    float v = readVoltageV();
    AVC_LOGI("bat", "init v=%.2f V cycles=%u last_pct=%.1f",
             v, g_cycles, g_lastPct);
}

extern "C" float battery_read_voltage() {
    return readVoltageV();
}

extern "C" float battery_read_pct() {
    float v = readVoltageV();
    float pct = (float)voltageToPct(v);
    updateCycles(pct);
    g_nvs.putFloat(kNvsLastPct, pct);

    // Low / critical warnings.
    uint32_t now = millis();
    if (pct < BAT_CRIT_PCT) {
        if (now - g_lastLowWarn > 1000) {
            AVC_LOGE("bat", "CRITICAL %.1f%% v=%.2f V — shutting down motors", pct, v);
            g_lastLowWarn = now;
            // Notify system (could publish an MQTT warning here).
        }
    } else if (pct < BAT_LOW_PCT) {
        if (now - g_lastLowWarn > 5000) {
            AVC_LOGW("bat", "LOW %.1f%% v=%.2f V", pct, v);
            g_lastLowWarn = now;
        }
    }
    return pct;
}

// ---------------------------------------------------------------------------
//  Optional: read temperature if NTC is wired to a second ADC channel.
//  Returns NaN if no sensor present.
// ---------------------------------------------------------------------------
extern "C" float battery_read_temperature() {
    return g_tempC;   // Placeholder — wire NTC if needed.
}

// ---------------------------------------------------------------------------
//  Optional: expose cycle count for status MQTT topic.
// ---------------------------------------------------------------------------
extern "C" uint32_t battery_cycles() { return g_cycles; }
