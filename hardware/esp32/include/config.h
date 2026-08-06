/**
 * @file config.h
 * @brief Global configuration for the ESP32 platform of the Autonomous
 *        Vehicle Control System.
 * @author AVC team
 * @date 2024
 * @license MIT
 *
 * Centralises all compile-time constants: pin maps, WiFi/MQTT defaults,
 * timing intervals, watchdog thresholds and debug flags. Per-variant pin
 * maps are selected via `CONFIG_IDF_TARGET_*` macros.
 */
#ifndef AVC_ESP32_CONFIG_H
#define AVC_ESP32_CONFIG_H

#include <Arduino.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
//  Build / target identification
// ---------------------------------------------------------------------------
#if defined(CONFIG_IDF_TARGET_ESP32S3)
  #define AVC_TARGET_S3  1
#elif defined(CONFIG_IDF_TARGET_ESP32C3)
  #define AVC_TARGET_C3  1
#elif defined(CONFIG_IDF_TARGET_ESP32) || !defined(CONFIG_IDF_TARGET_ESP32S3)
  #define AVC_TARGET_ESP32 1
#endif

// ---------------------------------------------------------------------------
//  Firmware version
// ---------------------------------------------------------------------------
constexpr const char* AVC_FW_VERSION   = "1.4.2";
constexpr const char* AVC_FW_GIT_HASH  = "deadbeef";
constexpr const char* AVC_NODE_NAME    = "avc-esp32";

// ---------------------------------------------------------------------------
//  I2C bus
// ---------------------------------------------------------------------------
#if defined(AVC_TARGET_S3)
  constexpr int I2C_SDA_PIN    = 8;
  constexpr int I2C_SCL_PIN    = 9;
#elif defined(AVC_TARGET_C3)
  constexpr int I2C_SDA_PIN    = 4;
  constexpr int I2C_SCL_PIN    = 5;
#else
  constexpr int I2C_SDA_PIN    = 21;
  constexpr int I2C_SCL_PIN    = 22;
#endif
constexpr uint32_t I2C_FREQ_HZ     = 400000;   // 400 kHz fast-mode
constexpr uint8_t  BNO055_I2C_ADDR = 0x28;
constexpr uint8_t  BMP280_I2C_ADDR = 0x76;
constexpr uint8_t  VL53L0X_I2C_ADDR= 0x29;

// ---------------------------------------------------------------------------
//  Motor driver (TB6612FNG)
// ---------------------------------------------------------------------------
#if defined(AVC_TARGET_S3)
  constexpr int MOTOR_PWMA_PIN  = 4;
  constexpr int MOTOR_PWMB_PIN  = 5;
  constexpr int MOTOR_AIN1_PIN  = 6;
  constexpr int MOTOR_AIN2_PIN  = 7;
  constexpr int MOTOR_BIN1_PIN  = 17;
  constexpr int MOTOR_BIN2_PIN  = 18;
#elif defined(AVC_TARGET_C3)
  constexpr int MOTOR_PWMA_PIN  = 1;
  constexpr int MOTOR_PWMB_PIN  = 2;
  constexpr int MOTOR_AIN1_PIN  = 3;
  constexpr int MOTOR_AIN2_PIN  = 8;
  constexpr int MOTOR_BIN1_PIN  = 9;
  constexpr int MOTOR_BIN2_PIN  = 10;
#else
  constexpr int MOTOR_PWMA_PIN  = 25;
  constexpr int MOTOR_PWMB_PIN  = 26;
  constexpr int MOTOR_AIN1_PIN  = 27;
  constexpr int MOTOR_AIN2_PIN  = 14;
  constexpr int MOTOR_BIN1_PIN  = 12;
  constexpr int MOTOR_BIN2_PIN  = 13;
#endif
constexpr int MOTOR_STBY_PIN     = 15;
constexpr uint32_t MOTOR_PWM_FREQ = 20000;    // 20 kHz (inaudible)
constexpr uint8_t  MOTOR_PWM_BITS = 10;        // 0..1023 duty
constexpr uint16_t MOTOR_PWM_MAX  = (1 << MOTOR_PWM_BITS) - 1;
constexpr float    MOTOR_RAMP_MS  = 80.0f;     // 0→full in 80 ms

// ---------------------------------------------------------------------------
//  Quadrature encoders (PCNT)
// ---------------------------------------------------------------------------
#if defined(AVC_TARGET_S3)
  constexpr int ENCODER_L_A_PIN = 36;
  constexpr int ENCODER_L_B_PIN = 37;
  constexpr int ENCODER_R_A_PIN = 38;
  constexpr int ENCODER_R_B_PIN = 39;
#elif defined(AVC_TARGET_C3)
  constexpr int ENCODER_L_A_PIN = 18;
  constexpr int ENCODER_L_B_PIN = 19;
  constexpr int ENCODER_R_A_PIN = 20;
  constexpr int ENCODER_R_B_PIN = 21;
#else
  constexpr int ENCODER_L_A_PIN = 32;
  constexpr int ENCODER_L_B_PIN = 33;
  constexpr int ENCODER_R_A_PIN = 34;
  constexpr int ENCODER_R_B_PIN = 35;
#endif
constexpr uint16_t ENCODER_PPR       = 1320;   // pulses-per-rev (CPR/4)
constexpr float    WHEEL_DIAMETER_M  = 0.065f;
constexpr float    WHEEL_BASE_M      = 0.135f;
constexpr float    GEAR_RATIO        = 29.0f;  // motor-shaft → wheel

// ---------------------------------------------------------------------------
//  Battery monitor (ADC + divider)
// ---------------------------------------------------------------------------
#if defined(AVC_TARGET_S3)
  constexpr int BAT_ADC_PIN         = 1;       // ADC1_CH0
#elif defined(AVC_TARGET_C3)
  constexpr int BAT_ADC_PIN         = 0;
#else
  constexpr int BAT_ADC_PIN         = 36;       // ADC1_CH0 (VP)
#endif
constexpr float    BAT_DIVIDER_RATIO = 11.0f;   // 100k + 10k → /11
constexpr float    BAT_LIPO_FULL_V   = 4.20f;   // per-cell
constexpr float    BAT_LIPO_EMPTY_V  = 3.20f;   // per-cell
constexpr uint8_t  BAT_CELLS         = 2;       // 2S LiPo
constexpr float    BAT_LOW_PCT       = 20.0f;   // warn below 20 %
constexpr float    BAT_CRIT_PCT      = 8.0f;    // shutdown below 8 %

// ---------------------------------------------------------------------------
//  Status LED
// ---------------------------------------------------------------------------
#if defined(AVC_TARGET_S3)
  constexpr int STATUS_LED_PIN = 48;
#elif defined(AVC_TARGET_C3)
  constexpr int STATUS_LED_PIN = 8;
#else
  constexpr int STATUS_LED_PIN = 2;
#endif

// ---------------------------------------------------------------------------
//  WiFi defaults (placeholders — overridden by NVS / config portal)
// ---------------------------------------------------------------------------
constexpr const char* WIFI_DEFAULT_SSID = "AVC-Setup";
constexpr const char* WIFI_DEFAULT_PASS = "configure-me";
constexpr const char* WIFI_AP_SSID_FMT = "AVC-Setup-%06X";
constexpr const char* WIFI_HOSTNAME     = "avc-esp32";
constexpr uint16_t WIFI_AP_PORTAL_PORT = 80;
constexpr uint32_t WIFI_CONNECT_TIMEOUT_MS = 10000;
constexpr uint32_t WIFI_RECONNECT_INTERVAL_MS = 5000;

// ---------------------------------------------------------------------------
//  MQTT
// ---------------------------------------------------------------------------
constexpr const char* MQTT_DEFAULT_BROKER  = "mqtt://192.168.1.10";
constexpr uint16_t    MQTT_DEFAULT_PORT     = 1883;
constexpr const char* MQTT_DEFAULT_USER     = "avc";
constexpr const char* MQTT_DEFAULT_PASS     = "avc-secret";
constexpr const char* MQTT_TOPIC_TELEMETRY  = "avc/%s/state/telemetry";
constexpr const char* MQTT_TOPIC_BATTERY    = "avc/%s/state/battery";
constexpr const char* MQTT_TOPIC_STATUS     = "avc/%s/state/status";
constexpr const char* MQTT_TOPIC_CMD_VEL    = "avc/%s/cmd/velocity";
constexpr const char* MQTT_TOPIC_CMD_OTA    = "avc/%s/cmd/ota";
constexpr const char* MQTT_TOPIC_CMD_CFG    = "avc/%s/cmd/config";
constexpr uint32_t    MQTT_KEEPALIVE_SEC    = 30;
constexpr uint16_t    MQTT_MAX_PACKET       = 2048;
constexpr uint32_t    MQTT_RECONNECT_MS     = 2000;

// ---------------------------------------------------------------------------
//  FreeRTOS task parameters
// ---------------------------------------------------------------------------
constexpr UBaseType_t TASK_PRIO_SENSOR    = 5;   // highest — 100 Hz
constexpr UBaseType_t TASK_PRIO_MOTOR     = 4;
constexpr UBaseType_t TASK_PRIO_TELEMETRY = 3;
constexpr UBaseType_t TASK_PRIO_WATCHDOG  = 2;
constexpr UBaseType_t TASK_PRIO_NETWORK   = 1;

constexpr uint32_t STACK_SENSOR    = 4096;
constexpr uint32_t STACK_MOTOR     = 4096;
constexpr uint32_t STACK_TELEMETRY = 6144;
constexpr uint32_t STACK_WATCHDOG  = 2048;
constexpr uint32_t STACK_NETWORK   = 8192;

constexpr BaseType_t CORE_SENSOR  = 1;   // sensor task pinned to core 1
constexpr BaseType_t CORE_MOTOR   = 1;
constexpr BaseType_t CORE_TELEM   = 0;   // WiFi/MQTT live on core 0
constexpr BaseType_t CORE_NET     = 0;

constexpr uint32_t TASK_SENSOR_HZ    = 100;
constexpr uint32_t TASK_MOTOR_HZ     = 50;
constexpr uint32_t TASK_TELEMETRY_HZ = 10;
constexpr uint32_t TASK_WATCHDOG_HZ  = 1;

// ---------------------------------------------------------------------------
//  NVS namespace & keys
// ---------------------------------------------------------------------------
constexpr const char* NVS_NAMESPACE   = "avc";
constexpr const char* NVS_KEY_WIFI_SSID = "wifi_ssid";
constexpr const char* NVS_KEY_WIFI_PASS = "wifi_pass";
constexpr const char* NVS_KEY_MQTT_HOST = "mqtt_host";
constexpr const char* NVS_KEY_MQTT_USER = "mqtt_user";
constexpr const char* NVS_KEY_MQTT_PASS = "mqtt_pass";
constexpr const char* NVS_KEY_NODE_ID   = "node_id";

// ---------------------------------------------------------------------------
//  Debug / logging
// ---------------------------------------------------------------------------
#ifndef AVC_LOG_LEVEL
  #define AVC_LOG_LEVEL 3   // 0=None 1=Error 2=Warn 3=Info 4=Debug 5=Verbose
#endif

#define AVC_LOGE(tag, fmt, ...) if (AVC_LOG_LEVEL>=1) log_e("[%s] " fmt, tag, ##__VA_ARGS__)
#define AVC_LOGW(tag, fmt, ...) if (AVC_LOG_LEVEL>=2) log_w("[%s] " fmt, tag, ##__VA_ARGS__)
#define AVC_LOGI(tag, fmt, ...) if (AVC_LOG_LEVEL>=3) log_i("[%s] " fmt, tag, ##__VA_ARGS__)
#define AVC_LOGD(tag, fmt, ...) if (AVC_LOG_LEVEL>=4) log_d("[%s] " fmt, tag, ##__VA_ARGS__)
#define AVC_LOGV(tag, fmt, ...) if (AVC_LOG_LEVEL>=5) log_v("[%s] " fmt, tag, ##__VA_ARGS__)

#endif // AVC_ESP32_CONFIG_H
