/**
 * @file main.cpp
 * @brief Entry point for the ESP32 platform of the Autonomous Vehicle
 *        Control System.
 * @author AVC team
 * @date 2024
 * @license MIT
 *
 * Wires together:
 *  - WiFiManager   (STA + config portal)
 *  - BluetoothManager (BLE telemetry)
 *  - MqttClient    (telemetry + commands)
 *  - SensorReader  (IMU + baro + ToF + encoders + battery)
 *  - MotorDriver   (TB6612FNG dual H-bridge + PID)
 *  - BatteryMonitor (LiPo SoC)
 *  - OtaUpdate     (signed OTA with rollback)
 *
 * FreeRTOS tasks (all pinned):
 *  - sensorTask    @ 100 Hz  prio 5  core 1
 *  - motorTask     @ 50 Hz   prio 4  core 1
 *  - telemetryTask @ 10 Hz   prio 3  core 0
 *  - watchdogTask  @ 1 Hz    prio 2  core 0
 *  - networkTask   (event-driven) prio 1 core 0
 */

#include <Arduino.h>
#include <Wire.h>
#include <ArduinoJson.h>
#include "config.h"
#include "wifi_manager.h"
#include "mqtt_client.h"
#include "sensor_reader.h"
#include "motor_driver.h"

// ---- C-callable entry points exposed by the sibling .cpp files ----
// bluetooth_manager.cpp / battery_monitor.cpp / ota_update.cpp each expose
// a small C-style API to avoid pulling in their full C++ headers here.
extern "C" void bluetooth_init();
extern "C" void bluetooth_publish_telemetry(const SensorFrame& f);
extern "C" void battery_init();
extern "C" float battery_read_voltage();
extern "C" float battery_read_pct();
extern "C" void ota_init();
extern "C" bool ota_check_pending();
extern "C" bool ota_apply(const String& url);

/// Aggregated "system status" struct shared between tasks.
struct SystemStatus {
    uint32_t   uptime_ms   = 0;
    uint32_t   loop_count  = 0;
    uint32_t   sensor_hz   = 0;
    uint32_t   motor_hz    = 0;
    uint32_t   telem_hz    = 0;
    uint32_t   free_heap   = 0;
    uint8_t    cpu0_load   = 0;
    uint8_t    cpu1_load   = 0;
    bool       wifi_ok     = false;
    bool       mqtt_ok     = false;
    bool       ble_ok      = false;
    bool       ota_pending = false;
    bool       estop       = false;
    char       fw[32]      = {0};
};

// ---------------------------------------------------------------------------
//  Globals shared across tasks (protected by mutex where needed)
// ---------------------------------------------------------------------------
SemaphoreHandle_t g_frameMutex = nullptr;
SensorFrame       g_latestFrame;
SystemStatus      g_status;
std::atomic<bool> g_estop{false};

WiFiManager   g_wifi;
MqttClient    g_mqtt;
SensorReader  g_sensors;
MotorDriver   g_motors;

// ---------------------------------------------------------------------------
//  Command callback (MQTT)
// ---------------------------------------------------------------------------
static void onMqttCommand(const String& topic, const JsonObject& payload) {
    if (topic.endsWith("/cmd/velocity")) {
        float v = payload["linear"]  | 0.0f;
        float w = payload["angular"] | 0.0f;
        if (!g_estop) g_motors.setDiffDrive(v, w);
        AVC_LOGI("main", "cmd/velocity v=%.2f w=%.2f", v, w);
    } else if (topic.endsWith("/cmd/ota")) {
        String url = payload["url"] | "";
        if (url.length() > 0) {
            AVC_LOGI("main", "cmd/ota url=%s", url.c_str());
            g_status.ota_pending = true;
            ota_apply(url);
        }
    } else if (topic.endsWith("/cmd/config")) {
        // Example: {"key":"kp_left","value":0.8}
        String key = payload["key"] | "";
        // Forward to motor PID / sensor gains (left as a stub for runtime tuning).
        AVC_LOGI("main", "cmd/config key=%s", key.c_str());
    } else if (topic.endsWith("/cmd/estop")) {
        bool stop = payload["stop"] | true;
        if (stop) {
            g_estop = true;
            g_motors.estop();
            AVC_LOGW("main", "E-STOP engaged");
        } else {
            g_estop = false;
            g_motors.clearEstop();
            AVC_LOGI("main", "E-STOP cleared");
        }
    }
}

// ---------------------------------------------------------------------------
//  Sensor task — 100 Hz on core 1
// ---------------------------------------------------------------------------
static void sensorTask(void* arg) {
    (void)arg;
    const TickType_t period = pdMS_TO_TICKS(1000 / TASK_SENSOR_HZ);
    TickType_t last = xTaskGetTickCount();
    uint32_t counter = 0;

    // Subscribe to the task watchdog.
    esp_task_wdt_add(NULL);

    while (true) {
        SensorFrame frame;
        if (g_sensors.read(frame)) {
            // Fill battery voltage (read by battery task elsewhere).
            frame.battery_v   = battery_read_voltage();
            frame.battery_pct = battery_read_pct();

            xSemaphoreTake(g_frameMutex, portMAX_DELAY);
            g_latestFrame = frame;
            xSemaphoreGive(g_frameMutex);

            g_status.sensor_hz = counter - g_status.sensor_hz;
            g_status.sensor_hz = TASK_SENSOR_HZ;  // simple heartbeat
        }

        counter++;
        esp_task_wdt_reset();
        vTaskDelayUntil(&last, period);
    }
}

// ---------------------------------------------------------------------------
//  Motor task — 50 Hz on core 1
// ---------------------------------------------------------------------------
static void motorTask(void* arg) {
    (void)arg;
    const TickType_t period = pdMS_TO_TICKS(1000 / TASK_MOTOR_HZ);
    TickType_t last = xTaskGetTickCount();
    esp_task_wdt_add(NULL);

    while (true) {
        SensorFrame snap;
        xSemaphoreTake(g_frameMutex, portMAX_DELAY);
        snap = g_latestFrame;
        xSemaphoreGive(g_frameMutex);

        if (!g_estop) {
            // Run PID using measured wheel speeds.
            g_motors.step(snap.enc_l.v_linear_ms,
                          snap.enc_r.v_linear_ms);
        }

        esp_task_wdt_reset();
        vTaskDelayUntil(&last, period);
    }
}

// ---------------------------------------------------------------------------
//  Telemetry task — 10 Hz on core 0
// ---------------------------------------------------------------------------
static void telemetryTask(void* arg) {
    (void)arg;
    const TickType_t period = pdMS_TO_TICKS(1000 / TASK_TELEMETRY_HZ);
    TickType_t last = xTaskGetTickCount();
    esp_task_wdt_add(NULL);
    uint32_t seq = 0;

    while (true) {
        SensorFrame snap;
        xSemaphoreTake(g_frameMutex, portMAX_DELAY);
        snap = g_latestFrame;
        xSemaphoreGive(g_frameMutex);

        // Publish over MQTT (if connected)
        if (g_mqtt.isConnected()) {
            JsonDocument doc;
            doc["seq"]      = seq++;
            doc["tick"]     = snap.tick;
            doc["imu"]["qx"]= snap.imu.qx;
            doc["imu"]["qy"]= snap.imu.qy;
            doc["imu"]["qz"]= snap.imu.qz;
            doc["imu"]["qw"]= snap.imu.qw;
            doc["imu"]["roll"]  = snap.imu.roll;
            doc["imu"]["pitch"] = snap.imu.pitch;
            doc["imu"]["yaw"]   = snap.imu.yaw;
            doc["imu"]["gx"]= snap.imu.gx;
            doc["imu"]["gy"]= snap.imu.gy;
            doc["imu"]["gz"]= snap.imu.gz;
            doc["baro"]["t"]  = snap.baro.temperature_c;
            doc["baro"]["p"]  = snap.baro.pressure_pa;
            doc["baro"]["alt"]= snap.baro.altitude_m;
            doc["tof"]["mm"]  = snap.tof.distance_mm;
            doc["tof"]["ok"]  = snap.tof.valid;
            doc["enc"]["l"]["v"] = snap.enc_l.v_linear_ms;
            doc["enc"]["r"]["v"] = snap.enc_r.v_linear_ms;
            doc["enc"]["l"]["n"] = snap.enc_l.count;
            doc["enc"]["r"]["n"] = snap.enc_r.count;
            doc["bat"]["v"]   = snap.battery_v;
            doc["bat"]["pct"] = snap.battery_pct;
            g_mqtt.publishTelemetry(doc);
        }

        // Publish over BLE (if connected)
        bluetooth_publish_telemetry(snap);

        esp_task_wdt_reset();
        vTaskDelayUntil(&last, period);
    }
}

// ---------------------------------------------------------------------------
//  Watchdog / status task — 1 Hz on core 0
// ---------------------------------------------------------------------------
static void watchdogTask(void* arg) {
    (void)arg;
    const TickType_t period = pdMS_TO_TICKS(1000 / TASK_WATCHDOG_HZ);
    TickType_t last = xTaskGetTickCount();
    esp_task_wdt_add(NULL);

    while (true) {
        g_status.uptime_ms  = millis();
        g_status.loop_count++;
        g_status.free_heap  = ESP.getFreeHeap();
        g_status.wifi_ok    = g_wifi.isConnected();
        g_status.mqtt_ok    = g_mqtt.isConnected();
        g_status.estop      = g_estop.load();

        // Publish status every 5 s
        if ((g_status.loop_count % 5) == 0 && g_mqtt.isConnected()) {
            JsonDocument doc;
            doc["uptime"]    = g_status.uptime_ms;
            doc["free_heap"] = g_status.free_heap;
            doc["wifi"]      = g_status.wifi_ok;
            doc["mqtt"]      = g_status.mqtt_ok;
            doc["ble"]       = g_status.ble_ok;
            doc["estop"]     = g_status.estop;
            doc["fw"]        = AVC_FW_VERSION;
            doc["node"]      = AVC_NODE_NAME;
            g_mqtt.publishStatus(doc);
        }

        // Blink status LED: solid on = OK, slow blink = no WiFi, fast = E-stop.
        static bool ledState = false;
        if (g_status.estop) {
            ledState = (g_status.loop_count % 2) == 0;
        } else if (!g_status.wifi_ok) {
            ledState = (g_status.loop_count % 4) == 0;
        } else {
            ledState = true;
        }
        digitalWrite(STATUS_LED_PIN, ledState);

        esp_task_wdt_reset();
        vTaskDelayUntil(&last, period);
    }
}

// ---------------------------------------------------------------------------
//  Network task — runs WiFi/MQTT loops + reconnect logic
// ---------------------------------------------------------------------------
static void networkTask(void* arg) {
    (void)arg;
    while (true) {
        g_wifi.loop();
        g_mqtt.loop();
        vTaskDelay(pdMS_TO_TICKS(10));
    }
}

// ---------------------------------------------------------------------------
//  Arduino setup() — equivalent to ESP-IDF app_main()
// ---------------------------------------------------------------------------
void setup() {
    Serial.begin(115200);
    delay(100);
    AVC_LOGI("main", "=== AVC ESP32 firmware v%s (git %s) ===",
             AVC_FW_VERSION, AVC_FW_GIT_HASH);

    // Configure status LED
    pinMode(STATUS_LED_PIN, OUTPUT);
    digitalWrite(STATUS_LED_PIN, LOW);

    // I²C bus
    Wire.begin(I2C_SDA_PIN, I2C_SCL_PIN, I2C_FREQ_HZ);

    // Mutex for shared sensor frame
    g_frameMutex = xSemaphoreCreateMutex();
    assert(g_frameMutex);

    // Initialise subsystems (order matters: sensors → motors → battery → radio)
    bool s_ok = g_sensors.begin();
    AVC_LOGI("main", "sensors init: %s", s_ok ? "OK" : "PARTIAL");

    bool m_ok = g_motors.begin();
    AVC_LOGI("main", "motors init: %s", m_ok ? "OK" : "FAIL");

    battery_init();
    ota_init();

    // Bring up WiFi — AP portal if no credentials.
    bool wifi_ok = g_wifi.begin(false);
    AVC_LOGI("main", "wifi begin: %s", wifi_ok ? "OK" : "PORTAL");

    // MQTT
    g_mqtt.configure(MQTT_DEFAULT_BROKER,
                     MQTT_DEFAULT_USER, MQTT_DEFAULT_PASS,
                     AVC_NODE_NAME);
    g_mqtt.onCommand(onMqttCommand);
    if (wifi_ok) {
        g_mqtt.connect();
        g_mqtt.subscribeDefaults();
    }

    // Bluetooth (BLE) — runs entirely in its own task internally.
    bluetooth_init();

    // Hardware watchdog: 8 s, panic on timeout
    esp_task_wdt_config_t wdt_cfg = {
        .timeout_ms = 8000,
        .idle_core_mask = (1 << 0) | (1 << 1),
        .trigger_panic = true
    };
    esp_task_wdt_init(&wdt_cfg);

    // Spawn FreeRTOS tasks
    xTaskCreatePinnedToCore(sensorTask,    "sensor",    STACK_SENSOR,
                            nullptr, TASK_PRIO_SENSOR,    nullptr, CORE_SENSOR);
    xTaskCreatePinnedToCore(motorTask,     "motor",     STACK_MOTOR,
                            nullptr, TASK_PRIO_MOTOR,     nullptr, CORE_MOTOR);
    xTaskCreatePinnedToCore(telemetryTask, "telem",     STACK_TELEMETRY,
                            nullptr, TASK_PRIO_TELEMETRY, nullptr, CORE_TELEM);
    xTaskCreatePinnedToCore(watchdogTask,  "watchdog",  STACK_WATCHDOG,
                            nullptr, TASK_PRIO_WATCHDOG,  nullptr, CORE_NET);
    xTaskCreatePinnedToCore(networkTask,   "network",   STACK_NETWORK,
                            nullptr, TASK_PRIO_NETWORK,   nullptr, CORE_NET);

    strncpy(g_status.fw, AVC_FW_VERSION, sizeof(g_status.fw) - 1);
    AVC_LOGI("main", "FreeRTOS tasks launched. Free heap: %u",
             (unsigned)ESP.getFreeHeap());
}

// ---------------------------------------------------------------------------
//  Arduino loop() — empty; everything is task-driven.
// ---------------------------------------------------------------------------
void loop() {
    vTaskDelay(pdMS_TO_TICKS(1000));
}
