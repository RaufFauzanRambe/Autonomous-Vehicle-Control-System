/**
 * @file bluetooth_manager.cpp
 * @brief BluetoothManager — BLE Nordic UART Service + telemetry GATT.
 * @author AVC team
 * @date 2024
 * @license MIT
 *
 * Exposes two BLE services:
 *  1. Nordic UART Service (NUS) — bidirectional serial-over-BLE.
 *     RX (write):    JSON commands like {"v":0.5,"w":0.1}
 *     TX (notify):   telemetry frames @ 10 Hz as JSON lines
 *  2. Device Information Service — manufacturer, model, firmware, hardware.
 *  3. AVC custom telemetry service — exposes individual characteristics:
 *       IMU quaternion, encoder speeds, battery, etc. for clients that prefer
 *       per-field reads/notifications instead of the JSON stream.
 *
 * Also publishes the C-callable entry points `bluetooth_init()` and
 * `bluetooth_publish_telemetry()` referenced from main.cpp.
 */

#include <Arduino.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>
#include <BLE2902.h>
#include <ArduinoJson.h>
#include "config.h"
#include "sensor_reader.h"

// ---- UUIDs ----------------------------------------------------------------
// Nordic UART Service (well-known)
static const BLEUUID kNusServiceUUID("6E400001-B5A3-F393-E0A9-E50E24DCCA9E");
static const BLEUUID kNusRxUUID     ("6E400002-B5A3-F393-E0A9-E50E24DCCA9E");
static const BLEUUID kNusTxUUID     ("6E400003-B5A3-F393-E0A9-E50E24DCCA9E");

// Device Information Service (well-known)
static const BLEUUID kDisServiceUUID("0000180A-0000-1000-8000-00805F9B34FB");
static const BLEUUID kDisMfrUUID    ("00002A29-0000-1000-8000-00805F9B34FB");
static const BLEUUID kDisModelUUID  ("00002A24-0000-1000-8000-00805F9B34FB");
static const BLEUUID kDisFwUUID     ("00002A26-0000-1000-8000-00805F9B34FB");
static const BLEUUID kDisHwUUID     ("00002A27-0000-1000-8000-00805F9B34FB");

// AVC custom telemetry service
static const BLEUUID kAvcServiceUUID("AFC10001-1234-5678-9ABC-DEF012345678");
static const BLEUUID kAvcImuUUID     ("AFC10002-1234-5678-9ABC-DEF012345678");
static const BLEUUID kAvcEncUUID     ("AFC10003-1234-5678-9ABC-DEF012345678");
static const BLEUUID kAvcBatUUID     ("AFC10004-1234-5678-9ABC-DEF012345678");
static const BLEUUID kAvcStatusUUID  ("AFC10005-1234-5678-9ABC-DEF012345678");

// ---- Globals --------------------------------------------------------------
static BLEServer*       g_server   = nullptr;
static BLECharacteristic* g_txChar  = nullptr;   // NUS TX (notify)
static BLECharacteristic* g_imuChar = nullptr;
static BLECharacteristic* g_encChar = nullptr;
static BLECharacteristic* g_batChar = nullptr;
static BLECharacteristic* g_stsChar = nullptr;
static bool g_bleConnected = false;
static QueueHandle_t g_cmdQueue = nullptr;       // RX commands → main task

// Optional hook that main.cpp can install to receive BLE commands.
static std::function<void(const String&)> g_cmdHandler;

// ---------------------------------------------------------------------------
//  Server callbacks — connection state tracking
// ---------------------------------------------------------------------------
class SrvCb : public BLEServerCallbacks {
    void onConnect(BLEServer* s) override {
        g_bleConnected = true;
        AVC_LOGI("ble", "client connected");
        // Reduce advertising interval while connected for snappier comms.
        s->getAdvertising()->stop();
    }
    void onDisconnect(BLEServer* s) override {
        g_bleConnected = false;
        AVC_LOGI("ble", "client disconnected — restarting advertising");
        s->startAdvertising();
    }
};

// ---------------------------------------------------------------------------
//  RX characteristic callback — push to queue / invoke handler
// ---------------------------------------------------------------------------
class RxCb : public BLECharacteristicCallbacks {
    void onWrite(BLECharacteristic* c) override {
        String rx((const char*)c->getValue().data(), c->getValue().length());
        if (g_cmdHandler) g_cmdHandler(rx);
        else if (g_cmdQueue) {
            // Make a heap copy because the queue stores pointers.
            char* dup = strdup(rx.c_str());
            if (dup) xQueueSend(g_cmdQueue, &dup, 0);
        }
        AVC_LOGD("ble", "RX '%s'", rx.c_str());
    }
};

// ---------------------------------------------------------------------------
//  Internal — start GATT server with all services
// ---------------------------------------------------------------------------
static void bleStart() {
    BLEDevice::init(AVC_NODE_NAME);
    BLEDevice::setMTU(185);   // typical negotiated MTU
    // Use a public (fixed) MAC for stable re-connection.
    BLEDevice::setOwnAddrType(BLE_ADDR_PUBLIC);

    g_server = BLEDevice::createServer();
    g_server->setCallbacks(new SrvCb());

    // --- Nordic UART Service ---
    BLEService* nus = g_server->createService(kNusServiceUUID);
    BLECharacteristic* rx = nus->createCharacteristic(
        kNusRxUUID, BLECharacteristic::PROPERTY_WRITE);
    rx->setCallbacks(new RxCb());
    g_txChar = nus->createCharacteristic(
        kNusTxUUID, BLECharacteristic::PROPERTY_NOTIFY);
    g_txChar->addDescriptor(new BLE2902());
    nus->start();

    // --- Device Information Service ---
    BLEService* dis = g_server->createService(kDisServiceUUID);
    auto addDis = [&](const BLEUUID& u, const char* v) {
        BLECharacteristic* c = dis->createCharacteristic(u, BLECharacteristic::PROPERTY_READ);
        c->setValue(v);
    };
    addDis(kDisMfrUUID,   "Autonomous Vehicle Control System");
    addDis(kDisModelUUID, "ESP32-Telemetry-Node");
    addDis(kDisFwUUID,    AVC_FW_VERSION);
    addDis(kDisHwUUID,    AVC_NODE_NAME);
    dis->start();

    // --- AVC custom telemetry service ---
    BLEService* avc = g_server->createService(kAvcServiceUUID);
    g_imuChar = avc->createCharacteristic(kAvcImuUUID, BLECharacteristic::PROPERTY_READ |
                                                       BLECharacteristic::PROPERTY_NOTIFY);
    g_imuChar->addDescriptor(new BLE2902());
    g_encChar = avc->createCharacteristic(kAvcEncUUID, BLECharacteristic::PROPERTY_READ |
                                                       BLECharacteristic::PROPERTY_NOTIFY);
    g_encChar->addDescriptor(new BLE2902());
    g_batChar = avc->createCharacteristic(kAvcBatUUID, BLECharacteristic::PROPERTY_READ |
                                                       BLECharacteristic::PROPERTY_NOTIFY);
    g_batChar->addDescriptor(new BLE2902());
    g_stsChar = avc->createCharacteristic(kAvcStatusUUID, BLECharacteristic::PROPERTY_READ);
    g_stsChar->setValue("idle");
    avc->start();

    // Advertise NUS + DIS.
    BLEAdvertising* adv = BLEDevice::getAdvertising();
    adv->addServiceUUID(kNusServiceUUID);
    adv->addServiceUUID(kDisServiceUUID);
    adv->setScanResponse(true);
    adv->setMinPreferred(0x06);  // helps with iPhone connection
    adv->setMinPreferred(0x12);
    BLEDevice::startAdvertising();

    AVC_LOGI("ble", "advertising started");
}

// ---------------------------------------------------------------------------
//  Public C-style API consumed by main.cpp
// ---------------------------------------------------------------------------
extern "C" void bluetooth_init() {
    g_cmdQueue = xQueueCreate(8, sizeof(char*));
    bleStart();
}

extern "C" void bluetooth_publish_telemetry(const SensorFrame& f) {
    if (!g_bleConnected || !g_txChar) return;

    // Compact JSON line for the NUS TX characteristic.
    JsonDocument doc;
    doc["seq"]   = f.seq;
    doc["tick"]  = f.tick;
    doc["imu"]["roll"]  = f.imu.roll;
    doc["imu"]["pitch"] = f.imu.pitch;
    doc["imu"]["yaw"]   = f.imu.yaw;
    doc["enc"]["l"]     = f.enc_l.v_linear_ms;
    doc["enc"]["r"]     = f.enc_r.v_linear_ms;
    doc["tof"]          = f.tof.distance_mm;
    doc["bat"]["v"]     = f.battery_v;
    doc["bat"]["pct"]   = f.battery_pct;

    String out;
    serializeJson(doc, out);
    out += "\n";
    g_txChar->setValue(out.c_str());
    g_txChar->notify();

    // Per-field GATT characteristics (binary packed for minimal overhead).
    #pragma pack(push, 1)
    struct ImuPacked { float roll, pitch, yaw; };
    struct EncPacked { float vL, vR; int32_t nL, nR; };
    struct BatPacked { float v, pct; };
    #pragma pack(pop)
    ImuPacked imu{f.imu.roll, f.imu.pitch, f.imu.yaw};
    EncPacked enc{f.enc_l.v_linear_ms, f.enc_r.v_linear_ms,
                  f.enc_l.count,      f.enc_r.count};
    BatPacked bat{f.battery_v, f.battery_pct};
    g_imuChar->setValue((uint8_t*)&imu, sizeof(imu));
    g_imuChar->notify();
    g_encChar->setValue((uint8_t*)&enc, sizeof(enc));
    g_encChar->notify();
    g_batChar->setValue((uint8_t*)&bat, sizeof(bat));
    g_batChar->notify();
}

// Convenience: install a command handler (called from main.cpp setup if needed).
extern "C" void bluetooth_set_command_handler(void (*h)(const char*)) {
    g_cmdHandler = [h](const String& s) { h(s.c_str()); };
}
