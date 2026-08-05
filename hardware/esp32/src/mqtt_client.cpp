/**
 * @file mqtt_client.cpp
 * @brief MqttClient implementation — PubSubClient wrapper with TLS + LWT.
 * @author AVC team
 * @date 2024
 * @license MIT
 */

#include "mqtt_client.h"
#include <Preferences.h>

// Singleton used by the PubSubClient callback (which has no user-data slot).
MqttClient* MqttClient::_instance = nullptr;

// ---------------------------------------------------------------------------
//  Constructor / configure
// ---------------------------------------------------------------------------
MqttClient::MqttClient()
    : _ps() {
    _instance = this;
    _ps.setKeepAlive(MQTT_KEEPALIVE_SEC);
    _ps.setSocketTimeout(5);
    _ps.setBufferSize(MQTT_MAX_PACKET);
    _ps.setCallback(_onMqttMessage);
}

void MqttClient::configure(const String& broker,
                           const String& user,
                           const String& pass,
                           const String& node_id) {
    // Parse "mqtt://host:port" or "mqtts://host:port".
    _user   = user;
    _pass   = pass;
    _nodeId = node_id;

    String b = broker;
    bool tls = false;
    if (b.startsWith("mqtts://")) { tls = true; b = b.substring(8); }
    else if (b.startsWith("mqtt://"))  { b = b.substring(7); }
    else if (b.startsWith("ssl://"))   { tls = true; }
    _useTls = tls;

    int colon = b.indexOf(':');
    if (colon > 0) {
        _broker = b.substring(0, colon);
        _port   = (uint16_t)b.substring(colon + 1).toInt();
    } else {
        _broker = b;
        _port   = tls ? 8883 : 1883;
    }
    if (_port == 0) _port = tls ? 8883 : 1883;

    // If no broker supplied, try NVS.
    if (_broker.length() == 0 || _broker == "mqtt://") {
        Preferences nvs;
        nvs.begin(NVS_NAMESPACE, true);
        _broker = nvs.getString(NVS_KEY_MQTT_HOST, MQTT_DEFAULT_BROKER);
        // Strip scheme prefix if present.
        if (_broker.startsWith("mqtt://"))  _broker = _broker.substring(7);
        if (_broker.startsWith("mqtts://")) { _broker = _broker.substring(8); _useTls = true; }
        _user = nvs.getString(NVS_KEY_MQTT_USER, MQTT_DEFAULT_USER);
        _pass = nvs.getString(NVS_KEY_MQTT_PASS, MQTT_DEFAULT_PASS);
        nvs.end();
    }
    AVC_LOGI("mqtt", "configured broker=%s:%u tls=%d node=%s",
             _broker.c_str(), _port, _useTls ? 1 : 0, _nodeId.c_str());
}

// ---------------------------------------------------------------------------
//  Connect / loop / disconnect
// ---------------------------------------------------------------------------
bool MqttClient::connect() {
    if (_broker.length() == 0) {
        AVC_LOGE("mqtt", "no broker configured");
        return false;
    }
    if (_useTls) {
        // Skip cert validation by default — production should load a CA bundle.
        _tls.setInsecure();
        _ps.setClient(_tls);
    } else {
        _ps.setClient(_plain);
    }
    _ps.setServer(_broker.c_str(), _port);

    // Last-will: notify other clients that we've gone offline.
    String willTopic = _expand(MQTT_TOPIC_STATUS);
    String willPayload = String("{\"node\":\"") + _nodeId +
                         "\",\"state\":\"offline\"}";

    _state = MqttState::kConnecting;
    AVC_LOGI("mqtt", "connecting to %s:%u as '%s' ...",
             _broker.c_str(), _port, _user.c_str());

    bool ok;
    if (_user.length() > 0) {
        ok = _ps.connect(_nodeId.c_str(),
                         _user.c_str(), _pass.c_str(),
                         willTopic.c_str(), 1, true, willPayload.c_str());
    } else {
        ok = _ps.connect(_nodeId.c_str(),
                         willTopic.c_str(), 1, true, willPayload.c_str());
    }

    if (ok) {
        _state = MqttState::kConnected;
        _backoffMs = MQTT_RECONNECT_MS;
        // Announce online.
        String online = String("{\"node\":\"") + _nodeId +
                        "\",\"state\":\"online\",\"fw\":\"" +
                        AVC_FW_VERSION + "\"}";
        _ps.publish(willTopic.c_str(), online.c_str(), true);
        AVC_LOGI("mqtt", "connected");
    } else {
        _state = MqttState::kError;
        AVC_LOGE("mqtt", "connect failed state=%d", _ps.state());
    }
    return ok;
}

void MqttClient::loop() {
    if (!_ps.loop()) {
        // PubSubClient returned false — connection dropped.
        if (_state == MqttState::kConnected) {
            AVC_LOGW("mqtt", "disconnected (state=%d)", _ps.state());
            _state = MqttState::kDisconnected;
        }
        uint32_t now = millis();
        if (now - _lastAttempt >= _backoffMs) {
            _lastAttempt = now;
            connect();
            // Exponential backoff capped at 60 s.
            _backoffMs = min<uint32_t>(_backoffMs * 2, 60000);
        }
    }
}

void MqttClient::disconnect() {
    if (_ps.connected()) {
        String offline = String("{\"node\":\"") + _nodeId +
                         "\",\"state\":\"offline_graceful\"}";
        _ps.publish(_expand(MQTT_TOPIC_STATUS).c_str(), offline.c_str(), true);
        _ps.disconnect();
    }
    _state = MqttState::kDisconnected;
}

// ---------------------------------------------------------------------------
//  Subscribe / publish
// ---------------------------------------------------------------------------
bool MqttClient::subscribe(const String& topic, MqttQos qos) {
    if (!isConnected()) return false;
    uint8_t q = (qos == MqttQos::kExactlyOnce) ? 1 : (uint8_t)qos;
    bool ok = _ps.subscribe(topic.c_str(), q);
    AVC_LOGI("mqtt", "subscribe %s qos=%u → %d", topic.c_str(), q, ok);
    return ok;
}

bool MqttClient::publish(const char* topic, const uint8_t* payload, size_t len,
                         MqttQos qos, bool retain) {
    if (!isConnected()) return false;
    uint8_t q = (qos == MqttQos::kExactlyOnce) ? 1 : (uint8_t)qos;
    return _ps.publish(topic, payload, len, false, retain);
}

bool MqttClient::publishTelemetry(const JsonDocument& doc) {
    String buf;
    serializeJson(doc, buf);
    String topic = _expand(MQTT_TOPIC_TELEMETRY);
    return publish(topic.c_str(), (const uint8_t*)buf.c_str(), buf.length(),
                   MqttQos::kAtMostOnce, false);
}

bool MqttClient::publishBattery(const JsonDocument& doc) {
    String buf;
    serializeJson(doc, buf);
    String topic = _expand(MQTT_TOPIC_BATTERY);
    return publish(topic.c_str(), (const uint8_t*)buf.c_str(), buf.length(),
                   MqttQos::kAtLeastOnce, false);
}

bool MqttClient::publishStatus(const JsonDocument& doc) {
    String buf;
    serializeJson(doc, buf);
    String topic = _expand(MQTT_TOPIC_STATUS);
    return publish(topic.c_str(), (const uint8_t*)buf.c_str(), buf.length(),
                   MqttQos::kAtLeastOnce, true);
}

void MqttClient::subscribeDefaults() {
    subscribe(_expand(MQTT_TOPIC_CMD_VEL),  MqttQos::kAtMostOnce);
    subscribe(_expand(MQTT_TOPIC_CMD_OTA),  MqttQos::kAtLeastOnce);
    subscribe(_expand(MQTT_TOPIC_CMD_CFG),  MqttQos::kAtLeastOnce);
}

// ---------------------------------------------------------------------------
//  Helpers
// ---------------------------------------------------------------------------
String MqttClient::_expand(const char* tmpl) const {
    char buf[64];
    snprintf(buf, sizeof(buf), tmpl, _nodeId.c_str());
    return String(buf);
}

void MqttClient::_onMqttMessage(char* topic, byte* payload, unsigned int len) {
    if (!_instance) return;
    // Copy payload into a null-terminated buffer.
    char* buf = (char*)malloc(len + 1);
    if (!buf) return;
    memcpy(buf, payload, len);
    buf[len] = '\0';

    JsonDocument doc;
    DeserializationError err = deserializeJson(doc, buf, len);
    if (err) {
        AVC_LOGW("mqtt", "JSON parse error on %s: %s", topic, err.c_str());
        free(buf);
        return;
    }
    if (_instance->_cmdCb) {
        _instance->_cmdCb(String(topic), doc.as<JsonObject>());
    }
    free(buf);
}
