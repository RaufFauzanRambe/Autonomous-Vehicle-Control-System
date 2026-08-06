/**
 * @file mqtt_client.h
 * @brief MqttClient — PubSubClient wrapper with TLS, last-will, reconnect.
 * @author AVC team
 * @date 2024
 * @license MIT
 */
#ifndef AVC_ESP32_MQTT_CLIENT_H
#define AVC_ESP32_MQTT_CLIENT_H

#include <Arduino.h>
#include <WiFi.h>
#include <WiFiClientSecure.h>
#include <PubSubClient.h>
#include <ArduinoJson.h>
#include "config.h"

/// Quality-of-service levels supported by the broker.
enum class MqttQos : uint8_t {
    kAtMostOnce  = 0,   ///< Fire-and-forget (default).
    kAtLeastOnce = 1,   ///< Acknowledged delivery.
    kExactlyOnce = 2,   ///< Not supported by PubSubClient; falls back to 1.
};

/// Connection state surfaced to the rest of the firmware.
enum class MqttState : uint8_t {
    kDisconnected = 0,
    kConnecting   = 1,
    kConnected    = 2,
    kError        = 3,
};

/// Type alias for the user-supplied command callback.
using CmdCallback = std::function<void(const String& topic, const JsonObject& payload)>;

/**
 * @class MqttClient
 * @brief Thin wrapper around PubSubClient providing:
 *        - secure (TLS) and plaintext transport,
 *        - NVS-persisted broker configuration,
 *        - last-will-and-testament on the status topic,
 *        - automatic reconnect with exponential backoff,
 *        - JSON-encoded telemetry publication.
 */
class MqttClient {
public:
    /**
     * @brief Constructor — wires PubSubClient to a WiFiClientSecure.
     */
    MqttClient();

    /**
     * @brief Configure broker.
     * @param[in] broker   URL like "mqtt://host:1883" or "mqtts://host:8883".
     * @param[in] user     Username (optional).
     * @param[in] pass     Password (optional).
     * @param[in] node_id  Node identifier used in topic templates.
     */
    void configure(const String& broker,
                   const String& user,
                   const String& pass,
                   const String& node_id);

    /**
     * @brief Blocking connect. Returns when connected or after 10 s.
     * @return true on success.
     */
    bool connect();

    /**
     * @brief Non-blocking — call from `loop()`. Reconnects with backoff.
     */
    void loop();

    /**
     * @brief Disconnect cleanly (sends DISCONNECT, transitions to kDisconnected).
     */
    void disconnect();

    /**
     * @brief Subscribe to a topic.
     * @param[in] topic Full topic string.
     * @param[in] qos   QoS level (default at-most-once).
     * @return true on success.
     */
    bool subscribe(const String& topic, MqttQos qos = MqttQos::kAtMostOnce);

    /**
     * @brief Publish a raw payload.
     * @param[in] topic   Full topic.
     * @param[in] payload Bytes to publish.
     * @param[in] len     Length of payload.
     * @param[in] qos     QoS level.
     * @param[in] retain  Broker should retain last message.
     * @return true on success.
     */
    bool publish(const char* topic, const uint8_t* payload, size_t len,
                 MqttQos qos = MqttQos::kAtMostOnce, bool retain = false);

    /**
     * @brief Publish a JSON document on the telemetry topic.
     * @param[in] doc Pre-populated JSON document.
     * @return true on success.
     */
    bool publishTelemetry(const JsonDocument& doc);

    /**
     * @brief Publish a JSON document on the battery topic.
     */
    bool publishBattery(const JsonDocument& doc);

    /**
     * @brief Publish a status JSON (also used as last-will).
     */
    bool publishStatus(const JsonDocument& doc);

    /**
     * @brief Register a single callback for ALL subscribed topics.
     * @param[in] cb Invoked on every inbound message.
     */
    inline void onCommand(CmdCallback cb) { _cmdCb = std::move(cb); }

    /**
     * @brief Current connection state.
     */
    inline MqttState state() const { return _state; }

    /**
     * @brief True if currently connected.
     */
    inline bool isConnected() const { return _state == MqttState::kConnected; }

    /**
     * @brief Subscribe to the AVC default command topics.
     */
    void subscribeDefaults();

private:
    /// Internal PubSubClient callback trampoline.
    static void _onMqttMessage(char* topic, byte* payload, unsigned int len);

    /// Expand a topic template like "avc/%s/state/telemetry".
    String _expand(const char* tmpl) const;

    WiFiClientSecure  _tls;       ///< Secure TCP client.
    WiFiClient        _plain;     ///< Plaintext TCP client.
    PubSubClient      _ps;        ///< Underlying MQTT client.
    String            _broker;    ///< Hostname or IP.
    uint16_t          _port = 1883;
    bool              _useTls = false;
    String            _user;
    String            _pass;
    String            _nodeId;

    MqttState         _state = MqttState::kDisconnected;
    uint32_t          _lastAttempt = 0;
    uint32_t          _backoffMs   = MQTT_RECONNECT_MS;
    CmdCallback       _cmdCb;

    static MqttClient* _instance; ///< Singleton for PubSubClient callback.
};

#endif // AVC_ESP32_MQTT_CLIENT_H
