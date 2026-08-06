/**
 * @file wifi_manager.h
 * @brief WiFiManager — STA + AP mode, NVS-backed credentials, config portal.
 * @author AVC team
 * @date 2024
 * @license MIT
 */
#ifndef AVC_ESP32_WIFI_MANAGER_H
#define AVC_ESP32_WIFI_MANAGER_H

#include <Arduino.h>
#include <WiFi.h>
#include <Preferences.h>
#include <ESPAsyncWebServer.h>
#include <DNSServer.h>

/**
 * @brief Operating mode of the WiFi stack.
 */
enum class WifiMode : uint8_t {
    kStation      = 0,  ///< Connected to an AP as a client.
    kAccessPoint  = 1,  ///< Acting as AP for the config portal.
    kOff          = 2,  ///< Radio off (deep sleep / shipping).
};

/**
 * @brief Aggregated WiFi status snapshot for telemetry.
 */
struct WifiStatus {
    WifiMode   mode;
    bool       connected;
    int8_t     rssi;          ///< dBm, -127 = unknown
    uint8_t    channel;
    IPAddress  ip;
    IPAddress  gateway;
    IPAddress  subnet;
    uint32_t   uptime_ms;
    uint32_t   reconnects;    ///< number of reconnect attempts since boot
};

/**
 * @class WiFiManager
 * @brief Connects the ESP32 to a configured AP, falls back to a captive
 *        portal for provisioning, and persists credentials in NVS.
 *
 * Lifecycle:
 * 1. `begin()` — load credentials from NVS; if missing, start AP+portal.
 * 2. `loop()` — non-blocking; triggers auto-reconnect on disconnect.
 * 3. `portal()` — manually start the captive portal (button press).
 * 4. `save()` — called by the portal's POST handler to persist credentials.
 */
class WiFiManager {
public:
    /**
     * @brief Constructor — opens NVS namespace `avc`.
     */
    WiFiManager();

    /**
     * @brief Initialise the WiFi stack.
     * @param[in] force_portal If true, skip NVS credentials and start AP.
     * @return true if STA connected within WIFI_CONNECT_TIMEOUT_MS.
     */
    bool begin(bool force_portal = false);

    /**
     * @brief Periodic handler — call from `loop()` (core 0).
     *        Manages auto-reconnect and DNS server when in AP mode.
     */
    void loop();

    /**
     * @brief Manually start the captive portal.
     * @param[in] timeout_ms Portal auto-shutdown time; 0 = never.
     */
    void startPortal(uint32_t timeout_ms = 180000);

    /**
     * @brief Disconnect and clear stored credentials (factory reset).
     */
    void resetCredentials();

    /**
     * @brief Save new credentials to NVS and attempt STA connect.
     * @param[in] ssid Network SSID (≤32 chars).
     * @param[in] pass WPA2/3 passphrase (8..63 chars).
     * @return true if connection succeeded.
     */
    bool save(const String& ssid, const String& pass);

    /**
     * @brief Snapshot of current WiFi status.
     */
    WifiStatus status() const;

    /**
     * @brief SSID stored in NVS (may be empty).
     */
    inline String ssid() const { return _ssid; }

    /**
     * @brief Is the device currently associated with an AP?
     */
    inline bool isConnected() const {
        return WiFi.status() == WL_CONNECTED;
    }

    /**
     * @brief mDNS hostname (avc-esp32.local by default).
     */
    inline String hostname() const { return _hostname; }

private:
    /// Attempt STA association using `_ssid`/`_pass`.
    bool connectSta();

    /// Bring up AP + DNS captive portal + web server.
    void bringUpPortal();

    /// Tear down portal resources.
    void tearDownPortal();

    /// AsyncWebServer routes for /, /save, /status.
    void registerRoutes();

    /// mDNS announce.
    void announceMDNS();

    Preferences _nvs;            ///< NVS handle, namespace "avc".
    String      _ssid;           ///< Loaded SSID.
    String      _pass;           ///< Loaded passphrase.
    String      _hostname;       ///< mDNS hostname.

    WifiMode    _mode = WifiMode::kOff;
    uint32_t    _lastReconnect = 0;
    uint32_t    _reconnects    = 0;
    uint32_t    _uptimeStart   = 0;
    uint32_t    _portalStart   = 0;
    uint32_t    _portalTimeout = 0;
    bool        _portalActive  = false;

    AsyncWebServer* _server = nullptr;
    DNSServer*      _dns    = nullptr;
};

#endif // AVC_ESP32_WIFI_MANAGER_H
