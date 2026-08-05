/**
 * @file wifi_manager.cpp
 * @brief WiFiManager implementation — STA + AP + captive portal + NVS.
 * @author AVC team
 * @date 2024
 * @license MIT
 */

#include "wifi_manager.h"
#include "config.h"
#include <esp_wifi.h>
#include <esp_task_wdt.h>
#include <DNSServer.h>

// Captive portal HTML — minimal, single file.
static const char kPortalHTML[] PROGMEM = R"HTML(
<!DOCTYPE html><html><head>
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AVC ESP32 Setup</title>
<style>
body{font-family:sans-serif;margin:20px;background:#f5f5f5}
h1{color:#2c3e50}
input{display:block;margin:8px 0;padding:8px;width:100%;box-sizing:border-box}
button{padding:10px 16px;background:#2c3e50;color:#fff;border:0;border-radius:4px}
</style></head><body>
<h1>AVC ESP32 Setup</h1>
<form action="/save" method="POST">
<label>WiFi SSID</label><input name="ssid" maxlength="32" required>
<label>WiFi Password</label><input name="pass" type="password" maxlength="63">
<label>MQTT Broker (optional)</label><input name="mqtt_host" placeholder="mqtt://192.168.1.10">
<button type="submit">Save &amp; Reboot</button>
</form>
</body></html>
)HTML";

WiFiManager::WiFiManager() {
    _nvs.begin(NVS_NAMESPACE, false);
    _ssid = _nvs.getString(NVS_KEY_WIFI_SSID, "");
    _pass = _nvs.getString(NVS_KEY_WIFI_PASS, "");
    _hostname = WIFI_HOSTNAME;
    _uptimeStart = millis();
}

bool WiFiManager::begin(bool force_portal) {
    WiFi.persistent(false);
    WiFi.setHostname(_hostname.c_str());
    WiFi.mode(WIFI_STA);

    if (!force_portal && _ssid.length() > 0) {
        if (connectSta()) {
            announceMDNS();
            _mode = WifiMode::kStation;
            return true;
        }
        AVC_LOGW("wifi", "STA connect failed, falling back to portal");
    }
    bringUpPortal();
    return false;
}

bool WiFiManager::connectSta() {
    AVC_LOGI("wifi", "STA connecting to '%s' ...", _ssid.c_str());
    WiFi.disconnect(true, true);
    delay(100);
    WiFi.begin(_ssid.c_str(), _pass.c_str());

    uint32_t t0 = millis();
    while (WiFi.status() != WL_CONNECTED &&
           (millis() - t0) < WIFI_CONNECT_TIMEOUT_MS) {
        delay(100);
        // Feed the watchdog during long connect attempts.
        esp_task_wdt_reset();
    }
    if (WiFi.status() == WL_CONNECTED) {
        AVC_LOGI("wifi", "STA up: %s / RSSI %d dBm / ch %u",
                 WiFi.localIP().toString().c_str(),
                 WiFi.RSSI(), WiFi.channel());
        return true;
    }
    AVC_LOGE("wifi", "STA timeout (status=%d)", WiFi.status());
    return false;
}

void WiFiManager::loop() {
    // STA reconnect handling.
    if (_mode == WifiMode::kStation) {
        if (WiFi.status() != WL_CONNECTED) {
            uint32_t now = millis();
            if (now - _lastReconnect >= WIFI_RECONNECT_INTERVAL_MS) {
                _lastReconnect = now;
                _reconnects++;
                AVC_LOGW("wifi", "auto-reconnect attempt %u", _reconnects);
                WiFi.reconnect();
            }
        }
    }

    // Captive portal DNS loop.
    if (_portalActive && _dns) {
        _dns->processNextRequest();
        // Auto-shutdown portal?
        if (_portalTimeout > 0 &&
            (millis() - _portalStart) > _portalTimeout) {
            AVC_LOGI("wifi", "portal timeout — shutting down");
            tearDownPortal();
        }
    }
}

void WiFiManager::startPortal(uint32_t timeout_ms) {
    if (_portalActive) return;
    _portalTimeout = timeout_ms;
    bringUpPortal();
}

void WiFiManager::bringUpPortal() {
    AVC_LOGI("wifi", "starting AP+portal");
    WiFi.mode(WIFI_AP);
    char apSsid[32];
    snprintf(apSsid, sizeof(apSsid), WIFI_AP_SSID_FMT,
             (uint32_t)ESP.getEfuseMac() & 0xFFFFFF);
    WiFi.softAP(apSsid, nullptr);
    delay(100);
    AVC_LOGI("wifi", "AP up: %s @ %s",
             apSsid, WiFi.softAPIP().toString().c_str());

    // Capture DNS for captive portal.
    _dns = new DNSServer();
    _dns->start(53, "*", WiFi.softAPIP());

    // Async web server.
    _server = new AsyncWebServer(WIFI_AP_PORTAL_PORT);
    registerRoutes();
    _server->begin();

    _portalActive = true;
    _portalStart  = millis();
    _mode         = WifiMode::kAccessPoint;
    announceMDNS();
}

void WiFiManager::tearDownPortal() {
    if (_server) { _server->end(); delete _server; _server = nullptr; }
    if (_dns)    { _dns->stop();   delete _dns;    _dns    = nullptr; }
    _portalActive = false;
    WiFi.softAPdisconnect(true);
}

void WiFiManager::registerRoutes() {
    _server->on("/", HTTP_GET, [](AsyncWebServerRequest* req) {
        req->send_P(200, "text/html", kPortalHTML);
    });

    _server->on("/save", HTTP_POST, [this](AsyncWebServerRequest* req) {
        String ssid, pass, mqtt_host;
        if (req->hasParam("ssid", true))       ssid = req->getParam("ssid", true)->value();
        if (req->hasParam("pass", true))       pass = req->getParam("pass", true)->value();
        if (req->hasParam("mqtt_host", true))  mqtt_host = req->getParam("mqtt_host", true)->value();

        req->send(200, "text/plain",
                  "Saved. Rebooting in 2 s ...");

        // Defer NVS write + reboot so the HTTP response goes out first.
        delay(500);
        if (ssid.length() > 0) {
            _nvs.putString(NVS_KEY_WIFI_SSID, ssid);
            _nvs.putString(NVS_KEY_WIFI_PASS, pass);
            if (mqtt_host.length() > 0) {
                _nvs.putString(NVS_KEY_MQTT_HOST, mqtt_host);
            }
            _nvs.end();
            AVC_LOGI("wifi", "saved credentials, rebooting");
            ESP.restart();
        }
    });

    _server->on("/status", HTTP_GET, [this](AsyncWebServerRequest* req) {
        AsyncResponseStream* r = req->beginResponseStream("application/json");
        r->printf("{\"mode\":%u,\"connected\":%s,\"rssi\":%d,\"uptime\":%lu}",
                  (unsigned)_mode,
                  isConnected() ? "true" : "false",
                  WiFi.RSSI(),
                  (unsigned long)(millis() - _uptimeStart));
        req->send(r);
    });

    // Captive portal — redirect any unknown URL to "/".
    _server->onNotFound([](AsyncWebServerRequest* req) {
        String url = "http://" + req->client()->localIP().toString() + "/";
        req->redirect(url);
    });
}

bool WiFiManager::save(const String& ssid, const String& pass) {
    _ssid = ssid;
    _pass = pass;
    _nvs.putString(NVS_KEY_WIFI_SSID, ssid);
    _nvs.putString(NVS_KEY_WIFI_PASS, pass);
    if (_portalActive) tearDownPortal();
    WiFi.mode(WIFI_STA);
    bool ok = connectSta();
    if (ok) {
        _mode = WifiMode::kStation;
        announceMDNS();
    }
    return ok;
}

void WiFiManager::resetCredentials() {
    _nvs.remove(NVS_KEY_WIFI_SSID);
    _nvs.remove(NVS_KEY_WIFI_PASS);
    _ssid = "";
    _pass = "";
    WiFi.disconnect(true, true);
    AVC_LOGI("wifi", "credentials cleared");
}

WifiStatus WiFiManager::status() const {
    WifiStatus s{};
    s.mode         = _mode;
    s.connected    = isConnected();
    s.rssi         = s.connected ? WiFi.RSSI() : -127;
    s.channel      = WiFi.channel();
    s.ip           = WiFi.localIP();
    s.gateway      = WiFi.gatewayIP();
    s.subnet       = WiFi.subnetMask();
    s.uptime_ms    = millis() - _uptimeStart;
    s.reconnects   = _reconnects;
    return s;
}

void WiFiManager::announceMDNS() {
    if (!MDNS.begin(_hostname.c_str())) {
        AVC_LOGW("wifi", "mDNS begin failed");
        return;
    }
    MDNS.addService("http",  "tcp", WIFI_AP_PORTAL_PORT);
    MDNS.addService("mqtt",  "tcp", MQTT_DEFAULT_PORT);
    MDNS.addServiceTxt("http", "tcp", "fw",   AVC_FW_VERSION);
    MDNS.addServiceTxt("http", "tcp", "node", AVC_NODE_NAME);
    AVC_LOGI("wifi", "mDNS: %s.local", _hostname.c_str());
}
