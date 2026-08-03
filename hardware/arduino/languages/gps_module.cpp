/**
 * @file    gps_module.cpp
 * @brief   Implementation of the GPS module wrapper.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#include "gps_module.h"
#include "utilities.h"

namespace avcs {

/* ========================================================================= *
 *  Constructor / destructor
 * ========================================================================= */
GpsModule::GpsModule() {
    _state.latitudeDeg  = 0.0;
    _state.longitudeDeg = 0.0;
    _state.speedMps     = 0.0f;
    _state.courseDeg    = 0.0f;
    _state.altitudeM    = 0.0f;
    _state.satellites   = 0;
    _state.hdop         = 99;
    _state.fixQuality   = 0;
    _state.lastFixMs    = 0;
    _state.hasFix       = false;
}

GpsModule::~GpsModule() { }

/* ========================================================================= *
 *  begin()
 * ========================================================================= */
void GpsModule::begin() {
    // GPS_SERIAL is defined in wiring_config.h — picks Serial1 on Mega/Due,
    // or a SoftwareSerial instance on boards without a free hardware UART.
    GPS_SERIAL.begin(GPS_BAUD);
    DBG_PRINT(F("[GPS ] UART started at "));
    DBG_PRINTLN(GPS_BAUD);
}

/* ========================================================================= *
 *  update()
 * ========================================================================= */
void GpsModule::update() {
    // Pump every available byte through the TinyGPSPlus state machine.
    // TinyGPSPlus parses incrementally and is interrupt-safe at the UART
    // level — there is no need to disable interrupts here.
    while (GPS_SERIAL.available() > 0) {
        int c = GPS_SERIAL.read();
        if (_gps.encode(static_cast<char>(c))) {
            // encode() returns true when a complete sentence was processed.
        }
    }

    // ---- Update the snapshot struct -----------------------------------
    if (_gps.location.isValid()) {
        _state.latitudeDeg  = _gps.location.lat();
        _state.longitudeDeg = _gps.location.lng();
        _state.hasFix       = true;
        _state.lastFixMs    = millis();
    } else {
        // If we haven't seen a valid fix within the timeout, clear hasFix.
        if (isTimedOut(_state.lastFixMs, GPS_FIX_TIMEOUT_MS)) {
            _state.hasFix = false;
        }
    }

    if (_gps.speed.isValid()) {
        // TinyGPSPlus returns speed in knots → convert to m/s (×0.514444).
        _state.speedMps = _gps.speed.mps();
    }
    if (_gps.course.isValid()) {
        _state.courseDeg = static_cast<float>(_gps.course.deg());
    }
    if (_gps.altitude.isValid()) {
        _state.altitudeM = static_cast<float>(_gps.altitude.meters());
    }
    if (_gps.satellites.isValid()) {
        _state.satellites = _gps.satellites.value();
    }
    if (_gps.hdop.isValid()) {
        _state.hdop = _gps.hdop.value() > 255 ? 255 : _gps.hdop.value();
    }

    // Fix quality: 0 = no fix, 1 = GPS, 2 = DGPS.  We can read this from the
    // underlying TinyGPSPlus integer field ($GPGGA sentence).
    // The TinyGPSPlus API exposes this via the .location object indirectly;
    // here we approximate: quality = (hasFix ? 1 : 0).
    _state.fixQuality = _state.hasFix ? 1u : 0u;
}

/* ========================================================================= *
 *  distanceToM / bearingTo
 * ========================================================================= */
float GpsModule::distanceToM(double targetLat, double targetLon) const {
    if (!_state.hasFix) return NAN;
    return haversineM(static_cast<float>(_state.latitudeDeg),
                      static_cast<float>(_state.longitudeDeg),
                      static_cast<float>(targetLat),
                      static_cast<float>(targetLon));
}

float GpsModule::bearingTo(double targetLat, double targetLon) const {
    if (!_state.hasFix) return NAN;
    return bearingDeg(static_cast<float>(_state.latitudeDeg),
                      static_cast<float>(_state.longitudeDeg),
                      static_cast<float>(targetLat),
                      static_cast<float>(targetLon));
}

/* ========================================================================= *
 *  isFresh / isNavigable
 * ========================================================================= */
bool GpsModule::isFresh() const {
    return !isTimedOut(_state.lastFixMs, GPS_FIX_TIMEOUT_MS);
}

bool GpsModule::isNavigable() const {
    // Require a fresh fix AND at least 4 satellites AND HDOP ≤ 5.
    return _state.hasFix
        && _state.satellites >= 4
        && _state.hdop       <= 5;
}

}  // namespace avcs
