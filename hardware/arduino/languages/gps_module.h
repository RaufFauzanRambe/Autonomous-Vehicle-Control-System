/**
 * @file    gps_module.h
 * @brief   Wrapper around TinyGPSPlus for NMEA GPS parsing.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#ifndef GPS_MODULE_H
#define GPS_MODULE_H
#pragma once

#include <Arduino.h>
#include <TinyGPSPlus.h>
#include "arduino_config.h"
#include "wiring_config.h"

namespace avcs {

/**
 * @brief Snapshot of the GPS receiver state.
 */
struct GpsState {
    double  latitudeDeg;   ///< WGS-84 latitude  (degrees).
    double  longitudeDeg;  ///< WGS-84 longitude (degrees).
    float   speedMps;      ///< Ground speed (m/s).
    float   courseDeg;     ///< Course over ground (degrees true).
    float   altitudeM;     ///< Altitude above MSL (m).
    uint8_t satellites;    ///< Number of satellites used in the fix.
    uint8_t hdop;          ///< Horizontal dilution of precision (×10).
    uint8_t fixQuality;    ///< 0=no fix, 1=GPS, 2=DGPS, ...
    uint32_t lastFixMs;    ///< millis() of the last valid fix.
    bool    hasFix;        ///< Convenience boolean.
};

/**
 * @brief TinyGPSPlus wrapper providing higher-level helpers.
 *
 * Usage:
 * @code
 *   avcs::GpsModule gps;
 *   gps.begin();
 *   ...
 *   gps.update();           // call from main loop
 *   if (gps.state().hasFix) {
 *       float d = gps.distanceToM(targetLat, targetLon);
 *       float b = gps.bearingTo(targetLat, targetLon);
 *   }
 * @endcode
 */
class GpsModule {
public:
    GpsModule();
    ~GpsModule();

    /// Start the GPS UART.  Call from setup().
    void begin();

    /**
     * @brief  Pump the GPS UART through the TinyGPSPlus parser.
     *
     * Call this as frequently as possible (ideally every loop iteration).
     * The parser is incremental and cheap.
     */
    void update();

    /// Read-only access to the parsed state.
    inline const GpsState& state() const { return _state; }

    /**
     * @brief  Great-circle distance from the current fix to a target.
     * @return Distance in metres (NaN if no fix).
     */
    float distanceToM(double targetLat, double targetLon) const;

    /**
     * @brief  Initial bearing from the current fix to a target.
     * @return Bearing in degrees true (NaN if no fix).
     */
    float bearingTo(double targetLat, double targetLon) const;

    /// True if the last fix is fresher than GPS_FIX_TIMEOUT_MS.
    bool  isFresh() const;

    /// True if the fix quality & HDOP meet minimum navigation requirements.
    bool  isNavigable() const;

private:
    TinyGPSPlus _gps;       ///< Underlying parser object.
    GpsState    _state;
};

}  // namespace avcs

#endif // GPS_MODULE_H
