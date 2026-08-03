/**
 * @file    utilities.cpp
 * @brief   Implementation of general-purpose helper functions.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#include "utilities.h"
#include <math.h>

namespace avcs {

/* ========================================================================= *
 *  Math helpers
 * ========================================================================= */

/**
 * @brief  Arduino's map() extended to floating-point numbers.
 * @param  x        Value to remap.
 * @param  inMin    Lower bound of the input range.
 * @param  inMax    Upper bound of the input range.
 * @param  outMin   Lower bound of the output range.
 * @param  outMax   Upper bound of the output range.
 * @return Remapped value (not clamped).
 */
float mapFloat(float x, float inMin, float inMax, float outMin, float outMax) {
    // Guard against division-by-zero (degenerate input range).
    if (fabsf(inMax - inMin) < 1e-6f) return outMin;
    return (x - inMin) * (outMax - outMin) / (inMax - inMin) + outMin;
}

/** @brief Clamp a float to [lo, hi]. */
float clamp(float value, float lo, float hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

/** @brief Clamp an int to [lo, hi]. */
int clampInt(int value, int lo, int hi) {
    if (value < lo) return lo;
    if (value > hi) return hi;
    return value;
}

/** @brief Degrees → radians. */
float deg2rad(float deg) { return deg * (M_PI / 180.0f); }

/** @brief Radians → degrees. */
float rad2deg(float rad) { return rad * (180.0f / M_PI); }

/**
 * @brief  Linear interpolation between two angles (degrees), taking the
 *         shortest path around the circle.
 * @param  a  Start angle (degrees, any range).
 * @param  b  End angle (degrees, any range).
 * @param  t  Interpolant in [0,1].
 * @return Interpolated angle in degrees.
 */
float angleLerp(float a, float b, float t) {
    float diff = fmodf(b - a + 540.0f, 360.0f) - 180.0f; // shortest delta
    return a + diff * t;
}

/** @brief Wrap an angle (degrees) to the range [-180, 180). */
float wrapAngle180(float angle) {
    while (angle < -180.0f) angle += 360.0f;
    while (angle >= 180.0f) angle -= 360.0f;
    return angle;
}

/** @brief Wrap an angle (degrees) to the range [0, 360). */
float wrapAngle360(float angle) {
    while (angle < 0.0f)    angle += 360.0f;
    while (angle >= 360.0f) angle -= 360.0f;
    return angle;
}

/**
 * @brief  Great-circle distance between two WGS-84 coordinates.
 * @param  lat1  Latitude of point 1 (degrees).
 * @param  lon1  Longitude of point 1 (degrees).
 * @param  lat2  Latitude of point 2 (degrees).
 * @param  lon2  Longitude of point 2 (degrees).
 * @return Distance in metres.
 *
 * Uses the haversine formula — numerically stable for small distances.
 */
float haversineM(float lat1, float lon1, float lat2, float lon2) {
    constexpr float R = 6371000.0f; // Earth radius (m)
    float phi1 = deg2rad(lat1);
    float phi2 = deg2rad(lat2);
    float dPhi = deg2rad(lat2 - lat1);
    float dLam = deg2rad(lon2 - lon1);
    float a = sinf(dPhi / 2.0f) * sinf(dPhi / 2.0f) +
              cosf(phi1) * cosf(phi2) *
              sinf(dLam / 2.0f) * sinf(dLam / 2.0f);
    float c = 2.0f * atan2f(sqrtf(a), sqrtf(1.0f - a));
    return R * c;
}

/**
 * @brief  Initial bearing (forward azimuth) from point 1 to point 2.
 * @return Bearing in degrees, true north = 0°, clockwise positive.
 */
float bearingDeg(float lat1, float lon1, float lat2, float lon2) {
    float phi1 = deg2rad(lat1);
    float phi2 = deg2rad(lat2);
    float dLam = deg2rad(lon2 - lon1);
    float y = sinf(dLam) * cosf(phi2);
    float x = cosf(phi1) * sinf(phi2) - sinf(phi1) * cosf(phi2) * cosf(dLam);
    return wrapAngle360(rad2deg(atan2f(y, x)));
}

/* ========================================================================= *
 *  Digital filters
 * ========================================================================= */

/**
 * @brief  First-order low-pass filter: y[n] = α·x[n] + (1-α)·y[n-1].
 * @param  prev    Previous output (y[n-1]).
 * @param  input   New sample (x[n]).
 * @param  alpha   Filter coefficient in [0,1]; smaller α = heavier smoothing.
 * @return Filtered value.
 *
 * α ≈ dt / (RC + dt) where RC is the filter time constant.
 */
float lowPassFilter(float prev, float input, float alpha) {
    return alpha * input + (1.0f - alpha) * prev;
}

/**
 * @brief  Simple moving-average filter over a circular buffer.
 * @param  buf     Pointer to a float array of length `size`.
 * @param  size    Number of slots in `buf`.
 * @param  newVal  New sample to insert.
 * @param  idx     Pointer to the current write index (maintained by caller).
 * @return Arithmetic mean of the buffer contents.
 */
float movingAverage(float *buf, uint8_t size, float newVal, uint8_t *idx) {
    buf[*idx] = newVal;
    *idx = (*idx + 1) % size;
    float sum = 0.0f;
    for (uint8_t i = 0; i < size; i++) sum += buf[i];
    return sum / static_cast<float>(size);
}

/* ========================================================================= *
 *  Conversion helpers
 * ========================================================================= */

/** @brief Convert motor-shaft RPM to linear speed (m/s) given wheel diameter. */
float rpmToMps(float rpm, float wheelDiameterMm) {
    float circumferenceM = (M_PI * wheelDiameterMm) / 1000.0f;
    return (rpm / 60.0f) * circumferenceM;
}

/** @brief Convert linear speed (m/s) to motor-shaft RPM. */
float mpsToRpm(float mps, float wheelDiameterMm) {
    float circumferenceM = (M_PI * wheelDiameterMm) / 1000.0f;
    if (circumferenceM < 1e-6f) return 0.0f;
    return (mps / circumferenceM) * 60.0f;
}

/**
 * @brief  Convert an ADC reading to a physical voltage.
 * @param  adc           Raw ADC count.
 * @param  vref          ADC reference voltage (V).
 * @param  bits          ADC resolution (10 or 12 typically).
 * @param  dividerRatio  Ratio of physical voltage presented to the pin
 *                       (R2 / (R1+R2)).  Pass 1.0f for direct measurement.
 * @return Voltage at the source (V).
 */
float adcToVoltage(uint16_t adc, float vref, uint8_t bits, float dividerRatio) {
    if (dividerRatio < 1e-6f) return 0.0f;
    float maxCount = static_cast<float>((1UL << bits) - 1UL);
    return (static_cast<float>(adc) / maxCount) * vref / dividerRatio;
}

/* ========================================================================= *
 *  Checksums
 * ========================================================================= */

/**
 * @brief  Compute CRC-8 using the Dallas/Maxim polynomial (X^8+X^5+X^4+1, 0x31).
 * @param  data  Pointer to the byte buffer.
 * @param  len   Number of bytes.
 * @return 8-bit CRC.
 *
 * Used for telemetry-packet integrity checks.
 */
uint8_t crc8(const uint8_t *data, size_t len) {
    uint8_t crc = 0x00;
    for (size_t i = 0; i < len; i++) {
        crc ^= data[i];
        for (uint8_t b = 0; b < 8; b++) {
            if (crc & 0x80) crc = (crc << 1) ^ 0x07;  // 0x07 = poly 0x31 reversed
            else            crc <<= 1;
        }
    }
    return crc;
}

/* ========================================================================= *
 *  Time helpers
 * ========================================================================= */

/** @brief True if `timeoutMillis` have elapsed since `lastMillis`. */
bool isTimedOut(uint32_t lastMillis, uint32_t timeoutMillis) {
    return (millis() - lastMillis) > timeoutMillis;
}

/** @brief Compute the delta between two micros() readings, handling wrap. */
uint32_t microsDelta(uint32_t prev, uint32_t now) {
    return now - prev;  // unsigned subtraction handles 32-bit wrap correctly
}

}  // namespace avcs
