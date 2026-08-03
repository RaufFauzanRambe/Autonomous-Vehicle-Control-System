/**
 * @file    sensors.h
 * @brief   Declarations for the sensor subsystem (ultrasonic + sensor fusion).
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#ifndef SENSORS_H
#define SENSORS_H
#pragma once

#include <Arduino.h>
#include "arduino_config.h"
#include "wiring_config.h"

namespace avcs {

/**
 * @brief Single ultrasonic range reading.
 */
struct UltrasonicSample {
    float   distanceM;      ///< Measured distance in metres (NaN if out of range).
    uint32_t timestampMs;   ///< millis() at the moment of measurement.
    bool    valid;          ///< True if echo was received within timeout.
};

/**
 * @brief Driver for a single HC-SR04 ultrasonic sensor.
 *
 * The HC-SR04 fires an 8-cycle 40 kHz burst on the rising edge of a ≥10 µs
 * trigger pulse.  The ECHO line then goes high until the burst is received
 * back (or until the internal 38 ms timeout).  Distance is:
 *
 *     d = (echo_pulse_width_us × speed_of_sound) / 2
 *
 * The factor of 2 accounts for the round-trip (echo) distance.
 */
class UltrasonicSensor {
public:
    /**
     * @brief Construct a sensor.
     * @param trigPin  Trigger output pin.
     * @param echoPin  Echo input pin (must support pulseInLong()).
     * @param maxRangeM Maximum meaningful range in metres (default 4.5 m).
     */
    UltrasonicSensor(uint8_t trigPin, uint8_t echoPin, float maxRangeM = 4.5f);

    /// Configure pin modes.  Call once in setup().
    void begin();

    /**
     * @brief  Fire one measurement and block until the echo is received.
     * @note   Blocking call (up to ULTRASONIC_TIMEOUT_US µs).
     * @return Populated UltrasonicSample.
     */
    UltrasonicSample read();

    /**
     * @brief  Return the most recent valid sample (does not re-fire).
     */
    inline const UltrasonicSample& last() const { return _last; }

private:
    uint8_t         _trigPin;
    uint8_t         _echoPin;
    float           _maxRangeM;
    UltrasonicSample _last;
};

/**
 * @brief Aggregated, fused view of all vehicle sensors.
 *
 * This structure is filled by SensorManager::update() each control cycle and
 * consumed by the controller and the telemetry transmitter.
 */
struct FusedSensorState {
    // Obstacle distances (metres), NaN if invalid
    float obstacleFrontM;
    float obstacleRearM;
    float obstacleLeftM;
    float obstacleRightM;

    /// Minimum of all forward-facing sensors — used by the safety layer.
    float obstacleMinForwardM;

    /// Battery pack voltage (V).
    float batteryV;

    /// Motor current draw (A).
    float motorCurrentA;

    uint32_t timestampMs;
};

/**
 * @brief Top-level coordinator that owns all sensor objects and produces a
 *        single fused state snapshot per control cycle.
 */
class SensorManager {
public:
    SensorManager();
    ~SensorManager();

    /// Initialise every owned sensor.  Call from Arduino setup().
    void begin();

    /**
     * @brief Poll every sensor at its native rate and fuse the results.
     *
     * Inside this method we apply:
     *   - moving-average filtering of the ultrasonic readings (5-tap)
     *   - low-pass filtering of the battery voltage (α = 0.05)
     *   - computation of obstacleMinForwardM
     */
    void update();

    /// Read-only access to the most recent fused state.
    inline const FusedSensorState& state() const { return _state; }

    /// Read-only access to individual ultrasonic sensors.
    inline const UltrasonicSensor& frontUS() const { return _front; }
    inline const UltrasonicSensor& rearUS()  const { return _rear; }
    inline const UltrasonicSensor& leftUS()  const { return _left; }
    inline const UltrasonicSensor& rightUS() const { return _right; }

private:
    UltrasonicSensor _front;
    UltrasonicSensor _rear;
    UltrasonicSensor _left;
    UltrasonicSensor _right;

    /// 5-tap moving-average buffers for each ultrasonic channel.
    static constexpr uint8_t US_FILTER_TAPS = 5;
    float _bufFront[US_FILTER_TAPS];
    float _bufRear [US_FILTER_TAPS];
    float _bufLeft [US_FILTER_TAPS];
    float _bufRight[US_FILTER_TAPS];
    uint8_t _idxFront, _idxRear, _idxLeft, _idxRight;

    /// Filtered battery voltage (persistent across calls).
    float _batteryFiltered;

    /// Last time each ultrasonic sensor was sampled (rate limiting).
    uint32_t _lastUsSampleMs;

    FusedSensorState _state;
};

}  // namespace avcs

#endif // SENSORS_H
