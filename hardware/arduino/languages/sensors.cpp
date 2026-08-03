/**
 * @file    sensors.cpp
 * @brief   Implementation of the ultrasonic driver and sensor-fusion manager.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#include "sensors.h"
#include "utilities.h"
#include <math.h>

namespace avcs {

/* ========================================================================= *
 *  UltrasonicSensor
 * ========================================================================= */

UltrasonicSensor::UltrasonicSensor(uint8_t trigPin, uint8_t echoPin, float maxRangeM)
    : _trigPin(trigPin)
    , _echoPin(echoPin)
    , _maxRangeM(maxRangeM)
{
    _last.distanceM   = NAN;
    _last.timestampMs = 0;
    _last.valid       = false;
}

void UltrasonicSensor::begin() {
    pinMode(_trigPin, OUTPUT);
    pinMode(_echoPin, INPUT);
    // Ensure trigger starts LOW so the first pulse is clean.
    digitalWrite(_trigPin, LOW);
    delayMicroseconds(5);
}

UltrasonicSample UltrasonicSensor::read() {
    // 1. Fire a 12 µs trigger pulse (HC-SR04 spec requires ≥10 µs).
    digitalWrite(_trigPin, LOW);
    delayMicroseconds(2);
    digitalWrite(_trigPin, HIGH);
    delayMicroseconds(12);
    digitalWrite(_trigPin, LOW);

    // 2. Wait for ECHO rising edge, then measure the high-time.
    //    pulseInLong() uses micros() internally — interrupt-safe on AVR.
    uint32_t echoUs = pulseInLong(_echoPin, HIGH, ULTRASONIC_TIMEOUT_US);

    UltrasonicSample s;
    s.timestampMs = millis();

    if (echoUs == 0) {
        // No echo received within timeout — out of range or sensor fault.
        s.distanceM = NAN;
        s.valid     = false;
    } else {
        // d = (t × c) / 2, with c = 343 m/s = 0.000343 m/µs
        s.distanceM = (echoUs * SPEED_OF_SOUND_MM_PER_US) / 2.0f / 1000.0f;
        // Sanity-check against the sensor's documented max range.
        s.valid = (s.distanceM > 0.02f && s.distanceM <= _maxRangeM);
        if (!s.valid) s.distanceM = NAN;
    }

    _last = s;
    return s;
}

/* ========================================================================= *
 *  SensorManager
 * ========================================================================= */

SensorManager::SensorManager()
    : _front(US_FRONT_TRIG_PIN, US_FRONT_ECHO_PIN)
    , _rear (US_REAR_TRIG_PIN,  US_REAR_ECHO_PIN)
    , _left (US_LEFT_TRIG_PIN,  US_LEFT_ECHO_PIN)
    , _right(US_RIGHT_TRIG_PIN, US_RIGHT_ECHO_PIN)
    , _idxFront(0), _idxRear(0), _idxLeft(0), _idxRight(0)
    , _batteryFiltered(NAN)
    , _lastUsSampleMs(0)
{
    // Zero the moving-average buffers so the first mean isn't garbage.
    for (uint8_t i = 0; i < US_FILTER_TAPS; i++) {
        _bufFront[i] = _bufRear[i] = _bufLeft[i] = _bufRight[i] = 0.0f;
    }
    _state.obstacleFrontM     = NAN;
    _state.obstacleRearM      = NAN;
    _state.obstacleLeftM      = NAN;
    _state.obstacleRightM     = NAN;
    _state.obstacleMinForwardM = NAN;
    _state.batteryV           = 0.0f;
    _state.motorCurrentA      = 0.0f;
    _state.timestampMs        = 0;
}

SensorManager::~SensorManager() { /* no dynamic allocations */ }

void SensorManager::begin() {
    _front.begin();
    _rear .begin();
    _left .begin();
    _right.begin();

    pinMode(BATTERY_VOLTAGE_PIN, INPUT);
    pinMode(MOTOR_CURRENT_PIN,   INPUT);

#if defined(ANALOG_READ_RESOLUTION)
    analogReadResolution(ADC_RESOLUTION_BITS);
#endif

    DBG_PRINTLN(F("[SENS] SensorManager initialised"));
}

void SensorManager::update() {
    /* ----------------------------------------------------------------- *
     *  Ultrasonic sensors are sampled at ULTRASONIC_SAMPLE_HZ to avoid
     *  acoustic cross-talk between adjacent transducers.
     * ----------------------------------------------------------------- */
    uint32_t now = millis();
    uint32_t usPeriodMs = 1000UL / ULTRASONIC_SAMPLE_HZ;
    if ((now - _lastUsSampleMs) >= usPeriodMs) {
        _lastUsSampleMs = now;

        // Sample each channel and push through a 5-tap moving average.
        UltrasonicSample f = _front.read();
        UltrasonicSample r = _rear .read();
        UltrasonicSample l = _left .read();
        UltrasonicSample ri = _right.read();

        // Replace NaN with a large sentinel (10 m, beyond HC-SR04 range)
        // so the moving-average doesn't collapse to NaN when one reading
        // fails — validity is still tracked by UltrasonicSample.valid.
        constexpr float OUT_OF_RANGE_M = 10.0f;
        float fD = f.valid ? f.distanceM : OUT_OF_RANGE_M;
        float rD = r.valid ? r.distanceM : OUT_OF_RANGE_M;
        float lD = l.valid ? l.distanceM : OUT_OF_RANGE_M;
        float riD = ri.valid ? ri.distanceM : OUT_OF_RANGE_M;

        _state.obstacleFrontM = movingAverage(_bufFront, US_FILTER_TAPS, fD,  &_idxFront);
        _state.obstacleRearM  = movingAverage(_bufRear,  US_FILTER_TAPS, rD,  &_idxRear);
        _state.obstacleLeftM  = movingAverage(_bufLeft,  US_FILTER_TAPS, lD,  &_idxLeft);
        _state.obstacleRightM = movingAverage(_bufRight, US_FILTER_TAPS, riD, &_idxRight);

        // Compute the minimum forward-facing distance — drives the safety
        // decision in the controller.
        float fwdMin = fminf(_state.obstacleFrontM, _state.obstacleLeftM);
        fwdMin = fminf(fwdMin, _state.obstacleRightM);
        _state.obstacleMinForwardM = fwdMin;
    }

    /* ----------------------------------------------------------------- *
     *  Battery voltage — heavy low-pass (α=0.05) because ADC noise on the
     *  AVR is ~±2 LSB and we don't want false low-battery alarms.
     * ----------------------------------------------------------------- */
    constexpr float VREF = 5.0f;            // AVR default ADC reference
    constexpr float DIV  = BATTERY_DIV_RATIO;
    uint16_t adcV = analogRead(BATTERY_VOLTAGE_PIN);
    float vRaw = adcToVoltage(adcV, VREF, ADC_RESOLUTION_BITS, DIV);

    if (isnan(_batteryFiltered)) {
        _batteryFiltered = vRaw;            // first sample — prime the filter
    } else {
        _batteryFiltered = lowPassFilter(_batteryFiltered, vRaw, 0.05f);
    }
    _state.batteryV = _batteryFiltered;

    /* ----------------------------------------------------------------- *
     *  Motor current — assume 0.1 Ω shunt + 50× gain → 5 V/A.  Just a
     *  rough estimate here; calibrate per build.
     * ----------------------------------------------------------------- */
    uint16_t adcI = analogRead(MOTOR_CURRENT_PIN);
    _state.motorCurrentA = (adcToVoltage(adcI, VREF, ADC_RESOLUTION_BITS, 1.0f)) * 1.0f;

    _state.timestampMs = now;
}

}  // namespace avcs
