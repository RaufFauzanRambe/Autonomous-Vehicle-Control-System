/**
 * @file    imu_module.h
 * @brief   Wrapper around the Adafruit BNO055 9-DOF IMU.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#ifndef IMU_MODULE_H
#define IMU_MODULE_H
#pragma once

#include <Arduino.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>
#include "arduino_config.h"
#include "wiring_config.h"

namespace avcs {

/**
 * @brief Calibration status of each BNO055 sensor subsystem.
 *
 * The BNO055 reports a 0-3 level for each sensor.  A "fully calibrated"
 * system has all four channels at level 3.
 */
struct ImuCalibration {
    uint8_t system;   ///< 0-3, where 3 = fully calibrated.
    uint8_t gyro;     ///< Gyroscope calibration level.
    uint8_t accel;    ///< Accelerometer calibration level.
    uint8_t mag;      ///< Magnetometer calibration level.
};

/**
 * @brief Snapshot of the IMU output.
 */
struct ImuState {
    // Euler angles (degrees)
    float yawDeg;
    float pitchDeg;
    float rollDeg;

    // Quaternion (w, x, y, z)
    float qw, qx, qy, qz;

    // Linear acceleration (m/s²) — gravity removed by the BNO055 fusion core
    float linAccelX, linAccelY, linAccelZ;

    // Angular velocity (rad/s)
    float gyroX, gyroY, gyroZ;

    ImuCalibration calib;
    uint32_t lastUpdateMs;
    bool valid;
};

/**
 * @brief Wrapper class providing a higher-level API around Adafruit_BNO055.
 *
 * The BNO055 includes an internal Cortex-M0 running a proprietary sensor
 * fusion algorithm, so we don't have to implement our own complementary or
 * Madgwick filter on the host MCU.  This class simply polls the sensor at
 * IMU_SAMPLE_HZ and stores the latest values.
 */
class ImuModule {
public:
    /**
     * @brief Construct an ImuModule.
     * @param address  I²C address of the BNO055 (default 0x28).
     */
    explicit ImuModule(uint8_t address = IMU_I2C_ADDRESS);

    ~ImuModule();

    /**
     * @brief Initialise the Wire bus and the BNO055 sensor.
     * @return True if the sensor was detected.
     */
    bool begin();

    /**
     * @brief  Poll the sensor at IMU_SAMPLE_HZ.
     *
     * Internally this method rate-limits itself based on the difference
     * between consecutive calls (using millis()), so it is safe to call
     * from the main loop at any frequency ≥ IMU_SAMPLE_HZ.
     */
    void update();

    /// Read-only access to the latest IMU snapshot.
    inline const ImuState& state() const { return _state; }

    /**
     * @brief  Trigger a fresh calibration-read from the sensor.
     * @return Populated ImuCalibration struct.
     *
     * Calibration procedure (Adafruit docs):
     *   1. Place the board flat and still until system == 3.
     *   2. Rotate slowly around all three axes until gyro == 3.
     *   3. Move in a figure-8 pattern until mag == 3.
     */
    ImuCalibration readCalibration();

private:
    Adafruit_BNO055 _bno;
    ImuState        _state;
    uint32_t        _lastSampleMs;
};

}  // namespace avcs

#endif // IMU_MODULE_H
