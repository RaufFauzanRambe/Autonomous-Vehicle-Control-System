/**
 * @file sensor_reader.h
 * @brief SensorReader — I²C + ADC + PCNT-encoder acquisition with fusion.
 * @author AVC team
 * @date 2024
 * @license MIT
 */
#ifndef AVC_ESP32_SENSOR_READER_H
#define AVC_ESP32_SENSOR_READER_H

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO055.h>
#include <Adafruit_BMP280.h>
#include <Adafruit_VL53L0X.h>
#include <driver/pcnt.h>
#include "config.h"

/// Quaternion + Euler + gyro + accel fused by the BNO055.
struct ImuData {
    float qx, qy, qz, qw;     ///< Quaternion (unit length).
    float roll, pitch, yaw;   ///< Euler angles (rad).
    float gx, gy, gz;         ///< Angular velocity (rad/s).
    float ax, ay, az;         ///< Linear acceleration (m/s²).
    uint8_t  calib_sys;       ///< 0..3 — system calibration.
    uint8_t  calib_gyro;
    uint8_t  calib_accel;
    uint8_t  calib_mag;
};

/// Barometric pressure + derived altitude.
struct BaroData {
    float temperature_c;   ///< °C
    float pressure_pa;     ///< Pa
    float altitude_m;      ///< m above sea level (needs sea-level reference)
};

/// Time-of-flight distance (VL53L0X).
struct TofData {
    float   distance_mm;   ///< Distance to obstacle.
    float   signal_rate;   ///< Return signal strength.
    bool    valid;         ///< Timeout / out-of-range → false.
};

/// Quadrature encoder + derived wheel speed.
struct EncoderData {
    int32_t  count;         ///< Raw PCNT count since last reset.
    float    rpm;           ///< Revolutions per minute (motor shaft).
    float    omega_rad_s;   ///< Wheel angular velocity (rad/s).
    float    v_linear_ms;   ///< Linear speed (m/s) at the wheel.
};

/// Aggregated sensor frame, sampled at TASK_SENSOR_HZ.
struct SensorFrame {
    uint32_t   tick;        ///< FreeRTOS tick at sample.
    uint32_t   seq;         ///< Monotonic counter.
    ImuData    imu;
    BaroData   baro;
    TofData    tof;
    EncoderData enc_l;
    EncoderData enc_r;
    float      battery_v;
    float      battery_pct;
};

/**
 * @class SensorReader
 * @brief Owns all sensor hardware and exposes a single `read()` that fills a
 *        SensorFrame. Designed to be called from a 100 Hz FreeRTOS task.
 *
 * Peripherals:
 *  - I²C bus on I2C_SDA_PIN / I2C_SCL_PIN at 400 kHz.
 *  - BNO055 IMU in NDOF mode (internal fusion).
 *  - BMP280 barometer (forced mode, oversampling ×1).
 *  - VL53L0X ToF (continuous, 33 ms timing budget).
 *  - Two PCNT units for quadrature encoders.
 *  - ADC1 channel for battery voltage.
 */
class SensorReader {
public:
    SensorReader();

    /**
     * @brief Initialise I²C, IMU, baro, ToF, PCNT and ADC.
     * @return true on success (partial success allowed — see warnings).
     */
    bool begin();

    /**
     * @brief Read a complete SensorFrame (atomic snapshot).
     * @param[out] out Destination frame.
     * @return true if all critical sensors returned fresh data.
     */
    bool read(SensorFrame& out);

    /**
     * @brief Reset PCNT counters to zero (e.g. on motor stop).
     */
    void resetEncoders();

    /**
     * @brief Helper — convert PCNT count to wheel radians.
     */
    static float countsToRad(int32_t count);

    /**
     * @brief Helper — convert wheel radians to linear metres.
     */
    static float radToMetres(float rad);

    /**
     * @brief Perform an I²C bus scan; writes found addresses to `out`.
     * @param[out] out Buffer of 128 bytes (one per address).
     * @return Number of devices found.
     */
    uint8_t i2cScan(uint8_t* out);

    // ---- Accessors ----
    inline bool imuOK()  const { return _imuOk; }
    inline bool baroOK() const { return _baroOk; }
    inline bool tofOK()  const { return _tofOk; }

private:
    /// Initialise the BNO055 in NDOF mode.
    bool initImu();
    /// Initialise the BMP280 in forced mode.
    bool initBaro();
    /// Initialise the VL53L0X in continuous mode.
    bool initTof();
    /// Configure two PCNT units for quadrature decoding.
    void initEncoders();
    /// Configure ADC1 for battery voltage.
    void initAdc();

    /// Snapshot an encoder unit (clears the count).
    int32_t readPcnt(pcnt_unit_t unit);

    Adafruit_BNO055  _bno;     ///< BNO055 driver (addr 0x28).
    Adafruit_BMP280  _bmp;     ///< BMP280 driver (addr 0x76).
    Adafruit_VL53L0X _lox;     ///< VL53L0X driver (addr 0x29).

    bool _imuOk   = false;
    bool _baroOk  = false;
    bool _tofOk   = false;

    uint32_t _seq = 0;
    float    _seaLevelPa = 101325.0f;   ///< Reference for altitude calc.

    /// Used by countsToRad() — precomputed.
    static constexpr float kCountsPerRev = ENCODER_PPR * 4.0f;
    static constexpr float kRadPerCount  = (2.0f * PI) / kCountsPerRev;
};

#endif // AVC_ESP32_SENSOR_READER_H
