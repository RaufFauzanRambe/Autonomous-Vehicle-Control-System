/**
 * @file    imu_module.cpp
 * @brief   Implementation of the BNO055 IMU wrapper.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 */
#include "imu_module.h"
#include "utilities.h"

namespace avcs {

/* ========================================================================= *
 *  Constructor / destructor
 * ========================================================================= */
ImuModule::ImuModule(uint8_t address)
    : _bno(Adafruit_BNO055(55, address))
{
    _state.yawDeg   = 0.0f;
    _state.pitchDeg = 0.0f;
    _state.rollDeg  = 0.0f;
    _state.qw = 1.0f; _state.qx = _state.qy = _state.qz = 0.0f;
    _state.linAccelX = _state.linAccelY = _state.linAccelZ = 0.0f;
    _state.gyroX = _state.gyroY = _state.gyroZ = 0.0f;
    _state.calib.system = _state.calib.gyro =
        _state.calib.accel = _state.calib.mag = 0;
    _state.lastUpdateMs = 0;
    _state.valid = false;
    _lastSampleMs = 0;
}

ImuModule::~ImuModule() { }

/* ========================================================================= *
 *  begin()
 * ========================================================================= */
bool ImuModule::begin() {
    // Initialise the Wire library — on AVR this is a no-op if already done.
    Wire.begin();

    // Adafruit_BNO055::begin() returns false if the sensor does not ACK.
    if (!_bno.begin()) {
        DBG_PRINTLN(F("[IMU ] BNO055 not detected — check wiring!"));
        _state.valid = false;
        return false;
    }

    // Use the internal oscillator and put the chip into NDOF fusion mode.
    // NDOF = nine degrees of freedom (accel + gyro + mag fusion).
    _bno.setExtCrystalUse(true);

    // Allow the chip to settle.
    delay(50);

    _state.valid = true;
    DBG_PRINTLN(F("[IMU ] BNO055 initialised, NDOF mode"));
    return true;
}

/* ========================================================================= *
 *  update()
 * ========================================================================= */
void ImuModule::update() {
    uint32_t now = millis();
    uint32_t periodMs = 1000UL / IMU_SAMPLE_HZ;
    if ((now - _lastSampleMs) < periodMs) return;
    _lastSampleMs = now;

    // ---- Euler angles (degrees) ----------------------------------------
    // getVector returns an imu::Vector<3> with x=heading(yaw), y=pitch, z=roll.
    imu::Vector<3> euler = _bno.getVector(Adafruit_BNO055::VECTOR_EULER);
    _state.yawDeg   = static_cast<float>(euler.x());
    _state.pitchDeg = static_cast<float>(euler.y());
    _state.rollDeg  = static_cast<float>(euler.z());

    // ---- Quaternion (normalised, w-first) ------------------------------
    imu::Quaternion q = _bno.getQuat();
    _state.qw = static_cast<float>(q.w());
    _state.qx = static_cast<float>(q.x());
    _state.qy = static_cast<float>(q.y());
    _state.qz = static_cast<float>(q.z());

    // ---- Linear acceleration (gravity removed) -------------------------
    imu::Vector<3> lin = _bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
    _state.linAccelX = static_cast<float>(lin.x());
    _state.linAccelY = static_cast<float>(lin.y());
    _state.linAccelZ = static_cast<float>(lin.z());

    // ---- Gyroscope (rad/s) ---------------------------------------------
    imu::Vector<3> gyr = _bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
    _state.gyroX = static_cast<float>(gyr.x());
    _state.gyroY = static_cast<float>(gyr.y());
    _state.gyroZ = static_cast<float>(gyr.z());

    // ---- Calibration status --------------------------------------------
    uint8_t sys, gyr, acc, mg;
    _bno.getCalibration(&sys, &gyr, &acc, &mg);
    _state.calib.system = sys;
    _state.calib.gyro   = gyr;
    _state.calib.accel  = acc;
    _state.calib.mag    = mg;

    _state.lastUpdateMs = now;
    _state.valid = true;
}

/* ========================================================================= *
 *  readCalibration()
 * ========================================================================= */
ImuCalibration ImuModule::readCalibration() {
    uint8_t sys, gyr, acc, mg;
    _bno.getCalibration(&sys, &gyr, &acc, &mg);
    _state.calib.system = sys;
    _state.calib.gyro   = gyr;
    _state.calib.accel  = acc;
    _state.calib.mag    = mg;
    return _state.calib;
}

}  // namespace avcs
