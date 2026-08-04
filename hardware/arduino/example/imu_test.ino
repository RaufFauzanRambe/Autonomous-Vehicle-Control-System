/**
 * @file    imu_test.ino
 * @brief   Adafruit BNO055 IMU test sketch.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 *
 * @details
 * Reads the BNO055's Euler angles, quaternion, linear acceleration,
 * gyroscope, and per-sensor calibration status.  Output is formatted
 * for the **Arduino Serial Plotter** so you can visualise the orientation
 * live:
 *
 *     YAW:<yaw>,PITCH:<pitch>,ROLL:<roll>,CAL_SYS:<sys>,CAL_GYR:<gyr>,...
 *
 * Calibration procedure (Adafruit BNO055 datasheet §3.11):
 *
 *   1. SYSTEM calibration requires all subsystems to be calibrated.
 *   2. GYRO: leave the board still for a few seconds.
 *   3. ACCEL: place the board in 6 stable positions (each face down).
 *   4. MAG:  move the board in a figure-8 pattern through the air.
 *
 * Once SYS=3 the calibration profile is stored automatically in the
 * BNO055's internal flash and survives power-cycles.
 */

#include <Arduino.h>
#include <Wire.h>
#include <Adafruit_BNO055.h>
#include <utility/imumaths.h>

static constexpr uint32_t DEBUG_BAUD       = 115200;
static constexpr uint32_t PRINT_INTERVAL   = 50;     // 20 Hz
static constexpr uint8_t  BNO055_ADDRESS   = 0x28;   // ADR tied low

static Adafruit_BNO055 bno = Adafruit_BNO055(55, BNO055_ADDRESS);

/**
 * @brief Pretty-print a 3-vector with prefix labels.
 */
static void printVec(const char *label, float x, float y, float z, int decimals) {
    Serial.print(label);
    Serial.print(F(":"));
    Serial.print(x, decimals); Serial.print(F(","));
    Serial.print(y, decimals); Serial.print(F(","));
    Serial.print(z, decimals);
}

void setup() {
    Serial.begin(DEBUG_BAUD);
    while (!Serial) { ; }
    Serial.println();
    Serial.println(F("============================================"));
    Serial.println(F(" AVCS — imu_test.ino (BNO055)"));
    Serial.println(F("============================================"));

    Wire.begin();
    Wire.setClock(400000);   // BNO055 supports fast-mode I²C

    if (!bno.begin()) {
        Serial.println(F("[IMU] BNO055 NOT detected — check wiring!"));
        Serial.println(F("      SDA = A4/20, SCL = A5/21, VCC = 3V3"));
        while (true) { delay(10); }  // halt
    }

    // Use the external 32.768 kHz crystal for best accuracy.
    bno.setExtCrystalUse(true);

    Serial.println(F("[IMU] BNO055 detected, NDOF mode enabled."));
    Serial.println(F("[IMU] Calibrate by moving the board:"));
    Serial.println(F("      - gyro : keep still until gyr=3"));
    Serial.println(F("      - accel: place in 6 stable positions"));
    Serial.println(F("      - mag  : figure-8 motion in air"));
    Serial.println();
    delay(500);
}

void loop() {
    static uint32_t lastPrint = 0;
    if ((millis() - lastPrint) < PRINT_INTERVAL) return;
    lastPrint = millis();

    // ---- Euler angles (degrees) --------------------------------------
    imu::Vector<3> euler = bno.getVector(Adafruit_BNO055::VECTOR_EULER);

    // ---- Quaternion (normalised, w-first) ----------------------------
    imu::Quaternion q = bno.getQuat();

    // ---- Linear acceleration (gravity removed) -----------------------
    imu::Vector<3> lin = bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);

    // ---- Gyroscope (rad/s) -------------------------------------------
    imu::Vector<3> gyr = bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);

    // ---- Calibration status ------------------------------------------
    uint8_t sys, gyr_cal, acc, mag;
    bno.getCalibration(&sys, &gyr_cal, &acc, &mag);

    // ---- Print plotter-friendly CSV line -----------------------------
    // The Arduino Serial Plotter will graph the first N labelled columns.
    Serial.print(F("YAW:"));   Serial.print(euler.x(), 2); Serial.print(F(","));
    Serial.print(F("PITCH:")); Serial.print(euler.y(), 2); Serial.print(F(","));
    Serial.print(F("ROLL:"));  Serial.print(euler.z(), 2); Serial.print(F(","));

    Serial.print(F("QW:")); Serial.print(q.w(), 3); Serial.print(F(","));
    Serial.print(F("QX:")); Serial.print(q.x(), 3); Serial.print(F(","));
    Serial.print(F("QY:")); Serial.print(q.y(), 3); Serial.print(F(","));
    Serial.print(F("QZ:")); Serial.print(q.z(), 3); Serial.print(F(","));

    printVec("LIN", lin.x(), lin.y(), lin.z(), 2); Serial.print(F(","));
    printVec("GYR", gyr.x(), gyr.y(), gyr.z(), 2); Serial.print(F(","));

    Serial.print(F("CAL_SYS:")); Serial.print(sys);     Serial.print(F(","));
    Serial.print(F("CAL_GYR:")); Serial.print(gyr_cal); Serial.print(F(","));
    Serial.print(F("CAL_ACC:")); Serial.print(acc);     Serial.print(F(","));
    Serial.print(F("CAL_MAG:")); Serial.println(mag);

    // ---- Human-readable status every 2 s -----------------------------
    static uint32_t lastVerbose = 0;
    if ((millis() - lastVerbose) >= 2000) {
        lastVerbose = millis();
        Serial.println();
        Serial.print(F("Euler (yaw/pitch/roll) = "));
        Serial.print(euler.x(), 1); Serial.print(F(" / "));
        Serial.print(euler.y(), 1); Serial.print(F(" / "));
        Serial.println(euler.z(), 1);

        Serial.print(F("Calibration  sys=")); Serial.print(sys);
        Serial.print(F("  gyr="));             Serial.print(gyr_cal);
        Serial.print(F("  acc="));             Serial.print(acc);
        Serial.print(F("  mag="));             Serial.println(mag);

        if (sys == 3 && gyr_cal == 3 && acc == 3 && mag == 3) {
            Serial.println(F("  -> FULLY CALIBRATED"));
        } else {
            Serial.println(F("  -> keep moving / repositioning"));
        }
        Serial.println();
    }
}
