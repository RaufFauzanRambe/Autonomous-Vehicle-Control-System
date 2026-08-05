/**
 * @file sensor_reader.cpp
 * @brief SensorReader implementation — IMU + baro + ToF + encoders + ADC.
 * @author AVC team
 * @date 2024
 * @license MIT
 */

#include "sensor_reader.h"
#include <esp_adc_cal.h>
#include <driver/adc.h>

// ---------------------------------------------------------------------------
//  PCNT configuration
// ---------------------------------------------------------------------------
static constexpr pcnt_unit_t kPcntL = PCNT_UNIT_0;
static constexpr pcnt_unit_t kPcntR = PCNT_UNIT_1;

static constexpr pcnt_channel_t kPcntLChA = PCNT_CHANNEL_0;
static constexpr pcnt_channel_t kPcntLChB = PCNT_CHANNEL_1;
static constexpr pcnt_channel_t kPcntRChA = PCNT_CHANNEL_2;
static constexpr pcnt_channel_t kPcntRChB = PCNT_CHANNEL_3;

// ---------------------------------------------------------------------------
//  Construction / begin()
// ---------------------------------------------------------------------------
SensorReader::SensorReader()
    : _bno(55, BNO055_I2C_ADDR),
      _bmp(),
      _lox() {
}

bool SensorReader::begin() {
    initAdc();
    initEncoders();

    bool ok = true;
    _imuOk  = initImu();   ok &= _imuOk;
    _baroOk = initBaro();  ok &= _baroOk;
    _tofOk  = initTof();   ok &= _tofOk;
    return ok;
}

bool SensorReader::initImu() {
    if (!_bno.begin(Adafruit_BNO055::OPERATION_MODE_NDOF)) {
        AVC_LOGE("imu", "BNO055 not found @ 0x%02X", BNO055_I2C_ADDR);
        return false;
    }
    delay(50);
    _bno.setExtCrystalUse(true);
    uint8_t sys=0, gyr=0, acc=0, mag=0;
    _bno.getCalibration(&sys, &gyr, &acc, &mag);
    AVC_LOGI("imu", "BNO055 up. calib sys=%u gyr=%u acc=%u mag=%u",
             sys, gyr, acc, mag);
    return true;
}

bool SensorReader::initBaro() {
    if (!_bmp.begin(BMP280_I2C_ADDR)) {
        AVC_LOGE("baro", "BMP280 not found @ 0x%02X", BMP280_I2C_ADDR);
        return false;
    }
    _bmp.setSampling(Adafruit_BMP280::MODE_FORCED,
                     Adafruit_BMP280::SAMPLING_X1,   // temp
                     Adafruit_BMP280::SAMPLING_X1,   // press
                     Adafruit_BMP280::FILTER_OFF,
                     Adafruit_BMP280::STANDBY_MS_1);
    AVC_LOGI("baro", "BMP280 up @ 0x%02X", BMP280_I2C_ADDR);
    return true;
}

bool SensorReader::initTof() {
    if (!_lox.begin()) {
        AVC_LOGE("tof", "VL53L0X not found @ 0x%02X", VL53L0X_I2C_ADDR);
        return false;
    }
    _lox.setTimeout(50);
    _lox.startRangeContinuous(33);   // 30 Hz
    AVC_LOGI("tof", "VL53L0X up @ 0x%02X", VL53L0X_I2C_ADDR);
    return true;
}

void SensorReader::initEncoders() {
    // Configure PCNT for quadrature decoding (count on both edges, B = sign).
    auto cfg = [&](pcnt_unit_t unit, pcnt_channel_t ch,
                   int pulse, int ctrl) {
        pcnt_config_t c = {};
        c.pulse_gpio_num = pulse;
        c.ctrl_gpio_num  = ctrl;
        c.channel        = PCNT_CHANNEL_0;
        c.unit           = unit;
        c.pos_mode       = PCNT_COUNT_INC;   // increment on rising edge
        c.neg_mode       = PCNT_COUNT_DEC;   // decrement on falling edge
        c.lctrl_mode     = PCNT_MODE_REVERSE;
        c.hctrl_mode     = PCNT_MODE_KEEP;
        pcnt_unit_config(&c);

        // Second channel swaps pulse/ctrl for proper quadrature.
        pcnt_config_t c2 = c;
        c2.channel        = PCNT_CHANNEL_1;
        c2.pulse_gpio_num = ctrl;
        c2.ctrl_gpio_num  = pulse;
        c2.pos_mode       = PCNT_COUNT_DEC;
        c2.neg_mode       = PCNT_COUNT_INC;
        pcnt_unit_config(&c2);

        // Glitch filter ~100 ns.
        pcnt_set_filter_value(unit, 10);
        pcnt_filter_enable(unit);
        pcnt_counter_clear(unit);
        pcnt_count_mode_t mode;
        pcnt_get_count_mode(unit, &mode);
        pcnt_set_count_mode(unit, PCNT_COUNT_INC);
        pcnt_counter_resume(unit);
    };

    cfg(kPcntL, kPcntLChA, ENCODER_L_A_PIN, ENCODER_L_B_PIN);
    cfg(kPcntR, kPcntRChA, ENCODER_R_A_PIN, ENCODER_R_B_PIN);

    AVC_LOGI("enc", "PCNT L=%d R=%d initialised", kPcntL, kPcntR);
}

void SensorReader::initAdc() {
    adc1_config_width(ADC_WIDTH_BIT_12);
    // Bat divider ~11:1; full-scale 3.3 V → 11*3.3 = 36 V max.
    adc1_config_channel_atten(ADC1_CHANNEL_0, ADC_ATTEN_DB_11);
    AVC_LOGI("adc", "battery ADC initialised");
}

// ---------------------------------------------------------------------------
//  read()
// ---------------------------------------------------------------------------
bool SensorReader::read(SensorFrame& out) {
    out.tick = xTaskGetTickCount();
    out.seq  = _seq++;

    // ---- IMU ----
    if (_imuOk) {
        imu::Quaternion q = _bno.getQuat();
        out.imu.qx = q.x(); out.imu.qy = q.y();
        out.imu.qz = q.z(); out.imu.qw = q.w();

        imu::Vector<3> euler = _bno.getVector(Adafruit_BNO055::VECTOR_EULER);
        out.imu.roll  = euler.x() * DEG_TO_RAD;
        out.imu.pitch = euler.y() * DEG_TO_RAD;
        out.imu.yaw   = euler.z() * DEG_TO_RAD;

        imu::Vector<3> gyr = _bno.getVector(Adafruit_BNO055::VECTOR_GYROSCOPE);
        out.imu.gx = gyr.x() * DEG_TO_RAD;
        out.imu.gy = gyr.y() * DEG_TO_RAD;
        out.imu.gz = gyr.z() * DEG_TO_RAD;

        imu::Vector<3> acc = _bno.getVector(Adafruit_BNO055::VECTOR_LINEARACCEL);
        out.imu.ax = acc.x();
        out.imu.ay = acc.y();
        out.imu.az = acc.z();

        _bno.getCalibration(&out.imu.calib_sys, &out.imu.calib_gyro,
                            &out.imu.calib_accel, &out.imu.calib_mag);
    } else {
        memset(&out.imu, 0, sizeof(out.imu));
    }

    // ---- Barometer (forced mode: take a single sample) ----
    if (_baroOk) {
        _bmp.takeForcedMeasurement();
        out.baro.temperature_c = _bmp.readTemperature();
        out.baro.pressure_pa   = _bmp.readPressure();
        out.baro.altitude_m    = _bmp.readAltitude(_seaLevelPa / 100.0f);
    } else {
        out.baro = {};
    }

    // ---- ToF ----
    if (_tofOk) {
        uint16_t mm = _lox.readRangeContinuousMillimeters();
        out.tof.distance_mm = (float)mm;
        out.tof.valid = !_lox.timeoutOccurred() && mm < 8000;
    } else {
        out.tof = {};
    }

    // ---- Encoders ----
    int32_t lc = readPcnt(kPcntL);
    int32_t rc = readPcnt(kPcntR);
    out.enc_l.count = lc;
    out.enc_r.count = rc;
    // Convert counts → angular vel using the sample dt (1/TASK_SENSOR_HZ).
    constexpr float dt = 1.0f / TASK_SENSOR_HZ;
    out.enc_l.omega_rad_s = countsToRad(lc) / dt;
    out.enc_r.omega_rad_s = countsToRad(rc) / dt;
    // Motor-shaft RPM = (omega * 60) / (2π) * GEAR_RATIO.
    out.enc_l.rpm = (out.enc_l.omega_rad_s * 60.0f) / (2.0f * PI) * GEAR_RATIO;
    out.enc_r.rpm = (out.enc_r.omega_rad_s * 60.0f) / (2.0f * PI) * GEAR_RATIO;
    // Linear speed at wheel = omega * radius.
    out.enc_l.v_linear_ms = out.enc_l.omega_rad_s * (WHEEL_DIAMETER_M * 0.5f);
    out.enc_r.v_linear_ms = out.enc_r.omega_rad_s * (WHEEL_DIAMETER_M * 0.5f);

    // Battery is filled in by main.cpp using the BatteryMonitor class.
    out.battery_v   = 0.0f;
    out.battery_pct = 0.0f;

    return _imuOk;   // IMU is the critical sensor.
}

void SensorReader::resetEncoders() {
    pcnt_counter_clear(kPcntL);
    pcnt_counter_clear(kPcntR);
}

int32_t SensorReader::readPcnt(pcnt_unit_t unit) {
    int16_t raw = 0;
    pcnt_counter_clear(unit);
    pcnt_get_counter_value(unit, &raw);
    return (int32_t)raw;
}

float SensorReader::countsToRad(int32_t count) {
    return (float)count * kRadPerCount;
}

float SensorReader::radToMetres(float rad) {
    return rad * (WHEEL_DIAMETER_M * 0.5f);
}

uint8_t SensorReader::i2cScan(uint8_t* out) {
    uint8_t found = 0;
    for (uint8_t addr = 1; addr < 127; ++addr) {
        Wire.beginTransmission(addr);
        if (Wire.endTransmission() == 0) {
            if (out) out[found] = addr;
            found++;
            AVC_LOGI("i2c", "found device @ 0x%02X", addr);
        }
    }
    AVC_LOGI("i2c", "scan complete: %u device(s)", found);
    return found;
}
