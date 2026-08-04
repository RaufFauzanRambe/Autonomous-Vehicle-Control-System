/**
 * @file    gps_test.ino
 * @brief   GPS module test sketch using TinyGPSPlus.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 *
 * @details
 * Opens `Serial1` at 9600 baud (NEO-6M / NEO-M8N default), pumps the
 * incoming NMEA stream through TinyGPSPlus, and prints a CSV-formatted
 * status line at 2 Hz:
 *
 *     [GPS] lat,lon,fix,sats,hdop,spd_mps,course_deg,alt_m,age_ms
 *
 * Usage:
 *   1. Upload to an Arduino Mega 2560 (Serial1 on pins 18/19).
 *   2. Wire GPS TX → Arduino pin 19 (RX1), GPS RX → pin 18 (TX1).
 *   3. Open the serial monitor at 115200 baud.
 *   4. Move outdoors — first cold-start fix can take 30–60 s.
 *
 * On boards without a free hardware UART (Uno), set USE_SOFTSERIAL below
 * and connect the GPS to the indicated software-serial pins.
 */

#include <Arduino.h>
#include <TinyGPSPlus.h>

// ---- Configuration ----------------------------------------------------
static constexpr uint32_t GPS_BAUD        = 9600;   // NEO-6M default
static constexpr uint32_t DEBUG_BAUD      = 115200;
static constexpr uint32_t PRINT_INTERVAL  = 500;    // 2 Hz

// ---- Choose serial port for the GPS ----------------------------------
// Define USE_SOFTSERIAL in platformio.ini build_flags for boards without
// a spare hardware UART (e.g. Uno).
#ifdef USE_SOFTSERIAL
#  include <SoftwareSerial.h>
static constexpr uint8_t GPS_RX_PIN = 4;
static constexpr uint8_t GPS_TX_PIN = 5;
SoftwareSerial gpsSerial(GPS_RX_PIN, GPS_TX_PIN);
#  define GPS_PORT  gpsSerial
#else
#  define GPS_PORT  Serial1
#endif

// ---- TinyGPSPlus parser instance -------------------------------------
static TinyGPSPlus gps;

// ---- Pretty-print helpers --------------------------------------------
static void printFloat(float val, int decimals) {
    Serial.print(val, decimals);
}

static void printInt(uint32_t val) {
    Serial.print(val);
}

void setup() {
    Serial.begin(DEBUG_BAUD);
    while (!Serial) { ; }
    Serial.println();
    Serial.println(F("=================================================="));
    Serial.println(F(" AVCS — gps_test.ino"));
    Serial.println(F(" Move outdoors with a clear view of the sky."));
    Serial.println(F(" First fix may take 30–60 seconds (cold start)."));
    Serial.println(F("==================================================\n"));

    GPS_PORT.begin(GPS_BAUD);
    Serial.print(F("[GPS] UART @ ")); Serial.print(GPS_BAUD);
    Serial.println(F(" baud"));
}

void loop() {
    // 1. Pump the GPS UART through the TinyGPSPlus parser.
    while (GPS_PORT.available() > 0) {
        gps.encode(GPS_PORT.read());
    }

    // 2. Periodic status print.
    static uint32_t lastPrint = 0;
    if ((millis() - lastPrint) < PRINT_INTERVAL) return;
    lastPrint = millis();

    Serial.print(F("[GPS] "));

    if (gps.location.isValid()) {
        Serial.print(gps.location.lat(), 6);
        Serial.print(F(","));
        Serial.print(gps.location.lng(), 6);
        Serial.print(F(","));
    } else {
        Serial.print(F(",,,,"));
    }

    Serial.print(gps.location.isValid() ? F("1") : F("0"));
    Serial.print(F(","));

    if (gps.satellites.isValid()) {
        printInt(gps.satellites.value());
    } else {
        Serial.print(F("?"));
    }
    Serial.print(F(","));

    if (gps.hdop.isValid()) {
        Serial.print(gps.hdop.hdop(), 2);
    } else {
        Serial.print(F("?"));
    }
    Serial.print(F(","));

    if (gps.speed.isValid()) {
        printFloat(gps.speed.mps(), 2);   // metres per second
    } else {
        Serial.print(F("0.00"));
    }
    Serial.print(F(","));

    if (gps.course.isValid()) {
        printFloat(gps.course.deg(), 1);
    } else {
        Serial.print(F("0.0"));
    }
    Serial.print(F(","));

    if (gps.altitude.isValid()) {
        printFloat(gps.altitude.meters(), 1);
    } else {
        Serial.print(F("0.0"));
    }
    Serial.print(F(","));

    // Fix age: how stale is the latest location fix (ms).
    Serial.print(gps.location.age());

    Serial.println();

    // 3. Once per minute, dump the raw NMEA sentences for debugging.
    static uint32_t lastDump = 0;
    if ((millis() - lastDump) >= 60000) {
        lastDump = millis();
        Serial.println(F("--- statistics ---"));
        Serial.print(F("chars processed : ")); Serial.println(gps.charsProcessed());
        Serial.print(F("sentences       : ")); Serial.println(gps.sentencesWithFix());
        Serial.print(F("failed checksum : ")); Serial.println(gps.failedChecksum());
        Serial.println(F("-------------------"));
    }
}
