/**
 * @file    blink_example.ino
 * @brief   Minimal "hello world" sketch for the AVCS firmware.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 *
 * @details
 * Blinks the onboard LED at 2 Hz and prints a status line over the
 * USB serial port every second.  Use this sketch to verify that the
 * Arduino IDE / PlatformIO toolchain is correctly installed before
 * moving on to the motor / GPS / IMU test sketches.
 */

// Status LED — uses the board's built-in LED (pin 13 on most Arduinos).
#ifndef LED_BUILTIN
#  define LED_BUILTIN 13
#endif

// Blink cadence (milliseconds per half-cycle → 2 Hz).
static constexpr unsigned long BLINK_INTERVAL_MS = 250;

void setup() {
    pinMode(LED_BUILTIN, OUTPUT);

    // Initialise the USB serial port for debug output.
    // 115200 baud is the AVCS project standard.
    Serial.begin(115200);
    while (!Serial) {
        ;  // Wait for the USB CDC port to enumerate (Leonardo / Micro / Due).
    }

    Serial.println();
    Serial.println(F("============================================"));
    Serial.println(F(" AVCS — blink_example.ino"));
    Serial.print  (F(" Build target : "));
    Serial.println(F(__FILE__));
    Serial.print  (F(" Compile time : "));
    Serial.println(F(__DATE__ " " __TIME__));
    Serial.println(F("============================================"));
}

void loop() {
    static bool ledState = false;
    static unsigned long lastToggle = 0;

    unsigned long now = millis();

    // Toggle the LED at BLINK_INTERVAL_MS cadence → 2 Hz blink.
    if ((now - lastToggle) >= BLINK_INTERVAL_MS) {
        lastToggle = now;
        ledState   = !ledState;
        digitalWrite(LED_BUILTIN, ledState ? HIGH : LOW);

        // Print a one-line status report on every toggle.
        // millis() wraps every ~49 days — fine for a demo sketch.
        Serial.print(F("[blink] t="));
        Serial.print(now / 1000.0f, 2);
        Serial.print(F("s  LED="));
        Serial.println(ledState ? F("ON ") : F("OFF"));
    }

    // The loop is empty most of the time — keep CPU free for future
    // additions (sensor polling, serial command handling, etc.).
}
