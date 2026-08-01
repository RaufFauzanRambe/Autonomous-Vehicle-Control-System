/**
 * @file    board_config.h
 * @brief   Board-specific pin abstraction layer.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 *
 * @details
 * The firmware supports three target boards.  Select the active board by
 * defining exactly one of the following macros at compile time
 * (either via `build_flags` in platformio.ini or by editing the line below):
 *
 *     #define BOARD_MEGA_2560
 *     #define BOARD_DUE
 *     #define BOARD_NANO_33_IOT
 *
 * Each block maps a set of **logical** pin names (used throughout the
 * firmware) to the physical Arduino pin number of the active board.
 */
#ifndef BOARD_CONFIG_H
#define BOARD_CONFIG_H
#pragma once

/* ------------------------------------------------------------------------- *
 *  Board selection — fallback to Mega 2560 if none specified.
 * ------------------------------------------------------------------------- */
#if !defined(BOARD_MEGA_2560) && !defined(BOARD_DUE) && !defined(BOARD_NANO_33_IOT)
#  define BOARD_MEGA_2560 1
#  warning "No BOARD_* macro defined — defaulting to Arduino Mega 2560."
#endif

/* ------------------------------------------------------------------------- *
 *  Arduino Mega 2560  (ATmega2560 @ 16 MHz)
 * ------------------------------------------------------------------------- */
#if defined(BOARD_MEGA_2560)

// --- Digital I/O ----------------------------------------------------------
#define PIN_STATUS_LED        13     // Onboard LED
#define PIN_E_STOP_IN         2      // INT0 — external e-stop button
#define PIN_MOTOR_PWM         4      // OC0B PWM (490 Hz)
#define PIN_MOTOR_IN1         5
#define PIN_MOTOR_IN2         6
#define PIN_MOTOR_BRAKE       7
#define PIN_STEERING_PWM      8      // Servo PWM (any digital pin via Servo lib)
#define PIN_ENCODER_A         18     // INT1 (TX1) — encoder phase A
#define PIN_ENCODER_B         19     // INT2 (RX1) — encoder phase B

// --- Ultrasonic sensors ---------------------------------------------------
#define PIN_ULTRASONIC_FRONT_TRIG   22
#define PIN_ULTRASONIC_FRONT_ECHO   23
#define PIN_ULTRASONIC_REAR_TRIG    24
#define PIN_ULTRASONIC_REAR_ECHO    25
#define PIN_ULTRASONIC_LEFT_TRIG    26
#define PIN_ULTRASONIC_LEFT_ECHO    27
#define PIN_ULTRASONIC_RIGHT_TRIG   28
#define PIN_ULTRASONIC_RIGHT_ECHO   29

// --- Analog inputs --------------------------------------------------------
#define PIN_BATTERY_V         A0
#define PIN_MOTOR_CURRENT     A1
#define PIN_POT_STEERING_FB   A2     // Optional pot feedback from servo horn

// --- Serial / UART --------------------------------------------------------
//   Serial  = USB debug                  (pin 0 = RX, pin 1 = TX)
//   Serial1 = GPS (9600 baud NMEA)       (pin 19 = RX, pin 18 = TX)
//   Serial2 = Companion-computer link    (pin 17 = RX, pin 16 = TX)
//   Serial3 = Spare / debug stream       (pin 15 = RX, pin 14 = TX)
#define HAS_SERIAL1 1
#define HAS_SERIAL2 1
#define HAS_SERIAL3 1

// --- I2C ------------------------------------------------------------------
//   SDA = pin 20,  SCL = pin 21  (dedicated TWI pins on ATmega2560)
#define PIN_I2C_SDA            20
#define PIN_I2C_SCL            21

// --- SPI ------------------------------------------------------------------
//   MOSI = 51, MISO = 50, SCK = 52, SS = 53
#define PIN_SPI_MOSI           51
#define PIN_SPI_MISO           50
#define PIN_SPI_SCK            52
#define PIN_SPI_SS             53

// --- Interrupt-capable pins (for attachInterrupt) -------------------------
#define INT0_PIN               2
#define INT1_PIN               3
#define INT2_PIN               21   // SCL
#define INT3_PIN               20   // SDA
#define INT4_PIN               19
#define INT5_PIN               18

#define BOARD_NAME             "Arduino Mega 2560"
#define ADC_RESOLUTION_BITS    10

/* ------------------------------------------------------------------------- *
 *  Arduino Due  (ATSAM3X8E Cortex-M3 @ 84 MHz)
 * ------------------------------------------------------------------------- */
#elif defined(BOARD_DUE)

#define PIN_STATUS_LED         13
#define PIN_E_STOP_IN          2
#define PIN_MOTOR_PWM          5     // PWM-capable
#define PIN_MOTOR_IN1          6
#define PIN_MOTOR_IN2          7
#define PIN_MOTOR_BRAKE        8
#define PIN_STEERING_PWM       9
#define PIN_ENCODER_A          18
#define PIN_ENCODER_B          19

#define PIN_ULTRASONIC_FRONT_TRIG   22
#define PIN_ULTRASONIC_FRONT_ECHO   23
#define PIN_ULTRASONIC_REAR_TRIG    24
#define PIN_ULTRASONIC_REAR_ECHO    25
#define PIN_ULTRASONIC_LEFT_TRIG    26
#define PIN_ULTRASONIC_LEFT_ECHO    27
#define PIN_ULTRASONIC_RIGHT_TRIG   28
#define PIN_ULTRASONIC_RIGHT_ECHO   29

#define PIN_BATTERY_V          A0
#define PIN_MOTOR_CURRENT      A1
#define PIN_POT_STEERING_FB    A2

#define HAS_SERIAL1 1
#define HAS_SERIAL2 1
#define HAS_SERIAL3 1
#define HAS_CAN_BUS  1           // due_can library

#define PIN_I2C_SDA            20
#define PIN_I2C_SCL            21
#define PIN_SPI_MOSI           75   // ICSP MOSI
#define PIN_SPI_MISO           74
#define PIN_SPI_SCK            76
#define PIN_SPI_SS             10

#define BOARD_NAME             "Arduino Due"
#define ADC_RESOLUTION_BITS    12

/* ------------------------------------------------------------------------- *
 *  Arduino Nano 33 IoT  (SAMD21G18 @ 48 MHz)
 * ------------------------------------------------------------------------- */
#elif defined(BOARD_NANO_33_IOT)

#define PIN_STATUS_LED         LED_BUILTIN   // typically 13
#define PIN_E_STOP_IN          2
#define PIN_MOTOR_PWM          3     // PWM-capable (TCC)
#define PIN_MOTOR_IN1          4
#define PIN_MOTOR_IN2          5
#define PIN_MOTOR_BRAKE        6
#define PIN_STEERING_PWM       9
#define PIN_ENCODER_A          7
#define PIN_ENCODER_B          8

#define PIN_ULTRASONIC_FRONT_TRIG   14
#define PIN_ULTRASONIC_FRONT_ECHO   15
#define PIN_ULTRASONIC_REAR_TRIG    16
#define PIN_ULTRASONIC_REAR_ECHO    17

#define PIN_BATTERY_V          A0
#define PIN_MOTOR_CURRENT      A1
#define PIN_POT_STEERING_FB    A2

// Nano 33 IoT has Serial1 on pins 0/1 (NINA UART).  Use Serial for USB.
#define HAS_SERIAL1 1
#define HAS_SERIAL2 0
#define HAS_SERIAL3 0

#define PIN_I2C_SDA            A4   // SAMD default Wire
#define PIN_I2C_SCL            A5
#define PIN_SPI_MOSI           11
#define PIN_SPI_MISO           12
#define PIN_SPI_SCK            13
#define PIN_SPI_SS             10

#define BOARD_NAME             "Arduino Nano 33 IoT"
#define ADC_RESOLUTION_BITS    12
#define USE_BUILTIN_IMU        1     // LSM6DS3 onboard — can substitute for BNO055

#endif

#endif // BOARD_CONFIG_H
