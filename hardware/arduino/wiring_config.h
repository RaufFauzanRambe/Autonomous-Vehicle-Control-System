/**
 * @file    wiring_config.h
 * @brief   Logical-name → physical-pin mapping for the AVCS Arduino firmware.
 * @author  AVCS Team
 * @date    2025
 * @license MIT
 *
 * @details
 * This file bridges the abstract component names (e.g. `MOTOR_PWM_PIN`) used
 * throughout the firmware and the board-specific pin numbers defined in
 * `board_config.h`.  Keeping this layer separate makes it trivial to swap
 * boards — only `board_config.h` needs editing.
 */
#ifndef WIRING_CONFIG_H
#define WIRING_CONFIG_H
#pragma once

#include "board_config.h"

/* ========================================================================= *
 *  Drive-motor subsystem (TB6612FNG / L298N)
 * ========================================================================= */
/// PWM pin feeding the motor-driver's PWMA/PWMB input.  Must be a PWM-capable
/// digital pin on the active board.
#define MOTOR_PWM_PIN        PIN_MOTOR_PWM
/// Direction-select bit 1 — together with MOTOR_IN2 selects forward/reverse/brake.
#define MOTOR_IN1_PIN        PIN_MOTOR_IN1
/// Direction-select bit 2 — complement of MOTOR_IN1.
#define MOTOR_IN2_PIN        PIN_MOTOR_IN2
/// Active-high brake pin (pulls both motor-driver inputs low → coast).
#define MOTOR_BRAKE_PIN      PIN_MOTOR_BRAKE
/// Quadrature-encoder phase A.  Must be on an interrupt-capable pin.
#define MOTOR_ENCODER_A_PIN  PIN_ENCODER_A
/// Quadrature-encoder phase B (direction); interrupt-capable optional.
#define MOTOR_ENCODER_B_PIN  PIN_ENCODER_B

/* ========================================================================= *
 *  Steering subsystem (RC hobby servo, 50 Hz PWM)
 * ========================================================================= */
/// Servo signal pin — typically any digital pin when using the Arduino Servo
/// library (software PWM timer is used internally).
#define STEERING_PWM_PIN     PIN_STEERING_PWM
/// Optional analog feedback from servo horn (1 kΩ pot).
#define STEERING_FEEDBACK_PIN PIN_POT_STEERING_FB

/* ========================================================================= *
 *  GPS subsystem (NMEA over UART, 9600 baud default)
 * ========================================================================= */
/// RX pin on the Arduino that connects to the GPS module's TX line.
#define GPS_RX_PIN           19   // Mega Serial1 RX1
/// TX pin on the Arduino that connects to the GPS module's RX line.
#define GPS_TX_PIN           18   // Mega Serial1 TX1
/// Hardware UART object for the GPS stream.  Use `GPS_SERIAL.begin(9600)`.
#if defined(HAS_SERIAL1)
#  define GPS_SERIAL         Serial1
#else
#  include <SoftwareSerial.h>
extern SoftwareSerial GPS_SERIAL_OBJ;
#  define GPS_SERIAL         GPS_SERIAL_OBJ
#endif

/* ========================================================================= *
 *  IMU subsystem (Adafruit BNO055 over I²C, address 0x28 or 0x29)
 * ========================================================================= */
/// I²C data line — shared with any other I²C device on the bus.
#define IMU_SDA_PIN          PIN_I2C_SDA
/// I²C clock line.
#define IMU_SCL_PIN          PIN_I2C_SCL
/// Primary I²C address of BNO055 (ADR pin tied to GND).
#define IMU_I2C_ADDRESS      0x28
/// Alternate address (ADR pin pulled high).
#define IMU_I2C_ADDRESS_ALT  0x29

/* ========================================================================= *
 *  Ultrasonic subsystem (HC-SR04 — 5 V trigger, 5 V echo)
 * ========================================================================= */
#define US_FRONT_TRIG_PIN    PIN_ULTRASONIC_FRONT_TRIG
#define US_FRONT_ECHO_PIN    PIN_ULTRASONIC_FRONT_ECHO
#define US_REAR_TRIG_PIN     PIN_ULTRASONIC_REAR_TRIG
#define US_REAR_ECHO_PIN     PIN_ULTRASONIC_REAR_ECHO
#define US_LEFT_TRIG_PIN     PIN_ULTRASONIC_LEFT_TRIG
#define US_LEFT_ECHO_PIN     PIN_ULTRASONIC_LEFT_ECHO
#define US_RIGHT_TRIG_PIN    PIN_ULTRASONIC_RIGHT_TRIG
#define US_RIGHT_ECHO_PIN    PIN_ULTRASONIC_RIGHT_ECHO

/* ========================================================================= *
 *  Power-monitoring subsystem
 * ========================================================================= */
/// ADC pin measuring the LiPo voltage through a 3:1 resistive divider.
#define BATTERY_VOLTAGE_PIN  PIN_BATTERY_V
/// ADC pin measuring motor-driver current via a 0.1 Ω shunt + op-amp.
#define MOTOR_CURRENT_PIN    PIN_MOTOR_CURRENT

/* ========================================================================= *
 *  Status & safety
 * ========================================================================= */
/// Heart-beat / status indicator LED (typically the onboard LED).
#define STATUS_LED_PIN       PIN_STATUS_LED
/// External e-stop button input (interrupt-capable).
#define E_STOP_PIN           PIN_E_STOP_IN

/* ========================================================================= *
 *  Companion-computer link  (Serial2 by default — 115200 baud JSON)
 * ========================================================================= */
#if defined(HAS_SERIAL2)
#  define COMPANION_SERIAL   Serial2
#elif defined(HAS_SERIAL3)
#  define COMPANION_SERIAL   Serial3
#else
#  define COMPANION_SERIAL   Serial   // Fallback: share the debug UART.
#endif

/* ========================================================================= *
 *  CAN-bus (Arduino Due only)
 * ========================================================================= */
#if defined(HAS_CAN_BUS)
/// CAN0 transceiver CS pin (SN65HVD230 / MCP2562).  Library default.
#define CAN0_CS_PIN          10
/// CAN0 interrupt pin.
#define CAN0_INT_PIN         2
#endif

#endif // WIRING_CONFIG_H
