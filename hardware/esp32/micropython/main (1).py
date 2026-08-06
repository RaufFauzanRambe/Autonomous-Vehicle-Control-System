# =============================================================================
#  main.py  —  Autonomous Vehicle Control System / ESP32 / MicroPython
# =============================================================================
#  Async (uasyncio) application: WiFi + MQTT + sensor sampling + motor control.
#
#  Tasks:
#    sensor_loop    @ 100 Hz  — poll IMU/baro/ToF/encoders, push to queue
#    motor_loop     @ 50 Hz   — pop setpoint, run PID, write PWM
#    telemetry_loop @ 10 Hz   — pop latest frame, publish JSON via MQTT/BLE
#    watchdog_loop  @ 1 Hz    — feed WDT, blink status LED, publish status
#
#  Author: AVC team   |  License: MIT   |  Date: 2024
# =============================================================================

import gc
import json
import machine
import uasyncio as asyncio
from machine import Pin, I2C, PWM, ADC, Timer
import sys

import wifi
import mqtt
import sensors

# -----------------------------------------------------------------------------
#  Globals & config
# -----------------------------------------------------------------------------
try:
    CFG = _avc_cfg     # type: ignore[name-defined]   # set by boot.py
    WDT = _avc_wdt     # type: ignore[name-defined]
except NameError:
    CFG = {"node_id": "avc-esp32-mpy",
           "tasks": {"sensor_hz": 100, "motor_hz": 50, "telem_hz": 10}}
    WDT = None

NODE_ID  = CFG.get("node_id", "avc-esp32-mpy")
HZ_S     = CFG["tasks"]["sensor_hz"]
HZ_M     = CFG["tasks"]["motor_hz"]
HZ_T     = CFG["tasks"]["telem_hz"]

LED_PIN  = Pin(2, Pin.OUT, value=0)

# Latest sensor frame (atomic from the producer's POV — single uasyncio task).
_latest = {"seq": 0, "imu": {}, "enc": {}, "tof": 0.0,
           "bat": {"v": 0.0, "pct": 0.0}}

# Latest motor setpoint (v in m/s, w in rad/s)
_setpoint = {"v": 0.0, "w": 0.0, "brake": False}

# MQTT message queue (inbound commands → motor loop)
_cmd_q = asyncio.Queue()

# -----------------------------------------------------------------------------
#  Hardware init
# -----------------------------------------------------------------------------
i2c  = I2C(scl=Pin(CFG["i2c"]["scl"]), sda=Pin(CFG["i2c"]["sda"]),
           freq=CFG["i2c"]["freq"])

# Motors (TB6612FNG): two PWM + four direction + standby
mot_pwma = PWM(Pin(CFG["motors"]["pwma"]), freq=CFG["motors"]["freq_hz"], duty=0)
mot_pwmb = PWM(Pin(CFG["motors"]["pwmb"]), freq=CFG["motors"]["freq_hz"], duty=0)
mot_a1   = Pin(CFG["motors"]["ain1"], Pin.OUT)
mot_a2   = Pin(CFG["motors"]["ain2"], Pin.OUT)
mot_b1   = Pin(CFG["motors"]["bin1"], Pin.OUT)
mot_b2   = Pin(CFG["motors"]["bin2"], Pin.OUT)
mot_stby = Pin(CFG["motors"]["stby"], Pin.OUT, value=1)   # bridge enabled

# Battery ADC
bat_adc = ADC(Pin(CFG["battery"]["adc"]))
bat_adc.atten(ADC.ATTN_11V)        # 0..3.6 V full-scale
bat_adc.width(ADC.WIDTH_12BIT)

# Encoders — polled here (no PCNT in MicroPython on classic ESP32 easily)
enc_la = Pin(CFG["encoders"]["la"], Pin.IN)
enc_lb = Pin(CFG["encoders"]["lb"], Pin.IN)
enc_ra = Pin(CFG["encoders"]["ra"], Pin.IN)
enc_rb = Pin(CFG["encoders"]["rb"], Pin.IN)
_enc = {"la":0, "lb":0, "ra":0, "rb":0, "l_count":0, "r_count":0,
        "la_last":0, "ra_last":0}

# -----------------------------------------------------------------------------
#  Encoder ISR (very simple — single channel per wheel, +1 per rising edge)
# -----------------------------------------------------------------------------
def _enc_isr_l(pin):
    _enc["l_count"] += 1
def _enc_isr_r(pin):
    _enc["r_count"] += 1

enc_la.irq(trigger=Pin.IRQ_RISING, handler=_enc_isr_l)
enc_ra.irq(trigger=Pin.IRQ_RISING, handler=_enc_isr_r)

# -----------------------------------------------------------------------------
#  Motor helpers
# -----------------------------------------------------------------------------
def _write_motor(pwm, in1, in2, duty_signed):
    """@brief Apply a -1..1 signed duty to one H-bridge channel."""
    duty = int(max(-1023, min(1023, duty_signed * 1023)))
    if duty >= 0:
        in1(1); in2(0); pwm.duty(duty)
    else:
        in1(0); in2(1); pwm.duty(-duty)

def motors_set(v, w):
    """@brief Differential-drive setpoint."""
    half = w * 0.135 / 2.0     # wheelbase / 2
    _write_motor(mot_pwma, mot_a1, mot_a2, v - half)
    _write_motor(mot_pwmb, mot_b1, mot_b2, v + half)

def motors_brake():
    mot_a1(1); mot_a2(1); mot_b1(1); mot_b2(1)
    mot_pwma.duty(0); mot_pwmb.duty(0)

def motors_estop():
    mot_stby(0)              # bridge standby
    motors_brake()

# -----------------------------------------------------------------------------
#  Async task — sensor loop (100 Hz)
# -----------------------------------------------------------------------------
async def sensor_loop():
    """@brief Poll all sensors at 100 Hz and publish to _latest."""
    period = 1.0 / HZ_S
    seq = 0
    # Initialise sensors (Adafruit-style drivers may be replaced with raw I2C)
    sensors.init(i2c, bat_adc)
    while True:
        try:
            seq += 1
            imu  = sensors.read_imu()
            baro = sensors.read_baro()
            tof  = sensors.read_tof()
            bat  = sensors.read_battery()

            # Encoder differential → convert to m/s
            l_count = _enc["l_count"]; _enc["l_count"] = 0
            r_count = _enc["r_count"]; _enc["r_count"] = 0
            # Simplified: 1 pulse ≈ (2π*r)/(4*ppr)  →  v = counts * (2π*r) / (4*ppr*dt)
            r = 0.065 / 2.0
            ppr = CFG["encoders"]["ppr"]
            dt  = period
            v_l = l_count * (2 * 3.14159265 * r) / (4 * ppr) / dt
            v_r = r_count * (2 * 3.14159265 * r) / (4 * ppr) / dt

            _latest.update({
                "seq": seq,
                "imu": imu,
                "baro": baro,
                "tof": tof,
                "enc": {"l": v_l, "r": v_r, "ln": l_count, "rn": r_count},
                "bat": bat,
            })
        except Exception as e:
            sys.print_exception(e)
        await asyncio.sleep(period)

# -----------------------------------------------------------------------------
#  Async task — motor loop (50 Hz)
# -----------------------------------------------------------------------------
async def motor_loop():
    """@brief Pop setpoint + handle inbound MQTT commands at 50 Hz."""
    period = 1.0 / HZ_M
    while True:
        # Drain command queue (non-blocking)
        try:
            while True:
                cmd = _cmd_q.get_nowait()
                if cmd is None: break
                _handle_cmd(cmd)
        except asyncio.QueueEmpty:
            pass

        sp = _setpoint
        if sp["brake"]:
            motors_brake()
        else:
            motors_set(sp["v"], sp["w"])

        await asyncio.sleep(period)

def _handle_cmd(cmd):
    """@brief Parse a JSON command dict and update _setpoint."""
    try:
        if "v" in cmd or "w" in cmd:
            _setpoint["v"] = float(cmd.get("v", 0.0))
            _setpoint["w"] = float(cmd.get("w", 0.0))
            _setpoint["brake"] = False
            print(f"[cmd] v={_setpoint['v']} w={_setpoint['w']}")
        elif cmd.get("estop"):
            motors_estop()
            print("[cmd] E-STOP")
        elif cmd.get("clear_estop"):
            mot_stby(1)
            print("[cmd] E-STOP cleared")
    except (TypeError, ValueError) as e:
        print(f"[cmd] bad payload: {e}")

# -----------------------------------------------------------------------------
#  Async task — telemetry loop (10 Hz)
# -----------------------------------------------------------------------------
async def telemetry_loop(mqtt_client):
    """@brief Publish _latest as JSON to MQTT every 100 ms."""
    period = 1.0 / HZ_T
    while True:
        try:
            topic = f"avc/{NODE_ID}/state/telemetry"
            payload = json.dumps(_latest)
            mqtt_client.publish(topic, payload)
        except Exception as e:
            print(f"[telem] publish failed: {e}")
        await asyncio.sleep(period)

# -----------------------------------------------------------------------------
#  Async task — watchdog loop (1 Hz)
# -----------------------------------------------------------------------------
async def watchdog_loop(mqtt_client):
    """@brief Feed WDT, blink LED, publish status."""
    n = 0
    while True:
        if WDT: WDT.feed()
        n += 1
        # Status LED: solid on if MQTT+WiFi, slow blink if WiFi only, fast if neither.
        if mqtt_client.is_connected():
            LED_PIN(1)
        elif wifi.is_connected():
            LED_PIN(n % 2)
        else:
            LED_PIN((n // 2) % 2)
        if n % 5 == 0 and mqtt_client.is_connected():
            st = {"uptime_ms": machine.ticks_ms(),
                  "free_heap": gc.mem_free(),
                  "node": NODE_ID}
            mqtt_client.publish(f"avc/{NODE_ID}/state/status", json.dumps(st))
        gc.collect()
        await asyncio.sleep(1.0)

# -----------------------------------------------------------------------------
#  MQTT callback — inbound command router
# -----------------------------------------------------------------------------
def on_mqtt_message(topic, payload):
    """@brief Decode JSON and enqueue into _cmd_q."""
    try:
        cmd = json.loads(payload)
        _cmd_q.put_nowait(cmd)
    except ValueError as e:
        print(f"[mqtt] JSON decode error: {e}")

# -----------------------------------------------------------------------------
#  Application entry
# -----------------------------------------------------------------------------
async def app_main():
    """@brief Top-level coroutine: bring up WiFi/MQTT, spawn tasks."""
    print("[main] starting app_main")
    gc.collect()

    # Connect WiFi
    await wifi.connect_async(CFG["wifi"]["ssid"], CFG["wifi"]["pass"],
                             timeout_s=15)
    if not wifi.is_connected():
        print("[main] WiFi failed — running sensor-only mode")

    # MQTT
    mqtt_client = mqtt.MqttClient(CFG["mqtt"]["broker"],
                                  CFG["mqtt"].get("port", 1883),
                                  CFG["mqtt"].get("user", ""),
                                  CFG["mqtt"].get("pass", ""),
                                  NODE_ID)
    mqtt_client.on_message = on_mqtt_message
    mqtt_client.subscribe(f"avc/{NODE_ID}/cmd/velocity")
    mqtt_client.subscribe(f"avc/{NODE_ID}/cmd/estop")
    await mqtt_client.connect_async()

    # Spawn tasks
    tasks = [
        asyncio.create_task(sensor_loop()),
        asyncio.create_task(motor_loop()),
        asyncio.create_task(telemetry_loop(mqtt_client)),
        asyncio.create_task(watchdog_loop(mqtt_client)),
        asyncio.create_task(mqtt_client.keep_alive_loop()),
    ]
    await asyncio.gather(*tasks)

# -----------------------------------------------------------------------------
#  Run!
# -----------------------------------------------------------------------------
def main():
    try:
        asyncio.run(app_main())
    except KeyboardInterrupt:
        print("[main] KeyboardInterrupt — shutting down")
    except Exception as e:
        sys.print_exception(e)
        # Don't let the WDT instantly kill us so the operator can read the trace.
        if WDT: WDT.feed()
    finally:
        motors_estop()
        asyncio.new_event_loop()

main()
