/**
 * @module vehicle_metrics
 * @description Vehicle metrics computation module for the Autonomous Vehicle
 * Control System. Calculates speed/distance/fuel/battery, acceleration and
 * deceleration rates, cornering forces, efficiency calculations, trip metrics
 * aggregation, and OBD-II style parameter groups.
 *
 * @author Autonomous Vehicle Control System
 * @version 2.0.0
 * @license MIT
 */

import { EventEmitter } from 'events';

// ─────────────────────────────────────────────────────────────────────────────
// Custom Error Classes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Error thrown by the vehicle metrics module.
 * @extends Error
 */
export class VehicleMetricsError extends Error {
  /**
   * @param {string} message - Error description
   * @param {string} [code='METRICS_ERROR'] - Machine-readable error code
   */
  constructor(message, code = 'METRICS_ERROR') {
    super(message);
    this.name = 'VehicleMetricsError';
    this.code = code;
    this.timestamp = Date.now();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** @enum {string} OBD-II Mode 1 PIDs (parameter IDs) */
export const OBD2PID = {
  SUPPORTED_PIDS_01_20: '0100',
  MONITOR_STATUS: '0101',
  FREEZE_DTC: '0102',
  FUEL_SYSTEM_STATUS: '0103',
  ENGINE_LOAD: '0104',
  ENGINE_COOLANT_TEMP: '0105',
  SHORT_TERM_FUEL_TRIM_1: '0106',
  LONG_TERM_FUEL_TRIM_1: '0107',
  ENGINE_RPM: '010C',
  VEHICLE_SPEED: '010D',
  TIMING_ADVANCE: '010E',
  INTAKE_AIR_TEMP: '010F',
  MAF_AIR_FLOW: '0110',
  THROTTLE_POSITION: '0111',
  O2_SENSOR_1: '0114',
  ODOMETER: '0121',
  FUEL_LEVEL: '012F',
  AMBIENT_AIR_TEMP: '0146',
  HYBRID_BATTERY_PERCENT: '015B',
  ENGINE_OIL_TEMP: '015C',
  FUEL_RATE: '015E'
};

/** @enum {string} Vehicle state types */
export const DriveMode = {
  PARK: 'P',
  REVERSE: 'R',
  NEUTRAL: 'N',
  DRIVE: 'D',
  SPORT: 'S',
  LOW: 'L'
};

/** @enum {string} Fuel system status codes */
export const FuelSystemStatus = {
  OPEN_LOW_TEMP: 'open_low_temp',
  OPEN_LOAD: 'open_load',
  CLOSED: 'closed',
  OPEN_FAILURE: 'open_failure'
};

/** @type {object} Default configuration */
const DEFAULT_CONFIG = {
  /** Vehicle mass in kg */
  vehicleMass: 1800,
  /** Drag coefficient */
  dragCoefficient: 0.28,
  /** Frontal area in m² */
  frontalArea: 2.2,
  /** Rolling resistance coefficient */
  rollingResistance: 0.012,
  /** Air density in kg/m³ */
  airDensity: 1.225,
  /** Drivetrain efficiency (0-1) */
  drivetrainEfficiency: 0.88,
  /** Regenerative braking efficiency (0-1) */
  regenEfficiency: 0.60,
  /** Battery capacity in kWh (for EV metrics) */
  batteryCapacityKWh: 75,
  /** Fuel tank capacity in liters */
  fuelTankCapacity: 55,
  /** Fuel energy density in kWh/liter */
  fuelEnergyDensity: 8.8,
  /** Gravity constant m/s² */
  gravity: 9.81,
  /** Update interval for derived metrics (ms) */
  updateIntervalMs: 100,
  /** Speed smoothing factor (0-1) */
  speedSmoothing: 0.3,
  /** Maximum speed for sanity check (m/s) */
  maxSpeed: 83.33 // 300 km/h
};

// ─────────────────────────────────────────────────────────────────────────────
// TripMetrics
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Aggregates trip-level metrics from start to current point.
 */
class TripMetrics {
  /**
   * @param {object} config - Vehicle configuration
   */
  constructor(config) {
    /** @type {object} */
    this._config = config;

    /** @type {number} */
    this._startTime = Date.now();
    /** @type {number} */
    this._endTime = null;
    /** @type {boolean} */
    this._active = true;

    /** @type {number} Total distance in meters */
    this.distance = 0;
    /** @type {number} Total energy consumed in kWh */
    this.energyConsumed = 0;
    /** @type {number} Total energy regenerated in kWh */
    this.energyRegenerated = 0;
    /** @type {number} Fuel consumed in liters */
    this.fuelConsumed = 0;
    /** @type {number} Maximum speed in m/s */
    this.maxSpeed = 0;
    /** @type {number} Maximum acceleration in m/s² */
    this.maxAcceleration = 0;
    /** @type {number} Maximum deceleration in m/s² */
    this.maxDeceleration = 0;
    /** @type {number} Maximum lateral acceleration in m/s² */
    this.maxLateralAcceleration = 0;
    /** @type {number} Time spent idling in seconds */
    this.idleTime = 0;
    /** @type {number} Time spent moving in seconds */
    this.movingTime = 0;
    /** @type {number} Number of acceleration events */
    this.accelerationEvents = 0;
    /** @type {number} Number of braking events */
    this.brakingEvents = 0;
    /** @type {number} Number of cornering events */
    this.corneringEvents = 0;

    /** @type {number} Previous speed for integration */
    this._prevSpeed = 0;
    /** @type {number} Previous timestamp */
    this._prevTimestamp = Date.now();
  }

  /**
   * Update trip metrics with new speed/acceleration data.
   * @param {object} params - Current vehicle state
   * @param {number} params.speed - Current speed (m/s)
   * @param {number} params.acceleration - Longitudinal acceleration (m/s²)
   * @param {number} params.lateralAccel - Lateral acceleration (m/s²)
   * @param {number} params.timestamp - Timestamp (ms)
   * @param {number} params.powerConsumption - Instantaneous power (kW)
   * @param {number} params.fuelRate - Instantaneous fuel rate (L/h)
   */
  update(params) {
    if (!this._active) return;

    const dt = (params.timestamp - this._prevTimestamp) / 1000;
    if (dt <= 0 || dt > 10) {
      this._prevTimestamp = params.timestamp;
      return;
    }

    // Distance integration (trapezoidal)
    const avgSpeed = (this._prevSpeed + params.speed) / 2;
    this.distance += avgSpeed * dt;

    // Speed tracking
    this.maxSpeed = Math.max(this.maxSpeed, params.speed);

    // Time tracking
    if (params.speed < 0.5) {
      this.idleTime += dt;
    } else {
      this.movingTime += dt;
    }

    // Acceleration tracking
    if (params.acceleration > 0.5) {
      this.maxAcceleration = Math.max(this.maxAcceleration, params.acceleration);
      if (params.acceleration > 2.0) this.accelerationEvents++;
    } else if (params.acceleration < -0.5) {
      this.maxDeceleration = Math.min(this.maxDeceleration, params.acceleration);
      if (params.acceleration < -2.0) this.brakingEvents++;
    }

    // Lateral acceleration tracking
    if (Math.abs(params.lateralAccel) > 0.3) {
      this.maxLateralAcceleration = Math.max(this.maxLateralAcceleration, Math.abs(params.lateralAccel));
      if (Math.abs(params.lateralAccel) > 2.0) this.corneringEvents++;
    }

    // Energy tracking
    if (params.powerConsumption > 0) {
      this.energyConsumed += (params.powerConsumption * dt) / 3600; // kWh
    } else if (params.powerConsumption < 0) {
      this.energyRegenerated += (Math.abs(params.powerConsumption) * dt) / 3600; // kWh
    }

    // Fuel tracking
    if (params.fuelRate > 0) {
      this.fuelConsumed += (params.fuelRate * dt) / 3600; // liters
    }

    this._prevSpeed = params.speed;
    this._prevTimestamp = params.timestamp;
  }

  /**
   * Finalize the trip.
   */
  finish() {
    this._active = false;
    this._endTime = Date.now();
  }

  /**
   * Get trip summary.
   * @returns {object}
   */
  getSummary() {
    const duration = ((this._endTime || Date.now()) - this._startTime) / 1000;
    const netEnergy = this.energyConsumed - this.energyRegenerated;

    return {
      duration,
      distance: this.distance,
      distanceKm: this.distance / 1000,
      maxSpeed: this.maxSpeed,
      maxSpeedKmh: this.maxSpeed * 3.6,
      avgSpeed: this.movingTime > 0 ? this.distance / this.movingTime : 0,
      avgSpeedKmh: this.movingTime > 0 ? (this.distance / this.movingTime) * 3.6 : 0,
      maxAcceleration: this.maxAcceleration,
      maxDeceleration: this.maxDeceleration,
      maxLateralAcceleration: this.maxLateralAcceleration,
      energyConsumed: this.energyConsumed,
      energyRegenerated: this.energyRegenerated,
      netEnergy,
      energyPerKm: this.distance > 0 ? (netEnergy / (this.distance / 1000)) : 0,
      fuelConsumed: this.fuelConsumed,
      fuelPer100km: this.distance > 0 ? (this.fuelConsumed / (this.distance / 1000)) * 100 : 0,
      idleTime: this.idleTime,
      movingTime: this.movingTime,
      idlePercent: duration > 0 ? (this.idleTime / duration) * 100 : 0,
      accelerationEvents: this.accelerationEvents,
      brakingEvents: this.brakingEvents,
      corneringEvents: this.corneringEvents
    };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// VehicleMetrics
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Vehicle metrics computation module. Calculates derived vehicle metrics from
 * raw telemetry data including speed, distance, fuel/battery, acceleration,
 * cornering forces, efficiency, and OBD-II style parameter groups.
 *
 * @extends EventEmitter
 *
 * @example
 * const metrics = new VehicleMetrics({ vehicleMass: 1500, batteryCapacityKWh: 60 });
 * metrics.update({ speed: 25, heading: 1.57, throttle: 0.5, brake: 0 });
 * const current = metrics.getCurrentMetrics();
 * const trip = metrics.getTripSummary();
 */
export class VehicleMetrics extends EventEmitter {
  /**
   * @param {object} [config={}] - Configuration overrides
   */
  constructor(config = {}) {
    super();

    /** @type {object} */
    this.config = { ...DEFAULT_CONFIG, ...config };

    // ── Current state ──────────────────────────────────────────────────
    /** @type {number} Current speed in m/s */
    this._speed = 0;
    /** @type {number} Smoothed speed */
    this._smoothedSpeed = 0;
    /** @type {number} Current heading in radians */
    this._heading = 0;
    /** @type {number} Current longitudinal acceleration m/s² */
    this._acceleration = 0;
    /** @type {number} Current lateral acceleration m/s² */
    this._lateralAccel = 0;
    /** @type {number} Current vertical acceleration m/s² */
    this._verticalAccel = 0;
    /** @type {number} Current steering angle in radians */
    this._steeringAngle = 0;
    /** @type {number} Throttle position (0-1) */
    this._throttle = 0;
    /** @type {number} Brake position (0-1) */
    this._brake = 0;
    /** @type {string} Current gear */
    this._gear = DriveMode.PARK;
    /** @type {number} Engine RPM */
    this._rpm = 0;
    /** @type {number} Engine load (0-100%) */
    this._engineLoad = 0;
    /** @type {number} Coolant temperature °C */
    this._coolantTemp = 25;
    /** @type {number} Battery SoC (0-100%) */
    this._batterySOC = 100;
    /** @type {number} Fuel level (0-100%) */
    this._fuelLevel = 100;
    /** @type {number} Odometer in meters */
    this._odometer = 0;
    /** @type {number} Instantaneous power consumption kW */
    this._powerConsumption = 0;
    /** @type {number} Instantaneous fuel rate L/h */
    this._fuelRate = 0;

    // ── Tracking state ─────────────────────────────────────────────────
    /** @type {number} Previous heading for yaw rate computation */
    this._prevHeading = 0;
    /** @type {number} Previous speed for acceleration computation */
    this._prevSpeed = 0;
    /** @type {number} Previous timestamp */
    this._prevTimestamp = Date.now();
    /** @type {number} Total distance */
    this._totalDistance = 0;

    // ── Trip metrics ───────────────────────────────────────────────────
    /** @type {TripMetrics} */
    this._currentTrip = new TripMetrics(this.config);
    /** @type {Array<object>} */
    this._completedTrips = [];

    /** @type {object} Statistics */
    this._stats = {
      updatesProcessed: 0,
      lastUpdateTime: null
    };
  }

  // ── Update Method ───────────────────────────────────────────────────────

  /**
   * Update vehicle metrics with new telemetry data.
   * @param {object} data - Telemetry data (parsed)
   * @param {object} [data.data] - Payload data
   * @param {number} [data.timestamp] - Data timestamp
   */
  update(data) {
    const payload = data.data || data;
    const timestamp = data.timestamp || Date.now();

    try {
      const dt = (timestamp - this._prevTimestamp) / 1000;
      if (dt <= 0 || dt > 30) {
        this._prevTimestamp = timestamp;
        return;
      }

      // Update raw values from telemetry
      if (payload.speed !== undefined) this._speed = this._clamp(payload.speed, 0, this.config.maxSpeed);
      if (payload.velocity !== undefined) this._speed = this._clamp(payload.velocity, 0, this.config.maxSpeed);
      if (payload.heading !== undefined) this._heading = payload.heading;
      if (payload.yaw !== undefined) this._heading = payload.yaw;
      if (payload.steeringAngle !== undefined) this._steeringAngle = payload.steeringAngle;
      if (payload.steer_angle !== undefined) this._steeringAngle = payload.steer_angle;
      if (payload.throttle !== undefined) this._throttle = this._clamp(payload.throttle, 0, 1);
      if (payload.brake !== undefined) this._brake = this._clamp(payload.brake, 0, 1);
      if (payload.gear !== undefined) this._gear = payload.gear;
      if (payload.rpm !== undefined) this._rpm = payload.rpm;
      if (payload.engineLoad !== undefined) this._engineLoad = payload.engineLoad;
      if (payload.coolantTemp !== undefined) this._coolantTemp = payload.coolantTemp;
      if (payload.batterySOC !== undefined) this._batterySOC = this._clamp(payload.batterySOC, 0, 100);
      if (payload.fuelLevel !== undefined) this._fuelLevel = this._clamp(payload.fuelLevel, 0, 100);
      if (payload.odometer !== undefined) this._odometer = payload.odometer;

      // Compute derived metrics

      // Smoothed speed (EMA)
      this._smoothedSpeed = ema(this._speed, this._smoothedSpeed, this.config.speedSmoothing);

      // Longitudinal acceleration
      this._acceleration = dt > 0 ? (this._speed - this._prevSpeed) / dt : 0;

      // Lateral (cornering) acceleration
      const yawRate = this._computeYawRate(this._heading, this._prevHeading, dt);
      this._lateralAccel = this._speed * yawRate;

      // Vertical acceleration (from IMU if available)
      if (payload.accelerometer?.z !== undefined) {
        this._verticalAccel = payload.accelerometer.z - this.config.gravity;
      }

      // Distance integration
      this._totalDistance += ((this._prevSpeed + this._speed) / 2) * dt;

      // Power consumption estimation
      this._powerConsumption = this._estimatePowerConsumption();

      // Fuel rate estimation
      this._fuelRate = this._estimateFuelRate();

      // Update trip metrics
      this._currentTrip.update({
        speed: this._speed,
        acceleration: this._acceleration,
        lateralAccel: this._lateralAccel,
        timestamp,
        powerConsumption: this._powerConsumption,
        fuelRate: this._fuelRate
      });

      // Update previous state
      this._prevSpeed = this._speed;
      this._prevHeading = this._heading;
      this._prevTimestamp = timestamp;

      this._stats.updatesProcessed++;
      this._stats.lastUpdateTime = timestamp;

      this.emit('update', this.getCurrentMetrics());
    } catch (error) {
      this.emit('error', new VehicleMetricsError(`Update failed: ${error.message}`, 'UPDATE_ERROR'));
    }
  }

  // ── Current Metrics ─────────────────────────────────────────────────────

  /**
   * Get all current vehicle metrics.
   * @returns {object}
   */
  getCurrentMetrics() {
    return {
      speed: {
        ms: this._speed,
        kmh: this._speed * 3.6,
        mph: this._speed * 2.237,
        smoothed: this._smoothedSpeed
      },
      acceleration: {
        longitudinal: this._acceleration,
        lateral: this._lateralAccel,
        vertical: this._verticalAccel,
        total: Math.sqrt(this._acceleration ** 2 + this._lateralAccel ** 2 + this._verticalAccel ** 2)
      },
      heading: {
        radians: this._heading,
        degrees: (this._heading * 180 / Math.PI + 360) % 360,
        compass: this._headingToCompass(this._heading)
      },
      steering: {
        angle: this._steeringAngle,
        angleDeg: this._steeringAngle * 180 / Math.PI
      },
      controls: {
        throttle: this._throttle,
        brake: this._brake,
        gear: this._gear
      },
      powertrain: {
        rpm: this._rpm,
        engineLoad: this._engineLoad,
        powerConsumptionKw: this._powerConsumption,
        fuelRateLh: this._fuelRate
      },
      energy: {
        batterySOC: this._batterySOC,
        fuelLevel: this._fuelLevel,
        batteryRangeKm: this._estimateBatteryRange(),
        fuelRangeKm: this._estimateFuelRange()
      },
      efficiency: {
        energyPerKm: this._computeEnergyEfficiency(),
        whPerKm: this._computeEnergyEfficiency() * 1000
      },
      odometer: {
        meters: this._odometer || this._totalDistance,
        km: (this._odometer || this._totalDistance) / 1000
      },
      thermal: {
        coolantTemp: this._coolantTemp
      },
      timestamp: this._stats.lastUpdateTime || Date.now()
    };
  }

  /**
   * Get metrics in OBD-II PID format.
   * @returns {object} Map of PID to decoded value
   */
  getOBD2Metrics() {
    return {
      [OBD2PID.VEHICLE_SPEED]: {
        pid: OBD2PID.VEHICLE_SPEED,
        name: 'Vehicle Speed',
        value: Math.round(this._speed * 3.6), // km/h
        unit: 'km/h'
      },
      [OBD2PID.ENGINE_RPM]: {
        pid: OBD2PID.ENGINE_RPM,
        name: 'Engine RPM',
        value: this._rpm,
        unit: 'RPM'
      },
      [OBD2PID.ENGINE_LOAD]: {
        pid: OBD2PID.ENGINE_LOAD,
        name: 'Engine Load',
        value: this._engineLoad,
        unit: '%'
      },
      [OBD2PID.THROTTLE_POSITION]: {
        pid: OBD2PID.THROTTLE_POSITION,
        name: 'Throttle Position',
        value: Math.round(this._throttle * 100),
        unit: '%'
      },
      [OBD2PID.ENGINE_COOLANT_TEMP]: {
        pid: OBD2PID.ENGINE_COOLANT_TEMP,
        name: 'Engine Coolant Temp',
        value: this._coolantTemp,
        unit: '°C'
      },
      [OBD2PID.FUEL_LEVEL]: {
        pid: OBD2PID.FUEL_LEVEL,
        name: 'Fuel Level',
        value: this._fuelLevel,
        unit: '%'
      },
      [OBD2PID.FUEL_RATE]: {
        pid: OBD2PID.FUEL_RATE,
        name: 'Fuel Rate',
        value: parseFloat((this._fuelRate).toFixed(2)),
        unit: 'L/h'
      },
      [OBD2PID.ODOMETER]: {
        pid: OBD2PID.ODOMETER,
        name: 'Odometer',
        value: parseFloat((this._totalDistance / 1000).toFixed(1)),
        unit: 'km'
      },
      [OBD2PID.HYBRID_BATTERY_PERCENT]: {
        pid: OBD2PID.HYBRID_BATTERY_PERCENT,
        name: 'Hybrid Battery %',
        value: this._batterySOC,
        unit: '%'
      }
    };
  }

  // ── Trip Metrics ────────────────────────────────────────────────────────

  /**
   * Start a new trip (resets trip metrics).
   */
  startTrip() {
    if (this._currentTrip) {
      this._currentTrip.finish();
      this._completedTrips.push(this._currentTrip.getSummary());
    }
    this._currentTrip = new TripMetrics(this.config);
    this.emit('trip:start');
  }

  /**
   * End the current trip.
   */
  endTrip() {
    if (this._currentTrip) {
      this._currentTrip.finish();
      const summary = this._currentTrip.getSummary();
      this._completedTrips.push(summary);
      this.emit('trip:end', summary);
    }
  }

  /**
   * Get the current trip summary.
   * @returns {object}
   */
  getTripSummary() {
    return this._currentTrip ? this._currentTrip.getSummary() : null;
  }

  /**
   * Get all completed trips.
   * @returns {object[]}
   */
  getCompletedTrips() {
    return [...this._completedTrips];
  }

  // ── Cornering Forces ────────────────────────────────────────────────────

  /**
   * Compute detailed cornering force analysis.
   * @returns {object}
   */
  getCorneringForces() {
    const lateralForce = this.config.vehicleMass * this._lateralAccel;
    const lateralG = this._lateralAccel / this.config.gravity;

    return {
      lateralAcceleration: this._lateralAccel,
      lateralForce,
      lateralG,
      yawRate: this._computeYawRate(this._heading, this._prevHeading, 0.1),
      understeerGradient: this._computeUndersteerGradient(),
      slipAngle: this._estimateSlipAngle()
    };
  }

  // ── Status ──────────────────────────────────────────────────────────────

  /**
   * Get module statistics.
   * @returns {object}
   */
  getStats() {
    return { ...this._stats };
  }

  /**
   * Get health status.
   * @returns {object}
   */
  getHealth() {
    return {
      status: 'healthy',
      updatesProcessed: this._stats.updatesProcessed,
      lastUpdateAge: this._stats.lastUpdateTime
        ? Date.now() - this._stats.lastUpdateTime
        : null
    };
  }

  // ── Private Methods ─────────────────────────────────────────────────────

  /**
   * Compute yaw rate from heading change.
   * @param {number} heading - Current heading
   * @param {number} prevHeading - Previous heading
   * @param {number} dt - Time delta
   * @returns {number} Yaw rate in rad/s
   * @private
   */
  _computeYawRate(heading, prevHeading, dt) {
    if (dt <= 0) return 0;
    let delta = heading - prevHeading;
    // Normalize to [-π, π]
    while (delta > Math.PI) delta -= 2 * Math.PI;
    while (delta < -Math.PI) delta += 2 * Math.PI;
    return delta / dt;
  }

  /**
   * Estimate instantaneous power consumption in kW.
   * Uses a simplified physical model: P = (F_aero + F_rolling + F_grade + F_accel) * v / η
   * @returns {number} Power in kW
   * @private
   */
  _estimatePowerConsumption() {
    const v = this._speed;
    if (v < 0.1) return this._throttle * 5; // Idle/creep power estimate

    // Aerodynamic drag force
    const fAero = 0.5 * this.config.airDensity * this.config.dragCoefficient *
                  this.config.frontalArea * v * v;

    // Rolling resistance force
    const fRolling = this.config.rollingResistance * this.config.vehicleMass * this.config.gravity;

    // Acceleration force
    const fAccel = this.config.vehicleMass * this._acceleration;

    // Total force at wheels
    const totalForce = fAero + fRolling + fAccel;

    // Power at motor (accounting for drivetrain efficiency)
    const powerW = totalForce * v / this.config.drivetrainEfficiency;

    // Negative power = regenerative braking
    return powerW / 1000; // Convert to kW
  }

  /**
   * Estimate instantaneous fuel consumption rate in L/h.
   * @returns {number}
   * @private
   */
  _estimateFuelRate() {
    if (this._fuelLevel <= 0) return 0;

    // Simple model: fuel rate proportional to power consumption
    // Average ICE efficiency ~25%, fuel energy density ~8.8 kWh/L
    const powerKw = Math.max(0, this._powerConsumption);
    const fuelRate = powerKw / (this.config.fuelEnergyDensity * 0.25);

    // Add idle fuel consumption
    const idleFuelRate = this._speed < 0.5 ? 0.8 : 0; // 0.8 L/h at idle

    return fuelRate + idleFuelRate;
  }

  /**
   * Estimate remaining battery range in km.
   * @returns {number}
   * @private
   */
  _estimateBatteryRange() {
    if (this._batterySOC <= 0) return 0;
    const remainingEnergy = (this._batterySOC / 100) * this.config.batteryCapacityKWh;
    const efficiency = this._computeEnergyEfficiency();
    if (efficiency <= 0) return 0;
    return remainingEnergy / efficiency; // km
  }

  /**
   * Estimate remaining fuel range in km.
   * @returns {number}
   * @private
   */
  _estimateFuelRange() {
    if (this._fuelLevel <= 0) return 0;
    const remainingFuel = (this._fuelLevel / 100) * this.config.fuelTankCapacity;
    const efficiency = this._fuelRate > 0 ? (this._speed * 3.6) / this._fuelRate : 10; // km/L
    return remainingFuel * efficiency;
  }

  /**
   * Compute current energy efficiency in kWh/km.
   * @returns {number}
   * @private
   */
  _computeEnergyEfficiency() {
    if (this._speed < 1) return 0.15; // Default efficiency estimate
    const powerKw = Math.max(0.1, this._powerConsumption);
    return powerKw / (this._speed * 3.6); // kWh/km
  }

  /**
   * Compute understeer gradient (simplified).
   * @returns {number}
   * @private
   */
  _computeUndersteerGradient() {
    // Simplified: positive = understeer, negative = oversteer
    const steeringYaw = this._steeringAngle * this._speed / 2.5; // wheelbase ~2.5m
    const actualYaw = this._computeYawRate(this._heading, this._prevHeading, 0.1);
    return steeringYaw - actualYaw;
  }

  /**
   * Estimate tire slip angle (simplified).
   * @returns {number} Slip angle in radians
   * @private
   */
  _estimateSlipAngle() {
    // Very simplified: proportional to lateral acceleration
    return Math.atan2(this._lateralAccel, this.config.gravity) * 0.1;
  }

  /**
   * Convert heading angle to compass direction.
   * @param {number} heading - Heading in radians
   * @returns {string}
   * @private
   */
  _headingToCompass(heading) {
    const deg = ((heading * 180 / Math.PI) % 360 + 360) % 360;
    const dirs = ['N', 'NNE', 'NE', 'ENE', 'E', 'ESE', 'SE', 'SSE',
                  'S', 'SSW', 'SW', 'WSW', 'W', 'WNW', 'NW', 'NNW'];
    const idx = Math.round(deg / 22.5) % 16;
    return dirs[idx];
  }

  /**
   * Clamp a value between min and max.
   * @param {number} value
   * @param {number} min
   * @param {number} max
   * @returns {number}
   * @private
   */
  _clamp(value, min, max) {
    return Math.max(min, Math.min(max, value));
  }
}

/**
 * Exponential moving average helper.
 * @param {number} value - Current value
 * @param {number} prev - Previous EMA
 * @param {number} alpha - Smoothing factor
 * @returns {number}
 */
function ema(value, prev, alpha) {
  return alpha * value + (1 - alpha) * prev;
}

export default VehicleMetrics;
