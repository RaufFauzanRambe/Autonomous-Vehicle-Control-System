/**
 * @module sensor_metrics
 * @description Sensor health monitoring module for the Autonomous Vehicle
 * Control System. Provides per-sensor health monitoring (data rate, latency,
 * error rate), calibration drift detection, sensor fusion quality score,
 * data completeness metrics, and uptime tracking.
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
 * Error thrown by the sensor metrics module.
 * @extends Error
 */
export class SensorMetricsError extends Error {
  /**
   * @param {string} message - Error description
   * @param {string} [code='SENSOR_METRICS_ERROR'] - Machine-readable error code
   */
  constructor(message, code = 'SENSOR_METRICS_ERROR') {
    super(message);
    this.name = 'SensorMetricsError';
    this.code = code;
    this.timestamp = Date.now();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** @enum {string} Sensor health status */
export const SensorHealthStatus = {
  HEALTHY: 'healthy',
  DEGRADED: 'degraded',
  CRITICAL: 'critical',
  OFFLINE: 'offline',
  UNKNOWN: 'unknown'
};

/** @enum {string} Sensor types */
export const SensorType = {
  GPS: 'gps',
  IMU: 'imu',
  LIDAR: 'lidar',
  RADAR: 'radar',
  CAMERA: 'camera',
  ULTRASONIC: 'ultrasonic',
  CAN_BUS: 'can_bus',
  WHEEL_SPEED: 'wheel_speed',
  STEERING_ANGLE: 'steering_angle',
  BRAKE_PRESSURE: 'brake_pressure'
};

/** @type {object} Default expected data rates per sensor type (Hz) */
const DEFAULT_DATA_RATES = {
  [SensorType.GPS]: 10,
  [SensorType.IMU]: 100,
  [SensorType.LIDAR]: 20,
  [SensorType.RADAR]: 15,
  [SensorType.CAMERA]: 30,
  [SensorType.ULTRASONIC]: 10,
  [SensorType.CAN_BUS]: 50,
  [SensorType.WHEEL_SPEED]: 100,
  [SensorType.STEERING_ANGLE]: 100,
  [SensorType.BRAKE_PRESSURE]: 100
};

/** @type {object} Default configuration */
const DEFAULT_CONFIG = {
  /** Maximum acceptable latency per sensor type (ms) */
  maxLatencyMs: 200,
  /** Minimum acceptable data rate as fraction of expected (0-1) */
  minRateFraction: 0.8,
  /** Maximum acceptable error rate (0-1) */
  maxErrorRate: 0.05,
  /** Calibration drift threshold (relative change) */
  calibrationDriftThreshold: 0.1,
  /** Window size for rate calculation (ms) */
  rateWindowMs: 5000,
  /** Sensor timeout (ms) - time before sensor considered offline */
  sensorTimeoutMs: 3000,
  /** Data completeness check interval (ms) */
  completenessIntervalMs: 1000,
  /** Expected data rates per sensor type */
  expectedRates: DEFAULT_DATA_RATES,
  /** Uptime tracking resolution (ms) */
  uptimeResolutionMs: 60000
};

// ─────────────────────────────────────────────────────────────────────────────
// SensorTracker
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Tracks health metrics for a single sensor.
 */
class SensorTracker {
  /**
   * @param {string} sensorId - Unique sensor identifier
   * @param {string} sensorType - Sensor type
   * @param {object} config - Configuration
   */
  constructor(sensorId, sensorType, config) {
    /** @type {string} */
    this.sensorId = sensorId;
    /** @type {string} */
    this.sensorType = sensorType;
    /** @type {object} */
    this.config = config;

    // ── Data Rate Tracking ─────────────────────────────────────────────
    /** @type {Array<number>} Timestamps of received data points */
    this._dataTimestamps = [];
    /** @type {number} Current measured data rate (Hz) */
    this._currentRate = 0;
    /** @type {number} Expected data rate (Hz) */
    this._expectedRate = config.expectedRates[sensorType] || 10;

    // ── Latency Tracking ───────────────────────────────────────────────
    /** @type {Array<number>} Recent latency measurements */
    this._latencies = [];
    /** @type {number} Average latency (ms) */
    this._avgLatency = 0;
    /** @type {number} Maximum latency (ms) */
    this._maxLatency = 0;

    // ── Error Tracking ─────────────────────────────────────────────────
    /** @type {number} Total data points received */
    this._totalDataPoints = 0;
    /** @type {number} Total error count */
    this._totalErrors = 0;
    /** @type {number} Current error rate (0-1) */
    this._errorRate = 0;
    /** @type {Array<{timestamp:number, type:string}>} Recent errors */
    this._recentErrors = [];

    // ── Calibration Tracking ───────────────────────────────────────────
    /** @type {number|null} Baseline calibration value */
    this._calibrationBaseline = null;
    /** @type {number} Current calibration drift (0-1) */
    this._calibrationDrift = 0;
    /** @type {Array<{timestamp:number, drift:number}>} Calibration history */
    this._calibrationHistory = [];

    // ── Data Completeness ──────────────────────────────────────────────
    /** @type {Map<string, number>} Expected fields and their received counts */
    this._fieldCounts = new Map();
    /** @type {number} Total expected field instances */
    this._totalExpectedFields = 0;
    /** @type {number} Data completeness score (0-1) */
    this._completeness = 1.0;

    // ── Uptime Tracking ────────────────────────────────────────────────
    /** @type {number} Timestamp of first data received */
    this._firstDataTime = null;
    /** @type {number} Timestamp of last data received */
    this._lastDataTime = null;
    /** @type {number} Total time sensor was online (ms) */
    this._totalOnlineTime = 0;
    /** @type {number} Number of offline periods */
    this._offlineCount = 0;
    /** @type {boolean} Current online status */
    this._isOnline = false;
    /** @type {number} Time of last state transition */
    this._lastStateChangeTime = null;

    // ── Fusion Quality ─────────────────────────────────────────────────
    /** @type {number} Fusion quality score (0-1) */
    this._fusionQuality = 1.0;
    /** @type {number} Consistency score with other sensors (0-1) */
    this._consistencyScore = 1.0;

    /** @type {string} */
    this._healthStatus = SensorHealthStatus.UNKNOWN;
  }

  /**
   * Record a data point from this sensor.
   * @param {object} data - Telemetry data
   * @param {number} latencyMs - Measured latency
   * @param {string[]} [expectedFields=[]] - Expected fields in the data
   */
  recordDataPoint(data, latencyMs = 0, expectedFields = []) {
    const now = Date.now();

    // Track data rate
    this._dataTimestamps.push(now);
    this._trimTimestamps();

    // Track latency
    if (latencyMs > 0) {
      this._latencies.push(latencyMs);
      if (this._latencies.length > 1000) this._latencies.shift();
      this._avgLatency = this._latencies.reduce((a, b) => a + b, 0) / this._latencies.length;
      this._maxLatency = Math.max(this._maxLatency, latencyMs);
    }

    // Track data points
    this._totalDataPoints++;

    // Track completeness
    if (expectedFields.length > 0) {
      this._totalExpectedFields += expectedFields.length;
      for (const field of expectedFields) {
        if (data[field] !== undefined && data[field] !== null) {
          this._fieldCounts.set(field, (this._fieldCounts.get(field) || 0) + 1);
        }
      }
      this._completeness = this._totalExpectedFields > 0
        ? Array.from(this._fieldCounts.values()).reduce((a, b) => a + b, 0) / this._totalExpectedFields
        : 1.0;
    }

    // Track uptime
    if (!this._firstDataTime) {
      this._firstDataTime = now;
      this._lastStateChangeTime = now;
    }

    if (!this._isOnline) {
      this._isOnline = true;
      if (this._lastStateChangeTime) {
        // Transitioning from offline to online
      }
      this._lastStateChangeTime = now;
    }
    this._lastDataTime = now;

    // Calculate data rate
    this._calculateDataRate();
  }

  /**
   * Record a sensor error.
   * @param {string} errorType - Error type classification
   * @param {string} [message=''] - Error message
   */
  recordError(errorType, message = '') {
    this._totalErrors++;
    this._recentErrors.push({ timestamp: Date.now(), type: errorType, message });
    if (this._recentErrors.length > 100) this._recentErrors.shift();

    // Calculate error rate
    this._errorRate = this._totalDataPoints > 0
      ? this._totalErrors / this._totalDataPoints
      : 0;
  }

  /**
   * Update calibration tracking.
   * @param {number} currentValue - Current calibration measurement
   */
  updateCalibration(currentValue) {
    if (this._calibrationBaseline === null) {
      this._calibrationBaseline = currentValue;
      return;
    }

    const baseline = this._calibrationBaseline;
    if (baseline === 0) {
      this._calibrationDrift = Math.abs(currentValue);
    } else {
      this._calibrationDrift = Math.abs((currentValue - baseline) / baseline);
    }

    this._calibrationHistory.push({
      timestamp: Date.now(),
      drift: this._calibrationDrift,
      value: currentValue
    });

    if (this._calibrationHistory.length > 1000) {
      this._calibrationHistory.shift();
    }
  }

  /**
   * Update the fusion quality score based on cross-sensor consistency.
   * @param {number} consistencyScore - Consistency with other sensors (0-1)
   */
  updateFusionQuality(consistencyScore) {
    this._consistencyScore = consistencyScore;
    // Fusion quality combines data rate, error rate, latency, and consistency
    const rateScore = this._expectedRate > 0
      ? Math.min(1, this._currentRate / this._expectedRate)
      : 1;
    const errorScore = 1 - this._errorRate;
    const latencyScore = this._avgLatency < this.config.maxLatencyMs ? 1 : 0.5;
    const consistencyWeight = 0.3;
    const otherWeight = (1 - consistencyWeight) / 3;

    this._fusionQuality = (
      rateScore * otherWeight +
      errorScore * otherWeight +
      latencyScore * otherWeight +
      consistencyScore * consistencyWeight
    );
  }

  /**
   * Check and update the sensor's online status.
   */
  checkOnlineStatus() {
    const now = Date.now();
    if (this._isOnline && this._lastDataTime &&
        (now - this._lastDataTime) > this.config.sensorTimeoutMs) {
      // Sensor went offline
      if (this._lastStateChangeTime) {
        this._totalOnlineTime += now - this._lastStateChangeTime;
      }
      this._isOnline = false;
      this._lastStateChangeTime = now;
      this._offlineCount++;
    }
  }

  /**
   * Compute overall health status.
   * @returns {SensorHealthStatus}
   */
  computeHealthStatus() {
    if (!this._isOnline) {
      this._healthStatus = SensorHealthStatus.OFFLINE;
      return this._healthStatus;
    }

    let score = 0;
    const rateOk = this._expectedRate > 0
      ? this._currentRate / this._expectedRate >= this.config.minRateFraction
      : true;
    const latencyOk = this._avgLatency < this.config.maxLatencyMs;
    const errorOk = this._errorRate < this.config.maxErrorRate;
    const calibOk = this._calibrationDrift < this.config.calibrationDriftThreshold;

    if (rateOk) score++;
    if (latencyOk) score++;
    if (errorOk) score++;
    if (calibOk) score++;

    if (score === 4) {
      this._healthStatus = SensorHealthStatus.HEALTHY;
    } else if (score >= 2) {
      this._healthStatus = SensorHealthStatus.DEGRADED;
    } else {
      this._healthStatus = SensorHealthStatus.CRITICAL;
    }

    return this._healthStatus;
  }

  /**
   * Get all metrics for this sensor.
   * @returns {object}
   */
  getMetrics() {
    const now = Date.now();

    // Update online time if currently online
    let currentOnlineTime = this._totalOnlineTime;
    if (this._isOnline && this._lastStateChangeTime) {
      currentOnlineTime += now - this._lastStateChangeTime;
    }

    const totalTrackingTime = this._firstDataTime ? now - this._firstDataTime : 0;
    const uptimePercent = totalTrackingTime > 0
      ? (currentOnlineTime / totalTrackingTime) * 100
      : 0;

    return {
      sensorId: this.sensorId,
      sensorType: this.sensorType,
      healthStatus: this._healthStatus,
      isOnline: this._isOnline,
      dataRate: {
        currentHz: parseFloat(this._currentRate.toFixed(1)),
        expectedHz: this._expectedRate,
        percent: this._expectedRate > 0
          ? parseFloat(((this._currentRate / this._expectedRate) * 100).toFixed(1))
          : 0
      },
      latency: {
        avgMs: parseFloat(this._avgLatency.toFixed(1)),
        maxMs: this._maxLatency,
        recent: this._latencies.slice(-10)
      },
      errors: {
        total: this._totalErrors,
        rate: parseFloat(this._errorRate.toFixed(4)),
        recent: this._recentErrors.slice(-5)
      },
      calibration: {
        baseline: this._calibrationBaseline,
        drift: parseFloat(this._calibrationDrift.toFixed(4)),
        isDrifted: this._calibrationDrift > this.config.calibrationDriftThreshold,
        history: this._calibrationHistory.slice(-20)
      },
      completeness: {
        score: parseFloat(this._completeness.toFixed(3)),
        totalExpectedFields: this._totalExpectedFields,
        receivedFields: Array.from(this._fieldCounts.values()).reduce((a, b) => a + b, 0)
      },
      fusion: {
        quality: parseFloat(this._fusionQuality.toFixed(3)),
        consistency: parseFloat(this._consistencyScore.toFixed(3))
      },
      uptime: {
        totalOnlineMs: currentOnlineTime,
        totalTrackingMs: totalTrackingTime,
        uptimePercent: parseFloat(uptimePercent.toFixed(1)),
        offlineCount: this._offlineCount,
        firstDataTime: this._firstDataTime,
        lastDataTime: this._lastDataTime
      },
      totalDataPoints: this._totalDataPoints
    };
  }

  // ── Private Methods ─────────────────────────────────────────────────────

  /**
   * Calculate current data rate from recent timestamps.
   * @private
   */
  _calculateDataRate() {
    this._trimTimestamps();

    if (this._dataTimestamps.length < 2) {
      this._currentRate = 0;
      return;
    }

    const timeSpan = (this._dataTimestamps[this._dataTimestamps.length - 1] - this._dataTimestamps[0]) / 1000;
    if (timeSpan > 0) {
      this._currentRate = (this._dataTimestamps.length - 1) / timeSpan;
    }
  }

  /**
   * Trim timestamps outside the rate calculation window.
   * @private
   */
  _trimTimestamps() {
    const cutoff = Date.now() - this.config.rateWindowMs;
    while (this._dataTimestamps.length > 0 && this._dataTimestamps[0] < cutoff) {
      this._dataTimestamps.shift();
    }
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// SensorMetrics
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Per-sensor health monitoring module. Tracks data rate, latency, error rate,
 * calibration drift, sensor fusion quality, data completeness, and uptime
 * for each sensor in the autonomous vehicle system.
 *
 * @extends EventEmitter
 *
 * @example
 * const metrics = new SensorMetrics({ sensorTimeoutMs: 5000 });
 * metrics.registerSensor('front_lidar', SensorType.LIDAR);
 * metrics.recordDataPoint('front_lidar', lidarData);
 * const health = metrics.getSensorHealth('front_lidar');
 * const report = metrics.generateHealthReport();
 */
export class SensorMetrics extends EventEmitter {
  /**
   * @param {object} [config={}] - Configuration overrides
   */
  constructor(config = {}) {
    super();

    /** @type {object} */
    this.config = { ...DEFAULT_CONFIG, ...config };

    /** @type {Map<string, SensorTracker>} Per-sensor trackers */
    this._sensors = new Map();

    /** @type {NodeJS.Timer|null} Health check timer */
    this._healthCheckTimer = null;

    /** @type {object} Module-level statistics */
    this._stats = {
      totalDataPoints: 0,
      totalErrors: 0,
      alertsFired: 0,
      lastHealthCheck: null
    };
  }

  // ── Sensor Registration ─────────────────────────────────────────────────

  /**
   * Register a new sensor for tracking.
   * @param {string} sensorId - Unique sensor identifier
   * @param {string} sensorType - Sensor type (from SensorType enum)
   * @param {object} [options={}] - Additional options
   * @param {number} [options.expectedRate] - Override expected data rate (Hz)
   * @param {number} [options.calibrationBaseline] - Initial calibration value
   * @throws {SensorMetricsError} If sensor already registered
   */
  registerSensor(sensorId, sensorType, options = {}) {
    if (this._sensors.has(sensorId)) {
      throw new SensorMetricsError(
        `Sensor '${sensorId}' is already registered`,
        'SENSOR_ALREADY_REGISTERED'
      );
    }

    const config = { ...this.config };
    if (options.expectedRate) {
      config.expectedRates = { ...config.expectedRates, [sensorType]: options.expectedRate };
    }

    const tracker = new SensorTracker(sensorId, sensorType, config);
    if (options.calibrationBaseline !== undefined) {
      tracker._calibrationBaseline = options.calibrationBaseline;
    }

    this._sensors.set(sensorId, tracker);
    this.emit('sensor:registered', { sensorId, sensorType });
  }

  /**
   * Unregister a sensor.
   * @param {string} sensorId - Sensor ID to remove
   */
  unregisterSensor(sensorId) {
    this._sensors.delete(sensorId);
    this.emit('sensor:unregistered', { sensorId });
  }

  // ── Data Recording ──────────────────────────────────────────────────────

  /**
   * Record a data point from a sensor.
   * @param {string} sensorId - Sensor identifier
   * @param {object} data - Telemetry data from the sensor
   * @param {object} [options={}] - Recording options
   * @param {number} [options.latencyMs] - Measured sensor latency
   * @param {string[]} [options.expectedFields] - Expected fields for completeness check
   */
  recordDataPoint(sensorId, data, options = {}) {
    const tracker = this._sensors.get(sensorId);
    if (!tracker) {
      // Auto-register unknown sensors
      const sensorType = data.type || data.sensorType || SensorType.UNKNOWN;
      this.registerSensor(sensorId, sensorType);
    }

    const t = this._sensors.get(sensorId);
    t.recordDataPoint(data, options.latencyMs || 0, options.expectedFields || []);
    this._stats.totalDataPoints++;

    // Check for calibration drift
    if (data.data) {
      const payload = data.data;
      const firstNumericKey = Object.keys(payload).find(k => typeof payload[k] === 'number');
      if (firstNumericKey) {
        t.updateCalibration(payload[firstNumericKey]);
      }
    }
  }

  /**
   * Record a sensor error.
   * @param {string} sensorId - Sensor identifier
   * @param {string} errorType - Error type classification
   * @param {string} [message=''] - Error message
   */
  recordError(sensorId, errorType, message = '') {
    const tracker = this._sensors.get(sensorId);
    if (!tracker) return;

    tracker.recordError(errorType, message);
    this._stats.totalErrors++;
    this.emit('sensor:error', { sensorId, errorType, message });
  }

  // ── Health Queries ──────────────────────────────────────────────────────

  /**
   * Get health metrics for a specific sensor.
   * @param {string} sensorId - Sensor identifier
   * @returns {object|null} Sensor metrics or null if not found
   */
  getSensorHealth(sensorId) {
    const tracker = this._sensors.get(sensorId);
    if (!tracker) return null;
    return tracker.getMetrics();
  }

  /**
   * Get the health status of all registered sensors.
   * @returns {object} Map of sensor ID to health status
   */
  getAllSensorHealth() {
    const result = {};
    for (const [id, tracker] of this._sensors) {
      tracker.checkOnlineStatus();
      tracker.computeHealthStatus();
      result[id] = tracker.getMetrics();
    }
    return result;
  }

  /**
   * Get overall sensor fleet health summary.
   * @returns {object}
   */
  getFleetHealth() {
    const now = Date.now();
    let healthy = 0, degraded = 0, critical = 0, offline = 0;

    for (const [, tracker] of this._sensors) {
      tracker.checkOnlineStatus();
      const status = tracker.computeHealthStatus();
      switch (status) {
        case SensorHealthStatus.HEALTHY: healthy++; break;
        case SensorHealthStatus.DEGRADED: degraded++; break;
        case SensorHealthStatus.CRITICAL: critical++; break;
        case SensorHealthStatus.OFFLINE: offline++; break;
      }
    }

    const total = this._sensors.size;
    const overallStatus = critical > 0 || offline > total / 2
      ? SensorHealthStatus.CRITICAL
      : degraded > total / 2
        ? SensorHealthStatus.DEGRADED
        : SensorHealthStatus.HEALTHY;

    return {
      overallStatus,
      totalSensors: total,
      healthy,
      degraded,
      critical,
      offline,
      healthPercent: total > 0 ? (healthy / total) * 100 : 0,
      timestamp: now
    };
  }

  /**
   * Generate a comprehensive health report.
   * @returns {object}
   */
  generateHealthReport() {
    const fleet = this.getFleetHealth();
    const sensors = this.getAllSensorHealth();

    // Identify problematic sensors
    const problematicSensors = Object.entries(sensors)
      .filter(([, m]) => m.healthStatus !== SensorHealthStatus.HEALTHY)
      .map(([id, m]) => ({
        sensorId: id,
        status: m.healthStatus,
        issues: this._identifyIssues(m)
      }));

    // Compute average fusion quality
    const fusionQualities = Object.values(sensors).map(m => m.fusion.quality);
    const avgFusionQuality = fusionQualities.length > 0
      ? fusionQualities.reduce((a, b) => a + b, 0) / fusionQualities.length
      : 0;

    return {
      fleet,
      sensors,
      problematicSensors,
      avgFusionQuality: parseFloat(avgFusionQuality.toFixed(3)),
      stats: this._stats,
      timestamp: Date.now()
    };
  }

  // ── Status ──────────────────────────────────────────────────────────────

  /**
   * Get module statistics.
   * @returns {object}
   */
  getStats() {
    return { ...this._stats, registeredSensors: this._sensors.size };
  }

  /**
   * Get health status.
   * @returns {object}
   */
  getHealth() {
    const fleet = this.getFleetHealth();
    return {
      status: fleet.overallStatus,
      totalSensors: fleet.totalSensors,
      healthyCount: fleet.healthy,
      degradedCount: fleet.degraded,
      criticalCount: fleet.critical,
      offlineCount: fleet.offline
    };
  }

  // ── Private Methods ─────────────────────────────────────────────────────

  /**
   * Identify issues from sensor metrics.
   * @param {object} metrics - Sensor metrics
   * @returns {string[]}
   * @private
   */
  _identifyIssues(metrics) {
    const issues = [];

    if (!metrics.isOnline) {
      issues.push('Sensor offline');
    }
    if (metrics.dataRate.percent < 80) {
      issues.push(`Low data rate: ${metrics.dataRate.currentHz}Hz (expected ${metrics.dataRate.expectedHz}Hz)`);
    }
    if (metrics.latency.avgMs > this.config.maxLatencyMs) {
      issues.push(`High latency: ${metrics.latency.avgMs}ms`);
    }
    if (metrics.errors.rate > this.config.maxErrorRate) {
      issues.push(`High error rate: ${(metrics.errors.rate * 100).toFixed(1)}%`);
    }
    if (metrics.calibration.isDrifted) {
      issues.push(`Calibration drift: ${(metrics.calibration.drift * 100).toFixed(1)}%`);
    }
    if (metrics.completeness.score < 0.9) {
      issues.push(`Data completeness: ${(metrics.completeness.score * 100).toFixed(1)}%`);
    }
    if (metrics.fusion.quality < 0.7) {
      issues.push(`Low fusion quality: ${(metrics.fusion.quality * 100).toFixed(1)}%`);
    }
    if (metrics.uptime.uptimePercent < 95) {
      issues.push(`Low uptime: ${metrics.uptime.uptimePercent.toFixed(1)}%`);
    }

    return issues;
  }
}

export default SensorMetrics;
