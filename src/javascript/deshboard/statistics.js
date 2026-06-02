/**
 * @fileoverview Statistics Computation
 * @description Moving averages, percentile calculations, min/max/std deviation,
 *   trip statistics (distance, duration, avg speed), sensor reliability metrics,
 *   and data quality scores for the Autonomous Vehicle Dashboard.
 * @module statistics
 */

// ─── Basic Statistics ─────────────────────────────────────────────────────────

/**
 * Compute the mean of a numeric array
 * @param {number[]} values - Input values
 * @returns {number} Mean value
 * @throws {Error} If array is empty
 */
export function mean(values) {
  if (!values || values.length === 0) throw new Error('Cannot compute mean of empty array');
  return values.reduce((sum, v) => sum + v, 0) / values.length;
}

/**
 * Compute the median of a numeric array
 * @param {number[]} values - Input values
 * @returns {number} Median value
 */
export function median(values) {
  if (!values || values.length === 0) throw new Error('Cannot compute median of empty array');
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 !== 0 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/**
 * Compute the mode (most frequent value) of a numeric array
 * @param {number[]} values - Input values
 * @returns {number} Mode value
 */
export function mode(values) {
  if (!values || values.length === 0) throw new Error('Cannot compute mode of empty array');
  const freq = new Map();
  for (const v of values) freq.set(v, (freq.get(v) || 0) + 1);
  let maxFreq = 0;
  let modeVal = values[0];
  for (const [val, count] of freq.entries()) {
    if (count > maxFreq) { maxFreq = count; modeVal = val; }
  }
  return modeVal;
}

/**
 * Compute the standard deviation of a numeric array
 * @param {number[]} values - Input values
 * @param {boolean} [population=false] - Use population formula (divide by N)
 * @returns {number} Standard deviation
 */
export function standardDeviation(values, population = false) {
  if (!values || values.length < 2) return 0;
  const avg = mean(values);
  const squareDiffs = values.map((v) => Math.pow(v - avg, 2));
  const variance = squareDiffs.reduce((sum, d) => sum + d, 0) / (population ? values.length : values.length - 1);
  return Math.sqrt(variance);
}

/**
 * Compute the variance of a numeric array
 * @param {number[]} values - Input values
 * @param {boolean} [population=false] - Use population formula
 * @returns {number} Variance
 */
export function variance(values, population = false) {
  if (!values || values.length < 2) return 0;
  const avg = mean(values);
  const squareDiffs = values.map((v) => Math.pow(v - avg, 2));
  return squareDiffs.reduce((sum, d) => sum + d, 0) / (population ? values.length : values.length - 1);
}

/**
 * Compute min, max, and range of a numeric array
 * @param {number[]} values - Input values
 * @returns {{ min: number, max: number, range: number }}
 */
export function minMaxRange(values) {
  if (!values || values.length === 0) return { min: 0, max: 0, range: 0 };
  const minVal = Math.min(...values);
  const maxVal = Math.max(...values);
  return { min: minVal, max: maxVal, range: maxVal - minVal };
}

// ─── Percentile Calculations ──────────────────────────────────────────────────

/**
 * Compute a specific percentile of a numeric array
 * @param {number[]} values - Input values
 * @param {number} percentile - Percentile (0-100)
 * @param {boolean} [interpolate=true] - Use linear interpolation
 * @returns {number} Percentile value
 */
export function percentile(values, percentile, interpolate = true) {
  if (!values || values.length === 0) throw new Error('Cannot compute percentile of empty array');
  if (percentile < 0 || percentile > 100) throw new Error('Percentile must be between 0 and 100');

  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];

  const index = (percentile / 100) * (sorted.length - 1);
  const lower = Math.floor(index);
  const upper = Math.ceil(index);

  if (!interpolate || lower === upper) return sorted[lower];

  const fraction = index - lower;
  return sorted[lower] + fraction * (sorted[upper] - sorted[lower]);
}

/**
 * Compute quartiles (Q1, Q2, Q3) and IQR
 * @param {number[]} values - Input values
 * @returns {{ q1: number, q2: number, q3: number, iqr: number }}
 */
export function quartiles(values) {
  const q1 = percentile(values, 25);
  const q2 = percentile(values, 50);
  const q3 = percentile(values, 75);
  return { q1, q2, q3, iqr: q3 - q1 };
}

/**
 * Identify outlier values using the IQR method
 * @param {number[]} values - Input values
 * @param {number} [k=1.5] - IQR multiplier for outlier bounds
 * @returns {{ outliers: number[], lowerBound: number, upperBound: number }}
 */
export function findOutliers(values, k = 1.5) {
  const { q1, q3, iqr } = quartiles(values);
  const lowerBound = q1 - k * iqr;
  const upperBound = q3 + k * iqr;
  const outliers = values.filter((v) => v < lowerBound || v > upperBound);
  return { outliers, lowerBound, upperBound };
}

// ─── Moving Averages ─────────────────────────────────────────────────────────

/**
 * Compute simple moving average
 * @param {number[]} values - Input values
 * @param {number} windowSize - Window size
 * @returns {number[]} SMA values (shorter than input by windowSize-1)
 */
export function simpleMovingAverage(values, windowSize) {
  if (!values || values.length < windowSize) return [];
  const result = [];
  for (let i = windowSize - 1; i < values.length; i++) {
    const window = values.slice(i - windowSize + 1, i + 1);
    result.push(mean(window));
  }
  return result;
}

/**
 * Compute exponential moving average
 * @param {number[]} values - Input values
 * @param {number} period - EMA period
 * @returns {number[]} EMA values
 */
export function exponentialMovingAverage(values, period) {
  if (!values || values.length === 0) return [];
  const k = 2 / (period + 1);
  const result = [values[0]];
  for (let i = 1; i < values.length; i++) {
    result.push(values[i] * k + result[i - 1] * (1 - k));
  }
  return result;
}

/**
 * Compute weighted moving average
 * @param {number[]} values - Input values
 * @param {number[]} weights - Weight array (must match window size)
 * @returns {number[]} WMA values
 */
export function weightedMovingAverage(values, weights) {
  if (!values || !weights || values.length < weights.length) return [];
  const weightSum = weights.reduce((a, b) => a + b, 0);
  const result = [];
  for (let i = weights.length - 1; i < values.length; i++) {
    let wma = 0;
    for (let j = 0; j < weights.length; j++) {
      wma += values[i - weights.length + 1 + j] * weights[j];
    }
    result.push(wma / weightSum);
  }
  return result;
}

// ─── Trip Statistics ──────────────────────────────────────────────────────────

/**
 * @typedef {Object} TripStats
 * @property {number} totalDistanceKm - Total trip distance
 * @property {number} totalDurationMs - Total trip duration
 * @property {number} avgSpeedKmh - Average speed
 * @property {number} maxSpeedKmh - Maximum speed
 * @property {number} minSpeedKmh - Minimum speed (when moving)
 * @property {number} movingDurationMs - Duration while vehicle was moving
 * @property {number} idleDurationMs - Duration while vehicle was stationary
 * @property {number} fuelEfficiency - Fuel/energy efficiency
 * @property {number} elevationGainM - Total elevation gain
 * @property {number} elevationLossM - Total elevation loss
 */

/**
 * Compute comprehensive trip statistics from telemetry data
 * @param {Object[]} telemetryData - Array of telemetry data points
 * @param {number} [movingThreshold=1] - Speed threshold for "moving" in km/h
 * @returns {TripStats}
 */
export function computeTripStatistics(telemetryData, movingThreshold = 1) {
  if (!telemetryData || telemetryData.length === 0) {
    return {
      totalDistanceKm: 0, totalDurationMs: 0, avgSpeedKmh: 0,
      maxSpeedKmh: 0, minSpeedKmh: 0, movingDurationMs: 0,
      idleDurationMs: 0, fuelEfficiency: 0, elevationGainM: 0, elevationLossM: 0,
    };
  }

  const speeds = telemetryData.map((d) => d.speed ?? 0).filter((s) => s >= 0);
  const timestamps = telemetryData.map((d) => d.timestamp || d._receivedAt || 0).filter((t) => t > 0);
  const distances = telemetryData.map((d) => d.distanceDelta ?? 0);
  const altitudes = telemetryData.map((d) => d.altitude ?? null).filter((a) => a !== null);

  const totalDistanceKm = distances.reduce((sum, d) => sum + d, 0);
  const totalDurationMs = timestamps.length >= 2 ? timestamps[timestamps.length - 1] - timestamps[0] : 0;
  const avgSpeedKmh = speeds.length > 0 ? mean(speeds) : 0;
  const maxSpeedKmh = speeds.length > 0 ? Math.max(...speeds) : 0;
  const minSpeedKmh = speeds.length > 0 ? Math.min(...speeds.filter((s) => s > movingThreshold)) : 0;

  // Compute moving vs idle duration
  let movingDurationMs = 0;
  let idleDurationMs = 0;
  for (let i = 1; i < telemetryData.length; i++) {
    const dt = (telemetryData[i].timestamp || 0) - (telemetryData[i - 1].timestamp || 0);
    if (dt > 0) {
      if ((telemetryData[i].speed ?? 0) > movingThreshold) {
        movingDurationMs += dt;
      } else {
        idleDurationMs += dt;
      }
    }
  }

  // Elevation change
  let elevationGainM = 0;
  let elevationLossM = 0;
  for (let i = 1; i < altitudes.length; i++) {
    const diff = altitudes[i] - altitudes[i - 1];
    if (diff > 0) elevationGainM += diff;
    else elevationLossM += Math.abs(diff);
  }

  // Fuel/energy efficiency (km/kWh equivalent)
  const energyUsed = telemetryData.reduce((sum, d) => sum + (d.energyDelta ?? 0), 0);
  const fuelEfficiency = energyUsed > 0 ? totalDistanceKm / energyUsed : 0;

  return {
    totalDistanceKm,
    totalDurationMs,
    avgSpeedKmh: Math.round(avgSpeedKmh * 100) / 100,
    maxSpeedKmh: Math.round(maxSpeedKmh * 100) / 100,
    minSpeedKmh: Math.round(minSpeedKmh * 100) / 100,
    movingDurationMs,
    idleDurationMs,
    fuelEfficiency: Math.round(fuelEfficiency * 100) / 100,
    elevationGainM: Math.round(elevationGainM * 10) / 10,
    elevationLossM: Math.round(elevationLossM * 10) / 10,
  };
}

// ─── Sensor Reliability Metrics ──────────────────────────────────────────────

/**
 * @typedef {Object} SensorReliability
 * @property {number} uptime - Uptime percentage (0-100)
 * @property {number} meanTimeBetweenFailures - MTBF in ms
 * @property {number} meanTimeToRecovery - MTTR in ms
 * @property {number} dataCompleteness - Data completeness score (0-1)
 * @property {number} dataFreshness - Data freshness score (0-1)
 * @property {number} overallScore - Overall reliability score (0-1)
 */

/**
 * Compute reliability metrics for a sensor
 * @param {Object[]} statusHistory - Array of sensor status entries
 * @param {number} expectedInterval - Expected data interval in ms
 * @returns {SensorReliability}
 */
export function computeSensorReliability(statusHistory, expectedInterval = 100) {
  if (!statusHistory || statusHistory.length === 0) {
    return { uptime: 0, meanTimeBetweenFailures: 0, meanTimeToRecovery: 0, dataCompleteness: 0, dataFreshness: 0, overallScore: 0 };
  }

  // Uptime: percentage of active readings
  const activeReadings = statusHistory.filter((s) => s.status === 'active' || s.active === true).length;
  const uptime = (activeReadings / statusHistory.length) * 100;

  // MTBF: average time between failures
  const failureTimestamps = [];
  for (let i = 1; i < statusHistory.length; i++) {
    const prev = statusHistory[i - 1];
    const curr = statusHistory[i];
    if ((prev.status === 'active' || prev.active) && curr.status !== 'active' && !curr.active) {
      failureTimestamps.push(curr.timestamp || i);
    }
  }
  const mtbf = failureTimestamps.length >= 2
    ? (failureTimestamps[failureTimestamps.length - 1] - failureTimestamps[0]) / (failureTimestamps.length - 1)
    : 0;

  // MTTR: average recovery time
  const recoveryTimes = [];
  let failureStart = null;
  for (let i = 0; i < statusHistory.length; i++) {
    const s = statusHistory[i];
    if (s.status !== 'active' && !s.active && failureStart === null) {
      failureStart = s.timestamp || i;
    } else if ((s.status === 'active' || s.active) && failureStart !== null) {
      recoveryTimes.push((s.timestamp || i) - failureStart);
      failureStart = null;
    }
  }
  const mttr = recoveryTimes.length > 0 ? mean(recoveryTimes) : 0;

  // Data completeness: ratio of non-null readings
  const nonNullReadings = statusHistory.filter((s) => s.value !== null && s.value !== undefined).length;
  const dataCompleteness = nonNullReadings / statusHistory.length;

  // Data freshness: how recent the last reading is
  const lastTimestamp = statusHistory[statusHistory.length - 1]?.timestamp || 0;
  const timeSinceLastReading = Date.now() - lastTimestamp;
  const dataFreshness = Math.max(0, 1 - timeSinceLastReading / (expectedInterval * 10));

  // Overall score: weighted combination
  const overallScore = (uptime / 100) * 0.3 + dataCompleteness * 0.3 + dataFreshness * 0.2 + (mtbf > 0 ? Math.min(mtbf / 3600000, 1) : 0.5) * 0.2;

  return {
    uptime: Math.round(uptime * 100) / 100,
    meanTimeBetweenFailures: Math.round(mtbf),
    meanTimeToRecovery: Math.round(mttr),
    dataCompleteness: Math.round(dataCompleteness * 1000) / 1000,
    dataFreshness: Math.round(dataFreshness * 1000) / 1000,
    overallScore: Math.round(overallScore * 1000) / 1000,
  };
}

// ─── Data Quality Scoring ─────────────────────────────────────────────────────

/**
 * @typedef {Object} DataQualityReport
 * @property {number} overallScore - Overall quality score (0-1)
 * @property {number} completeness - Data completeness (0-1)
 * @property {number} consistency - Data consistency (0-1)
 * @property {number} timeliness - Data timeliness (0-1)
 * @property {number} validity - Data validity (0-1)
 * @property {string[]} issues - List of detected quality issues
 */

/**
 * Compute a data quality report for telemetry data
 * @param {Object[]} dataPoints - Array of telemetry data points
 * @param {Object} [schema={}] - Expected data schema with field types
 * @param {number} [expectedInterval=100] - Expected data interval in ms
 * @returns {DataQualityReport}
 */
export function computeDataQuality(dataPoints, schema = {}, expectedInterval = 100) {
  if (!dataPoints || dataPoints.length === 0) {
    return { overallScore: 0, completeness: 0, consistency: 0, timeliness: 0, validity: 0, issues: ['No data points provided'] };
  }

  const issues = [];
  const expectedFields = Object.keys(schema).length > 0 ? Object.keys(schema) : null;

  // Completeness: ratio of filled fields
  let totalFields = 0;
  let filledFields = 0;
  for (const dp of dataPoints) {
    const fields = expectedFields || Object.keys(dp);
    totalFields += fields.length;
    for (const field of fields) {
      if (dp[field] !== null && dp[field] !== undefined && dp[field] !== '') {
        filledFields++;
      }
    }
  }
  const completeness = totalFields > 0 ? filledFields / totalFields : 1;

  // Consistency: check for sudden jumps and type mismatches
  let inconsistentCount = 0;
  for (const dp of dataPoints) {
    if (dp.speed !== undefined && dp.speed < 0) { inconsistentCount++; issues.push('Negative speed detected'); }
    if (dp.batteryLevel !== undefined && (dp.batteryLevel < 0 || dp.batteryLevel > 100)) { inconsistentCount++; issues.push('Battery level out of range'); }
    if (dp.heading !== undefined && (dp.heading < 0 || dp.heading > 360)) { inconsistentCount++; issues.push('Heading out of range'); }
  }
  const consistency = dataPoints.length > 0 ? 1 - inconsistentCount / dataPoints.length : 1;

  // Timeliness: check data interval consistency
  let intervalDeviations = 0;
  const timestamps = dataPoints.map((d) => d.timestamp || d._receivedAt || 0).filter((t) => t > 0);
  for (let i = 1; i < timestamps.length; i++) {
    const interval = timestamps[i] - timestamps[i - 1];
    if (Math.abs(interval - expectedInterval) > expectedInterval * 3) {
      intervalDeviations++;
    }
  }
  const timeliness = timestamps.length > 1 ? 1 - intervalDeviations / (timestamps.length - 1) : 1;
  if (intervalDeviations > 0) issues.push(`${intervalDeviations} interval deviations detected`);

  // Validity: check schema conformance
  let invalidCount = 0;
  if (expectedFields) {
    for (const dp of dataPoints) {
      for (const [field, type] of Object.entries(schema)) {
        if (dp[field] !== undefined && typeof dp[field] !== type) {
          invalidCount++;
        }
      }
    }
  }
  const validity = totalFields > 0 ? 1 - invalidCount / totalFields : 1;

  const overallScore = completeness * 0.3 + consistency * 0.25 + timeliness * 0.25 + validity * 0.2;

  return {
    overallScore: Math.round(overallScore * 1000) / 1000,
    completeness: Math.round(completeness * 1000) / 1000,
    consistency: Math.round(consistency * 1000) / 1000,
    timeliness: Math.round(timeliness * 1000) / 1000,
    validity: Math.round(validity * 1000) / 1000,
    issues: [...new Set(issues)],
  };
}

// ─── Statistics Computation Class ─────────────────────────────────────────────

/**
 * Aggregator class for computing statistics over rolling windows
 */
export class StatisticsComputer {
  /**
   * @param {Object} [options={}]
   * @param {number} [options.windowSize=100] - Rolling window size
   * @param {number} [options.emaPeriod=20] - EMA period
   */
  constructor(options = {}) {
    /** @private */ this._windowSize = options.windowSize ?? 100;
    /** @private */ this._emaPeriod = options.emaPeriod ?? 20;
    /** @private @type {Map<string, number[]>} */ this._buffers = new Map();
  }

  /**
   * Add a value to a named buffer
   * @param {string} name - Buffer name
   * @param {number} value - Value to add
   */
  addValue(name, value) {
    if (!this._buffers.has(name)) this._buffers.set(name, []);
    const buffer = this._buffers.get(name);
    buffer.push(value);
    if (buffer.length > this._windowSize) buffer.shift();
  }

  /**
   * Compute statistics for a named buffer
   * @param {string} name - Buffer name
   * @returns {Object} Statistics summary
   */
  compute(name) {
    const values = this._buffers.get(name);
    if (!values || values.length === 0) {
      return { count: 0, mean: 0, median: 0, stdDev: 0, min: 0, max: 0, sma: [], ema: [] };
    }

    return {
      count: values.length,
      mean: mean(values),
      median: median(values),
      stdDev: standardDeviation(values),
      ...minMaxRange(values),
      ...quartiles(values),
      sma: simpleMovingAverage(values, Math.min(20, values.length)),
      ema: exponentialMovingAverage(values, this._emaPeriod),
    };
  }

  /**
   * Get all buffer names
   * @returns {string[]}
   */
  getBufferNames() {
    return Array.from(this._buffers.keys());
  }

  /**
   * Clear a specific buffer
   * @param {string} name - Buffer name
   */
  clearBuffer(name) {
    this._buffers.delete(name);
  }

  /**
   * Clear all buffers
   */
  clearAll() {
    this._buffers.clear();
  }
}
