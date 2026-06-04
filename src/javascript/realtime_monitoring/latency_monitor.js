/**
 * @fileoverview LatencyMonitor - End-to-end latency measurement and tracking
 * for the autonomous vehicle control system. Provides round-trip time tracking,
 * percentile statistics (P50/P95/P99), latency budget tracking per pipeline
 * stage, threshold breach alerts, and latency heatmap data generation.
 *
 * @module realtime_monitoring/latency_monitor
 * @version 2.1.0
 * @author Autonomous Vehicle Control System Team
 */

import { EventEmitter } from 'events';

/** @typedef {'perception'|'planning'|'control'|'communication'|'full_pipeline'} PipelineStage */

/**
 * @typedef {Object} LatencyMonitorConfig
 * @property {number} [sampleIntervalMs=50] - Default sampling interval
 * @property {number} [maxSamples=10000] - Maximum stored latency samples
 * @property {number} [historyRetentionMs=300000] - How long to keep samples (5 min)
 * @property {Object} [budgets] - Latency budgets per pipeline stage (ms)
 * @property {number} [budgets.perception=50] - Perception stage budget
 * @property {number} [budgets.planning=30] - Planning stage budget
 * @property {number} [budgets.control=20] - Control stage budget
 * @property {number} [budgets.communication=10] - Communication stage budget
 * @property {number} [budgets.full_pipeline=150] - Full pipeline budget
 * @property {Object} [thresholds] - Alert thresholds
 * @property {number} [thresholds.warningPercent=80] - Budget usage % for warning
 * @property {number} [thresholds.criticalPercent=95] - Budget usage % for critical
 * @property {number} [heatmapBucketCount=20] - Number of buckets for heatmap
 */

/**
 * @typedef {Object} LatencySample
 * @property {number} value - Latency value in milliseconds
 * @property {PipelineStage} stage - Pipeline stage
 * @property {number} timestamp - Sample timestamp
 * @property {string} [source] - Source identifier
 * @property {string} [requestId] - Associated request ID
 */

/**
 * @typedef {Object} LatencyStats
 * @property {number} p50 - 50th percentile latency
 * @property {number} p95 - 95th percentile latency
 * @property {number} p99 - 99th percentile latency
 * @property {number} avg - Average latency
 * @property {number} min - Minimum latency
 * @property {number} max - Maximum latency
 * @property {number} samples - Total sample count
 * @property {number} stddev - Standard deviation
 */

/**
 * @typedef {Object} StageBudget
 * @property {PipelineStage} stage - Pipeline stage
 * @property {number} budget - Allocated budget in ms
 * @property {number} used - Current usage in ms (P95)
 * @property {number} percentUsed - Percentage of budget used
 * @property {'ok'|'warning'|'critical'} status - Budget status
 */

/**
 * @typedef {Object} HeatmapBucket
 * @property {number} rangeStart - Bucket range start (ms)
 * @property {number} rangeEnd - Bucket range end (ms)
 * @property {number} count - Number of samples in bucket
 * @property {number} percentage - Percentage of total samples
 */

/**
 * @typedef {Object} RoundTripMeasurement
 * @property {string} id - Measurement ID
 * @property {number} startTime - Start timestamp
 * @property {number} [endTime] - End timestamp
 * @property {number} [duration] - Duration in ms
 * @property {boolean} completed - Whether the RTT is complete
 */

/**
 * LatencyMonitor provides comprehensive latency measurement with percentile
 * tracking, budget monitoring, and heatmap generation.
 *
 * @extends EventEmitter
 *
 * @example
 * const latency = new LatencyMonitor({
 *   budgets: { perception: 50, planning: 30, control: 20 },
 * });
 *
 * latency.on('latency:threshold_breach', (metric) => {
 *   console.warn(`Latency breach on ${metric.stage}: ${metric.p99}ms`);
 * });
 *
 * latency.start();
 * latency.recordSample({ value: 35, stage: 'perception' });
 */
export class LatencyMonitor extends EventEmitter {
  /** @type {LatencyMonitorConfig} */
  #config;

  /** @type {LatencySample[]} */
  #samples = [];

  /** @type {Map<string, RoundTripMeasurement>} */
  #roundTrips = new Map();

  /** @type {Map<PipelineStage, LatencySample[]>} */
  #stageSamples = new Map();

  /** @type {NodeJS.Timeout|null} */
  #cleanupTimer = null;

  /** @type {boolean} */
  #running = false;

  /** @type {number} */
  #totalSamplesRecorded = 0;

  /**
   * Creates a new LatencyMonitor.
   *
   * @param {LatencyMonitorConfig} [config={}] - Configuration
   */
  constructor(config = {}) {
    super();
    this.setMaxListeners(20);

    this.#config = {
      sampleIntervalMs: 50,
      maxSamples: 10000,
      historyRetentionMs: 300000,
      budgets: {
        perception: 50,
        planning: 30,
        control: 20,
        communication: 10,
        full_pipeline: 150,
      },
      thresholds: {
        warningPercent: 80,
        criticalPercent: 95,
      },
      heatmapBucketCount: 20,
      ...config,
    };

    // Initialize stage sample stores
    const stages = ['perception', 'planning', 'control', 'communication', 'full_pipeline'];
    for (const stage of stages) {
      this.#stageSamples.set(stage, []);
    }
  }

  /**
   * Whether the monitor is running.
   * @type {boolean}
   */
  get isRunning() {
    return this.#running;
  }

  /**
   * Total number of samples recorded.
   * @type {number}
   */
  get totalSamples() {
    return this.#totalSamplesRecorded;
  }

  /**
   * Current sample count in the buffer.
   * @type {number}
   */
  get bufferSize() {
    return this.#samples.length;
  }

  /**
   * Starts the latency monitor and cleanup interval.
   *
   * @returns {void}
   */
  start() {
    if (this.#running) return;
    this.#running = true;

    this.#cleanupTimer = setInterval(() => {
      this.#cleanupOldSamples();
    }, 10000);
  }

  /**
   * Stops the latency monitor.
   *
   * @returns {void}
   */
  stop() {
    this.#running = false;

    if (this.#cleanupTimer) {
      clearInterval(this.#cleanupTimer);
      this.#cleanupTimer = null;
    }
  }

  /**
   * Records a latency sample.
   *
   * @param {Object} sample - Sample data
   * @param {number} sample.value - Latency value in ms
   * @param {PipelineStage} [sample.stage='full_pipeline'] - Pipeline stage
   * @param {string} [sample.source] - Source identifier
   * @param {string} [sample.requestId] - Associated request ID
   * @returns {void}
   *
   * @fires LatencyMonitor#latency:sample
   */
  recordSample(sample) {
    if (!this.#running) return;
    if (typeof sample?.value !== 'number' || sample.value < 0) return;

    const enriched = {
      value: sample.value,
      stage: sample.stage || 'full_pipeline',
      timestamp: sample.timestamp || Date.now(),
      source: sample.source || 'unknown',
      requestId: sample.requestId || null,
    };

    this.#samples.push(enriched);
    this.#totalSamplesRecorded++;

    // Add to stage-specific store
    const stageStore = this.#stageSamples.get(enriched.stage);
    if (stageStore) {
      stageStore.push(enriched);
    }

    // Check budget
    this.#checkBudgetBreach(enriched.stage);

    /**
     * @event LatencyMonitor#latency:sample
     * @type {LatencySample}
     */
    this.emit('latency:sample', enriched);
  }

  /**
   * Starts a round-trip time measurement.
   *
   * @param {string} id - Unique measurement ID
   * @returns {void}
   */
  startRoundTrip(id) {
    this.#roundTrips.set(id, {
      id,
      startTime: Date.now(),
      completed: false,
    });
  }

  /**
   * Ends a round-trip time measurement and records the sample.
   *
   * @param {string} id - Measurement ID
   * @param {PipelineStage} [stage='communication'] - Pipeline stage
   * @returns {number|null} RTT in milliseconds, or null if not found
   */
  endRoundTrip(id, stage = 'communication') {
    const measurement = this.#roundTrips.get(id);
    if (!measurement || measurement.completed) return null;

    measurement.endTime = Date.now();
    measurement.duration = measurement.endTime - measurement.startTime;
    measurement.completed = true;

    this.recordSample({
      value: measurement.duration,
      stage,
      source: `rtt_${id}`,
    });

    this.#roundTrips.delete(id);
    return measurement.duration;
  }

  /**
   * Gets aggregated latency statistics across all samples.
   *
   * @returns {LatencyStats} Latency statistics
   */
  getStats() {
    return this.#computeStats(this.#samples);
  }

  /**
   * Gets latency statistics for a specific pipeline stage.
   *
   * @param {PipelineStage} stage - Pipeline stage
   * @returns {LatencyStats} Stage-specific latency statistics
   */
  getStageStats(stage) {
    const samples = this.#stageSamples.get(stage) || [];
    return this.#computeStats(samples);
  }

  /**
   * Gets all pipeline stage statistics.
   *
   * @returns {Object<string, LatencyStats>} Stats keyed by stage
   */
  getAllStageStats() {
    const result = {};
    for (const [stage, samples] of this.#stageSamples) {
      result[stage] = this.#computeStats(samples);
    }
    return result;
  }

  /**
   * Gets latency budget tracking for all pipeline stages.
   *
   * @returns {StageBudget[]} Budget tracking for each stage
   */
  getBudgetTracking() {
    const budgets = [];

    for (const [stage, budgetMs] of Object.entries(this.#config.budgets)) {
      const stats = this.getStageStats(stage);
      const used = stats.p95;
      const percentUsed = budgetMs > 0 ? (used / budgetMs) * 100 : 0;

      let status = 'ok';
      if (percentUsed >= this.#config.thresholds.criticalPercent) {
        status = 'critical';
      } else if (percentUsed >= this.#config.thresholds.warningPercent) {
        status = 'warning';
      }

      budgets.push({
        stage,
        budget: budgetMs,
        used,
        percentUsed: Math.round(percentUsed * 10) / 10,
        status,
      });
    }

    return budgets;
  }

  /**
   * Generates latency heatmap data.
   * Creates a distribution of latency values across configurable buckets.
   *
   * @param {Object} [options={}] - Heatmap options
   * @param {PipelineStage} [options.stage] - Filter by stage
   * @param {number} [options.buckets] - Override bucket count
   * @param {number} [options.minValue=0] - Heatmap range start
   * @param {number} [options.maxValue] - Heatmap range end (auto-detected)
   * @returns {HeatmapBucket[]} Heatmap data
   */
  getHeatmap(options = {}) {
    let samples = options.stage
      ? (this.#stageSamples.get(options.stage) || [])
      : this.#samples;

    if (samples.length === 0) return [];

    const values = samples.map((s) => s.value);
    const bucketCount = options.buckets || this.#config.heatmapBucketCount;
    const minValue = options.minValue ?? 0;
    const maxValue = options.maxValue ?? Math.max(...values);
    const bucketSize = (maxValue - minValue) / bucketCount;

    if (bucketSize <= 0) return [];

    const buckets = Array.from({ length: bucketCount }, (_, i) => ({
      rangeStart: minValue + i * bucketSize,
      rangeEnd: minValue + (i + 1) * bucketSize,
      count: 0,
      percentage: 0,
    }));

    for (const value of values) {
      const index = Math.min(
        Math.floor((value - minValue) / bucketSize),
        bucketCount - 1
      );
      if (index >= 0 && index < bucketCount) {
        buckets[index].count++;
      }
    }

    const total = values.length;
    for (const bucket of buckets) {
      bucket.percentage = total > 0
        ? Math.round((bucket.count / total) * 10000) / 100
        : 0;
    }

    return buckets;
  }

  /**
   * Gets recent latency trend (moving average over time windows).
   *
   * @param {Object} [options={}] - Trend options
   * @param {number} [options.windowMs=10000] - Time window for each data point
   * @param {number} [options.points=30] - Number of trend data points
   * @returns {Array<{timestamp: number, avg: number, p95: number, p99: number}>} Trend data
   */
  getTrend(options = {}) {
    const windowMs = options.windowMs ?? 10000;
    const points = options.points ?? 30;
    const now = Date.now();
    const trend = [];

    for (let i = points - 1; i >= 0; i--) {
      const windowEnd = now - i * windowMs;
      const windowStart = windowEnd - windowMs;
      const windowSamples = this.#samples.filter(
        (s) => s.timestamp >= windowStart && s.timestamp < windowEnd
      );

      if (windowSamples.length === 0) {
        trend.push({ timestamp: windowEnd, avg: 0, p95: 0, p99: 0 });
        continue;
      }

      const stats = this.#computeStats(windowSamples);
      trend.push({
        timestamp: windowEnd,
        avg: Math.round(stats.avg * 100) / 100,
        p95: Math.round(stats.p95 * 100) / 100,
        p99: Math.round(stats.p99 * 100) / 100,
      });
    }

    return trend;
  }

  // ─── Private Methods ───────────────────────────────────────────────

  /**
   * Computes latency statistics from a set of samples.
   * @private
   * @param {LatencySample[]} samples - Latency samples
   * @returns {LatencyStats} Computed statistics
   */
  #computeStats(samples) {
    if (samples.length === 0) {
      return { p50: 0, p95: 0, p99: 0, avg: 0, min: 0, max: 0, samples: 0, stddev: 0 };
    }

    const values = samples.map((s) => s.value).sort((a, b) => a - b);
    const n = values.length;
    const sum = values.reduce((a, b) => a + b, 0);
    const avg = sum / n;

    const variance = values.reduce((acc, v) => acc + Math.pow(v - avg, 2), 0) / n;
    const stddev = Math.sqrt(variance);

    return {
      p50: this.#percentile(values, 50),
      p95: this.#percentile(values, 95),
      p99: this.#percentile(values, 99),
      avg: Math.round(avg * 100) / 100,
      min: values[0],
      max: values[n - 1],
      samples: n,
      stddev: Math.round(stddev * 100) / 100,
    };
  }

  /**
   * Computes a percentile value from a sorted array.
   * @private
   * @param {number[]} sorted - Sorted array of values
   * @param {number} p - Percentile (0-100)
   * @returns {number} Percentile value
   */
  #percentile(sorted, p) {
    if (sorted.length === 0) return 0;
    if (sorted.length === 1) return sorted[0];

    const index = (p / 100) * (sorted.length - 1);
    const lower = Math.floor(index);
    const upper = Math.ceil(index);
    const weight = index - lower;

    if (lower === upper) return sorted[lower];
    return Math.round((sorted[lower] * (1 - weight) + sorted[upper] * weight) * 100) / 100;
  }

  /**
   * Checks if a stage's latency exceeds budget thresholds.
   * @private
   * @param {PipelineStage} stage - Pipeline stage
   */
  #checkBudgetBreach(stage) {
    const budgetMs = this.#config.budgets[stage];
    if (!budgetMs) return;

    const stats = this.getStageStats(stage);
    const percentUsed = budgetMs > 0 ? (stats.p99 / budgetMs) * 100 : 0;

    if (percentUsed >= this.#config.thresholds.criticalPercent) {
      /**
       * @event LatencyMonitor#latency:threshold_breach
       * @type {Object}
       * @property {PipelineStage} stage
       * @property {number} p99 - P99 latency
       * @property {number} budget - Budget in ms
       * @property {string} severity - 'warning' or 'critical'
       */
      this.emit('latency:threshold_breach', {
        stage,
        p99: stats.p99,
        p95: stats.p95,
        avg: stats.avg,
        budget: budgetMs,
        percentUsed: Math.round(percentUsed),
        severity: 'critical',
      });
    } else if (percentUsed >= this.#config.thresholds.warningPercent) {
      this.emit('latency:threshold_breach', {
        stage,
        p99: stats.p99,
        p95: stats.p95,
        avg: stats.avg,
        budget: budgetMs,
        percentUsed: Math.round(percentUsed),
        severity: 'warning',
      });
    }
  }

  /**
   * Removes samples older than the retention period.
   * @private
   */
  #cleanupOldSamples() {
    const cutoff = Date.now() - this.#config.historyRetentionMs;

    this.#samples = this.#samples.filter((s) => s.timestamp >= cutoff);

    for (const [stage, samples] of this.#stageSamples) {
      const filtered = samples.filter((s) => s.timestamp >= cutoff);
      this.#stageSamples.set(stage, filtered);
    }

    // Enforce max sample count
    if (this.#samples.length > this.#config.maxSamples) {
      this.#samples = this.#samples.slice(-this.#config.maxSamples);
    }
  }
}

export default LatencyMonitor;
