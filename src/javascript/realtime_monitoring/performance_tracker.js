/**
 * @fileoverview PerformanceTracker - Performance monitoring and tracking for
 * the autonomous vehicle control system. Provides FPS tracking, memory usage
 * trends, CPU utilization, event loop lag detection, GC pressure monitoring,
 * throughput measurement, performance regression detection, and report generation.
 *
 * @module realtime_monitoring/performance_tracker
 * @version 2.1.0
 * @author Autonomous Vehicle Control System Team
 */

import { EventEmitter } from 'events';
import os from 'os';

/** @typedef {'fps'|'memory'|'cpu'|'eventLoop'|'gc'|'throughput'} MetricType */

/**
 * @typedef {Object} PerformanceTrackerConfig
 * @property {number} [samplingIntervalMs=100] - Metric sampling interval
 * @property {number} [historySize=600] - Max data points per metric (1 min at 100ms)
 * @property {number} [fpsTarget=30] - Target frames per second
 * @property {number} [fpsWarningThreshold=20] - FPS warning threshold
 * @property {number} [fpsCriticalThreshold=10] - FPS critical threshold
 * @property {number} [memoryWarningPercent=80] - Memory usage warning %
 * @property {number} [cpuWarningPercent=85] - CPU usage warning %
 * @property {number} [eventLoopLagWarningMs=50] - Event loop lag warning ms
 * @property {number} [regressionWindowSize=30] - Window for regression detection
 * @property {number} [regressionThreshold=0.15] - 15% degradation triggers regression
 */

/**
 * @typedef {Object} PerformanceReport
 * @property {number} timestamp - Report timestamp
 * @property {number} fps - Current FPS
 * @property {Object} memoryUsage - Memory usage details
 * @property {number} memoryUsage.heapUsed - Heap used in MB
 * @property {number} memoryUsage.heapTotal - Heap total in MB
 * @property {number} memoryUsage.rss - Resident set size in MB
 * @property {number} memoryUsage.external - External memory in MB
 * @property {number} memoryUsage.usagePercent - Usage percentage
 * @property {number} cpuUsage - CPU usage percentage
 * @property {number} eventLoopLag - Event loop lag in ms
 * @property {number} gcPressure - GC pressure indicator (0-1)
 * @property {number} throughput - Operations per second
 * @property {Object} regressions - Detected regressions
 */

/**
 * @typedef {Object} MetricDataPoint
 * @property {number} timestamp - Data point timestamp
 * @property {number} value - Metric value
 */

/**
 * @typedef {Object} RegressionInfo
 * @property {string} metric - Metric name
 * @property {number} baseline - Baseline value
 * @property {number} current - Current value
 * @property {number} degradation - Degradation percentage
 * @property {number} detectedAt - Detection timestamp
 */

/**
 * PerformanceTracker provides continuous performance monitoring with
 * regression detection and comprehensive reporting.
 *
 * @extends EventEmitter
 *
 * @example
 * const tracker = new PerformanceTracker({
 *   fpsTarget: 30,
 *   regressionThreshold: 0.15,
 * });
 *
 * tracker.on('performance:regression', (info) => {
 *   console.warn(`Regression in ${info.metric}: ${info.degradation}% degradation`);
 * });
 *
 * tracker.start();
 */
export class PerformanceTracker extends EventEmitter {
  /** @type {PerformanceTrackerConfig} */
  #config;

  /** @type {Map<MetricType, MetricDataPoint[]>} */
  #metrics = new Map();

  /** @type {Map<MetricType, number>} */
  #baselines = new Map();

  /** @type {NodeJS.Timeout|null} */
  #samplingTimer = null;

  /** @type {boolean} */
  #running = false;

  /** @type {number} */
  #frameCount = 0;

  /** @type {number} */
  #lastFrameTime = 0;

  /** @type {number} */
  #currentFps = 0;

  /** @type {number} */
  #fpsAccumulator = 0;

  /** @type {number} */
  #fpsLastCalcTime = 0;

  /** @type {number} */
  #operationCount = 0;

  /** @type {number} */
  #throughputLastCalcTime = 0;

  /** @type {number} */
  #currentThroughput = 0;

  /** @type {number[]} */
  #gcDurations = [];

  /** @type {number} */
  #lastGcTime = 0;

  /** @type {number} */
  #lastCpuUsage = null;

  /** @type {number} */
  #lastCpuSampleTime = 0;

  /** @type {RegressionInfo[]} */
  #detectedRegressions = [];

  /**
   * Creates a new PerformanceTracker.
   *
   * @param {PerformanceTrackerConfig} [config={}] - Configuration
   */
  constructor(config = {}) {
    super();
    this.setMaxListeners(20);

    this.#config = {
      samplingIntervalMs: 100,
      historySize: 600,
      fpsTarget: 30,
      fpsWarningThreshold: 20,
      fpsCriticalThreshold: 10,
      memoryWarningPercent: 80,
      cpuWarningPercent: 85,
      eventLoopLagWarningMs: 50,
      regressionWindowSize: 30,
      regressionThreshold: 0.15,
      ...config,
    };

    // Initialize metric stores
    const metricTypes = ['fps', 'memory', 'cpu', 'eventLoop', 'gc', 'throughput'];
    for (const type of metricTypes) {
      this.#metrics.set(type, []);
    }
  }

  /**
   * Whether the tracker is running.
   * @type {boolean}
   */
  get isRunning() {
    return this.#running;
  }

  /**
   * Current FPS value.
   * @type {number}
   */
  get currentFps() {
    return this.#currentFps;
  }

  /**
   * Current throughput (ops/sec).
   * @type {number}
   */
  get currentThroughput() {
    return this.#currentThroughput;
  }

  /**
   * Number of detected regressions.
   * @type {number}
   */
  get regressionCount() {
    return this.#detectedRegressions.length;
  }

  /**
   * Starts performance tracking.
   *
   * @returns {void}
   */
  start() {
    if (this.#running) return;
    this.#running = true;

    this.#fpsLastCalcTime = Date.now();
    this.#throughputLastCalcTime = Date.now();
    this.#lastCpuSampleTime = Date.now();

    // Set baselines from initial readings
    this.#initializeBaselines();

    this.#samplingTimer = setInterval(() => {
      this.#sampleMetrics();
    }, this.#config.samplingIntervalMs);

    // Monitor GC if available
    this.#setupGCMonitoring();
  }

  /**
   * Stops performance tracking.
   *
   * @returns {void}
   */
  stop() {
    this.#running = false;

    if (this.#samplingTimer) {
      clearInterval(this.#samplingTimer);
      this.#samplingTimer = null;
    }
  }

  /**
   * Records a sensor frame tick for FPS calculation.
   *
   * @returns {void}
   */
  recordSensorFrame() {
    this.#frameCount++;
    this.#operationCount++;

    const now = Date.now();
    const elapsed = now - this.#fpsLastCalcTime;

    if (elapsed >= 1000) {
      this.#currentFps = Math.round((this.#frameCount / elapsed) * 1000);
      this.#frameCount = 0;
      this.#fpsLastCalcTime = now;

      this.#addMetric('fps', this.#currentFps);
      this.#checkFpsThreshold();
    }
  }

  /**
   * Records a tick for throughput measurement.
   *
   * @param {number} [count=1] - Number of operations in this tick
   * @returns {void}
   */
  recordTick(count = 1) {
    this.#operationCount += count;

    const now = Date.now();
    const elapsed = now - this.#throughputLastCalcTime;

    if (elapsed >= 1000) {
      this.#currentThroughput = Math.round((this.#operationCount / elapsed) * 1000);
      this.#operationCount = 0;
      this.#throughputLastCalcTime = now;

      this.#addMetric('throughput', this.#currentThroughput);
    }
  }

  /**
   * Generates a comprehensive performance report.
   *
   * @returns {PerformanceReport} Current performance report
   */
  getReport() {
    const memData = process.memoryUsage();
    const timestamp = Date.now();

    return {
      timestamp,
      fps: this.#currentFps,
      memoryUsage: {
        heapUsed: Math.round(memData.heapUsed / (1024 * 1024) * 100) / 100,
        heapTotal: Math.round(memData.heapTotal / (1024 * 1024) * 100) / 100,
        rss: Math.round(memData.rss / (1024 * 1024) * 100) / 100,
        external: Math.round(memData.external / (1024 * 1024) * 100) / 100,
        usagePercent: Math.round((memData.heapUsed / memData.heapTotal) * 10000) / 100,
      },
      cpuUsage: this.#getCpuUsage(),
      eventLoopLag: this.#getLastMetricValue('eventLoop'),
      gcPressure: this.#computeGCPressure(),
      throughput: this.#currentThroughput,
      regressions: this.#getActiveRegressions(),
    };
  }

  /**
   * Gets metric data points for a specific metric type.
   *
   * @param {MetricType} type - Metric type
   * @param {number} [limit=100] - Max data points
   * @returns {MetricDataPoint[]} Metric data
   */
  getMetricData(type, limit = 100) {
    const data = this.#metrics.get(type) || [];
    return data.slice(-limit);
  }

  /**
   * Gets all metric types and their latest values.
   *
   * @returns {Object<string, number>} Latest metric values
   */
  getCurrentMetrics() {
    const result = {};
    for (const [type] of this.#metrics) {
      result[type] = this.#getLastMetricValue(type);
    }
    return result;
  }

  /**
   * Gets detected performance regressions.
   *
   * @returns {RegressionInfo[]} Active regressions
   */
  getRegressions() {
    return [...this.#detectedRegressions];
  }

  /**
   * Gets metric trend data (moving average).
   *
   * @param {MetricType} type - Metric type
   * @param {number} [windowSize=10] - Moving average window
   * @returns {MetricDataPoint[]} Trend data
   */
  getTrend(type, windowSize = 10) {
    const data = this.#metrics.get(type) || [];
    if (data.length < windowSize) return data;

    const trend = [];
    for (let i = windowSize - 1; i < data.length; i++) {
      const window = data.slice(i - windowSize + 1, i + 1);
      const avg = window.reduce((sum, dp) => sum + dp.value, 0) / window.length;
      trend.push({
        timestamp: data[i].timestamp,
        value: Math.round(avg * 100) / 100,
      });
    }

    return trend;
  }

  /**
   * Clears all metric data and resets baselines.
   *
   * @returns {void}
   */
  reset() {
    for (const [type] of this.#metrics) {
      this.#metrics.set(type, []);
    }
    this.#baselines.clear();
    this.#detectedRegressions = [];
    this.#frameCount = 0;
    this.#operationCount = 0;
    this.#currentFps = 0;
    this.#currentThroughput = 0;
    this.#gcDurations = [];

    this.#initializeBaselines();
  }

  /**
   * Sets a baseline for a metric (for regression detection).
   *
   * @param {MetricType} type - Metric type
   * @param {number} value - Baseline value
   * @returns {void}
   */
  setBaseline(type, value) {
    this.#baselines.set(type, value);
  }

  // ─── Private Methods ───────────────────────────────────────────────

  /**
   * Samples all performance metrics.
   * @private
   */
  #sampleMetrics() {
    // Memory
    const memData = process.memoryUsage();
    const memPercent = (memData.heapUsed / memData.heapTotal) * 100;
    this.#addMetric('memory', memPercent);

    // CPU
    const cpuPercent = this.#getCpuUsage();
    this.#addMetric('cpu', cpuPercent);

    // Event loop lag
    const lagStart = Date.now();
    setImmediate(() => {
      const lag = Date.now() - lagStart;
      this.#addMetric('eventLoop', lag);
      this.#checkEventLoopLag(lag);
    });

    // Check memory threshold
    if (memPercent > this.#config.memoryWarningPercent) {
      this.emit('performance:warning', {
        metric: 'memory',
        value: memPercent,
        threshold: this.#config.memoryWarningPercent,
      });
    }

    // Check CPU threshold
    if (cpuPercent > this.#config.cpuWarningPercent) {
      this.emit('performance:warning', {
        metric: 'cpu',
        value: cpuPercent,
        threshold: this.#config.cpuWarningPercent,
      });
    }

    // Check regressions
    this.#checkRegressions();
  }

  /**
   * Adds a metric data point.
   * @private
   * @param {MetricType} type - Metric type
   * @param {number} value - Metric value
   */
  #addMetric(type, value) {
    const data = this.#metrics.get(type);
    if (!data) return;

    data.push({ timestamp: Date.now(), value });

    while (data.length > this.#config.historySize) {
      data.shift();
    }
  }

  /**
   * Gets the last recorded value for a metric.
   * @private
   * @param {MetricType} type - Metric type
   * @returns {number} Last value or 0
   */
  #getLastMetricValue(type) {
    const data = this.#metrics.get(type);
    if (!data || data.length === 0) return 0;
    return data[data.length - 1].value;
  }

  /**
   * Initializes baseline values from current system state.
   * @private
   */
  #initializeBaselines() {
    const memData = process.memoryUsage();
    this.#baselines.set('fps', this.#config.fpsTarget);
    this.#baselines.set('memory', (memData.heapUsed / memData.heapTotal) * 100);
    this.#baselines.set('cpu', 30); // Assume 30% baseline CPU
    this.#baselines.set('eventLoop', 5); // 5ms baseline
    this.#baselines.set('gc', 0.1); // Low GC pressure baseline
    this.#baselines.set('throughput', 100); // 100 ops/sec baseline
  }

  /**
   * Gets current CPU usage percentage.
   * @private
   * @returns {number} CPU usage percentage
   */
  #getCpuUsage() {
    const now = Date.now();
    const cpus = os.cpus();
    let totalIdle = 0;
    let totalTick = 0;

    for (const cpu of cpus) {
      for (const type of Object.keys(cpu.times)) {
        totalTick += cpu.times[type];
      }
      totalIdle += cpu.times.idle;
    }

    const currentUsage = totalTick - totalIdle;
    const percent = totalTick > 0 ? Math.round((currentUsage / totalTick) * 100) : 0;

    return percent;
  }

  /**
   * Computes GC pressure based on recent GC durations.
   * @private
   * @returns {number} GC pressure (0-1)
   */
  #computeGCPressure() {
    if (this.#gcDurations.length === 0) return 0;

    const recentGc = this.#gcDurations.filter((d) => Date.now() - d.timestamp < 10000);
    if (recentGc.length === 0) return 0;

    const totalGcTime = recentGc.reduce((sum, d) => sum + d.duration, 0);
    const windowMs = 10000;
    return Math.min(1, totalGcTime / windowMs);
  }

  /**
   * Sets up GC monitoring if the v8 module is available.
   * @private
   */
  #setupGCMonitoring() {
    try {
      if (typeof globalThis.gc !== 'undefined') {
        // GC hooks available - monitor via periodic memory snapshots
        const gcMonitorInterval = setInterval(() => {
          if (!this.#running) {
            clearInterval(gcMonitorInterval);
            return;
          }

          const memBefore = process.memoryUsage().heapUsed;
          setImmediate(() => {
            const memAfter = process.memoryUsage().heapUsed;
            const freed = memBefore - memAfter;

            if (freed > 0) {
              const gcDuration = Math.min(100, freed / 1000000); // Estimate
              this.#gcDurations.push({
                timestamp: Date.now(),
                duration: gcDuration,
              });

              if (this.#gcDurations.length > 100) {
                this.#gcDurations.shift();
              }

              this.#addMetric('gc', this.#computeGCPressure());
            }
          });
        }, 1000);
      }
    } catch {
      // GC monitoring not available
    }
  }

  /**
   * Checks FPS against thresholds.
   * @private
   */
  #checkFpsThreshold() {
    if (this.#currentFps < this.#config.fpsCriticalThreshold) {
      this.emit('performance:critical', {
        metric: 'fps',
        value: this.#currentFps,
        threshold: this.#config.fpsCriticalThreshold,
        message: `FPS critically low: ${this.#currentFps}`,
      });
    } else if (this.#currentFps < this.#config.fpsWarningThreshold) {
      this.emit('performance:warning', {
        metric: 'fps',
        value: this.#currentFps,
        threshold: this.#config.fpsWarningThreshold,
        message: `FPS below target: ${this.#currentFps}`,
      });
    }
  }

  /**
   * Checks event loop lag against threshold.
   * @private
   * @param {number} lag - Current event loop lag in ms
   */
  #checkEventLoopLag(lag) {
    if (lag > this.#config.eventLoopLagWarningMs) {
      this.emit('performance:warning', {
        metric: 'eventLoop',
        value: lag,
        threshold: this.#config.eventLoopLagWarningMs,
        message: `Event loop lag: ${lag}ms`,
      });
    }
  }

  /**
   * Checks for performance regressions by comparing current values to baselines.
   * @private
   */
  #checkRegressions() {
    const windowSize = this.#config.regressionWindowSize;

    for (const [type, data] of this.#metrics) {
      if (data.length < windowSize * 2) continue;

      const baseline = this.#baselines.get(type);
      if (baseline === undefined || baseline === 0) continue;

      const recentWindow = data.slice(-windowSize);
      const recentAvg = recentWindow.reduce((sum, dp) => sum + dp.value, 0) / recentWindow.length;

      // For metrics where lower is better (fps, throughput)
      const lowerIsBetter = type === 'fps' || type === 'throughput';

      let degradation;
      if (lowerIsBetter) {
        degradation = (baseline - recentAvg) / baseline;
      } else {
        degradation = (recentAvg - baseline) / baseline;
      }

      if (degradation > this.#config.regressionThreshold) {
        // Check if already detected
        const existing = this.#detectedRegressions.find((r) => r.metric === type);
        if (existing) {
          existing.current = recentAvg;
          existing.degradation = Math.round(degradation * 10000) / 100;
          existing.detectedAt = Date.now();
        } else {
          const regression = {
            metric: type,
            baseline,
            current: Math.round(recentAvg * 100) / 100,
            degradation: Math.round(degradation * 10000) / 100,
            detectedAt: Date.now(),
          };

          this.#detectedRegressions.push(regression);

          /**
           * @event PerformanceTracker#performance:regression
           * @type {RegressionInfo}
           */
          this.emit('performance:regression', regression);
        }
      } else {
        // Remove resolved regression
        this.#detectedRegressions = this.#detectedRegressions.filter(
          (r) => r.metric !== type
        );
      }
    }
  }

  /**
   * Gets currently active (unresolved) regressions.
   * @private
   * @returns {RegressionInfo[]}
   */
  #getActiveRegressions() {
    const result = {};
    for (const r of this.#detectedRegressions) {
      result[r.metric] = {
        baseline: r.baseline,
        current: r.current,
        degradation: r.degradation,
        detectedAt: r.detectedAt,
      };
    }
    return result;
  }
}

export default PerformanceTracker;
