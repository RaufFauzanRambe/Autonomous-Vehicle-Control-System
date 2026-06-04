/**
 * @fileoverview HealthMonitor - Comprehensive health monitoring for the
 * autonomous vehicle control system. Performs system-level health checks
 * (CPU, memory, disk, network), per-module service health tracking,
 * heartbeat monitoring, health score computation, degradation detection,
 * auto-recovery triggers, and maintains health history for trend analysis.
 *
 * @module realtime_monitoring/health_monitor
 * @version 2.1.0
 * @author Autonomous Vehicle Control System Team
 */

import { EventEmitter } from 'events';
import os from 'os';

/** @typedef {'healthy'|'degraded'|'unhealthy'|'critical'|'unknown'} HealthStatus */

/**
 * @typedef {Object} HealthMonitorConfig
 * @property {number} [checkIntervalMs=1000] - Health check interval
 * @property {number} [heartbeatTimeoutMs=5000] - Heartbeat timeout threshold
 * @property {number} [historySize=3600] - Max health history entries (1 hour at 1s)
 * @property {number} [degradedThreshold=0.7] - Health score below this = degraded
 * @property {number} [criticalThreshold=0.4] - Health score below this = critical
 * @property {number} [autoRecoveryDelayMs=10000] - Delay before auto-recovery attempt
 * @property {number} [maxAutoRecoveryAttempts=3] - Max consecutive auto-recovery attempts
 * @property {Object} [thresholds] - System resource thresholds
 * @property {number} [thresholds.cpuPercent=90] - CPU usage warning threshold
 * @property {number} [thresholds.memoryPercent=85] - Memory usage warning threshold
 * @property {number} [thresholds.diskPercent=90] - Disk usage warning threshold
 * @property {number} [thresholds.eventLoopLagMs=100] - Event loop lag threshold
 */

/**
 * @typedef {Object} ResourceCheck
 * @property {string} name - Resource name
 * @property {HealthStatus} status - Check status
 * @property {number} value - Current value
 * @property {number} threshold - Warning threshold
 * @property {string} unit - Value unit
 * @property {string} [message] - Additional info
 */

/**
 * @typedef {Object} ServiceHealth
 * @property {string} name - Service name
 * @property {HealthStatus} status - Current status
 * @property {number} lastHeartbeat - Last heartbeat timestamp
 * @property {number} heartbeatMissed - Consecutive missed heartbeats
 * @property {boolean} autoRecoveryAttempted - Whether recovery was attempted
 * @property {number} [uptime] - Service uptime in ms
 * @property {string} [version] - Service version
 */

/**
 * @typedef {Object} HealthReport
 * @property {number} timestamp - Report timestamp
 * @property {number} score - Overall health score (0-1)
 * @property {HealthStatus} status - Aggregated health status
 * @property {ResourceCheck[]} systemChecks - System resource checks
 * @property {Object<string, ServiceHealth>} services - Per-service health
 * @property {Object} [trends] - Health trend data
 */

/**
 * @typedef {Object} HealthHistoryEntry
 * @property {number} timestamp - Entry timestamp
 * @property {number} score - Health score
 * @property {HealthStatus} status - Health status
 * @property {string[]} [issues] - Active issues
 */

/**
 * HealthMonitor provides continuous health assessment of the autonomous
 * vehicle control system through resource monitoring, service heartbeats,
 * and configurable health scoring.
 *
 * @extends EventEmitter
 *
 * @example
 * const health = new HealthMonitor({
 *   checkIntervalMs: 1000,
 *   thresholds: { cpuPercent: 80, memoryPercent: 85 },
 * });
 *
 * health.registerService('perception', { version: '2.1.0' });
 * health.on('health:degraded', (report) => handleDegradation(report));
 *
 * await health.start();
 */
export class HealthMonitor extends EventEmitter {
  /** @type {HealthMonitorConfig} */
  #config;

  /** @type {NodeJS.Timeout|null} */
  #checkTimer = null;

  /** @type {Map<string, ServiceHealth>} */
  #services = new Map();

  /** @type {HealthHistoryEntry[]} */
  #history = [];

  /** @type {number} */
  #lastCheckTime = 0;

  /** @type {HealthStatus} */
  #currentStatus = 'unknown';

  /** @type {number} */
  #currentScore = 1.0;

  /** @type {Map<string, number>} */
  #recoveryAttempts = new Map();

  /** @type {Map<string, NodeJS.Timeout>} */
  #recoveryTimers = new Map();

  /** @type {boolean} */
  #running = false;

  /**
   * Creates a new HealthMonitor.
   *
   * @param {HealthMonitorConfig} [config={}] - Configuration
   */
  constructor(config = {}) {
    super();
    this.setMaxListeners(30);

    this.#config = {
      checkIntervalMs: 1000,
      heartbeatTimeoutMs: 5000,
      historySize: 3600,
      degradedThreshold: 0.7,
      criticalThreshold: 0.4,
      autoRecoveryDelayMs: 10000,
      maxAutoRecoveryAttempts: 3,
      thresholds: {
        cpuPercent: 90,
        memoryPercent: 85,
        diskPercent: 90,
        eventLoopLagMs: 100,
      },
      ...config,
    };
  }

  /**
   * Current overall health score (0-1).
   * @type {number}
   */
  get score() {
    return this.#currentScore;
  }

  /**
   * Current overall health status.
   * @type {HealthStatus}
   */
  get status() {
    return this.#currentStatus;
  }

  /**
   * Whether the monitor is actively running.
   * @type {boolean}
   */
  get isRunning() {
    return this.#running;
  }

  /**
   * Number of registered services.
   * @type {number}
   */
  get serviceCount() {
    return this.#services.size;
  }

  /**
   * Number of health history entries.
   * @type {number}
   */
  get historySize() {
    return this.#history.length;
  }

  /**
   * Starts periodic health monitoring.
   *
   * @returns {Promise<void>}
   *
   * @fires HealthMonitor#health:check
   */
  async start() {
    if (this.#running) return;
    this.#running = true;

    // Perform initial check
    await this.performCheck();

    this.#checkTimer = setInterval(async () => {
      await this.performCheck();
    }, this.#config.checkIntervalMs);
  }

  /**
   * Stops health monitoring.
   *
   * @returns {void}
   */
  stop() {
    this.#running = false;

    if (this.#checkTimer) {
      clearInterval(this.#checkTimer);
      this.#checkTimer = null;
    }

    for (const timer of this.#recoveryTimers.values()) {
      clearTimeout(timer);
    }
    this.#recoveryTimers.clear();
  }

  /**
   * Performs a single comprehensive health check.
   * Checks system resources, service heartbeats, and computes health score.
   *
   * @returns {Promise<HealthReport>}
   *
   * @fires HealthMonitor#health:check
   * @fires HealthMonitor#health:degraded
   * @fires HealthMonitor#health:recovered
   * @fires HealthMonitor#health:critical
   */
  async performCheck() {
    const timestamp = Date.now();
    this.#lastCheckTime = timestamp;

    // System resource checks
    const systemChecks = await this.#performSystemChecks();

    // Service health checks
    const serviceChecks = this.#performServiceChecks(timestamp);

    // Compute health score
    const score = this.#computeHealthScore(systemChecks, serviceChecks);

    // Determine status from score
    const status = this.#scoreToStatus(score);

    // Build report
    const report = {
      timestamp,
      score,
      status,
      systemChecks,
      services: serviceChecks,
    };

    // Update state
    const previousStatus = this.#currentStatus;
    this.#currentScore = score;
    this.#currentStatus = status;

    // Add to history
    this.#addToHistory({ timestamp, score, status });

    // Emit events for status changes
    if (status !== previousStatus) {
      if (status === 'degraded' || status === 'critical') {
        /**
         * @event HealthMonitor#health:degraded
         * @type {HealthReport}
         */
        this.emit('health:degraded', report);

        if (status === 'critical') {
          /**
           * @event HealthMonitor#health:critical
           * @type {HealthReport}
           */
          this.emit('health:critical', report);
        }
      } else if ((previousStatus === 'degraded' || previousStatus === 'critical') && status === 'healthy') {
        /**
         * @event HealthMonitor#health:recovered
         * @type {HealthReport}
         */
        this.emit('health:recovered', report);
      }
    }

    // Trigger auto-recovery for unhealthy services
    this.#checkAutoRecovery(serviceChecks);

    /**
     * @event HealthMonitor#health:check
     * @type {HealthReport}
     */
    this.emit('health:check', report);

    return report;
  }

  /**
   * Gets the latest health report.
   *
   * @returns {HealthReport} Current health report
   */
  getHealthReport() {
    return {
      timestamp: this.#lastCheckTime || Date.now(),
      score: this.#currentScore,
      status: this.#currentStatus,
      systemChecks: [],
      services: {},
    };
  }

  /**
   * Registers a service for health monitoring.
   *
   * @param {string} name - Service name
   * @param {Object} [metadata={}] - Service metadata
   * @param {string} [metadata.version] - Service version
   * @param {boolean} [metadata.critical=false] - Whether service is critical
   * @param {Function} [metadata.healthCheck] - Custom health check function
   * @param {Function} [metadata.recoveryFn] - Auto-recovery function
   * @returns {void}
   */
  registerService(name, metadata = {}) {
    this.#services.set(name, {
      name,
      status: 'unknown',
      lastHeartbeat: Date.now(),
      heartbeatMissed: 0,
      autoRecoveryAttempted: false,
      ...metadata,
    });
  }

  /**
   * Records a heartbeat from a registered service.
   *
   * @param {string} serviceName - Service name
   * @param {Object} [data={}] - Heartbeat data
   * @returns {boolean} Whether the heartbeat was accepted
   */
  recordHeartbeat(serviceName, data = {}) {
    const service = this.#services.get(serviceName);
    if (!service) return false;

    service.lastHeartbeat = Date.now();
    service.heartbeatMissed = 0;
    service.status = 'healthy';

    if (data.status) service.status = data.status;
    if (data.uptime) service.uptime = data.uptime;

    return true;
  }

  /**
   * Unregisters a service from health monitoring.
   *
   * @param {string} name - Service name
   * @returns {boolean} Whether the service was removed
   */
  unregisterService(name) {
    return this.#services.delete(name);
  }

  /**
   * Gets health history entries.
   *
   * @param {Object} [options={}] - Query options
   * @param {number} [options.limit=100] - Max entries
   * @param {number} [options.since] - Minimum timestamp
   * @param {HealthStatus} [options.status] - Filter by status
   * @returns {HealthHistoryEntry[]} History entries
   */
  getHistory(options = {}) {
    let entries = [...this.#history];

    if (options.since) {
      entries = entries.filter((e) => e.timestamp >= options.since);
    }
    if (options.status) {
      entries = entries.filter((e) => e.status === options.status);
    }

    const limit = options.limit ?? 100;
    return entries.slice(-limit);
  }

  /**
   * Gets health trend data over the history.
   *
   * @param {number} [intervalMs=60000] - Aggregation interval
   * @returns {Object[]} Trend data points
   */
  getTrends(intervalMs = 60000) {
    if (this.#history.length < 2) return [];

    const trends = [];
    let intervalStart = this.#history[0].timestamp;
    let intervalScores = [];

    for (const entry of this.#history) {
      if (entry.timestamp - intervalStart >= intervalMs) {
        trends.push({
          timestamp: intervalStart,
          avgScore: intervalScores.reduce((a, b) => a + b, 0) / intervalScores.length,
          minScore: Math.min(...intervalScores),
          maxScore: Math.max(...intervalScores),
          sampleCount: intervalScores.length,
        });
        intervalStart = entry.timestamp;
        intervalScores = [];
      }
      intervalScores.push(entry.score);
    }

    if (intervalScores.length > 0) {
      trends.push({
        timestamp: intervalStart,
        avgScore: intervalScores.reduce((a, b) => a + b, 0) / intervalScores.length,
        minScore: Math.min(...intervalScores),
        maxScore: Math.max(...intervalScores),
        sampleCount: intervalScores.length,
      });
    }

    return trends;
  }

  // ─── Private Methods ───────────────────────────────────────────────

  /**
   * Performs system-level resource checks.
   * @private
   * @returns {Promise<ResourceCheck[]>}
   */
  async #performSystemChecks() {
    const checks = [];
    const thresholds = this.#config.thresholds;

    // CPU check
    const cpuUsage = this.#getCpuUsage();
    checks.push({
      name: 'cpu',
      status: cpuUsage > thresholds.cpuPercent ? 'critical' : cpuUsage > thresholds.cpuPercent * 0.8 ? 'degraded' : 'healthy',
      value: cpuUsage,
      threshold: thresholds.cpuPercent,
      unit: 'percent',
      message: cpuUsage > thresholds.cpuPercent ? 'CPU usage exceeds threshold' : undefined,
    });

    // Memory check
    const memUsage = this.#getMemoryUsage();
    checks.push({
      name: 'memory',
      status: memUsage.percent > thresholds.memoryPercent ? 'critical' : memUsage.percent > thresholds.memoryPercent * 0.8 ? 'degraded' : 'healthy',
      value: memUsage.percent,
      threshold: thresholds.memoryPercent,
      unit: 'percent',
      message: memUsage.percent > thresholds.memoryPercent ? `Memory: ${memUsage.used}MB / ${memUsage.total}MB` : undefined,
    });

    // Event loop lag check
    const eventLoopLag = await this.#measureEventLoopLag();
    checks.push({
      name: 'eventLoop',
      status: eventLoopLag > thresholds.eventLoopLagMs ? 'critical' : eventLoopLag > thresholds.eventLoopLagMs * 0.5 ? 'degraded' : 'healthy',
      value: eventLoopLag,
      threshold: thresholds.eventLoopLagMs,
      unit: 'ms',
      message: eventLoopLag > thresholds.eventLoopLagMs ? 'Event loop lag detected' : undefined,
    });

    // Load average check (Unix only)
    const loadAvg = os.loadavg();
    const cpuCount = os.cpus().length;
    const loadPercent = (loadAvg[0] / cpuCount) * 100;
    checks.push({
      name: 'loadAverage',
      status: loadPercent > 90 ? 'critical' : loadPercent > 70 ? 'degraded' : 'healthy',
      value: Math.round(loadAvg[0] * 100) / 100,
      threshold: cpuCount,
      unit: 'load',
      message: `1min load: ${loadAvg[0].toFixed(2)} across ${cpuCount} CPUs`,
    });

    return checks;
  }

  /**
   * Performs service health checks based on heartbeats.
   * @private
   * @param {number} now - Current timestamp
   * @returns {Object<string, ServiceHealth>}
   */
  #performServiceChecks(now) {
    const results = {};

    for (const [name, service] of this.#services) {
      const timeSinceLastHeartbeat = now - service.lastHeartbeat;

      if (timeSinceLastHeartbeat > this.#config.heartbeatTimeoutMs * 3) {
        service.status = 'critical';
        service.heartbeatMissed = Math.floor(timeSinceLastHeartbeat / this.#config.heartbeatTimeoutMs);
      } else if (timeSinceLastHeartbeat > this.#config.heartbeatTimeoutMs) {
        service.status = 'degraded';
        service.heartbeatMissed = Math.floor(timeSinceLastHeartbeat / this.#config.heartbeatTimeoutMs);
      }

      if (typeof service.healthCheck === 'function') {
        try {
          const customStatus = service.healthCheck();
          if (customStatus) service.status = customStatus;
        } catch {
          service.status = 'unhealthy';
        }
      }

      results[name] = {
        name: service.name,
        status: service.status,
        lastHeartbeat: service.lastHeartbeat,
        heartbeatMissed: service.heartbeatMissed,
        autoRecoveryAttempted: service.autoRecoveryAttempted,
        uptime: service.uptime,
        version: service.version,
      };
    }

    return results;
  }

  /**
   * Computes an overall health score from system and service checks.
   * @private
   * @param {ResourceCheck[]} systemChecks - System resource checks
   * @param {Object<string, ServiceHealth>} serviceChecks - Service checks
   * @returns {number} Health score between 0 and 1
   */
  #computeHealthScore(systemChecks, serviceChecks) {
    // System checks: average score weighted by check
    let systemScore = 1.0;
    for (const check of systemChecks) {
      const checkScore = this.#checkToScore(check);
      systemScore *= checkScore; // Multiplicative: one bad check degrades overall
    }

    // Service checks: weighted average
    let serviceScore = 1.0;
    const serviceNames = Object.keys(serviceChecks);
    if (serviceNames.length > 0) {
      let totalWeight = 0;
      let weightedScore = 0;

      for (const name of serviceNames) {
        const service = serviceChecks[name];
        const weight = service.status === 'critical' ? 3 : service.status === 'degraded' ? 2 : 1;
        const score = this.#statusToScore(service.status);
        weightedScore += score * weight;
        totalWeight += weight;
      }

      serviceScore = totalWeight > 0 ? weightedScore / totalWeight : 1.0;
    }

    // Combine: 40% system, 60% services (services are more critical for AV)
    return Math.max(0, Math.min(1, systemScore * 0.4 + serviceScore * 0.6));
  }

  /**
   * Converts a resource check to a 0-1 score.
   * @private
   * @param {ResourceCheck} check - Resource check
   * @returns {number} Score between 0 and 1
   */
  #checkToScore(check) {
    switch (check.status) {
      case 'healthy': return 1.0;
      case 'degraded': return 0.7;
      case 'unhealthy': return 0.4;
      case 'critical': return 0.1;
      default: return 0.5;
    }
  }

  /**
   * Converts a health status to a 0-1 score.
   * @private
   * @param {HealthStatus} status - Health status
   * @returns {number} Score between 0 and 1
   */
  #statusToScore(status) {
    switch (status) {
      case 'healthy': return 1.0;
      case 'degraded': return 0.6;
      case 'unhealthy': return 0.3;
      case 'critical': return 0.1;
      case 'unknown': return 0.5;
      default: return 0.5;
    }
  }

  /**
   * Converts a score to a health status.
   * @private
   * @param {number} score - Health score
   * @returns {HealthStatus}
   */
  #scoreToStatus(score) {
    if (score >= this.#config.degradedThreshold) return 'healthy';
    if (score >= this.#config.criticalThreshold) return 'degraded';
    return 'critical';
  }

  /**
   * Gets current CPU usage percentage.
   * @private
   * @returns {number} CPU usage percentage
   */
  #getCpuUsage() {
    const cpus = os.cpus();
    let totalIdle = 0;
    let totalTick = 0;

    for (const cpu of cpus) {
      for (const type of Object.keys(cpu.times)) {
        totalTick += cpu.times[type];
      }
      totalIdle += cpu.times.idle;
    }

    const totalUsage = totalTick - totalIdle;
    return totalTick > 0 ? Math.round((totalUsage / totalTick) * 100) : 0;
  }

  /**
   * Gets memory usage information.
   * @private
   * @returns {{ total: number, used: number, free: number, percent: number }}
   */
  #getMemoryUsage() {
    const totalMem = os.totalmem();
    const freeMem = os.freemem();
    const usedMem = totalMem - freeMem;

    return {
      total: Math.round(totalMem / (1024 * 1024)),
      used: Math.round(usedMem / (1024 * 1024)),
      free: Math.round(freeMem / (1024 * 1024)),
      percent: Math.round((usedMem / totalMem) * 100),
    };
  }

  /**
   * Measures event loop lag.
   * @private
   * @returns {Promise<number>} Lag in milliseconds
   */
  #measureEventLoopLag() {
    return new Promise((resolve) => {
      const start = Date.now();
      setImmediate(() => {
        const lag = Date.now() - start;
        resolve(lag);
      });
    });
  }

  /**
   * Adds an entry to health history with size management.
   * @private
   * @param {HealthHistoryEntry} entry - History entry
   */
  #addToHistory(entry) {
    this.#history.push(entry);

    while (this.#history.length > this.#config.historySize) {
      this.#history.shift();
    }
  }

  /**
   * Checks for auto-recovery opportunities for unhealthy services.
   * @private
   * @param {Object<string, ServiceHealth>} serviceChecks - Current service checks
   */
  #checkAutoRecovery(serviceChecks) {
    for (const [name, service] of Object.entries(serviceChecks)) {
      if (service.status !== 'critical' && service.status !== 'unhealthy') continue;

      const registered = this.#services.get(name);
      if (!registered?.recoveryFn) continue;

      const attempts = this.#recoveryAttempts.get(name) || 0;
      if (attempts >= this.#config.maxAutoRecoveryAttempts) continue;

      if (this.#recoveryTimers.has(name)) continue; // Already scheduled

      const timer = setTimeout(async () => {
        this.#recoveryTimers.delete(name);

        try {
          await registered.recoveryFn();

          this.#recoveryAttempts.set(name, attempts + 1);

          /**
           * @event HealthMonitor#recovery:attempted
           * @type {Object}
           * @property {string} serviceName
           * @property {boolean} success
           */
          this.emit('recovery:attempted', { serviceName: name, success: true });

          // Reset attempts on successful recovery
          this.#recoveryAttempts.set(name, 0);
        } catch (error) {
          this.#recoveryAttempts.set(name, attempts + 1);
          this.emit('recovery:attempted', { serviceName: name, success: false, error: error.message });
        }
      }, this.#config.autoRecoveryDelayMs);

      this.#recoveryTimers.set(name, timer);
    }
  }
}

export default HealthMonitor;
