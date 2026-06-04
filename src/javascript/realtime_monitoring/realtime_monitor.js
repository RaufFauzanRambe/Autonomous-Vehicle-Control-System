/**
 * @fileoverview RealtimeMonitor - Main orchestrator for the autonomous vehicle
 * realtime monitoring subsystem. Coordinates all sub-modules including health
 * monitoring, latency tracking, alert management, event logging, performance
 * tracking, and system status aggregation. Manages the data pipeline from
 * ingestion through processing to dashboard push.
 *
 * @module realtime_monitoring/realtime_monitor
 * @version 2.1.0
 * @author Autonomous Vehicle Control System Team
 */

import { EventEmitter } from 'events';
import { HealthMonitor } from './health_monitor.js';
import { LatencyMonitor } from './latency_monitor.js';
import { EventLogger } from './event_logger.js';
import { AlertManager } from './alert_manager.js';
import { SystemStatus } from './system_status.js';
import { PerformanceTracker } from './performance_tracker.js';
import { WebSocketServer } from './websocket_server.js';
import { WebSocketClient } from './websocket_client.js';

/** @typedef {'stopped'|'initializing'|'running'|'degraded'|'error'|'shutting_down'} MonitorState */

/** @typedef {'perception'|'planning'|'control'|'communication'|'all'} PipelineStage */

/**
 * @typedef {Object} MonitorConfig
 * @property {number} [tickIntervalMs=100] - Main loop tick interval in milliseconds
 * @property {number} [dashboardPushIntervalMs=200] - Dashboard data push interval
 * @property {number} [healthCheckIntervalMs=1000] - Health check interval
 * @property {number} [latencySampleIntervalMs=50] - Latency sampling interval
 * @property {boolean} [enableWebSocketServer=true] - Whether to start the WS server
 * @property {number} [wsServerPort=8090] - WebSocket server port
 * @property {boolean} [enableCloudUpload=false] - Upload telemetry to cloud
 * @property {string} [cloudEndpoint=''] - Cloud endpoint for telemetry upload
 * @property {number} [maxDataBufferSize=10000] - Maximum data points in buffer
 * @property {number} [alertThrottleMs=5000] - Minimum time between same alerts
 * @property {Object} [moduleConfig={}] - Per-module configuration overrides
 */

/**
 * @typedef {Object} DataPoint
 * @property {string} id - Unique data point identifier
 * @property {string} source - Source module name
 * @property {string} type - Data type (e.g., 'sensor', 'health', 'latency')
 * @property {number} timestamp - Unix timestamp in milliseconds
 * @property {Object} payload - Data payload
 * @property {PipelineStage} [pipelineStage] - Associated pipeline stage
 */

/**
 * @typedef {Object} DashboardSnapshot
 * @property {number} timestamp - Snapshot timestamp
 * @property {MonitorState} state - Current monitor state
 * @property {import('./system_status.js').SystemStatusReport} systemStatus - Overall system status
 * @property {import('./health_monitor.js').HealthReport} healthReport - Health report
 * @property {import('./latency_monitor.js').LatencyStats} latencyStats - Latency statistics
 * @property {import('./performance_tracker.js').PerformanceReport} performanceReport - Performance metrics
 * @property {Array<import('./alert_manager.js').Alert>} activeAlerts - Active alerts
 * @property {Object} customMetrics - Custom metrics from sub-modules
 */

/**
 * RealtimeMonitor orchestrates all monitoring sub-modules for the autonomous
 * vehicle control system. It manages lifecycle, data flow, and provides a
 * unified dashboard data stream.
 *
 * @extends EventEmitter
 *
 * @example
 * const monitor = new RealtimeMonitor({
 *   tickIntervalMs: 100,
 *   wsServerPort: 8090,
 * });
 *
 * monitor.on('dashboard:update', (snapshot) => {
 *   console.log('Dashboard update:', snapshot.systemStatus.mode);
 * });
 *
 * monitor.on('alert:critical', (alert) => {
 *   emergencyHandler(alert);
 * });
 *
 * await monitor.start();
 */
export class RealtimeMonitor extends EventEmitter {
  /** @type {MonitorConfig} */
  #config;

  /** @type {MonitorState} */
  #state = 'stopped';

  /** @type {HealthMonitor|null} */
  #healthMonitor = null;

  /** @type {LatencyMonitor|null} */
  #latencyMonitor = null;

  /** @type {EventLogger|null} */
  #eventLogger = null;

  /** @type {AlertManager|null} */
  #alertManager = null;

  /** @type {SystemStatus|null} */
  #systemStatus = null;

  /** @type {PerformanceTracker|null} */
  #performanceTracker = null;

  /** @type {WebSocketServer|null} */
  #wsServer = null;

  /** @type {WebSocketClient|null} */
  #wsClient = null;

  /** @type {DataPoint[]} */
  #dataBuffer = [];

  /** @type {NodeJS.Timeout|null} */
  #tickTimer = null;

  /** @type {NodeJS.Timeout|null} */
  #dashboardPushTimer = null;

  /** @type {number} */
  #tickCount = 0;

  /** @type {number} */
  #startTime = 0;

  /** @type {Map<string, Function>} */
  #dataProcessors = new Map();

  /** @type {Map<string, Object>} */
  #moduleRegistry = new Map();

  /**
   * Creates a new RealtimeMonitor instance.
   *
   * @param {MonitorConfig} [config={}] - Monitor configuration
   */
  constructor(config = {}) {
    super();
    this.setMaxListeners(50);

    this.#config = {
      tickIntervalMs: 100,
      dashboardPushIntervalMs: 200,
      healthCheckIntervalMs: 1000,
      latencySampleIntervalMs: 50,
      enableWebSocketServer: true,
      wsServerPort: 8090,
      enableCloudUpload: false,
      cloudEndpoint: '',
      maxDataBufferSize: 10000,
      alertThrottleMs: 5000,
      moduleConfig: {},
      ...config,
    };

    this.#initializeDataProcessors();
  }

  /**
   * Current monitor state.
   * @type {MonitorState}
   */
  get state() {
    return this.#state;
  }

  /**
   * Whether the monitor is currently running.
   * @type {boolean}
   */
  get isRunning() {
    return this.#state === 'running' || this.#state === 'degraded';
  }

  /**
   * Uptime in milliseconds since monitor was started.
   * @type {number}
   */
  get uptime() {
    return this.#startTime > 0 ? Date.now() - this.#startTime : 0;
  }

  /**
   * Current data buffer size.
   * @type {number}
   */
  get bufferSize() {
    return this.#dataBuffer.length;
  }

  /**
   * Reference to the alert manager sub-module.
   * @type {AlertManager|null}
   */
  get alertManager() {
    return this.#alertManager;
  }

  /**
   * Reference to the health monitor sub-module.
   * @type {HealthMonitor|null}
   */
  get healthMonitor() {
    return this.#healthMonitor;
  }

  /**
   * Reference to the event logger sub-module.
   * @type {EventLogger|null}
   */
  get eventLogger() {
    return this.#eventLogger;
  }

  /**
   * Reference to the performance tracker sub-module.
   * @type {PerformanceTracker|null}
   */
  get performanceTracker() {
    return this.#performanceTracker;
  }

  /**
   * Starts the realtime monitor and all sub-modules.
   * Initializes the data pipeline, starts the tick loop, and begins
   * pushing dashboard data.
   *
   * @returns {Promise<void>}
   * @throws {Error} If monitor is already running or initialization fails
   *
   * @fires RealtimeMonitor#started
   * @fires RealtimeMonitor#state:change
   */
  async start() {
    if (this.#state !== 'stopped') {
      throw new Error(`Cannot start monitor in state: ${this.#state}`);
    }

    this.#setState('initializing');
    this.#startTime = Date.now();
    this.#tickCount = 0;

    try {
      await this.#initializeSubModules();
      this.#setupSubModuleEventHandlers();
      this.#startTickLoop();
      this.#startDashboardPush();

      this.#setState('running');

      if (this.#eventLogger) {
        this.#eventLogger.info('RealtimeMonitor', 'Monitor started successfully', {
          tickInterval: this.#config.tickIntervalMs,
          dashboardPushInterval: this.#config.dashboardPushIntervalMs,
          wsServerEnabled: this.#config.enableWebSocketServer,
        });
      }

      /**
       * @event RealtimeMonitor#started
       * @type {Object}
       * @property {number} timestamp - Start timestamp
       */
      this.emit('started', { timestamp: Date.now() });
    } catch (error) {
      this.#setState('error');
      await this.#cleanupSubModules();

      if (this.#eventLogger) {
        this.#eventLogger.critical('RealtimeMonitor', 'Failed to start monitor', {
          error: error.message,
          stack: error.stack,
        });
      }

      throw new Error(`Monitor initialization failed: ${error.message}`);
    }
  }

  /**
   * Gracefully stops the monitor and all sub-modules.
   * Flushes remaining data, closes connections, and cleans up resources.
   *
   * @param {number} [gracePeriodMs=5000] - Grace period for cleanup
   * @returns {Promise<void>}
   *
   * @fires RealtimeMonitor#stopped
   */
  async stop(gracePeriodMs = 5000) {
    if (this.#state === 'stopped') {
      return;
    }

    this.#setState('shutting_down');

    if (this.#eventLogger) {
      this.#eventLogger.info('RealtimeMonitor', 'Monitor shutting down', {
        gracePeriodMs,
        uptime: this.uptime,
      });
    }

    clearInterval(this.#tickTimer);
    clearInterval(this.#dashboardPushTimer);
    this.#tickTimer = null;
    this.#dashboardPushTimer = null;

    const shutdownTimeout = setTimeout(() => {
      console.warn('Monitor shutdown exceeded grace period, forcing cleanup');
    }, gracePeriodMs);

    try {
      await this.#cleanupSubModules();
      clearTimeout(shutdownTimeout);
    } catch (error) {
      clearTimeout(shutdownTimeout);
      console.error('Error during monitor shutdown:', error);
    }

    this.#dataBuffer = [];
    this.#tickCount = 0;
    this.#startTime = 0;
    this.#setState('stopped');

    /**
     * @event RealtimeMonitor#stopped
     * @type {Object}
     * @property {number} timestamp - Stop timestamp
     */
    this.emit('stopped', { timestamp: Date.now() });
  }

  /**
   * Ingests a data point into the monitoring pipeline.
   * The data point will be processed, buffered, and routed to
   * relevant sub-modules.
   *
   * @param {DataPoint} dataPoint - The data point to ingest
   * @returns {boolean} Whether the data point was accepted
   *
   * @fires RealtimeMonitor#data:ingest
   */
  ingestData(dataPoint) {
    if (!this.isRunning) {
      return false;
    }

    if (!dataPoint || !dataPoint.source || !dataPoint.type) {
      if (this.#eventLogger) {
        this.#eventLogger.warn('RealtimeMonitor', 'Invalid data point rejected', {
          dataPoint: dataPoint?.id || 'unknown',
        });
      }
      return false;
    }

    if (this.#dataBuffer.length >= this.#config.maxDataBufferSize) {
      this.#dataBuffer.shift();
      if (this.#eventLogger) {
        this.#eventLogger.warn('RealtimeMonitor', 'Data buffer overflow, oldest entry dropped');
      }
    }

    const enriched = {
      ...dataPoint,
      timestamp: dataPoint.timestamp || Date.now(),
      id: dataPoint.id || `dp_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`,
    };

    this.#dataBuffer.push(enriched);
    this.#routeDataPoint(enriched);

    /**
     * @event RealtimeMonitor#data:ingest
     * @type {DataPoint}
     */
    this.emit('data:ingest', enriched);

    return true;
  }

  /**
   * Generates a complete dashboard snapshot for the current state.
   *
   * @returns {DashboardSnapshot} Current dashboard snapshot
   */
  getDashboardSnapshot() {
    const timestamp = Date.now();

    return {
      timestamp,
      state: this.#state,
      systemStatus: this.#systemStatus?.getStatus() ?? {
        mode: 'unknown',
        overallStatus: 'offline',
        modules: {},
        timestamp,
      },
      healthReport: this.#healthMonitor?.getHealthReport() ?? {
        score: 0,
        status: 'unknown',
        checks: {},
        timestamp,
      },
      latencyStats: this.#latencyMonitor?.getStats() ?? {
        p50: 0,
        p95: 0,
        p99: 0,
        avg: 0,
        min: 0,
        max: 0,
        samples: 0,
      },
      performanceReport: this.#performanceTracker?.getReport() ?? {
        fps: 0,
        memoryUsage: {},
        cpuUsage: 0,
        eventLoopLag: 0,
        gcPressure: 0,
        throughput: 0,
        timestamp,
      },
      activeAlerts: this.#alertManager?.getActiveAlerts() ?? [],
      customMetrics: this.#collectCustomMetrics(),
    };
  }

  /**
   * Registers a custom data processor for a specific data type.
   *
   * @param {string} dataType - Data type to process
   * @param {Function} processor - Processing function (dataPoint) => void
   * @returns {void}
   */
  registerDataProcessor(dataType, processor) {
    if (typeof processor !== 'function') {
      throw new TypeError('Processor must be a function');
    }
    this.#dataProcessors.set(dataType, processor);
  }

  /**
   * Registers a sub-module with the monitor for status tracking.
   *
   * @param {string} name - Module name
   * @param {Object} module - Module instance
   * @param {Object} [metadata={}] - Additional module metadata
   * @returns {void}
   */
  registerModule(name, module, metadata = {}) {
    this.#moduleRegistry.set(name, {
      instance: module,
      metadata: {
        name,
        registeredAt: Date.now(),
        ...metadata,
      },
    });
  }

  /**
   * Gets the configuration object.
   *
   * @returns {Readonly<MonitorConfig>} Frozen copy of current config
   */
  getConfig() {
    return Object.freeze({ ...this.#config });
  }

  /**
   * Updates configuration at runtime (limited to non-critical settings).
   *
   * @param {Partial<MonitorConfig>} updates - Configuration updates
   * @returns {boolean} Whether updates were applied
   */
  updateConfig(updates) {
    const allowedKeys = new Set([
      'dashboardPushIntervalMs',
      'alertThrottleMs',
      'enableCloudUpload',
      'cloudEndpoint',
    ]);

    let applied = false;
    for (const [key, value] of Object.entries(updates)) {
      if (allowedKeys.has(key)) {
        this.#config[key] = value;
        applied = true;
      }
    }

    if (applied && this.#eventLogger) {
      this.#eventLogger.info('RealtimeMonitor', 'Configuration updated', { updates });
    }

    return applied;
  }

  /**
   * Forces a health check across all sub-modules.
   *
   * @returns {Promise<import('./health_monitor.js').HealthReport>}
   */
  async forceHealthCheck() {
    if (!this.#healthMonitor) {
      throw new Error('Health monitor not initialized');
    }
    return this.#healthMonitor.performCheck();
  }

  /**
   * Retrieves recent data points from the buffer.
   *
   * @param {Object} [filter={}] - Filter criteria
   * @param {string} [filter.source] - Filter by source
   * @param {string} [filter.type] - Filter by type
   * @param {number} [filter.limit=100] - Max results
   * @param {number} [filter.since] - Minimum timestamp
   * @returns {DataPoint[]} Filtered data points
   */
  getRecentData(filter = {}) {
    let results = [...this.#dataBuffer];

    if (filter.source) {
      results = results.filter((dp) => dp.source === filter.source);
    }
    if (filter.type) {
      results = results.filter((dp) => dp.type === filter.type);
    }
    if (filter.since) {
      results = results.filter((dp) => dp.timestamp >= filter.since);
    }

    const limit = filter.limit ?? 100;
    return results.slice(-limit);
  }

  // ─── Private Methods ───────────────────────────────────────────────

  /**
   * Initializes all sub-modules with their respective configurations.
   * @private
   * @returns {Promise<void>}
   */
  async #initializeSubModules() {
    const moduleConfig = this.#config.moduleConfig;

    this.#eventLogger = new EventLogger({
      maxEntries: 50000,
      rotationIntervalMs: 3600000,
      ...moduleConfig.eventLogger,
    });

    this.#alertManager = new AlertManager({
      throttleMs: this.#config.alertThrottleMs,
      ...moduleConfig.alertManager,
    });

    this.#healthMonitor = new HealthMonitor({
      checkIntervalMs: this.#config.healthCheckIntervalMs,
      ...moduleConfig.healthMonitor,
    });

    this.#latencyMonitor = new LatencyMonitor({
      sampleIntervalMs: this.#config.latencySampleIntervalMs,
      ...moduleConfig.latencyMonitor,
    });

    this.#systemStatus = new SystemStatus({
      ...moduleConfig.systemStatus,
    });

    this.#performanceTracker = new PerformanceTracker({
      ...moduleConfig.performanceTracker,
    });

    if (this.#config.enableWebSocketServer) {
      this.#wsServer = new WebSocketServer({
        port: this.#config.wsServerPort,
        ...moduleConfig.wsServer,
      });
      await this.#wsServer.start();
    }

    if (this.#config.enableCloudUpload && this.#config.cloudEndpoint) {
      this.#wsClient = new WebSocketClient({
        url: this.#config.cloudEndpoint,
        ...moduleConfig.wsClient,
      });
      await this.#wsClient.connect();
    }

    this.registerModule('healthMonitor', this.#healthMonitor, { critical: true });
    this.registerModule('latencyMonitor', this.#latencyMonitor);
    this.registerModule('eventLogger', this.#eventLogger, { critical: true });
    this.registerModule('alertManager', this.#alertManager, { critical: true });
    this.registerModule('systemStatus', this.#systemStatus, { critical: true });
    this.registerModule('performanceTracker', this.#performanceTracker);
  }

  /**
   * Sets up event forwarding from sub-modules to the main monitor.
   * @private
   */
  #setupSubModuleEventHandlers() {
    if (this.#alertManager) {
      this.#alertManager.on('alert:created', (alert) => {
        this.emit('alert:new', alert);
        if (alert.severity === 'critical') {
          this.emit('alert:critical', alert);
          this.#handleCriticalAlert(alert);
        }
      });

      this.#alertManager.on('alert:escalated', (alert) => {
        this.emit('alert:escalated', alert);
      });
    }

    if (this.#healthMonitor) {
      this.#healthMonitor.on('health:degraded', (report) => {
        this.#setState('degraded');
        this.emit('health:degraded', report);
      });

      this.#healthMonitor.on('health:recovered', (report) => {
        if (this.#state === 'degraded') {
          this.#setState('running');
        }
        this.emit('health:recovered', report);
      });

      this.#healthMonitor.on('health:check', (report) => {
        this.ingestData({
          source: 'healthMonitor',
          type: 'health',
          payload: report,
        });
      });
    }

    if (this.#latencyMonitor) {
      this.#latencyMonitor.on('latency:threshold_breach', (metric) => {
        if (this.#alertManager) {
          this.#alertManager.createAlert({
            severity: metric.p99 > 200 ? 'critical' : 'warning',
            source: 'latencyMonitor',
            message: `Latency threshold breached: P99=${metric.p99}ms`,
            metadata: metric,
          });
        }
      });
    }

    if (this.#performanceTracker) {
      this.#performanceTracker.on('performance:regression', (details) => {
        if (this.#alertManager) {
          this.#alertManager.createAlert({
            severity: 'warning',
            source: 'performanceTracker',
            message: `Performance regression detected: ${details.metric}`,
            metadata: details,
          });
        }
      });
    }
  }

  /**
   * Starts the main tick loop for periodic processing.
   * @private
   */
  #startTickLoop() {
    this.#tickTimer = setInterval(() => {
      this.#tickCount++;

      if (this.#tickCount % 10 === 0) {
        this.#processBufferedData();
      }

      if (this.#performanceTracker) {
        this.#performanceTracker.recordTick();
      }
    }, this.#config.tickIntervalMs);
  }

  /**
   * Starts the periodic dashboard data push.
   * @private
   */
  #startDashboardPush() {
    this.#dashboardPushTimer = setInterval(() => {
      const snapshot = this.getDashboardSnapshot();

      if (this.#wsServer) {
        this.#wsServer.broadcast('dashboard', {
          type: 'dashboard:update',
          data: snapshot,
        });
      }

      if (this.#wsClient?.isConnected) {
        this.#wsClient.send('telemetry', snapshot);
      }

      /**
       * @event RealtimeMonitor#dashboard:update
       * @type {DashboardSnapshot}
       */
      this.emit('dashboard:update', snapshot);
    }, this.#config.dashboardPushIntervalMs);
  }

  /**
   * Routes a data point to relevant sub-modules and custom processors.
   * @private
   * @param {DataPoint} dataPoint - Data point to route
   */
  #routeDataPoint(dataPoint) {
    const processor = this.#dataProcessors.get(dataPoint.type);
    if (processor) {
      try {
        processor(dataPoint);
      } catch (error) {
        if (this.#eventLogger) {
          this.#eventLogger.error('RealtimeMonitor', 'Data processor error', {
            dataType: dataPoint.type,
            error: error.message,
          });
        }
      }
    }

    if (dataPoint.type === 'latency' && this.#latencyMonitor) {
      this.#latencyMonitor.recordSample(dataPoint.payload);
    }

    if (dataPoint.type === 'health' && this.#systemStatus) {
      this.#systemStatus.updateModuleHealth(dataPoint.source, dataPoint.payload);
    }

    if (dataPoint.type === 'sensor' && this.#performanceTracker) {
      this.#performanceTracker.recordSensorFrame();
    }
  }

  /**
   * Processes accumulated data in the buffer (aggregation, compaction).
   * @private
   */
  #processBufferedData() {
    if (this.#dataBuffer.length === 0) return;

    const now = Date.now();
    const cutoff = now - 60000;
    const oldLength = this.#dataBuffer.length;

    this.#dataBuffer = this.#dataBuffer.filter((dp) => dp.timestamp >= cutoff);

    if (this.#dataBuffer.length < oldLength && this.#eventLogger) {
      this.#eventLogger.debug('RealtimeMonitor', 'Buffer compacted', {
        removed: oldLength - this.#dataBuffer.length,
        remaining: this.#dataBuffer.length,
      });
    }
  }

  /**
   * Collects custom metrics from registered modules.
   * @private
   * @returns {Object} Custom metrics keyed by module name
   */
  #collectCustomMetrics() {
    const metrics = {};
    for (const [name, entry] of this.#moduleRegistry) {
      if (typeof entry.instance?.getMetrics === 'function') {
        try {
          metrics[name] = entry.instance.getMetrics();
        } catch {
          metrics[name] = { error: 'Failed to collect metrics' };
        }
      }
    }
    return metrics;
  }

  /**
   * Handles a critical alert by triggering emergency protocols.
   * @private
   * @param {import('./alert_manager.js').Alert} alert - The critical alert
   */
  #handleCriticalAlert(alert) {
    if (this.#systemStatus) {
      this.#systemStatus.setMode('emergency');
    }

    if (this.#wsServer) {
      this.#wsServer.broadcast('alerts', {
        type: 'alert:critical',
        data: alert,
      });
    }

    /**
     * @event RealtimeMonitor#emergency
     * @type {import('./alert_manager.js').Alert}
     */
    this.emit('emergency', alert);
  }

  /**
   * Initializes built-in data processors.
   * @private
   */
  #initializeDataProcessors() {
    this.#dataProcessors.set('sensor', (dp) => {
      if (dp.payload?.anomalyDetected && this.#alertManager) {
        this.#alertManager.createAlert({
          severity: 'warning',
          source: dp.source,
          message: `Sensor anomaly detected: ${dp.payload.anomalyType || 'unknown'}`,
          metadata: dp.payload,
        });
      }
    });

    this.#dataProcessors.set('control', (dp) => {
      if (dp.payload?.commandFailed && this.#alertManager) {
        this.#alertManager.createAlert({
          severity: 'critical',
          source: dp.source,
          message: `Control command failed: ${dp.payload.commandId || 'unknown'}`,
          metadata: dp.payload,
        });
      }
    });
  }

  /**
   * Cleans up and stops all sub-modules.
   * @private
   * @returns {Promise<void>}
   */
  async #cleanupSubModules() {
    const cleanupOrder = [
      () => this.#wsClient?.disconnect(),
      () => this.#wsServer?.stop(),
      () => this.#performanceTracker?.stop(),
      () => this.#latencyMonitor?.stop(),
      () => this.#healthMonitor?.stop(),
      () => this.#alertManager?.clearAll(),
      () => this.#eventLogger?.flush(),
    ];

    for (const cleanupFn of cleanupOrder) {
      try {
        const result = cleanupFn();
        if (result instanceof Promise) {
          await result;
        }
      } catch (error) {
        console.error('Sub-module cleanup error:', error.message);
      }
    }
  }

  /**
   * Updates the monitor state and emits state change event.
   * @private
   * @param {MonitorState} newState - New state
   */
  #setState(newState) {
    const oldState = this.#state;
    if (oldState === newState) return;

    this.#state = newState;

    /**
     * @event RealtimeMonitor#state:change
     * @type {Object}
     * @property {MonitorState} oldState - Previous state
     * @property {MonitorState} newState - New state
     */
    this.emit('state:change', { oldState, newState });
  }
}

export default RealtimeMonitor;
