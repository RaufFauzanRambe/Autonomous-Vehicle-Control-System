/**
 * @module telemetry
 * @description Main Telemetry Manager module for the Autonomous Vehicle Control System.
 * Orchestrates all telemetry sub-modules including client, server, parser, storage,
 * streaming, analytics, and metrics. Manages the complete data pipeline:
 * collect → parse → store → stream → analyze.
 *
 * @author Autonomous Vehicle Control System
 * @version 2.0.0
 * @license MIT
 */

import { EventEmitter } from 'events';
import { TelemetryClient } from './telemetry_client.js';
import { TelemetryParser } from './telemetry_parser.js';
import { TelemetryStorage } from './telemetry_storage.js';
import { TelemetryStream } from './telemetry_stream.js';
import { TelemetryAnalytics } from './telemetry_analytics.js';
import { VehicleMetrics } from './vehicle_metrics.js';
import { SensorMetrics } from './sensor_metrics.js';

// ─────────────────────────────────────────────────────────────────────────────
// Custom Error Classes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Base error class for telemetry system errors.
 * @extends Error
 */
export class TelemetryError extends Error {
  /**
   * @param {string} message - Error description
   * @param {string} [code='TELEMETRY_ERROR'] - Machine-readable error code
   * @param {Error} [cause] - Underlying cause
   */
  constructor(message, code = 'TELEMETRY_ERROR', cause = null) {
    super(message);
    this.name = 'TelemetryError';
    this.code = code;
    this.cause = cause;
    this.timestamp = Date.now();
  }
}

/**
 * Error thrown when a sub-module fails to initialize.
 * @extends TelemetryError
 */
export class ModuleInitializationError extends TelemetryError {
  constructor(moduleName, cause) {
    super(`Failed to initialize module: ${moduleName}`, 'MODULE_INIT_ERROR', cause);
    this.name = 'ModuleInitializationError';
    this.moduleName = moduleName;
  }
}

/**
 * Error thrown when the data pipeline encounters a failure.
 * @extends TelemetryError
 */
export class PipelineError extends TelemetryError {
  constructor(stage, message, cause) {
    super(`Pipeline error at stage '${stage}': ${message}`, 'PIPELINE_ERROR', cause);
    this.name = 'PipelineError';
    this.stage = stage;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants & Configuration
// ─────────────────────────────────────────────────────────────────────────────

/** @type {Set<string>} Valid lifecycle states for the TelemetryManager */
const VALID_STATES = new Set([
  'uninitialized',
  'initializing',
  'ready',
  'running',
  'paused',
  'stopping',
  'stopped',
  'error'
]);

/** @type {object} Default configuration values */
const DEFAULT_CONFIG = {
  /** Maximum number of concurrent data pipelines */
  maxPipelines: 8,
  /** Buffer size for incoming raw data (bytes) */
  receiveBufferSize: 1048576,
  /** Interval (ms) for periodic health checks */
  healthCheckIntervalMs: 5000,
  /** Interval (ms) for flushing buffered data to storage */
  flushIntervalMs: 1000,
  /** Maximum time (ms) to wait for graceful shutdown */
  shutdownTimeoutMs: 10000,
  /** Whether to auto-start sub-modules on init */
  autoStart: true,
  /** Enable verbose diagnostic logging */
  verboseLogging: false,
  /** Maximum pipeline processing latency before alert (ms) */
  maxPipelineLatencyMs: 100,
  /** Data sampling rate for high-frequency sensors (1 = every sample) */
  samplingRate: 1,
  /** Number of retry attempts for failed pipeline stages */
  pipelineRetryAttempts: 3,
  /** Delay between pipeline retry attempts (ms) */
  pipelineRetryDelayMs: 500,
  /** Sub-module configurations */
  modules: {
    client: { enabled: true },
    parser: { enabled: true },
    storage: { enabled: true, maxBufferSize: 100000 },
    stream: { enabled: true, highWaterMark: 16384 },
    analytics: { enabled: true, windowSizeMs: 60000 },
    vehicleMetrics: { enabled: true },
    sensorMetrics: { enabled: true }
  }
};

/** @enum {string} Pipeline stages */
export const PipelineStage = {
  COLLECT: 'collect',
  PARSE: 'parse',
  STORE: 'store',
  STREAM: 'stream',
  ANALYZE: 'analyze'
};

/** @enum {string} Telemetry event names */
export const TelemetryEvent = {
  DATA_RECEIVED: 'data:received',
  DATA_PARSED: 'data:parsed',
  DATA_STORED: 'data:stored',
  DATA_STREAMED: 'data:streamed',
  DATA_ANALYZED: 'data:analyzed',
  PIPELINE_COMPLETE: 'pipeline:complete',
  PIPELINE_ERROR: 'pipeline:error',
  ALERT: 'alert',
  STATE_CHANGED: 'state:changed',
  MODULE_READY: 'module:ready',
  MODULE_ERROR: 'module:error',
  HEALTH_CHECK: 'health:check',
  SHUTDOWN: 'shutdown'
};

// ─────────────────────────────────────────────────────────────────────────────
// Pipeline Tracker
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Tracks the state and timing of a single data item through the pipeline.
 */
class PipelineTracker {
  /**
   * @param {string} id - Unique pipeline execution ID
   * @param {object} metadata - Additional metadata about the data item
   */
  constructor(id, metadata = {}) {
    /** @type {string} */
    this.id = id;
    /** @type {object} */
    this.metadata = metadata;
    /** @type {Map<string, {start: number, end: number|null, duration: number|null}>} */
    this.stages = new Map();
    /** @type {string} */
    this.currentStage = PipelineStage.COLLECT;
    /** @type {number} */
    this.createdAt = Date.now();
    /** @type {number|null} */
    this.completedAt = null;
    /** @type {Error|null} */
    this.error = null;
  }

  /**
   * Mark entry into a pipeline stage.
   * @param {string} stage - The pipeline stage
   */
  enterStage(stage) {
    this.currentStage = stage;
    this.stages.set(stage, { start: Date.now(), end: null, duration: null });
  }

  /**
   * Mark exit from a pipeline stage.
   * @param {string} stage - The pipeline stage
   */
  exitStage(stage) {
    const record = this.stages.get(stage);
    if (record) {
      record.end = Date.now();
      record.duration = record.end - record.start;
    }
  }

  /**
   * Get total pipeline latency in milliseconds.
   * @returns {number}
   */
  getTotalLatency() {
    return this.completedAt ? this.completedAt - this.createdAt : Date.now() - this.createdAt;
  }

  /**
   * Get the duration of a specific stage.
   * @param {string} stage - The pipeline stage
   * @returns {number|null}
   */
  getStageDuration(stage) {
    const record = this.stages.get(stage);
    return record?.duration ?? null;
  }

  /**
   * Mark pipeline as completed.
   */
  complete() {
    this.completedAt = Date.now();
  }

  /**
   * Mark pipeline as failed.
   * @param {Error} error - The error that caused the failure
   */
  fail(error) {
    this.error = error;
    this.completedAt = Date.now();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// TelemetryManager
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Main telemetry manager class. Orchestrates all sub-modules and manages
 * the complete telemetry data pipeline from collection through analysis.
 *
 * @extends EventEmitter
 *
 * @example
 * const manager = new TelemetryManager({ verboseLogging: true });
 * await manager.initialize();
 * manager.on(TelemetryEvent.ALERT, (alert) => console.warn(alert));
 * await manager.start();
 */
export class TelemetryManager extends EventEmitter {
  /**
   * Create a new TelemetryManager instance.
   * @param {object} [config={}] - Configuration overrides
   * @param {number} [config.maxPipelines=8] - Maximum concurrent pipeline executions
   * @param {number} [config.flushIntervalMs=1000] - Data flush interval
   * @param {boolean} [config.autoStart=true] - Auto-start sub-modules on init
   * @param {object} [config.modules] - Per-module configuration overrides
   */
  constructor(config = {}) {
    super();

    /** @type {object} Merged configuration */
    this.config = this._mergeConfig(config);

    /** @type {string} Current lifecycle state */
    this._state = 'uninitialized';

    /** @type {Map<string, object>} Initialized sub-module instances */
    this._modules = new Map();

    /** @type {Map<string, PipelineTracker>} Active pipeline trackers */
    this._activePipelines = new Map();

    /** @type {number} Monotonic counter for pipeline IDs */
    this._pipelineCounter = 0;

    /** @type {NodeJS.Timer|null} Health check interval handle */
    this._healthCheckTimer = null;

    /** @type {NodeJS.Timer|null} Flush interval handle */
    this._flushTimer = null;

    /** @type {object} Runtime statistics */
    this._stats = {
      totalReceived: 0,
      totalParsed: 0,
      totalStored: 0,
      totalStreamed: 0,
      totalAnalyzed: 0,
      pipelineErrors: 0,
      avgPipelineLatencyMs: 0,
      lastHealthCheck: null,
      startTime: null
    };

    // Bind methods for use as event handlers
    this._handleModuleError = this._handleModuleError.bind(this);
    this._handleHealthCheck = this._handleHealthCheck.bind(this);
    this._handleFlush = this._handleFlush.bind(this);
  }

  // ── Lifecycle Methods ───────────────────────────────────────────────────

  /**
   * Initialize the telemetry manager and all enabled sub-modules.
   * Transitions state: uninitialized → initializing → ready
   *
   * @returns {Promise<void>}
   * @throws {ModuleInitializationError} If any sub-module fails to initialize
   */
  async initialize() {
    this._setState('initializing');

    try {
      const moduleConfigs = this.config.modules;

      // Initialize parser first (stateless, no dependencies)
      if (moduleConfigs.parser.enabled) {
        await this._initModule('parser', () => new TelemetryParser(moduleConfigs.parser));
      }

      // Initialize storage
      if (moduleConfigs.storage.enabled) {
        await this._initModule('storage', () => new TelemetryStorage(moduleConfigs.storage));
      }

      // Initialize stream processor
      if (moduleConfigs.stream.enabled) {
        await this._initModule('stream', () => new TelemetryStream(moduleConfigs.stream));
      }

      // Initialize analytics engine
      if (moduleConfigs.analytics.enabled) {
        await this._initModule('analytics', () => new TelemetryAnalytics(moduleConfigs.analytics));
      }

      // Initialize vehicle metrics
      if (moduleConfigs.vehicleMetrics.enabled) {
        await this._initModule('vehicleMetrics', () => new VehicleMetrics(moduleConfigs.vehicleMetrics));
      }

      // Initialize sensor metrics
      if (moduleConfigs.sensorMetrics.enabled) {
        await this._initModule('sensorMetrics', () => new SensorMetrics(moduleConfigs.sensorMetrics));
      }

      // Initialize client last (depends on parser for data handling)
      if (moduleConfigs.client.enabled) {
        await this._initModule('client', () => new TelemetryClient(moduleConfigs.client));
      }

      this._setState('ready');
      this.emit(TelemetryEvent.MODULE_READY, { module: 'telemetry-manager' });
    } catch (error) {
      this._setState('error');
      throw new ModuleInitializationError('TelemetryManager', error);
    }
  }

  /**
   * Start the telemetry data pipeline.
   * Transitions state: ready → running
   *
   * @returns {Promise<void>}
   * @throws {TelemetryError} If manager is not in a startable state
   */
  async start() {
    if (this._state !== 'ready' && this._state !== 'paused') {
      throw new TelemetryError(
        `Cannot start from state '${this._state}'. Expected 'ready' or 'paused'.`,
        'INVALID_STATE'
      );
    }

    this._setState('running');
    this._stats.startTime = Date.now();

    // Start health check timer
    this._healthCheckTimer = setInterval(
      this._handleHealthCheck,
      this.config.healthCheckIntervalMs
    );

    // Start flush timer
    this._flushTimer = setInterval(this._handleFlush, this.config.flushIntervalMs);

    // Connect client if available
    const client = this._modules.get('client');
    if (client) {
      try {
        await client.connect();
      } catch (error) {
        this.emit(TelemetryEvent.MODULE_ERROR, {
          module: 'client',
          error: new TelemetryError('Client connection failed on start', 'CLIENT_CONNECT_ERROR', error)
        });
      }
    }

    // Wire up streaming pipelines
    this._setupPipelineConnections();
  }

  /**
   * Pause data collection and pipeline processing.
   * Transitions state: running → paused
   *
   * @returns {void}
   */
  pause() {
    if (this._state !== 'running') {
      return;
    }

    this._setState('paused');
    this._clearTimers();

    const client = this._modules.get('client');
    if (client) {
      client.disconnect();
    }
  }

  /**
   * Resume data collection from a paused state.
   * Transitions state: paused → running
   *
   * @returns {Promise<void>}
   */
  async resume() {
    if (this._state !== 'paused') {
      return;
    }
    await this.start();
  }

  /**
   * Gracefully shut down the telemetry manager and all sub-modules.
   * Transitions state: * → stopping → stopped
   *
   * @param {object} [options={}] - Shutdown options
   * @param {number} [options.timeout=this.config.shutdownTimeoutMs] - Graceful shutdown timeout
   * @param {boolean} [options.force=false] - Force immediate shutdown
   * @returns {Promise<void>}
   */
  async shutdown(options = {}) {
    const { timeout = this.config.shutdownTimeoutMs, force = false } = options;
    this._setState('stopping');
    this._clearTimers();

    if (!force) {
      // Wait for active pipelines to complete (with timeout)
      const shutdownStart = Date.now();
      while (this._activePipelines.size > 0 && (Date.now() - shutdownStart) < timeout) {
        await new Promise(resolve => setTimeout(resolve, 100));
      }

      if (this._activePipelines.size > 0) {
        this.emit(TelemetryEvent.PIPELINE_ERROR, {
          message: `Shutdown timeout with ${this._activePipelines.size} active pipelines`
        });
      }
    }

    // Flush remaining data to storage
    const storage = this._modules.get('storage');
    if (storage) {
      try {
        await storage.flush();
      } catch (_) { /* best effort flush */ }
    }

    // Disconnect client
    const client = this._modules.get('client');
    if (client) {
      client.disconnect();
    }

    // Destroy streams
    const stream = this._modules.get('stream');
    if (stream) {
      stream.destroy();
    }

    this._modules.clear();
    this._activePipelines.clear();
    this._setState('stopped');
    this.emit(TelemetryEvent.SHUTDOWN, { timestamp: Date.now(), stats: this._stats });
    this.removeAllListeners();
  }

  // ── Pipeline Processing ─────────────────────────────────────────────────

  /**
   * Process a raw telemetry data packet through the complete pipeline.
   * Pipeline stages: collect → parse → store → stream → analyze
   *
   * @param {Buffer|object} rawData - Raw telemetry data packet
   * @param {object} [options={}] - Processing options
   * @param {string} [options.source='unknown'] - Data source identifier
   * @param {boolean} [options.skipStorage=false] - Skip storage stage
   * @param {boolean} [options.skipAnalytics=false] - Skip analytics stage
   * @returns {Promise<{tracker: PipelineTracker, result: object}>}
   */
  async processData(rawData, options = {}) {
    const pipelineId = `pipe_${++this._pipelineCounter}`;
    const tracker = new PipelineTracker(pipelineId, { source: options.source || 'unknown' });

    if (this._state !== 'running') {
      throw new TelemetryError(
        `Cannot process data in state '${this._state}'. Manager must be running.`,
        'INVALID_STATE'
      );
    }

    if (this._activePipelines.size >= this.config.maxPipelines) {
      throw new TelemetryError(
        'Maximum concurrent pipelines reached',
        'PIPELINE_CAPACITY_EXCEEDED'
      );
    }

    this._activePipelines.set(pipelineId, tracker);
    let parsedData = null;

    try {
      // Stage 1: Collect
      tracker.enterStage(PipelineStage.COLLECT);
      this._stats.totalReceived++;
      this.emit(TelemetryEvent.DATA_RECEIVED, {
        pipelineId,
        source: options.source,
        size: Buffer.isBuffer(rawData) ? rawData.length : JSON.stringify(rawData).length,
        timestamp: Date.now()
      });
      tracker.exitStage(PipelineStage.COLLECT);

      // Stage 2: Parse
      tracker.enterStage(PipelineStage.PARSE);
      const parser = this._modules.get('parser');
      if (parser) {
        parsedData = await this._retryStage(
          () => parser.parse(rawData),
          PipelineStage.PARSE
        );
      } else {
        parsedData = rawData;
      }
      this._stats.totalParsed++;
      this.emit(TelemetryEvent.DATA_PARSED, { pipelineId, data: parsedData });
      tracker.exitStage(PipelineStage.PARSE);

      // Stage 3: Store
      if (!options.skipStorage) {
        tracker.enterStage(PipelineStage.STORE);
        const storage = this._modules.get('storage');
        if (storage) {
          await this._retryStage(
            () => storage.write(parsedData),
            PipelineStage.STORE
          );
        }
        this._stats.totalStored++;
        this.emit(TelemetryEvent.DATA_STORED, { pipelineId, data: parsedData });
        tracker.exitStage(PipelineStage.STORE);
      }

      // Stage 4: Stream
      tracker.enterStage(PipelineStage.STREAM);
      const stream = this._modules.get('stream');
      if (stream) {
        stream.push(parsedData);
      }
      this._stats.totalStreamed++;
      this.emit(TelemetryEvent.DATA_STREAMED, { pipelineId });
      tracker.exitStage(PipelineStage.STREAM);

      // Stage 5: Analyze
      if (!options.skipAnalytics) {
        tracker.enterStage(PipelineStage.ANALYZE);
        const analytics = this._modules.get('analytics');
        if (analytics) {
          const analysisResult = await this._retryStage(
            () => analytics.process(parsedData),
            PipelineStage.ANALYZE
          );

          if (analysisResult?.alerts?.length > 0) {
            for (const alert of analysisResult.alerts) {
              this.emit(TelemetryEvent.ALERT, alert);
            }
          }
        }
        this._stats.totalAnalyzed++;
        this.emit(TelemetryEvent.DATA_ANALYZED, { pipelineId });
        tracker.exitStage(PipelineStage.ANALYZE);
      }

      // Update vehicle & sensor metrics
      const vehicleMetrics = this._modules.get('vehicleMetrics');
      if (vehicleMetrics && parsedData) {
        vehicleMetrics.update(parsedData);
      }

      const sensorMetrics = this._modules.get('sensorMetrics');
      if (sensorMetrics && parsedData) {
        sensorMetrics.recordDataPoint(options.source, parsedData);
      }

      tracker.complete();
      this._updateAvgLatency(tracker.getTotalLatency());

      this.emit(TelemetryEvent.PIPELINE_COMPLETE, {
        pipelineId,
        latencyMs: tracker.getTotalLatency(),
        stages: Object.fromEntries(tracker.stages)
      });

      return { tracker, result: parsedData };
    } catch (error) {
      tracker.fail(error);
      this._stats.pipelineErrors++;
      this.emit(TelemetryEvent.PIPELINE_ERROR, {
        pipelineId,
        stage: tracker.currentStage,
        error
      });
      throw new PipelineError(tracker.currentStage, error.message, error);
    } finally {
      this._activePipelines.delete(pipelineId);
    }
  }

  // ── Module Access ───────────────────────────────────────────────────────

  /**
   * Get a sub-module instance by name.
   * @param {string} name - Module name (parser, storage, stream, analytics, client, vehicleMetrics, sensorMetrics)
   * @returns {object|null} Module instance or null if not initialized
   */
  getModule(name) {
    return this._modules.get(name) || null;
  }

  /**
   * Get current lifecycle state.
   * @returns {string}
   */
  getState() {
    return this._state;
  }

  /**
   * Get runtime statistics.
   * @returns {object}
   */
  getStats() {
    return {
      ...this._stats,
      activePipelines: this._activePipelines.size,
      uptimeMs: this._stats.startTime ? Date.now() - this._stats.startTime : 0,
      state: this._state
    };
  }

  /**
   * Get all active pipeline trackers.
   * @returns {PipelineTracker[]}
   */
  getActivePipelines() {
    return Array.from(this._activePipelines.values());
  }

  /**
   * Query stored telemetry data.
   * @param {object} query - Query parameters
   * @param {string} [query.sensorType] - Filter by sensor type
   * @param {number} [query.startTime] - Start of time range (ms epoch)
   * @param {number} [query.endTime] - End of time range (ms epoch)
   * @param {number} [query.limit] - Maximum results
   * @returns {Promise<object[]>}
   */
  async queryData(query) {
    const storage = this._modules.get('storage');
    if (!storage) {
      throw new TelemetryError('Storage module not available', 'MODULE_NOT_FOUND');
    }
    return storage.query(query);
  }

  // ── Private Methods ─────────────────────────────────────────────────────

  /**
   * Initialize a single sub-module with error handling.
   * @param {string} name - Module name
   * @param {Function} factory - Factory function to create module instance
   * @returns {Promise<void>}
   * @private
   */
  async _initModule(name, factory) {
    try {
      const instance = factory();

      // Wire up module error events
      if (instance instanceof EventEmitter) {
        instance.on('error', (error) => this._handleModuleError(name, error));
      }

      // Initialize module if it has an init method
      if (typeof instance.initialize === 'function') {
        await instance.initialize();
      }

      this._modules.set(name, instance);
      this.emit(TelemetryEvent.MODULE_READY, { module: name });
    } catch (error) {
      throw new ModuleInitializationError(name, error);
    }
  }

  /**
   * Set up data flow connections between sub-modules via events.
   * @private
   */
  _setupPipelineConnections() {
    const client = this._modules.get('client');
    const stream = this._modules.get('stream');
    const analytics = this._modules.get('analytics');

    // Client → Manager pipeline
    if (client) {
      client.on('data', async (data) => {
        try {
          await this.processData(data.raw, { source: data.source });
        } catch (error) {
          this.emit(TelemetryEvent.PIPELINE_ERROR, {
            stage: PipelineStage.COLLECT,
            error
          });
        }
      });
    }

    // Stream → Analytics passthrough
    if (stream && analytics) {
      stream.on('data', (data) => {
        if (typeof analytics.process === 'function') {
          analytics.process(data).catch((error) => {
            this.emit(TelemetryEvent.MODULE_ERROR, {
              module: 'analytics',
              error
            });
          });
        }
      });
    }
  }

  /**
   * Retry a pipeline stage on failure.
   * @param {Function} fn - Async function to execute
   * @param {string} stage - Pipeline stage name
   * @returns {Promise<*>}
   * @private
   */
  async _retryStage(fn, stage) {
    let lastError;
    for (let attempt = 0; attempt <= this.config.pipelineRetryAttempts; attempt++) {
      try {
        return await fn();
      } catch (error) {
        lastError = error;
        if (attempt < this.config.pipelineRetryAttempts) {
          await new Promise(r => setTimeout(r, this.config.pipelineRetryDelayMs * (attempt + 1)));
        }
      }
    }
    throw new PipelineError(stage, `Failed after ${this.config.pipelineRetryAttempts + 1} attempts`, lastError);
  }

  /**
   * Handle module error events.
   * @param {string} moduleName - Name of the module that errored
   * @param {Error} error - The error
   * @private
   */
  _handleModuleError(moduleName, error) {
    this.emit(TelemetryEvent.MODULE_ERROR, { module: moduleName, error });
  }

  /**
   * Periodic health check handler.
   * @private
   */
  _handleHealthCheck() {
    const health = {
      state: this._state,
      activePipelines: this._activePipelines.size,
      modules: {},
      stats: this._stats,
      timestamp: Date.now()
    };

    // Check module health
    for (const [name, mod] of this._modules) {
      health.modules[name] = typeof mod.getHealth === 'function' ? mod.getHealth() : { status: 'unknown' };
    }

    // Alert on high pipeline latency
    if (this._stats.avgPipelineLatencyMs > this.config.maxPipelineLatencyMs) {
      this.emit(TelemetryEvent.ALERT, {
        level: 'warning',
        type: 'HIGH_LATENCY',
        message: `Average pipeline latency ${this._stats.avgPipelineLatencyMs.toFixed(1)}ms exceeds threshold ${this.config.maxPipelineLatencyMs}ms`,
        timestamp: Date.now()
      });
    }

    this._stats.lastHealthCheck = Date.now();
    this.emit(TelemetryEvent.HEALTH_CHECK, health);
  }

  /**
   * Periodic flush handler for buffered data.
   * @private
   */
  _handleFlush() {
    const storage = this._modules.get('storage');
    if (storage && typeof storage.flush === 'function') {
      storage.flush().catch((error) => {
        this.emit(TelemetryEvent.MODULE_ERROR, {
          module: 'storage',
          error: new TelemetryError('Flush failed', 'FLUSH_ERROR', error)
        });
      });
    }
  }

  /**
   * Update the rolling average pipeline latency.
   * @param {number} latencyMs - Latest pipeline latency in ms
   * @private
   */
  _updateAvgLatency(latencyMs) {
    const totalProcessed = this._stats.totalReceived;
    this._stats.avgPipelineLatencyMs =
      (this._stats.avgPipelineLatencyMs * (totalProcessed - 1) + latencyMs) / totalProcessed;
  }

  /**
   * Transition to a new lifecycle state with validation.
   * @param {string} newState - Target state
   * @private
   * @throws {TelemetryError} If state transition is invalid
   */
  _setState(newState) {
    if (!VALID_STATES.has(newState)) {
      throw new TelemetryError(`Invalid state: ${newState}`, 'INVALID_STATE');
    }
    const oldState = this._state;
    this._state = newState;
    this.emit(TelemetryEvent.STATE_CHANGED, { from: oldState, to: newState, timestamp: Date.now() });
  }

  /**
   * Clear all interval timers.
   * @private
   */
  _clearTimers() {
    if (this._healthCheckTimer) {
      clearInterval(this._healthCheckTimer);
      this._healthCheckTimer = null;
    }
    if (this._flushTimer) {
      clearInterval(this._flushTimer);
      this._flushTimer = null;
    }
  }

  /**
   * Deep merge user config with defaults.
   * @param {object} userConfig - User-provided configuration
   * @returns {object} Merged configuration
   * @private
   */
  _mergeConfig(userConfig) {
    const merged = { ...DEFAULT_CONFIG, ...userConfig };
    merged.modules = { ...DEFAULT_CONFIG.modules, ...(userConfig.modules || {}) };
    for (const key of Object.keys(DEFAULT_CONFIG.modules)) {
      merged.modules[key] = { ...DEFAULT_CONFIG.modules[key], ...(userConfig.modules?.[key] || {}) };
    }
    return merged;
  }
}

export default TelemetryManager;
