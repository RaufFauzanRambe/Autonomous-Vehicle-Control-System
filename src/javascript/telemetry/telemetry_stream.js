/**
 * @module telemetry_stream
 * @description Streaming module for the Autonomous Vehicle Control System telemetry.
 * Provides Transform streams for telemetry data with backpressure handling,
 * data sampling, rate limiting, stream multiplexing, and replay from storage.
 *
 * @author Autonomous Vehicle Control System
 * @version 2.0.0
 * @license MIT
 */

import { Transform, PassThrough, Readable } from 'stream';
import { EventEmitter } from 'events';

// ─────────────────────────────────────────────────────────────────────────────
// Custom Error Classes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Error thrown by the telemetry stream module.
 * @extends Error
 */
export class StreamError extends Error {
  /**
   * @param {string} message - Error description
   * @param {string} [code='STREAM_ERROR'] - Machine-readable error code
   */
  constructor(message, code = 'STREAM_ERROR') {
    super(message);
    this.name = 'StreamError';
    this.code = code;
    this.timestamp = Date.now();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** @type {object} Default stream configuration */
const DEFAULT_CONFIG = {
  /** High-water mark for Transform streams (bytes) */
  highWaterMark: 16384,
  /** Default sampling rate (1 = every sample, 10 = every 10th) */
  defaultSamplingRate: 1,
  /** Default rate limit (messages per second, 0 = unlimited) */
  defaultRateLimit: 0,
  /** Maximum multiplexed streams */
  maxMultiplexedStreams: 16,
  /** Object mode for streams */
  objectMode: true,
  /** Enable backpressure logging */
  backpressureLogging: false
};

// ─────────────────────────────────────────────────────────────────────────────
// TelemetryTransform
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Transform stream that processes telemetry data with sampling, rate limiting,
 * and data enrichment.
 *
 * @extends Transform
 *
 * @example
 * const transform = new TelemetryTransform({ samplingRate: 5 });
 * transform.on('data', (chunk) => console.log(chunk));
 * source.pipe(transform).pipe(destination);
 */
export class TelemetryTransform extends Transform {
  /**
   * @param {object} [options={}] - Transform options
   * @param {number} [options.samplingRate=1] - Pass every Nth sample
   * @param {number} [options.rateLimit=0] - Max messages per second (0 = unlimited)
   * @param {Function} [options.transformFn] - Custom transform function
   * @param {boolean} [options.objectMode=true] - Object mode
   */
  constructor(options = {}) {
    super({
      objectMode: options.objectMode !== false,
      highWaterMark: options.highWaterMark || DEFAULT_CONFIG.highWaterMark
    });

    /** @type {number} Sampling rate */
    this.samplingRate = options.samplingRate || DEFAULT_CONFIG.defaultSamplingRate;

    /** @type {number} Rate limit (messages/sec) */
    this.rateLimit = options.rateLimit || DEFAULT_CONFIG.defaultRateLimit;

    /** @type {Function|null} Custom transform function */
    this.transformFn = options.transformFn || null;

    /** @type {number} Sample counter */
    this._sampleCounter = 0;

    /** @type {number} Rate limit token bucket */
    this._rateTokens = this.rateLimit;

    /** @type {number} Last rate limit refill time */
    this._lastRefillTime = Date.now();

    /** @type {object} Stream statistics */
    this._stats = {
      received: 0,
      passed: 0,
      sampled: 0,
      rateLimited: 0,
      errors: 0,
      backpressureCount: 0
    };
  }

  /**
   * Transform implementation with sampling and rate limiting.
   * @param {*} chunk - Input chunk
   * @param {string} encoding - Chunk encoding
   * @param {Function} callback - Transform callback
   * @private
   */
  _transform(chunk, encoding, callback) {
    this._stats.received++;

    try {
      // Sampling: pass every Nth item
      this._sampleCounter++;
      if (this.samplingRate > 1 && this._sampleCounter % this.samplingRate !== 0) {
        this._stats.sampled++;
        callback();
        return;
      }

      // Rate limiting: token bucket algorithm
      if (this.rateLimit > 0) {
        this._refillRateTokens();
        if (this._rateTokens < 1) {
          this._stats.rateLimited++;
          callback();
          return;
        }
        this._rateTokens--;
      }

      // Apply custom transform or pass through
      let result = chunk;
      if (this.transformFn) {
        result = this.transformFn(chunk);
        if (result === null || result === undefined) {
          callback();
          return;
        }
      }

      this._stats.passed++;
      this.push(result);
      callback();
    } catch (error) {
      this._stats.errors++;
      callback(new StreamError(`Transform error: ${error.message}`, 'TRANSFORM_ERROR'));
    }
  }

  /**
   * Refill rate limit tokens based on elapsed time.
   * @private
   */
  _refillRateTokens() {
    const now = Date.now();
    const elapsed = (now - this._lastRefillTime) / 1000;
    this._rateTokens = Math.min(this.rateLimit, this._rateTokens + elapsed * this.rateLimit);
    this._lastRefillTime = now;
  }

  /**
   * Get stream statistics.
   * @returns {object}
   */
  getStats() {
    return { ...this._stats };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// DataSampler
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Sampling transform that reduces data frequency using various strategies.
 */
export class DataSampler extends Transform {
  /**
   * @param {object} [options={}] - Sampling options
   * @param {string} [options.strategy='uniform'] - Sampling strategy: 'uniform', 'random', 'change', 'adaptive'
   * @param {number} [options.rate=1] - Sampling rate (for uniform: every Nth, for random: probability 1/N)
   * @param {string} [options.changeField] - Field to monitor for change-based sampling
   * @param {number} [options.changeThreshold=0.01] - Minimum change to pass through
   * @param {number} [options.adaptiveMinRate=1] - Minimum rate for adaptive sampling
   * @param {number} [options.adaptiveMaxRate=100] - Maximum rate for adaptive sampling
   */
  constructor(options = {}) {
    super({ objectMode: true });

    /** @type {string} */
    this.strategy = options.strategy || 'uniform';
    /** @type {number} */
    this.rate = options.rate || 1;
    /** @type {string|null} */
    this.changeField = options.changeField || null;
    /** @type {number} */
    this.changeThreshold = options.changeThreshold || 0.01;
    /** @type {number} */
    this.adaptiveMinRate = options.adaptiveMinRate || 1;
    /** @type {number} */
    this.adaptiveMaxRate = options.adaptiveMaxRate || 100;

    /** @type {number} */
    this._counter = 0;
    /** @type {*} Last value for change detection */
    this._lastValue = null;
    /** @type {number} Current adaptive rate */
    this._adaptiveRate = this.adaptiveMinRate;
    /** @type {number} Variance accumulator for adaptive */
    this._varianceAccum = 0;
  }

  /**
   * @param {*} chunk - Input data
   * @param {string} encoding - Encoding
   * @param {Function} callback - Callback
   * @private
   */
  _transform(chunk, encoding, callback) {
    this._counter++;

    let pass = false;

    switch (this.strategy) {
      case 'uniform':
        pass = this._counter % this.rate === 0;
        break;

      case 'random':
        pass = Math.random() < (1 / this.rate);
        break;

      case 'change': {
        const value = this._getFieldValue(chunk, this.changeField);
        if (this._lastValue === null || Math.abs(value - this._lastValue) >= this.changeThreshold) {
          pass = true;
          this._lastValue = value;
        }
        break;
      }

      case 'adaptive': {
        const value = this._getFieldValue(chunk, this.changeField);
        if (value !== null) {
          if (this._lastValue !== null) {
            const delta = Math.abs(value - this._lastValue);
            this._varianceAccum = this._varianceAccum * 0.95 + delta * 0.05;
            // High variance → lower rate (more frequent sampling)
            // Low variance → higher rate (less frequent sampling)
            this._adaptiveRate = Math.round(
              this.adaptiveMinRate + (1 - Math.min(this._varianceAccum / this.changeThreshold, 1)) *
              (this.adaptiveMaxRate - this.adaptiveMinRate)
            );
            this._adaptiveRate = Math.max(this.adaptiveMinRate, Math.min(this.adaptiveMaxRate, this._adaptiveRate));
          }
          this._lastValue = value;
        }
        pass = this._counter % Math.max(1, this._adaptiveRate) === 0;
        break;
      }

      default:
        pass = true;
    }

    if (pass) {
      this.push({ ...chunk, _sampled: true, _sampleStrategy: this.strategy });
    }
    callback();
  }

  /**
   * Extract a field value from a data object.
   * @param {object} obj - Data object
   * @param {string} path - Dot-notation field path
   * @returns {*}
   * @private
   */
  _getFieldValue(obj, path) {
    if (!path || !obj) return null;
    const parts = path.split('.');
    let current = obj;
    for (const part of parts) {
      if (current === null || current === undefined) return null;
      current = current[part];
    }
    return typeof current === 'number' ? current : null;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// StreamMultiplexer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Merges multiple sensor streams into a single output stream with
 * time-ordered interleaving and backpressure coordination.
 *
 * @extends Readable
 */
export class StreamMultiplexer extends Readable {
  /**
   * @param {object} [options={}] - Multiplexer options
   * @param {number} [options.maxStreams=16] - Maximum input streams
   * @param {number} [options.highWaterMark=16384] - Buffer size
   */
  constructor(options = {}) {
    super({ objectMode: true, highWaterMark: options.highWaterMark || 16384 });

    /** @type {number} */
    this.maxStreams = options.maxStreams || DEFAULT_CONFIG.maxMultiplexedStreams;

    /** @type {Map<string, {stream:Readable, buffer:Array, ended:boolean}>} */
    this._inputs = new Map();

    /** @type {boolean} */
    this._destroyed = false;

    /** @type {object} */
    this._stats = {
      totalMerged: 0,
      activeStreams: 0,
      bufferOverflows: 0
    };
  }

  /**
   * Add an input stream to the multiplexer.
   * @param {string} name - Stream identifier
   * @param {Readable} stream - Input readable stream
   * @throws {StreamError} If max streams exceeded
   */
  addInput(name, stream) {
    if (this._inputs.size >= this.maxStreams) {
      throw new StreamError(`Maximum streams (${this.maxStreams}) exceeded`, 'MAX_STREAMS');
    }

    const entry = { stream, buffer: [], ended: false };
    this._inputs.set(name, entry);
    this._stats.activeStreams++;

    stream.on('data', (chunk) => {
      entry.buffer.push({ ...chunk, _source: name, _mergedAt: Date.now() });
      this._tryPush();
    });

    stream.on('end', () => {
      entry.ended = true;
      this._checkAllEnded();
    });

    stream.on('error', (error) => {
      this.emit('error', new StreamError(`Input stream '${name}' error: ${error.message}`, 'INPUT_ERROR'));
    });
  }

  /**
   * Remove an input stream.
   * @param {string} name - Stream identifier
   */
  removeInput(name) {
    const entry = this._inputs.get(name);
    if (entry) {
      entry.stream.destroy();
      this._inputs.delete(name);
      this._stats.activeStreams--;
    }
  }

  /**
   * Read implementation for merged stream.
   * @private
   */
  _read() {
    this._tryPush();
  }

  /**
   * Try to push the oldest buffered item.
   * @private
   */
  _tryPush() {
    if (this._destroyed) return;

    // Find the oldest item across all input buffers
    let oldest = null;
    let oldestTime = Infinity;
    let oldestName = null;

    for (const [name, entry] of this._inputs) {
      if (entry.buffer.length > 0) {
        const item = entry.buffer[0];
        const time = item.timestamp || item._mergedAt || 0;
        if (time < oldestTime) {
          oldestTime = time;
          oldest = item;
          oldestName = name;
        }
      }
    }

    if (oldest) {
      this._inputs.get(oldestName).buffer.shift();
      this._stats.totalMerged++;
      this.push(oldest);
    }
  }

  /**
   * Check if all input streams have ended.
   * @private
   */
  _checkAllEnded() {
    const allEnded = Array.from(this._inputs.values()).every(entry => entry.ended && entry.buffer.length === 0);
    if (allEnded) {
      this.push(null);
    }
  }

  /**
   * Destroy the multiplexer and all input streams.
   * @param {Error} [err] - Optional error
   */
  destroy(err) {
    if (this._destroyed) return;
    this._destroyed = true;

    for (const [name, entry] of this._inputs) {
      try {
        entry.stream.destroy();
      } catch (_) { /* ignore */ }
    }
    this._inputs.clear();
    this._stats.activeStreams = 0;

    super.destroy(err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// StorageReplay
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Reads stored telemetry data and replays it as a readable stream,
 * optionally at original or accelerated timing.
 *
 * @extends Readable
 */
export class StorageReplay extends Readable {
  /**
   * @param {object} storage - TelemetryStorage instance
   * @param {object} [options={}] - Replay options
   * @param {string} [options.sensorType] - Sensor type to replay
   * @param {number} [options.startTime] - Start time for replay range
   * @param {number} [options.endTime] - End time for replay range
   * @param {number} [options.speed=1.0] - Replay speed multiplier (1 = real-time, 10 = 10x)
   * @param {boolean} [options.realtime=false] - Use real-time pacing
   * @param {number} [options.highWaterMark=16384] - Buffer size
   */
  constructor(storage, options = {}) {
    super({ objectMode: true, highWaterMark: options.highWaterMark || 16384 });

    /** @type {object} */
    this._storage = storage;

    /** @type {number} */
    this._speed = options.speed || 1.0;

    /** @type {boolean} */
    this._realtime = options.realtime || false;

    /** @type {object} */
    this._query = {
      sensorType: options.sensorType,
      startTime: options.startTime,
      endTime: options.endTime,
      limit: 100000,
      order: 'asc'
    };

    /** @type {Array<object>} Replay data */
    this._data = [];

    /** @type {number} Current replay index */
    this._index = 0;

    /** @type {NodeJS.Timer|null} */
    this._replayTimer = null;

    /** @type {boolean} */
    this._destroyed = false;

    /** @type {object} */
    this._stats = { replayed: 0, skipped: 0 };
  }

  /**
   * Start the replay process.
   * @returns {Promise<void>}
   */
  async start() {
    this._data = await this._storage.query(this._query);

    if (this._data.length === 0) {
      this.push(null);
      return;
    }

    if (this._realtime) {
      this._scheduleNext();
    } else {
      // Burst mode: push all data as fast as possible
      this._pushAll();
    }
  }

  /**
   * Read implementation.
   * @private
   */
  _read() {
    // In burst mode, data is pushed by _pushAll
    // In realtime mode, data is pushed by _scheduleNext
  }

  /**
   * Push all data immediately (burst mode).
   * @private
   */
  _pushAll() {
    while (this._index < this._data.length && !this._destroyed) {
      const item = this._data[this._index++];
      if (!this.push({ ...item, _replayed: true, _replayIndex: this._index })) {
        // Backpressure: schedule resume
        setImmediate(() => this._pushAll());
        return;
      }
      this._stats.replayed++;
    }

    if (this._index >= this._data.length) {
      this.push(null);
    }
  }

  /**
   * Schedule the next data point with real-time pacing.
   * @private
   */
  _scheduleNext() {
    if (this._destroyed || this._index >= this._data.length) {
      this.push(null);
      return;
    }

    const current = this._data[this._index];

    if (this._index === 0) {
      // Push first item immediately
      this.push({ ...current, _replayed: true, _replayIndex: this._index });
      this._stats.replayed++;
      this._index++;
      this._scheduleNext();
      return;
    }

    const prev = this._data[this._index - 1];
    const delay = ((current.timestamp - prev.timestamp) / this._speed);

    if (delay <= 0 || !isFinite(delay)) {
      // Same timestamp or invalid, push immediately
      this.push({ ...current, _replayed: true, _replayIndex: this._index });
      this._stats.replayed++;
      this._index++;
      this._scheduleNext();
      return;
    }

    this._replayTimer = setTimeout(() => {
      if (this._destroyed) return;
      this.push({ ...current, _replayed: true, _replayIndex: this._index });
      this._stats.replayed++;
      this._index++;
      this._scheduleNext();
    }, delay);
  }

  /**
   * Destroy the replay stream.
   * @param {Error} [err] - Optional error
   */
  destroy(err) {
    if (this._destroyed) return;
    this._destroyed = true;

    if (this._replayTimer) {
      clearTimeout(this._replayTimer);
      this._replayTimer = null;
    }

    super.destroy(err);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// TelemetryStream (Main Module)
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Main telemetry stream module. Creates and manages transform pipelines
 * for telemetry data with backpressure handling, sampling, rate limiting,
 * stream multiplexing, and replay capabilities.
 *
 * @extends EventEmitter
 *
 * @example
 * const stream = new TelemetryStream({ highWaterMark: 32768 });
 * stream.addPipeline('gps', { samplingRate: 5, rateLimit: 100 });
 * stream.addPipeline('imu', { samplingRate: 10 });
 * stream.push({ type: 'gps_nmea', data: {...} });
 */
export class TelemetryStream extends EventEmitter {
  /**
   * @param {object} [config={}] - Stream configuration
   */
  constructor(config = {}) {
    super();

    /** @type {object} */
    this.config = { ...DEFAULT_CONFIG, ...config };

    /** @type {Map<string, TelemetryTransform>} Named transform pipelines */
    this._pipelines = new Map();

    /** @type {StreamMultiplexer|null} */
    this._multiplexer = new StreamMultiplexer({
      maxStreams: this.config.maxMultiplexedStreams,
      highWaterMark: this.config.highWaterMark
    });

    /** @type {PassThrough} Main output pass-through */
    this._output = new PassThrough({ objectMode: this.config.objectMode });

    /** @type {object} */
    this._stats = {
      totalPushed: 0,
      totalPassed: 0,
      totalDropped: 0
    };

    // Wire multiplexer output to main output
    this._multiplexer.on('data', (data) => {
      this._stats.totalPassed++;
      this._output.push(data);
      this.emit('data', data);
    });

    this._multiplexer.on('error', (error) => {
      this.emit('error', error);
    });
  }

  /**
   * Add a named transform pipeline.
   * @param {string} name - Pipeline name (typically sensor type)
   * @param {object} [options={}] - Transform options
   * @returns {TelemetryTransform} The created transform
   */
  addPipeline(name, options = {}) {
    const transform = new TelemetryTransform({
      ...options,
      objectMode: this.config.objectMode,
      highWaterMark: this.config.highWaterMark
    });

    this._pipelines.set(name, transform);

    // Connect to multiplexer
    this._multiplexer.addInput(name, transform);

    this.emit('pipeline:added', { name });
    return transform;
  }

  /**
   * Remove a named pipeline.
   * @param {string} name - Pipeline name
   */
  removePipeline(name) {
    const transform = this._pipelines.get(name);
    if (transform) {
      this._multiplexer.removeInput(name);
      transform.destroy();
      this._pipelines.delete(name);
      this.emit('pipeline:removed', { name });
    }
  }

  /**
   * Push data into the appropriate pipeline based on data type.
   * @param {object} data - Telemetry data with a `type` field
   * @returns {boolean} True if data was accepted
   */
  push(data) {
    this._stats.totalPushed++;

    const typeName = data.type || data.sensorType || 'default';
    let pipeline = this._pipelines.get(typeName);

    // Auto-create pipeline if not exists
    if (!pipeline) {
      pipeline = this.addPipeline(typeName);
    }

    if (pipeline.write(data)) {
      return true;
    }

    this._stats.totalDropped++;
    return false;
  }

  /**
   * Get the main output stream (for piping to consumers).
   * @returns {PassThrough}
   */
  getOutputStream() {
    return this._output;
  }

  /**
   * Create a replay stream from stored data.
   * @param {object} storage - TelemetryStorage instance
   * @param {object} [options={}] - Replay options
   * @returns {StorageReplay}
   */
  createReplay(storage, options = {}) {
    return new StorageReplay(storage, options);
  }

  /**
   * Get stream statistics.
   * @returns {object}
   */
  getStats() {
    const pipelineStats = {};
    for (const [name, pipeline] of this._pipelines) {
      pipelineStats[name] = pipeline.getStats();
    }
    return {
      ...this._stats,
      pipelines: pipelineStats,
      activePipelines: this._pipelines.size
    };
  }

  /**
   * Get health status.
   * @returns {object}
   */
  getHealth() {
    const dropRate = this._stats.totalPushed > 0
      ? this._stats.totalDropped / this._stats.totalPushed
      : 0;
    return {
      status: dropRate < 0.01 ? 'healthy' : dropRate < 0.1 ? 'degraded' : 'unhealthy',
      dropRate,
      activePipelines: this._pipelines.size
    };
  }

  /**
   * Destroy all streams and pipelines.
   */
  destroy() {
    for (const [name, pipeline] of this._pipelines) {
      try { pipeline.destroy(); } catch (_) { /* ignore */ }
    }
    this._pipelines.clear();

    try { this._multiplexer.destroy(); } catch (_) { /* ignore */ }
    try { this._output.destroy(); } catch (_) { /* ignore */ }

    this.emit('destroyed');
    this.removeAllListeners();
  }
}

export default TelemetryStream;
