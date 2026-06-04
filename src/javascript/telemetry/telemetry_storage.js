/**
 * @module telemetry_storage
 * @description Storage layer for the Autonomous Vehicle Control System telemetry.
 * Provides in-memory ring buffer with configurable size, time-series data storage,
 * delta encoding compression, query interface, export capabilities, and cleanup policy.
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
 * Error thrown by the telemetry storage module.
 * @extends Error
 */
export class StorageError extends Error {
  /**
   * @param {string} message - Error description
   * @param {string} [code='STORAGE_ERROR'] - Machine-readable error code
   */
  constructor(message, code = 'STORAGE_ERROR') {
    super(message);
    this.name = 'StorageError';
    this.code = code;
    this.timestamp = Date.now();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Ring Buffer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Fixed-size circular buffer implementation for efficient data rotation.
 * When the buffer is full, oldest entries are automatically overwritten.
 *
 * @template T
 */
class RingBuffer {
  /**
   * @param {number} capacity - Maximum number of items the buffer can hold
   */
  constructor(capacity) {
    /** @type {number} */
    this.capacity = capacity;
    /** @type {Array<T|null>} */
    this._buffer = new Array(capacity).fill(null);
    /** @type {number} Write head position */
    this._head = 0;
    /** @type {number} Current item count */
    this._size = 0;
    /** @type {number} Total items ever written (for indexing) */
    this._totalWritten = 0;
  }

  /**
   * Push an item into the ring buffer.
   * @param {T} item - Item to add
   * @returns {T|null} The evicted item if buffer was full, null otherwise
   */
  push(item) {
    const evicted = this._buffer[this._head];
    this._buffer[this._head] = item;
    this._head = (this._head + 1) % this.capacity;
    if (this._size < this.capacity) {
      this._size++;
    }
    this._totalWritten++;
    return evicted;
  }

  /**
   * Get the item at a relative index (0 = oldest, size-1 = newest).
   * @param {number} index - Relative index
   * @returns {T|null}
   */
  get(index) {
    if (index < 0 || index >= this._size) return null;
    const actualIndex = (this._head - this._size + index + this.capacity) % this.capacity;
    return this._buffer[actualIndex];
  }

  /**
   * Get the most recently added item.
   * @returns {T|null}
   */
  latest() {
    if (this._size === 0) return null;
    return this.get(this._size - 1);
  }

  /**
   * Get all items in order (oldest first).
   * @returns {T[]}
   */
  toArray() {
    const result = [];
    for (let i = 0; i < this._size; i++) {
      result.push(this.get(i));
    }
    return result;
  }

  /**
   * Get items within a relative range.
   * @param {number} start - Start index (inclusive)
   * @param {number} end - End index (exclusive)
   * @returns {T[]}
   */
  slice(start, end) {
    const result = [];
    const actualEnd = Math.min(end, this._size);
    for (let i = start; i < actualEnd; i++) {
      const item = this.get(i);
      if (item !== null) result.push(item);
    }
    return result;
  }

  /**
   * Clear the buffer.
   */
  clear() {
    this._buffer = new Array(this.capacity).fill(null);
    this._head = 0;
    this._size = 0;
  }

  /** @type {number} Current number of items */
  get size() { return this._size; }

  /** @type {number} Total items ever written */
  get totalWritten() { return this._totalWritten; }

  /** @type {boolean} Whether buffer is at capacity */
  get isFull() { return this._size >= this.capacity; }
}

// ─────────────────────────────────────────────────────────────────────────────
// Delta Encoder
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Delta encoding compressor for numeric time-series data.
 * Stores only the difference between consecutive values, reducing
 * memory usage for slowly-changing telemetry data.
 */
class DeltaEncoder {
  constructor() {
    /** @type {Map<string, {prev:number, base:number, baseTime:number}>} */
    this._state = new Map();
  }

  /**
   * Encode a numeric value using delta compression.
   * @param {string} key - Metric key
   * @param {number} value - Current value
   * @param {number} timestamp - Current timestamp
   * @returns {{delta:number, base:number, isBase:boolean}}
   */
  encode(key, value, timestamp) {
    const state = this._state.get(key);

    if (!state) {
      this._state.set(key, { prev: value, base: value, baseTime: timestamp });
      return { delta: value, base: value, isBase: true };
    }

    const delta = value - state.prev;
    state.prev = value;

    // Reset base periodically to prevent drift accumulation
    if (Math.abs(delta) > Math.abs(state.base) * 0.5 || timestamp - state.baseTime > 60000) {
      state.base = value;
      state.baseTime = timestamp;
      return { delta: value, base: value, isBase: true };
    }

    return { delta, base: state.base, isBase: false };
  }

  /**
   * Decode a delta-encoded value.
   * @param {string} key - Metric key
   * @param {number} delta - Delta or absolute value
   * @param {number} base - Base value
   * @param {boolean} isBase - Whether this is a base (absolute) value
   * @returns {number} Decoded absolute value
   */
  decode(key, delta, base, isBase) {
    if (isBase) {
      this._state.set(key, { prev: delta, base: delta, baseTime: Date.now() });
      return delta;
    }
    const state = this._state.get(key);
    if (!state) return base + delta;
    const value = state.prev + delta;
    state.prev = value;
    return value;
  }

  /**
   * Reset encoder state.
   */
  reset() {
    this._state.clear();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Time-Series Entry
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Represents a single time-series data point with metadata.
 */
class TimeSeriesEntry {
  /**
   * @param {string} sensorType - Source sensor type
   * @param {object} data - Telemetry data
   * @param {number} timestamp - Data timestamp (ms epoch)
   */
  constructor(sensorType, data, timestamp) {
    /** @type {string} */
    this.sensorType = sensorType;
    /** @type {object} */
    this.data = data;
    /** @type {number} */
    this.timestamp = timestamp;
    /** @type {number} */
    this.storedAt = Date.now();
    /** @type {number} Entry size estimate in bytes */
    this.size = JSON.stringify(data).length + 64; // overhead estimate
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// TelemetryStorage
// ─────────────────────────────────────────────────────────────────────────────

/**
 * In-memory telemetry storage with ring buffers, delta encoding compression,
 * time-range queries, aggregation, and export capabilities.
 *
 * @extends EventEmitter
 *
 * @example
 * const storage = new TelemetryStorage({ maxBufferSize: 50000 });
 * await storage.initialize();
 * await storage.write({ type: 'gps_nmea', data: { lat: 37.77, lng: -122.41 } });
 * const recent = await storage.query({ sensorType: 'gps_nmea', limit: 10 });
 * const csv = storage.exportCSV('gps_nmea');
 */
export class TelemetryStorage extends EventEmitter {
  /**
   * @param {object} [config={}] - Configuration
   * @param {number} [config.maxBufferSize=100000] - Maximum entries per sensor type
   * @param {boolean} [config.deltaEncoding=true] - Enable delta encoding compression
   * @param {number} [config.cleanupIntervalMs=30000] - Cleanup interval
   * @param {number} [config.maxAgeMs=3600000] - Maximum data age (1 hour)
   * @param {number} [config.maxTotalMemoryBytes=536870912] - Max memory (512MB)
   */
  constructor(config = {}) {
    super();

    /** @type {object} */
    this.config = {
      maxBufferSize: config.maxBufferSize || 100000,
      deltaEncoding: config.deltaEncoding !== false,
      cleanupIntervalMs: config.cleanupIntervalMs || 30000,
      maxAgeMs: config.maxAgeMs || 3600000,
      maxTotalMemoryBytes: config.maxTotalMemoryBytes || 536870912
    };

    /** @type {Map<string, RingBuffer<TimeSeriesEntry>>} Per-sensor ring buffers */
    this._buffers = new Map();

    /** @type {DeltaEncoder} */
    this._deltaEncoder = new DeltaEncoder();

    /** @type {NodeJS.Timer|null} */
    this._cleanupTimer = null;

    /** @type {number} Estimated total memory usage in bytes */
    this._totalMemoryBytes = 0;

    /** @type {object} Storage statistics */
    this._stats = {
      totalWrites: 0,
      totalReads: 0,
      totalEvictions: 0,
      compressionRatio: 1.0,
      bySensorType: {}
    };
  }

  // ── Lifecycle ───────────────────────────────────────────────────────────

  /**
   * Initialize the storage layer.
   * @returns {Promise<void>}
   */
  async initialize() {
    this._startCleanup();
    this.emit('initialized');
  }

  /**
   * Flush any buffered data and persist (in-memory, this is a no-op but
   * exists for API consistency with persistent backends).
   * @returns {Promise<void>}
   */
  async flush() {
    this.emit('flushed', { totalEntries: this._getTotalEntries() });
  }

  /**
   * Close storage and release resources.
   */
  close() {
    if (this._cleanupTimer) {
      clearInterval(this._cleanupTimer);
      this._cleanupTimer = null;
    }
    this._buffers.clear();
    this._deltaEncoder.reset();
    this.emit('closed');
  }

  // ── Write Operations ────────────────────────────────────────────────────

  /**
   * Write a telemetry data point to storage.
   * @param {object} data - Parsed telemetry data
   * @param {string} [data.type] - Sensor/data type
   * @param {object} [data.data] - Payload data
   * @param {number} [data.timestamp] - Data timestamp
   * @returns {Promise<void>}
   */
  async write(data) {
    const sensorType = data.type || data.sensorType || 'unknown';
    const timestamp = data.timestamp || data.data?.timestamp || Date.now();
    const payload = data.data || data;

    // Get or create ring buffer for this sensor type
    if (!this._buffers.has(sensorType)) {
      this._buffers.set(sensorType, new RingBuffer(this.config.maxBufferSize));
      this._stats.bySensorType[sensorType] = { writes: 0, reads: 0, evictions: 0 };
    }

    const buffer = this._buffers.get(sensorType);
    const stats = this._stats.bySensorType[sensorType];

    // Apply delta encoding if enabled for numeric fields
    let compressedPayload = payload;
    if (this.config.deltaEncoding && typeof payload === 'object') {
      compressedPayload = this._compressPayload(payload, sensorType, timestamp);
    }

    // Create entry
    const entry = new TimeSeriesEntry(sensorType, compressedPayload, timestamp);

    // Write to ring buffer
    const evicted = buffer.push(entry);
    if (evicted) {
      this._totalMemoryBytes -= evicted.size;
      stats.evictions++;
      this._stats.totalEvictions++;
    }

    this._totalMemoryBytes += entry.size;
    this._stats.totalWrites++;
    stats.writes++;

    // Check memory limits
    if (this._totalMemoryBytes > this.config.maxTotalMemoryBytes) {
      this._enforceMemoryLimit();
    }

    this.emit('write', { sensorType, timestamp, size: entry.size });
  }

  /**
   * Write multiple data points in batch.
   * @param {object[]} dataList - Array of data points
   * @returns {Promise<number>} Number of items written
   */
  async writeBatch(dataList) {
    let count = 0;
    for (const data of dataList) {
      await this.write(data);
      count++;
    }
    return count;
  }

  // ── Read Operations ─────────────────────────────────────────────────────

  /**
   * Query stored telemetry data.
   * @param {object} [query={}] - Query parameters
   * @param {string} [query.sensorType] - Filter by sensor type
   * @param {number} [query.startTime] - Start of time range (ms epoch)
   * @param {number} [query.endTime] - End of time range (ms epoch)
   * @param {number} [query.limit=1000] - Maximum results
   * @param {string} [query.order='desc'] - Sort order ('asc' or 'desc')
   * @returns {Promise<object[]>} Matching data entries
   */
  async query(query = {}) {
    const {
      sensorType,
      startTime = 0,
      endTime = Date.now(),
      limit = 1000,
      order = 'desc'
    } = query;

    this._stats.totalReads++;

    if (sensorType) {
      return this._querySensor(sensorType, startTime, endTime, limit, order);
    }

    // Query across all sensor types
    const results = [];
    for (const [type] of this._buffers) {
      const entries = this._querySensor(type, startTime, endTime, limit, order);
      results.push(...entries);
      if (results.length >= limit) break;
    }

    // Sort by timestamp
    results.sort((a, b) => order === 'desc'
      ? b.timestamp - a.timestamp
      : a.timestamp - b.timestamp
    );

    return results.slice(0, limit);
  }

  /**
   * Get the latest data point for a sensor type.
   * @param {string} sensorType - Sensor type
   * @returns {object|null}
   */
  getLatest(sensorType) {
    const buffer = this._buffers.get(sensorType);
    if (!buffer) return null;

    const entry = buffer.latest();
    if (!entry) return null;

    this._stats.totalReads++;
    const stats = this._stats.bySensorType[sensorType];
    if (stats) stats.reads++;

    return this._decompressEntry(entry);
  }

  /**
   * Get aggregate statistics for a sensor type within a time range.
   * @param {object} query - Query parameters
   * @param {string} query.sensorType - Sensor type
   * @param {string} query.field - Numeric field path (e.g., 'accelerometer.x')
   * @param {number} [query.startTime] - Start time
   * @param {number} [query.endTime] - End time
   * @returns {Promise<{min:number, max:number, avg:number, count:number, sum:number, stdDev:number}>}
   */
  async aggregate(query) {
    const { sensorType, field, startTime = 0, endTime = Date.now() } = query;

    const entries = await this.query({ sensorType, startTime, endTime, limit: 100000, order: 'asc' });
    const values = [];

    for (const entry of entries) {
      const value = this._getNestedValue(entry.data, field);
      if (typeof value === 'number' && !isNaN(value)) {
        values.push(value);
      }
    }

    if (values.length === 0) {
      return { min: 0, max: 0, avg: 0, count: 0, sum: 0, stdDev: 0 };
    }

    const sum = values.reduce((a, b) => a + b, 0);
    const avg = sum / values.length;
    const variance = values.reduce((acc, v) => acc + Math.pow(v - avg, 2), 0) / values.length;
    const stdDev = Math.sqrt(variance);

    return {
      min: Math.min(...values),
      max: Math.max(...values),
      avg,
      count: values.length,
      sum,
      stdDev
    };
  }

  // ── Export ──────────────────────────────────────────────────────────────

  /**
   * Export sensor data as JSON string.
   * @param {string} sensorType - Sensor type to export
   * @param {object} [options={}] - Export options
   * @param {number} [options.startTime] - Start time filter
   * @param {number} [options.endTime] - End time filter
   * @param {number} [options.limit] - Maximum entries
   * @param {boolean} [options.pretty=false] - Pretty-print JSON
   * @returns {string} JSON string
   */
  exportJSON(sensorType, options = {}) {
    const entries = this._querySensor(
      sensorType,
      options.startTime || 0,
      options.endTime || Date.now(),
      options.limit || 100000,
      'asc'
    );

    const exportData = entries.map(entry => ({
      timestamp: entry.timestamp,
      sensorType: entry.sensorType,
      data: entry.data
    }));

    return options.pretty
      ? JSON.stringify(exportData, null, 2)
      : JSON.stringify(exportData);
  }

  /**
   * Export sensor data as CSV string.
   * @param {string} sensorType - Sensor type to export
   * @param {object} [options={}] - Export options
   * @param {string[]} [options.fields] - Fields to include (auto-detected if omitted)
   * @param {number} [options.limit] - Maximum entries
   * @returns {string} CSV string
   */
  exportCSV(sensorType, options = {}) {
    const entries = this._querySensor(
      sensorType,
      options.startTime || 0,
      options.endTime || Date.now(),
      options.limit || 100000,
      'asc'
    );

    if (entries.length === 0) return '';

    // Auto-detect fields from first entry
    const fields = options.fields || this._flattenKeys(entries[0].data);
    const header = ['timestamp', ...fields].join(',');

    const rows = entries.map(entry => {
      const values = fields.map(field => {
        const value = this._getNestedValue(entry.data, field);
        return typeof value === 'number' ? value.toFixed(6) : String(value ?? '');
      });
      return [entry.timestamp, ...values].join(',');
    });

    return [header, ...rows].join('\n');
  }

  // ── Status ──────────────────────────────────────────────────────────────

  /**
   * Get storage statistics.
   * @returns {object}
   */
  getStats() {
    return {
      ...this._stats,
      totalEntries: this._getTotalEntries(),
      sensorTypes: Array.from(this._buffers.keys()),
      memoryUsageBytes: this._totalMemoryBytes,
      memoryUsageMB: (this._totalMemoryBytes / (1024 * 1024)).toFixed(2)
    };
  }

  /**
   * Get health status.
   * @returns {object}
   */
  getHealth() {
    const memoryPercent = this._totalMemoryBytes / this.config.maxTotalMemoryBytes;
    return {
      status: memoryPercent < 0.8 ? 'healthy' : memoryPercent < 0.95 ? 'degraded' : 'critical',
      memoryUsage: memoryPercent,
      sensorTypes: this._buffers.size,
      totalEntries: this._getTotalEntries()
    };
  }

  // ── Private Methods ─────────────────────────────────────────────────────

  /**
   * Query a single sensor type's buffer.
   * @param {string} sensorType - Sensor type
   * @param {number} startTime - Start time
   * @param {number} endTime - End time
   * @param {number} limit - Max results
   * @param {string} order - Sort order
   * @returns {object[]}
   * @private
   */
  _querySensor(sensorType, startTime, endTime, limit, order) {
    const buffer = this._buffers.get(sensorType);
    if (!buffer) return [];

    const stats = this._stats.bySensorType[sensorType];
    if (stats) stats.reads++;

    const allEntries = buffer.toArray();
    const filtered = allEntries.filter(entry =>
      entry.timestamp >= startTime && entry.timestamp <= endTime
    );

    filtered.sort((a, b) => order === 'desc'
      ? b.timestamp - a.timestamp
      : a.timestamp - b.timestamp
    );

    return filtered.slice(0, limit).map(entry => this._decompressEntry(entry));
  }

  /**
   * Compress payload using delta encoding for numeric fields.
   * @param {object} payload - Data payload
   * @param {string} sensorType - Sensor type (used as key prefix)
   * @param {number} timestamp - Timestamp
   * @returns {object} Compressed payload
   * @private
   */
  _compressPayload(payload, sensorType, timestamp) {
    if (typeof payload !== 'object' || payload === null) return payload;

    const compressed = Array.isArray(payload) ? [...payload] : { ...payload };
    let originalSize = 0;
    let compressedSize = 0;

    const compressRecursive = (obj, prefix = '') => {
      for (const key of Object.keys(obj)) {
        const fullKey = `${sensorType}:${prefix}${key}`;
        const value = obj[key];

        if (typeof value === 'number' && !isNaN(value)) {
          originalSize += 8; // 64-bit float
          const encoded = this._deltaEncoder.encode(fullKey, value, timestamp);
          if (!encoded.isBase) {
            obj[key] = { __delta: encoded.delta, __base: encoded.base };
            compressedSize += 8; // smaller delta typically
          } else {
            compressedSize += 8;
          }
        } else if (typeof value === 'object' && value !== null) {
          compressRecursive(value, `${prefix}${key}.`);
        }
      }
    };

    compressRecursive(compressed);

    // Update compression ratio
    if (originalSize > 0 && compressedSize > 0) {
      const newRatio = compressedSize / originalSize;
      this._stats.compressionRatio =
        this._stats.compressionRatio * 0.9 + newRatio * 0.1; // EMA
    }

    return compressed;
  }

  /**
   * Decompress a stored entry.
   * @param {TimeSeriesEntry} entry - Stored entry
   * @returns {object} Decompressed entry
   * @private
   */
  _decompressEntry(entry) {
    if (!this.config.deltaEncoding) {
      return { sensorType: entry.sensorType, timestamp: entry.timestamp, data: entry.data };
    }

    const data = this._decompressPayload(entry.data, entry.sensorType, entry.timestamp);
    return { sensorType: entry.sensorType, timestamp: entry.timestamp, data };
  }

  /**
   * Decompress payload by resolving delta-encoded values.
   * @param {object} payload - Compressed payload
   * @param {string} sensorType - Sensor type
   * @param {number} timestamp - Timestamp
   * @returns {object} Decompressed payload
   * @private
   */
  _decompressPayload(payload, sensorType, timestamp) {
    if (typeof payload !== 'object' || payload === null) return payload;

    const decompressed = Array.isArray(payload) ? [...payload] : { ...payload };

    const decompressRecursive = (obj, prefix = '') => {
      for (const key of Object.keys(obj)) {
        const value = obj[key];
        if (typeof value === 'object' && value !== null && value.__delta !== undefined) {
          const fullKey = `${sensorType}:${prefix}${key}`;
          obj[key] = this._deltaEncoder.decode(fullKey, value.__delta, value.__base, false);
        } else if (typeof value === 'object' && value !== null) {
          decompressRecursive(value, `${prefix}${key}.`);
        }
      }
    };

    decompressRecursive(decompressed);
    return decompressed;
  }

  /**
   * Get the total number of entries across all buffers.
   * @returns {number}
   * @private
   */
  _getTotalEntries() {
    let total = 0;
    for (const buffer of this._buffers.values()) {
      total += buffer.size;
    }
    return total;
  }

  /**
   * Enforce memory limit by evicting oldest entries.
   * @private
   */
  _enforceMemoryLimit() {
    const targetBytes = this.config.maxTotalMemoryBytes * 0.8;
    while (this._totalMemoryBytes > targetBytes && this._buffers.size > 0) {
      // Find the buffer with the oldest data
      let oldestBuffer = null;
      let oldestTime = Infinity;

      for (const [type, buffer] of this._buffers) {
        const first = buffer.get(0);
        if (first && first.timestamp < oldestTime) {
          oldestTime = first.timestamp;
          oldestBuffer = { type, buffer };
        }
      }

      if (oldestBuffer) {
        const entry = oldestBuffer.buffer.get(0);
        if (entry) {
          this._totalMemoryBytes -= entry.size;
        }
        // Simulate removal by creating new smaller buffer
        const items = oldestBuffer.buffer.slice(1, oldestBuffer.buffer.size);
        const newBuffer = new RingBuffer(this.config.maxBufferSize);
        for (const item of items) {
          newBuffer.push(item);
        }
        this._buffers.set(oldestBuffer.type, newBuffer);
      } else {
        break;
      }
    }
  }

  /**
   * Start the periodic cleanup timer.
   * @private
   */
  _startCleanup() {
    if (this.config.cleanupIntervalMs > 0) {
      this._cleanupTimer = setInterval(() => this._cleanup(), this.config.cleanupIntervalMs);
    }
  }

  /**
   * Remove expired entries from all buffers.
   * @private
   */
  _cleanup() {
    const cutoff = Date.now() - this.config.maxAgeMs;
    let removed = 0;

    for (const [type, buffer] of this._buffers) {
      const items = buffer.toArray().filter(entry => {
        if (entry.timestamp < cutoff) {
          this._totalMemoryBytes -= entry.size;
          removed++;
          return false;
        }
        return true;
      });

      if (items.length < buffer.size) {
        const newBuffer = new RingBuffer(this.config.maxBufferSize);
        for (const item of items) {
          newBuffer.push(item);
        }
        this._buffers.set(type, newBuffer);
      }
    }

    if (removed > 0) {
      this.emit('cleanup', { removed, timestamp: Date.now() });
    }
  }

  /**
   * Get a nested value from an object using dot notation.
   * @param {object} obj - Source object
   * @param {string} path - Dot-separated path (e.g., 'accelerometer.x')
   * @returns {*}
   * @private
   */
  _getNestedValue(obj, path) {
    const parts = path.split('.');
    let current = obj;
    for (const part of parts) {
      if (current === null || current === undefined) return undefined;
      current = current[part];
    }
    return current;
  }

  /**
   * Flatten an object's keys into dot-notation paths.
   * @param {object} obj - Object to flatten
   * @param {string} [prefix=''] - Key prefix
   * @returns {string[]}
   * @private
   */
  _flattenKeys(obj, prefix = '') {
    const keys = [];
    if (typeof obj !== 'object' || obj === null) return keys;

    for (const key of Object.keys(obj)) {
      const fullKey = prefix ? `${prefix}.${key}` : key;
      if (typeof obj[key] === 'object' && obj[key] !== null && !Array.isArray(obj[key])) {
        keys.push(...this._flattenKeys(obj[key], fullKey));
      } else {
        keys.push(fullKey);
      }
    }
    return keys;
  }
}

export default TelemetryStorage;
