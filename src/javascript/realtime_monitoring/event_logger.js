/**
 * @fileoverview EventLogger - Structured event logging for the autonomous
 * vehicle control system. Supports log levels (debug/info/warn/error/critical),
 * log rotation, contextual logging with request IDs and session tracking,
 * log search/filter, export to JSON/CSV, and log aggregation.
 *
 * @module realtime_monitoring/event_logger
 * @version 2.1.0
 * @author Autonomous Vehicle Control System Team
 */

import { EventEmitter } from 'events';
import { createWriteStream, mkdirSync, existsSync } from 'fs';
import { join, dirname } from 'path';

/** @typedef {'debug'|'info'|'warn'|'error'|'critical'} LogLevel */

/** @typedef {'json'|'csv'} ExportFormat */

/**
 * @typedef {Object} EventLoggerConfig
 * @property {number} [maxEntries=50000] - Maximum entries in memory
 * @property {number} [rotationIntervalMs=3600000] - Log rotation interval (1 hour)
 * @property {number} [maxFileSizeBytes=52428800] - Max file size before rotation (50MB)
 * @property {string} [logDirectory='./logs'] - Directory for log files
 * @property {boolean} [enableFileOutput=false] - Whether to write logs to file
 * @property {boolean} [enableConsoleOutput=true] - Whether to output to console
 * @property {LogLevel} [minLevel='debug'] - Minimum log level to record
 * @property {boolean} [includeStackTrace=true] - Include stack traces for errors
 * @property {string} [serviceId='vehicle-monitor'] - Service identifier for log entries
 * @property {number} [flushIntervalMs=5000] - File flush interval
 */

/**
 * @typedef {Object} LogEntry
 * @property {string} id - Unique log entry ID
 * @property {LogLevel} level - Log level
 * @property {string} source - Source module/component
 * @property {string} message - Log message
 * @property {Object} [context] - Additional context data
 * @property {string} [requestId] - Associated request ID
 * @property {string} [sessionId] - Associated session ID
 * @property {string} [traceId] - Distributed tracing ID
 * @property {number} timestamp - Entry timestamp
 * @property {string} serviceId - Service identifier
 * @property {string} [stackTrace] - Stack trace for errors
 */

/**
 * @typedef {Object} LogFilter
 * @property {LogLevel} [level] - Filter by level
 * @property {LogLevel} [minLevel] - Filter by minimum level
 * @property {string} [source] - Filter by source (substring match)
 * @property {string} [message] - Filter by message (substring match)
 * @property {string} [requestId] - Filter by request ID
 * @property {string} [sessionId] - Filter by session ID
 * @property {number} [since] - Filter entries after timestamp
 * @property {number} [until] - Filter entries before timestamp
 * @property {number} [limit=100] - Max results
 */

/**
 * @typedef {Object} LogStats
 * @property {number} totalEntries - Total entries in memory
 * @property {Object<string, number>} countsByLevel - Count per log level
 * @property {Object<string, number>} countsBySource - Count per source
 * @property {number} oldestEntry - Oldest entry timestamp
 * @property {number} newestEntry - Newest entry timestamp
 */

/** @type {Record<LogLevel, number>} */
const LOG_LEVEL_PRIORITY = {
  debug: 0,
  info: 1,
  warn: 2,
  error: 3,
  critical: 4,
};

/** @type {number} */
let entryIdCounter = 0;

/**
 * EventLogger provides structured, level-based logging with context tracking,
 * rotation, search/filter, and export capabilities.
 *
 * @extends EventEmitter
 *
 * @example
 * const logger = new EventLogger({
 *   minLevel: 'info',
 *   enableFileOutput: true,
 *   logDirectory: '/var/log/vehicle',
 * });
 *
 * logger.setContext({ requestId: 'req-123', sessionId: 'sess-456' });
 * logger.info('Perception', 'Object detected', { type: 'vehicle', distance: 45 });
 * logger.error('Control', 'Steering command failed', { commandId: 'cmd-789' });
 */
export class EventLogger extends EventEmitter {
  /** @type {EventLoggerConfig} */
  #config;

  /** @type {LogEntry[]} */
  #entries = [];

  /** @type {Map<string, string>} */
  #context = new Map();

  /** @type {Object<string, number>} */
  #levelCounts = { debug: 0, info: 0, warn: 0, error: 0, critical: 0 };

  /** @type {Map<string, number>} */
  #sourceCounts = new Map();

  /** @type {import('fs').WriteStream|null} */
  #fileStream = null;

  /** @type {NodeJS.Timeout|null} */
  #rotationTimer = null;

  /** @type {string|null} */
  #currentLogFile = null;

  /** @type {number} */
  #currentFileSize = 0;

  /** @type {number} */
  #rotationCount = 0;

  /**
   * Creates a new EventLogger.
   *
   * @param {EventLoggerConfig} [config={}] - Logger configuration
   */
  constructor(config = {}) {
    super();
    this.setMaxListeners(20);

    this.#config = {
      maxEntries: 50000,
      rotationIntervalMs: 3600000,
      maxFileSizeBytes: 52428800,
      logDirectory: './logs',
      enableFileOutput: false,
      enableConsoleOutput: true,
      minLevel: 'debug',
      includeStackTrace: true,
      serviceId: 'vehicle-monitor',
      flushIntervalMs: 5000,
      ...config,
    };
  }

  /**
   * Total number of log entries in memory.
   * @type {number}
   */
  get entryCount() {
    return this.#entries.length;
  }

  /**
   * Current log level counts.
   * @type {Readonly<Object<string, number>>}
   */
  get levelCounts() {
    return Object.freeze({ ...this.#levelCounts });
  }

  /**
   * Opens file output stream and starts rotation timer.
   *
   * @returns {void}
   */
  init() {
    if (this.#config.enableFileOutput) {
      this.#ensureLogDirectory();
      this.#openNewLogFile();

      this.#rotationTimer = setInterval(() => {
        this.#rotateLogFile();
      }, this.#config.rotationIntervalMs);
    }
  }

  /**
   * Logs a debug-level message.
   *
   * @param {string} source - Source module
   * @param {string} message - Log message
   * @param {Object} [context={}] - Additional context
   * @returns {LogEntry} The created log entry
   */
  debug(source, message, context = {}) {
    return this.#log('debug', source, message, context);
  }

  /**
   * Logs an info-level message.
   *
   * @param {string} source - Source module
   * @param {string} message - Log message
   * @param {Object} [context={}] - Additional context
   * @returns {LogEntry} The created log entry
   */
  info(source, message, context = {}) {
    return this.#log('info', source, message, context);
  }

  /**
   * Logs a warn-level message.
   *
   * @param {string} source - Source module
   * @param {string} message - Log message
   * @param {Object} [context={}] - Additional context
   * @returns {LogEntry} The created log entry
   */
  warn(source, message, context = {}) {
    return this.#log('warn', source, message, context);
  }

  /**
   * Logs an error-level message.
   *
   * @param {string} source - Source module
   * @param {string} message - Log message
   * @param {Object} [context={}] - Additional context
   * @returns {LogEntry} The created log entry
   */
  error(source, message, context = {}) {
    return this.#log('error', source, message, context);
  }

  /**
   * Logs a critical-level message.
   *
   * @param {string} source - Source module
   * @param {string} message - Log message
   * @param {Object} [context={}] - Additional context
   * @returns {LogEntry} The created log entry
   */
  critical(source, message, context = {}) {
    return this.#log('critical', source, message, context);
  }

  /**
   * Sets persistent context that will be included in all subsequent log entries.
   *
   * @param {Object} context - Context key-value pairs
   * @returns {void}
   *
   * @example
   * logger.setContext({ requestId: 'req-123', sessionId: 'sess-456' });
   * // All subsequent logs will include requestId and sessionId
   */
  setContext(context) {
    for (const [key, value] of Object.entries(context)) {
      this.#context.set(key, String(value));
    }
  }

  /**
   * Clears one or more context keys, or clears all context.
   *
   * @param {string|string[]} [keys] - Keys to clear (omit to clear all)
   * @returns {void}
   */
  clearContext(keys) {
    if (!keys) {
      this.#context.clear();
      return;
    }
    const keyArray = Array.isArray(keys) ? keys : [keys];
    for (const key of keyArray) {
      this.#context.delete(key);
    }
  }

  /**
   * Creates a child logger with preset context.
   *
   * @param {Object} context - Context for the child logger
   * @returns {Object} Child logger with bound context
   */
  child(context) {
    const parent = this;
    return {
      debug: (source, message, ctx = {}) => parent.debug(source, message, { ...context, ...ctx }),
      info: (source, message, ctx = {}) => parent.info(source, message, { ...context, ...ctx }),
      warn: (source, message, ctx = {}) => parent.warn(source, message, { ...context, ...ctx }),
      error: (source, message, ctx = {}) => parent.error(source, message, { ...context, ...ctx }),
      critical: (source, message, ctx = {}) => parent.critical(source, message, { ...context, ...ctx }),
    };
  }

  /**
   * Searches and filters log entries.
   *
   * @param {LogFilter} filter - Filter criteria
   * @returns {LogEntry[]} Matching log entries
   */
  search(filter) {
    let results = [...this.#entries];

    if (filter.level) {
      results = results.filter((e) => e.level === filter.level);
    }

    if (filter.minLevel) {
      const minPriority = LOG_LEVEL_PRIORITY[filter.minLevel] ?? 0;
      results = results.filter((e) => (LOG_LEVEL_PRIORITY[e.level] ?? 0) >= minPriority);
    }

    if (filter.source) {
      const srcLower = filter.source.toLowerCase();
      results = results.filter((e) => e.source.toLowerCase().includes(srcLower));
    }

    if (filter.message) {
      const msgLower = filter.message.toLowerCase();
      results = results.filter((e) => e.message.toLowerCase().includes(msgLower));
    }

    if (filter.requestId) {
      results = results.filter((e) => e.requestId === filter.requestId);
    }

    if (filter.sessionId) {
      results = results.filter((e) => e.sessionId === filter.sessionId);
    }

    if (filter.since) {
      results = results.filter((e) => e.timestamp >= filter.since);
    }

    if (filter.until) {
      results = results.filter((e) => e.timestamp <= filter.until);
    }

    const limit = filter.limit ?? 100;
    return results.slice(-limit);
  }

  /**
   * Gets log statistics.
   *
   * @returns {LogStats} Log statistics
   */
  getStats() {
    const entries = this.#entries;
    return {
      totalEntries: entries.length,
      countsByLevel: { ...this.#levelCounts },
      countsBySource: Object.fromEntries(this.#sourceCounts),
      oldestEntry: entries.length > 0 ? entries[0].timestamp : 0,
      newestEntry: entries.length > 0 ? entries[entries.length - 1].timestamp : 0,
    };
  }

  /**
   * Exports log entries to the specified format.
   *
   * @param {ExportFormat} format - Export format ('json' or 'csv')
   * @param {LogFilter} [filter={}] - Filter criteria for export
   * @returns {string} Formatted log data
   */
  export(format, filter = {}) {
    const entries = this.search(filter);

    if (format === 'csv') {
      return this.#exportToCSV(entries);
    }

    return JSON.stringify(entries, null, 2);
  }

  /**
   * Flushes buffered log entries to file.
   *
   * @returns {void}
   */
  flush() {
    if (this.#fileStream) {
      this.#fileStream.write('', () => {
        // Flush callback
      });
    }
  }

  /**
   * Closes the logger, flushing and closing file streams.
   *
   * @returns {void}
   */
  close() {
    if (this.#rotationTimer) {
      clearInterval(this.#rotationTimer);
      this.#rotationTimer = null;
    }

    this.flush();

    if (this.#fileStream) {
      this.#fileStream.end();
      this.#fileStream = null;
    }
  }

  // ─── Private Methods ───────────────────────────────────────────────

  /**
   * Core logging method.
   * @private
   * @param {LogLevel} level - Log level
   * @param {string} source - Source module
   * @param {string} message - Log message
   * @param {Object} context - Additional context
   * @returns {LogEntry}
   */
  #log(level, source, message, context) {
    // Level filtering
    const minPriority = LOG_LEVEL_PRIORITY[this.#config.minLevel] ?? 0;
    if ((LOG_LEVEL_PRIORITY[level] ?? 0) < minPriority) {
      return null;
    }

    const entry = {
      id: `log_${Date.now()}_${++entryIdCounter}`,
      level,
      source,
      message,
      context: { ...context },
      requestId: context.requestId || this.#context.get('requestId') || null,
      sessionId: context.sessionId || this.#context.get('sessionId') || null,
      traceId: context.traceId || this.#context.get('traceId') || null,
      timestamp: Date.now(),
      serviceId: this.#config.serviceId,
    };

    // Include stack trace for error and critical
    if (this.#config.includeStackTrace && (level === 'error' || level === 'critical')) {
      const error = context.error || context.err;
      if (error instanceof Error) {
        entry.stackTrace = error.stack;
      }
    }

    // Add to in-memory store
    this.#entries.push(entry);
    this.#levelCounts[level] = (this.#levelCounts[level] || 0) + 1;
    this.#sourceCounts.set(source, (this.#sourceCounts.get(source) || 0) + 1);

    // Enforce max entries
    while (this.#entries.length > this.#config.maxEntries) {
      this.#entries.shift();
    }

    // Console output
    if (this.#config.enableConsoleOutput) {
      this.#consoleOutput(entry);
    }

    // File output
    if (this.#config.enableFileOutput && this.#fileStream) {
      const line = JSON.stringify(entry) + '\n';
      this.#currentFileSize += Buffer.byteLength(line);
      this.#fileStream.write(line);

      // Check file size rotation
      if (this.#currentFileSize >= this.#config.maxFileSizeBytes) {
        this.#rotateLogFile();
      }
    }

    /**
     * @event EventLogger#log
     * @type {LogEntry}
     */
    this.emit('log', entry);

    // Emit level-specific events
    this.emit(`log:${level}`, entry);

    return entry;
  }

  /**
   * Outputs a log entry to the console.
   * @private
   * @param {LogEntry} entry - Log entry
   */
  #consoleOutput(entry) {
    const time = new Date(entry.timestamp).toISOString();
    const levelStr = entry.level.toUpperCase().padEnd(8);
    const prefix = `[${time}] ${levelStr} [${entry.source}]`;

    switch (entry.level) {
      case 'debug':
        console.debug(prefix, entry.message, entry.context);
        break;
      case 'info':
        console.info(prefix, entry.message, entry.context);
        break;
      case 'warn':
        console.warn(prefix, entry.message, entry.context);
        break;
      case 'error':
      case 'critical':
        console.error(prefix, entry.message, entry.context);
        if (entry.stackTrace) {
          console.error(entry.stackTrace);
        }
        break;
    }
  }

  /**
   * Ensures the log directory exists.
   * @private
   */
  #ensureLogDirectory() {
    if (!existsSync(this.#config.logDirectory)) {
      try {
        mkdirSync(this.#config.logDirectory, { recursive: true });
      } catch {
        console.error(`Failed to create log directory: ${this.#config.logDirectory}`);
      }
    }
  }

  /**
   * Opens a new log file for writing.
   * @private
   */
  #openNewLogFile() {
    if (this.#fileStream) {
      this.#fileStream.end();
    }

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    this.#currentLogFile = join(
      this.#config.logDirectory,
      `vehicle-monitor-${timestamp}.log`
    );

    try {
      this.#fileStream = createWriteStream(this.#currentLogFile, { flags: 'a' });
      this.#currentFileSize = 0;
      this.#rotationCount++;
    } catch (error) {
      console.error(`Failed to open log file: ${error.message}`);
      this.#fileStream = null;
    }
  }

  /**
   * Rotates the log file by closing the current one and opening a new one.
   * @private
   */
  #rotateLogFile() {
    if (!this.#config.enableFileOutput) return;

    const oldFile = this.#currentLogFile;
    this.#openNewLogFile();

    /**
     * @event EventLogger#log:rotated
     * @type {Object}
     * @property {string} oldFile - Previous log file path
     * @property {string} newFile - New log file path
     * @property {number} rotationCount - Total rotations
     */
    this.emit('log:rotated', {
      oldFile,
      newFile: this.#currentLogFile,
      rotationCount: this.#rotationCount,
    });
  }

  /**
   * Exports entries to CSV format.
   * @private
   * @param {LogEntry[]} entries - Log entries to export
   * @returns {string} CSV string
   */
  #exportToCSV(entries) {
    const headers = ['id', 'timestamp', 'level', 'source', 'message', 'requestId', 'sessionId', 'serviceId'];
    const rows = [headers.join(',')];

    for (const entry of entries) {
      const row = [
        entry.id,
        new Date(entry.timestamp).toISOString(),
        entry.level,
        `"${(entry.source || '').replace(/"/g, '""')}"`,
        `"${(entry.message || '').replace(/"/g, '""')}"`,
        entry.requestId || '',
        entry.sessionId || '',
        entry.serviceId || '',
      ];
      rows.push(row.join(','));
    }

    return rows.join('\n');
  }
}

export default EventLogger;
