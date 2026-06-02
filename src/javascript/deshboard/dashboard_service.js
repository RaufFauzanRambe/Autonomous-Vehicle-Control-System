/**
 * @fileoverview Dashboard Business Logic Service
 * @description Dashboard creation/update/deletion, widget data aggregation,
 *   real-time data fetching, caching layer, and data transformation pipeline
 *   for the Autonomous Vehicle Dashboard.
 * @module dashboard_service
 */

import { EventEmitter } from 'events';

// ─── Cache Implementation ────────────────────────────────────────────────────

/**
 * In-memory LRU cache with TTL support
 * @template T
 */
class LRUCache {
  /**
   * @param {Object} [options={}]
   * @param {number} [options.maxSize=100] - Maximum cache entries
   * @param {number} [options.defaultTTL=60000] - Default TTL in milliseconds
   */
  constructor(options = {}) {
    /** @private @type {Map<string, { value: T, expires: number }>} */ this._cache = new Map();
    /** @private */ this._maxSize = options.maxSize ?? 100;
    /** @private */ this._defaultTTL = options.defaultTTL ?? 60000;
    /** @private */ this._hits = 0;
    /** @private */ this._misses = 0;
  }

  /**
   * Get a cached value
   * @param {string} key - Cache key
   * @returns {T|null} Cached value or null if expired/missing
   */
  get(key) {
    const entry = this._cache.get(key);
    if (!entry) {
      this._misses++;
      return null;
    }
    if (Date.now() > entry.expires) {
      this._cache.delete(key);
      this._misses++;
      return null;
    }
    // Move to end (most recently used)
    this._cache.delete(key);
    this._cache.set(key, entry);
    this._hits++;
    return entry.value;
  }

  /**
   * Set a cached value
   * @param {string} key - Cache key
   * @param {T} value - Value to cache
   * @param {number} [ttl] - TTL in ms (uses default if omitted)
   */
  set(key, value, ttl) {
    if (this._cache.size >= this._maxSize) {
      // Evict least recently used (first entry)
      const firstKey = this._cache.keys().next().value;
      this._cache.delete(firstKey);
    }
    this._cache.set(key, {
      value,
      expires: Date.now() + (ttl ?? this._defaultTTL),
    });
  }

  /**
   * Invalidate a cache entry
   * @param {string} key - Cache key
   */
  invalidate(key) {
    this._cache.delete(key);
  }

  /**
   * Invalidate entries matching a pattern
   * @param {string|RegExp} pattern - Key pattern
   */
  invalidatePattern(pattern) {
    const regex = typeof pattern === 'string' ? new RegExp(pattern) : pattern;
    for (const key of this._cache.keys()) {
      if (regex.test(key)) this._cache.delete(key);
    }
  }

  /** Clear all cache entries */
  clear() {
    this._cache.clear();
    this._hits = 0;
    this._misses = 0;
  }

  /** @returns {{ size: number, hits: number, misses: number, hitRate: string }} */
  get stats() {
    const total = this._hits + this._misses;
    return {
      size: this._cache.size,
      hits: this._hits,
      misses: this._misses,
      hitRate: total > 0 ? `${((this._hits / total) * 100).toFixed(1)}%` : '0%',
    };
  }
}

// ─── Data Transformation Pipeline ────────────────────────────────────────────

/**
 * Pipeline for transforming raw vehicle data into dashboard-ready format
 */
class DataTransformPipeline {
  constructor() {
    /** @private @type {Function[]} */ this._stages = [];
    this._registerDefaultStages();
  }

  /**
   * Register default data transformation stages
   * @private
   */
  _registerDefaultStages() {
    // Stage 1: Validate incoming data structure
    this._stages.push((data) => {
      if (!data || typeof data !== 'object') {
        throw new Error('Invalid telemetry data: expected object');
      }
      return {
        ...data,
        _receivedAt: Date.now(),
        _valid: true,
      };
    });

    // Stage 2: Normalize units (metric base)
    this._stages.push((data) => {
      if (data.speed !== undefined && data.speedUnit === 'mph') {
        data.speed = data.speed * 1.60934;
        data.speedUnit = 'km/h';
      }
      if (data.temperature !== undefined && data.tempUnit === 'fahrenheit') {
        data.temperature = (data.temperature - 32) * 5 / 9;
        data.tempUnit = 'celsius';
      }
      return data;
    });

    // Stage 3: Compute derived fields
    this._stages.push((data) => {
      if (data.speed !== undefined && data.heading !== undefined) {
        data.velocityVector = {
          vx: data.speed * Math.cos((data.heading * Math.PI) / 180),
          vy: data.speed * Math.sin((data.heading * Math.PI) / 180),
        };
      }
      if (data.batteryLevel !== undefined && data.batteryDrainRate !== undefined) {
        data.estimatedRangeKm = data.batteryLevel / Math.max(data.batteryDrainRate, 0.001);
      }
      return data;
    });

    // Stage 4: Add quality score
    this._stages.push((data) => {
      const fields = Object.keys(data).filter((k) => !k.startsWith('_'));
      const filledFields = fields.filter((k) => data[k] !== null && data[k] !== undefined);
      data._qualityScore = filledFields.length / Math.max(fields.length, 1);
      return data;
    });
  }

  /**
   * Add a custom transformation stage
   * @param {Function} stage - Transform function (data) => data
   */
  addStage(stage) {
    if (typeof stage !== 'function') throw new Error('Stage must be a function');
    this._stages.push(stage);
  }

  /**
   * Process data through all pipeline stages
   * @param {Object} data - Raw input data
   * @returns {Object} Transformed data
   */
  process(data) {
    return this._stages.reduce((current, stage) => {
      try {
        return stage(current);
      } catch (err) {
        console.error(`Pipeline stage failed: ${err.message}`);
        return current;
      }
    }, data);
  }
}

// ─── Dashboard Service ───────────────────────────────────────────────────────

/**
 * Business logic service for dashboard operations
 * @extends EventEmitter
 */
export class DashboardService extends EventEmitter {
  /**
   * @param {Object} store - Dashboard store instance
   * @param {Object} alertManager - Alert manager instance
   * @param {Object} settingsManager - Settings manager instance
   */
  constructor(store, alertManager, settingsManager) {
    super();
    /** @private */ this.store = store;
    /** @private */ this.alertManager = alertManager;
    /** @private */ this.settingsManager = settingsManager;
    /** @private */ this.cache = new LRUCache({ maxSize: 200, defaultTTL: 30000 });
    /** @private */ this.pipeline = new DataTransformPipeline();
    /** @private @type {Map<string, Object>} */ this._telemetryBuffer = new Map();
    /** @private @type {Map<string, number>} */ this._lastFetchTimes = new Map();
    /** @private */ this._refreshInterval = null;
    this._startAutoRefresh();
  }

  // ─── Dashboard CRUD ─────────────────────────────────────────────────────

  /**
   * Create a new dashboard
   * @param {Object} config - Dashboard configuration
   * @returns {Object} Created dashboard
   */
  createDashboard(config) {
    const dashboard = this.store.createDashboard(config);
    this.cache.invalidatePattern(/^dashboards:/);
    this.emit('dashboard:created', dashboard);
    return dashboard;
  }

  /**
   * Retrieve a dashboard by ID
   * @param {string} id - Dashboard ID
   * @returns {Object|null}
   */
  getDashboard(id) {
    const cacheKey = `dashboards:${id}`;
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    const dashboard = this.store.getDashboard(id);
    if (dashboard) this.cache.set(cacheKey, dashboard);
    return dashboard;
  }

  /**
   * List dashboards with optional filters
   * @param {Object} [filter={}] - Filter criteria
   * @returns {Object[]}
   */
  listDashboards(filter = {}) {
    const cacheKey = `dashboards:list:${JSON.stringify(filter)}`;
    const cached = this.cache.get(cacheKey);
    if (cached) return cached;

    const dashboards = this.store.listDashboards(filter);
    this.cache.set(cacheKey, dashboards, 10000);
    return dashboards;
  }

  /**
   * Update a dashboard
   * @param {string} id - Dashboard ID
   * @param {Object} updates - Fields to update
   * @returns {Object} Updated dashboard
   */
  updateDashboard(id, updates) {
    const dashboard = this.store.updateDashboard(id, updates);
    this.cache.invalidate(`dashboards:${id}`);
    this.cache.invalidatePattern(/^dashboards:list/);
    this.emit('dashboard:updated', dashboard);
    return dashboard;
  }

  /**
   * Delete a dashboard
   * @param {string} id - Dashboard ID
   * @returns {boolean}
   */
  deleteDashboard(id) {
    const result = this.store.deleteDashboard(id);
    this.cache.invalidate(`dashboards:${id}`);
    this.cache.invalidatePattern(/^dashboards:list/);
    if (result) this.emit('dashboard:deleted', { id });
    return result;
  }

  // ─── Widget Operations ──────────────────────────────────────────────────

  /**
   * Get widgets for a dashboard
   * @param {string} dashboardId - Dashboard ID
   * @returns {Object[]|null}
   */
  getWidgets(dashboardId) {
    const dashboard = this.store.getDashboard(dashboardId);
    if (!dashboard) return null;
    return dashboard.widgets;
  }

  /**
   * Add a widget to a dashboard
   * @param {string} dashboardId - Dashboard ID
   * @param {Object} widgetConfig - Widget configuration
   * @returns {Object} Added widget
   */
  addWidget(dashboardId, widgetConfig) {
    const widget = this.store.addWidget(dashboardId, widgetConfig);
    this.cache.invalidate(`dashboards:${dashboardId}`);
    this.cache.invalidatePattern(/^widget_data:/);
    this.emit('widget:added', { dashboardId, widget });
    return widget;
  }

  /**
   * Remove a widget from a dashboard
   * @param {string} dashboardId - Dashboard ID
   * @param {string} widgetId - Widget ID
   * @returns {boolean}
   */
  removeWidget(dashboardId, widgetId) {
    const result = this.store.removeWidget(dashboardId, widgetId);
    this.cache.invalidate(`dashboards:${dashboardId}`);
    if (result) this.emit('widget:removed', { dashboardId, widgetId });
    return result;
  }

  // ─── Widget Data Aggregation ────────────────────────────────────────────

  /**
   * Fetch aggregated data for a specific widget
   * @param {string} dashboardId - Dashboard ID
   * @param {string} widgetId - Widget ID
   * @param {Object} [options={}] - Fetch options
   * @param {number} [options.timeRange=300000] - Time range in ms
   * @param {boolean} [options.forceRefresh=false] - Skip cache
   * @returns {Promise<Object>} Aggregated widget data
   */
  async getWidgetData(dashboardId, widgetId, options = {}) {
    const { timeRange = 300000, forceRefresh = false } = options;
    const cacheKey = `widget_data:${dashboardId}:${widgetId}:${timeRange}`;

    if (!forceRefresh) {
      const cached = this.cache.get(cacheKey);
      if (cached) return cached;
    }

    const dashboard = this.store.getDashboard(dashboardId);
    if (!dashboard) throw new Error(`Dashboard not found: ${dashboardId}`);

    const widget = dashboard.widgets.find((w) => w.id === widgetId);
    if (!widget) throw new Error(`Widget not found: ${widgetId}`);

    const now = Date.now();
    const cutoff = now - timeRange;

    // Get telemetry data within the time range
    const telemetryData = this._getTelemetryRange(widget.type, cutoff, now);

    // Transform data through the pipeline
    const transformedData = telemetryData.map((d) => this.pipeline.process(d));

    // Aggregate based on widget type
    const aggregated = this._aggregateForWidget(widget.type, transformedData);

    const result = {
      widgetId,
      widgetType: widget.type,
      timeRange: { start: cutoff, end: now },
      data: aggregated,
      dataPoints: transformedData.length,
      qualityScore: transformedData.length > 0
        ? transformedData.reduce((sum, d) => sum + (d._qualityScore || 0), 0) / transformedData.length
        : 0,
      fetchedAt: new Date().toISOString(),
    };

    this.cache.set(cacheKey, result, 5000);
    return result;
  }

  /**
   * Get layout for a dashboard
   * @param {string} dashboardId - Dashboard ID
   * @param {number} [viewportWidth] - Viewport width
   * @returns {Object[]}
   */
  getComputedLayout(dashboardId, viewportWidth) {
    return this.store.getComputedLayout(dashboardId, viewportWidth);
  }

  // ─── Export / Import ────────────────────────────────────────────────────

  /**
   * Export dashboard configuration
   * @param {string} dashboardId - Dashboard ID
   * @returns {string} JSON configuration
   */
  exportDashboard(dashboardId) {
    return this.store.exportDashboard(dashboardId);
  }

  /**
   * Import dashboard configuration
   * @param {string} configStr - JSON configuration string
   * @returns {Object} Imported dashboard
   */
  importDashboard(configStr) {
    return this.store.importDashboard(configStr);
  }

  // ─── Real-time Data ─────────────────────────────────────────────────────

  /**
   * Process incoming telemetry data
   * @param {Object} data - Raw telemetry data
   */
  processTelemetry(data) {
    const transformed = this.pipeline.process(data);
    const source = data.source || data.sensorId || 'default';

    if (!this._telemetryBuffer.has(source)) {
      this._telemetryBuffer.set(source, []);
    }
    const buffer = this._telemetryBuffer.get(source);
    buffer.push(transformed);

    // Keep only last 1000 entries per source
    if (buffer.length > 1000) {
      buffer.splice(0, buffer.length - 1000);
    }

    this._lastFetchTimes.set(source, Date.now());

    // Check alert thresholds
    this._checkThresholds(transformed);

    this.emit('telemetry:processed', transformed);
  }

  // ─── Cleanup ────────────────────────────────────────────────────────────

  /**
   * Clean up stale data from buffers
   */
  cleanupStaleData() {
    const cutoff = Date.now() - 3600000; // 1 hour
    for (const [source, buffer] of this._telemetryBuffer.entries()) {
      const staleIndex = buffer.findIndex((d) => (d._receivedAt || 0) >= cutoff);
      if (staleIndex > 0) {
        buffer.splice(0, staleIndex);
      }
      if (buffer.length === 0) {
        this._telemetryBuffer.delete(source);
      }
    }
    this.cache.invalidatePattern(/^widget_data:/);
  }

  // ─── Private Methods ────────────────────────────────────────────────────

  /**
   * Get telemetry data for a time range
   * @private
   * @param {string} widgetType - Widget type
   * @param {number} start - Start timestamp
   * @param {number} end - End timestamp
   * @returns {Object[]}
   */
  _getTelemetryRange(widgetType, start, end) {
    const allData = [];
    for (const buffer of this._telemetryBuffer.values()) {
      for (const entry of buffer) {
        const ts = entry._receivedAt || entry.timestamp || 0;
        if (ts >= start && ts <= end) {
          allData.push(entry);
        }
      }
    }
    return allData.sort((a, b) => (a._receivedAt || 0) - (b._receivedAt || 0));
  }

  /**
   * Aggregate data based on widget type
   * @private
   * @param {string} widgetType - Widget type
   * @param {Object[]} data - Transformed data points
   * @returns {Object}
   */
  _aggregateForWidget(widgetType, data) {
    if (data.length === 0) return { values: [], summary: {} };

    switch (widgetType) {
      case 'speed_gauge':
      case 'speed_chart': {
        const speeds = data.map((d) => d.speed).filter((s) => s !== undefined);
        return {
          values: speeds,
          summary: {
            current: speeds[speeds.length - 1] ?? 0,
            avg: speeds.reduce((a, b) => a + b, 0) / speeds.length,
            max: Math.max(...speeds),
            min: Math.min(...speeds),
          },
          timestamps: data.map((d) => d._receivedAt),
        };
      }
      case 'battery_indicator': {
        const levels = data.map((d) => d.batteryLevel).filter((l) => l !== undefined);
        const drainRates = data.map((d) => d.batteryDrainRate).filter((r) => r !== undefined);
        return {
          values: levels,
          summary: {
            current: levels[levels.length - 1] ?? 0,
            avgDrainRate: drainRates.length > 0 ? drainRates.reduce((a, b) => a + b, 0) / drainRates.length : 0,
          },
        };
      }
      case 'sensor_status': {
        const sensors = {};
        for (const d of data) {
          if (d.sensors) {
            for (const [id, val] of Object.entries(d.sensors)) {
              if (!sensors[id]) sensors[id] = [];
              sensors[id].push(val);
            }
          }
        }
        return { values: sensors, summary: { sensorCount: Object.keys(sensors).length } };
      }
      case 'object_detection': {
        const objects = data.flatMap((d) => d.detectedObjects || []);
        return {
          values: objects,
          summary: {
            totalCount: objects.length,
            types: [...new Set(objects.map((o) => o.type))],
          },
        };
      }
      default:
        return { values: data, summary: { count: data.length } };
    }
  }

  /**
   * Check telemetry data against alert thresholds
   * @private
   * @param {Object} data - Transformed telemetry data
   */
  _checkThresholds(data) {
    if (data.speed > 180) {
      this.alertManager.fireAlert('speed_exceeded', 'critical', `Speed ${data.speed.toFixed(1)} km/h exceeds 180 km/h threshold`);
    }
    if (data.batteryLevel !== undefined && data.batteryLevel < 10) {
      this.alertManager.fireAlert('low_battery', 'warning', `Battery level at ${data.batteryLevel.toFixed(1)}%`);
    }
    if (data._qualityScore !== undefined && data._qualityScore < 0.5) {
      this.alertManager.fireAlert('data_quality', 'info', `Data quality score low: ${(data._qualityScore * 100).toFixed(0)}%`);
    }
  }

  /**
   * Start automatic data refresh cycle
   * @private
   */
  _startAutoRefresh() {
    this._refreshInterval = setInterval(() => {
      this.cache.invalidatePattern(/^widget_data:/);
      this.emit('data:refresh');
    }, 15000);
  }

  /**
   * Stop auto-refresh
   */
  stopAutoRefresh() {
    if (this._refreshInterval) {
      clearInterval(this._refreshInterval);
      this._refreshInterval = null;
    }
  }

  /**
   * Get service health status
   * @returns {Object}
   */
  getHealthStatus() {
    return {
      cache: this.cache.stats,
      telemetryBuffers: this._telemetryBuffer.size,
      totalDataPoints: Array.from(this._telemetryBuffer.values()).reduce((sum, buf) => sum + buf.length, 0),
      activeSources: this._lastFetchTimes.size,
      autoRefresh: this._refreshInterval !== null,
    };
  }
}

export { LRUCache, DataTransformPipeline };
