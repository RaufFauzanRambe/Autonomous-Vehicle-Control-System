/**
 * @fileoverview Widget System
 * @description Base Widget class and concrete implementations: SpeedGauge,
 *   BatteryIndicator, GpsMap, SensorStatus, ObjectDetectionFeed,
 *   LaneDepartureIndicator, WeatherWidget, TripInfoWidget — each with
 *   render/update/destroy lifecycle for the Autonomous Vehicle Dashboard.
 * @module widgets
 */

import { EventEmitter } from 'events';

// ─── Widget Lifecycle States ─────────────────────────────────────────────────

/** @enum {string} */
export const WidgetState = {
  CREATED: 'created',
  MOUNTED: 'mounted',
  UPDATED: 'updated',
  ERROR: 'error',
  DESTROYED: 'destroyed',
};

// ─── Base Widget Class ───────────────────────────────────────────────────────

/**
 * Abstract base class for all dashboard widgets
 * @extends EventEmitter
 */
export class Widget extends EventEmitter {
  /**
   * @param {string} id - Unique widget identifier
   * @param {Object} [config={}] - Widget configuration
   * @param {string} [config.title='Widget'] - Display title
   * @param {string} [config.size='2x1'] - Grid size
   * @param {boolean} [config.enabled=true] - Whether widget is active
   * @param {number} [config.refreshInterval=1000] - Data refresh interval in ms
   */
  constructor(id, config = {}) {
    super();
    /** @protected */ this.id = id;
    /** @protected */ this.title = config.title || 'Widget';
    /** @protected */ this.size = config.size || '2x1';
    /** @protected */ this.enabled = config.enabled !== false;
    /** @protected */ this.refreshInterval = config.refreshInterval ?? 1000;
    /** @protected */ this.state = WidgetState.CREATED;
    /** @protected @type {Object} */ this.data = {};
    /** @protected @type {HTMLElement|null} */ this.element = null;
    /** @protected @type {number|null} */ this._refreshTimer = null;
    /** @protected @type {Error|null} */ this.lastError = null;
    /** @protected */ this.createdAt = Date.now();
    /** @protected */ this.updatedAt = Date.now();
    /** @protected */ this.updateCount = 0;
  }

  /**
   * Render the widget to a DOM element
   * @param {HTMLElement} container - Parent container element
   * @returns {HTMLElement} The rendered widget element
   */
  render(container) {
    if (this.state === WidgetState.DESTROYED) {
      throw new Error(`Cannot render destroyed widget: ${this.id}`);
    }

    this.element = document.createElement('div');
    this.element.id = `widget-${this.id}`;
    this.element.className = `av-widget av-widget--${this.type} av-widget--${this.size.replace('x', '-')}`;
    this.element.setAttribute('role', 'region');
    this.element.setAttribute('aria-label', this.title);

    this._renderHeader();
    this._renderContent();
    this._renderFooter();

    container.appendChild(this.element);
    this.state = WidgetState.MOUNTED;
    this.emit('mounted', { id: this.id });

    this._startRefresh();
    return this.element;
  }

  /**
   * Update widget with new data
   * @param {Object} newData - New data to display
   */
  update(newData) {
    if (this.state === WidgetState.DESTROYED) return;

    try {
      const oldData = { ...this.data };
      this.data = this._transformData(newData);
      this.updatedAt = Date.now();
      this.updateCount++;

      if (this.element && this.state === WidgetState.MOUNTED) {
        this._updateContent(oldData);
      }

      this.state = WidgetState.UPDATED;
      this.emit('updated', { id: this.id, data: this.data, oldData });
    } catch (err) {
      this.lastError = err;
      this.state = WidgetState.ERROR;
      this.emit('error', { id: this.id, error: err.message });
      this._renderError(err);
    }
  }

  /**
   * Destroy the widget and clean up resources
   */
  destroy() {
    this._stopRefresh();
    if (this.element) {
      this.element.remove();
      this.element = null;
    }
    this.state = WidgetState.DESTROYED;
    this.removeAllListeners();
    this.emit('destroyed', { id: this.id });
  }

  /**
   * Get the widget type identifier
   * @returns {string}
   */
  get type() {
    return 'base';
  }

  /**
   * Get widget health/status summary
   * @returns {Object}
   */
  getStatus() {
    return {
      id: this.id,
      type: this.type,
      title: this.title,
      state: this.state,
      updateCount: this.updateCount,
      lastError: this.lastError?.message || null,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      uptime: Date.now() - this.createdAt,
    };
  }

  // ─── Protected Lifecycle Hooks ─────────────────────────────────────────

  /**
   * Render the widget header
   * @protected
   */
  _renderHeader() {
    const header = document.createElement('div');
    header.className = 'av-widget__header';
    header.innerHTML = `
      <h3 class="av-widget__title">${this.title}</h3>
      <span class="av-widget__status av-widget__status--${this.state}"></span>
    `;
    this.element.appendChild(header);
  }

  /**
   * Render the widget content area — override in subclasses
   * @protected
   */
  _renderContent() {
    const content = document.createElement('div');
    content.className = 'av-widget__content';
    content.innerHTML = '<div class="av-widget__placeholder">Awaiting data...</div>';
    this.element.appendChild(content);
  }

  /**
   * Update the widget content — override in subclasses
   * @protected
   * @param {Object} _oldData - Previous data
   */
  _updateContent(_oldData) {
    // Default: re-render content
    const content = this.element.querySelector('.av-widget__content');
    if (content) content.innerHTML = `<pre>${JSON.stringify(this.data, null, 2)}</pre>`;
  }

  /**
   * Render the widget footer
   * @protected
   */
  _renderFooter() {
    const footer = document.createElement('div');
    footer.className = 'av-widget__footer';
    footer.innerHTML = `<span class="av-widget__updated">--</span>`;
    this.element.appendChild(footer);
  }

  /**
   * Render an error state
   * @protected
   * @param {Error} err - The error
   */
  _renderError(err) {
    const content = this.element?.querySelector('.av-widget__content');
    if (content) {
      content.innerHTML = `<div class="av-widget__error">Error: ${err.message}</div>`;
    }
  }

  /**
   * Transform incoming data — override in subclasses
   * @protected
   * @param {Object} rawData - Raw data
   * @returns {Object} Transformed data
   */
  _transformData(rawData) {
    return rawData;
  }

  /**
   * Start the refresh interval
   * @private
   */
  _startRefresh() {
    if (this.refreshInterval > 0) {
      this._refreshTimer = setInterval(() => {
        this.emit('refresh', { id: this.id });
      }, this.refreshInterval);
    }
  }

  /**
   * Stop the refresh interval
   * @private
   */
  _stopRefresh() {
    if (this._refreshTimer) {
      clearInterval(this._refreshTimer);
      this._refreshTimer = null;
    }
  }

  /**
   * Update the timestamp in the footer
   * @protected
   */
  _updateFooterTimestamp() {
    const ts = this.element?.querySelector('.av-widget__updated');
    if (ts) ts.textContent = new Date(this.updatedAt).toLocaleTimeString();
  }
}

// ─── SpeedGauge Widget ───────────────────────────────────────────────────────

/**
 * Displays current vehicle speed with a gauge visualization
 * @extends Widget
 */
export class SpeedGauge extends Widget {
  /**
   * @param {string} id - Widget ID
   * @param {Object} [config={}]
   * @param {number} [config.maxSpeed=220] - Maximum speed on the gauge
   * @param {string} [config.unit='km/h'] - Speed unit
   */
  constructor(id, config = {}) {
    super(id, { title: config.title || 'Speed', ...config });
    /** @private */ this._maxSpeed = config.maxSpeed ?? 220;
    /** @private */ this._unit = config.unit || 'km/h';
    /** @private */ this._speedHistory = [];
  }

  /** @returns {string} */ get type() { return 'speed_gauge'; }

  /** @protected */ _transformData(raw) {
    const speed = typeof raw.speed === 'number' ? raw.speed : 0;
    this._speedHistory.push(speed);
    if (this._speedHistory.length > 300) this._speedHistory.shift();
    return {
      speed: Math.max(0, Math.min(speed, this._maxSpeed)),
      unit: this._unit,
      percentage: (speed / this._maxSpeed) * 100,
      history: [...this._speedHistory],
      timestamp: raw.timestamp || Date.now(),
    };
  }

  /** @protected */ _renderContent() {
    const content = document.createElement('div');
    content.className = 'av-widget__content av-speed-gauge';
    content.innerHTML = `
      <div class="av-speed-gauge__display">
        <svg class="av-speed-gauge__svg" viewBox="0 0 200 120">
          <path class="av-speed-gauge__track" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#2d3348" stroke-width="12" stroke-linecap="round"/>
          <path class="av-speed-gauge__fill" d="M 20 100 A 80 80 0 0 1 180 100" fill="none" stroke="#00d4aa" stroke-width="12" stroke-linecap="round" stroke-dasharray="251" stroke-dashoffset="251"/>
        </svg>
        <div class="av-speed-gauge__value">0</div>
        <div class="av-speed-gauge__unit">${this._unit}</div>
      </div>
    `;
    this.element.appendChild(content);
  }

  /** @protected */ _updateContent() {
    const valueEl = this.element?.querySelector('.av-speed-gauge__value');
    const fillEl = this.element?.querySelector('.av-speed-gauge__fill');
    if (valueEl) valueEl.textContent = this.data.speed.toFixed(0);
    if (fillEl) {
      const circumference = 251;
      const offset = circumference - (this.data.percentage / 100) * circumference;
      fillEl.style.strokeDashoffset = offset;
      fillEl.style.stroke = this.data.percentage > 80 ? '#ef4444' : this.data.percentage > 60 ? '#f59e0b' : '#00d4aa';
    }
    this._updateFooterTimestamp();
  }
}

// ─── BatteryIndicator Widget ─────────────────────────────────────────────────

/**
 * Displays battery level with charge status and estimated range
 * @extends Widget
 */
export class BatteryIndicator extends Widget {
  /**
   * @param {string} id - Widget ID
   * @param {Object} [config={}]
   * @param {number} [config.criticalThreshold=10] - Critical battery percentage
   * @param {number} [config.warningThreshold=25] - Warning battery percentage
   */
  constructor(id, config = {}) {
    super(id, { title: config.title || 'Battery', size: '1x1', ...config });
    /** @private */ this._criticalThreshold = config.criticalThreshold ?? 10;
    /** @private */ this._warningThreshold = config.warningThreshold ?? 25;
  }

  /** @returns {string} */ get type() { return 'battery_indicator'; }

  /** @protected */ _transformData(raw) {
    const level = typeof raw.batteryLevel === 'number' ? raw.batteryLevel : 0;
    const drainRate = raw.batteryDrainRate || 0;
    return {
      level: Math.max(0, Math.min(level, 100)),
      drainRate,
      isCharging: raw.isCharging || false,
      estimatedRangeKm: level > 0 && drainRate > 0 ? level / drainRate : null,
      status: level <= this._criticalThreshold ? 'critical' : level <= this._warningThreshold ? 'warning' : 'normal',
      temperature: raw.batteryTemperature || null,
      timestamp: raw.timestamp || Date.now(),
    };
  }

  /** @protected */ _renderContent() {
    const content = document.createElement('div');
    content.className = 'av-widget__content av-battery';
    content.innerHTML = `
      <div class="av-battery__icon">
        <div class="av-battery__body"><div class="av-battery__fill" style="width:0%"></div></div>
        <div class="av-battery__cap"></div>
      </div>
      <div class="av-battery__level">0%</div>
      <div class="av-battery__status">--</div>
    `;
    this.element.appendChild(content);
  }

  /** @protected */ _updateContent() {
    const fill = this.element?.querySelector('.av-battery__fill');
    const level = this.element?.querySelector('.av-battery__level');
    const status = this.element?.querySelector('.av-battery__status');
    if (fill) {
      fill.style.width = `${this.data.level}%`;
      fill.style.backgroundColor = this.data.status === 'critical' ? '#ef4444' : this.data.status === 'warning' ? '#f59e0b' : '#22c55e';
    }
    if (level) level.textContent = `${this.data.level.toFixed(1)}%`;
    if (status) {
      const rangeStr = this.data.estimatedRangeKm ? ` ~${this.data.estimatedRangeKm.toFixed(0)}km` : '';
      status.textContent = this.data.isCharging ? 'Charging' : `Drain: ${this.data.drainRate.toFixed(2)}/h${rangeStr}`;
    }
    this._updateFooterTimestamp();
  }
}

// ─── GpsMap Widget ───────────────────────────────────────────────────────────

/**
 * Displays GPS position on a map with route tracking
 * @extends Widget
 */
export class GpsMap extends Widget {
  /**
   * @param {string} id - Widget ID
   * @param {Object} [config={}]
   * @param {number} [config.zoomLevel=15] - Map zoom level
   * @param {boolean} [config.showRoute=true] - Show route trail
   * @param {number} [config.maxTrailPoints=500] - Maximum route trail points
   */
  constructor(id, config = {}) {
    super(id, { title: config.title || 'GPS Map', size: '2x2', ...config });
    /** @private */ this._zoom = config.zoomLevel ?? 15;
    /** @private */ this._showRoute = config.showRoute ?? true;
    /** @private @type {{ lat: number, lng: number }[]} */ this._trail = [];
    /** @private */ this._maxTrail = config.maxTrailPoints ?? 500;
  }

  /** @returns {string} */ get type() { return 'gps_map'; }

  /** @protected */ _transformData(raw) {
    const lat = raw.latitude ?? raw.lat ?? 0;
    const lng = raw.longitude ?? raw.lng ?? 0;
    if (lat !== 0 || lng !== 0) {
      this._trail.push({ lat, lng });
      if (this._trail.length > this._maxTrail) this._trail.shift();
    }
    return {
      latitude: lat,
      longitude: lng,
      heading: raw.heading ?? 0,
      speed: raw.speed ?? 0,
      altitude: raw.altitude ?? null,
      satelliteCount: raw.satelliteCount ?? null,
      hdop: raw.hdop ?? null,
      trail: [...this._trail],
      timestamp: raw.timestamp || Date.now(),
    };
  }

  /** @protected */ _renderContent() {
    const content = document.createElement('div');
    content.className = 'av-widget__content av-gps-map';
    content.innerHTML = `
      <canvas class="av-gps-map__canvas" width="400" height="300"></canvas>
      <div class="av-gps-map__overlay">
        <span class="av-gps-map__coords">0.0000, 0.0000</span>
        <span class="av-gps-map__heading">Heading: 0°</span>
        <span class="av-gps-map__satellites">Satellites: --</span>
      </div>
    `;
    this.element.appendChild(content);
  }

  /** @protected */ _updateContent() {
    const coords = this.element?.querySelector('.av-gps-map__coords');
    const heading = this.element?.querySelector('.av-gps-map__heading');
    const sats = this.element?.querySelector('.av-gps-map__satellites');
    if (coords) coords.textContent = `${this.data.latitude.toFixed(4)}, ${this.data.longitude.toFixed(4)}`;
    if (heading) heading.textContent = `Heading: ${this.data.heading.toFixed(1)}°`;
    if (sats) sats.textContent = `Satellites: ${this.data.satelliteCount ?? '--'}`;
    this._drawMap();
    this._updateFooterTimestamp();
  }

  /** @private */ _drawMap() {
    const canvas = this.element?.querySelector('.av-gps-map__canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#1a1d27';
    ctx.fillRect(0, 0, w, h);
    if (this._trail.length < 2) return;
    let minLat = Infinity, maxLat = -Infinity, minLng = Infinity, maxLng = -Infinity;
    for (const p of this._trail) {
      minLat = Math.min(minLat, p.lat); maxLat = Math.max(maxLat, p.lat);
      minLng = Math.min(minLng, p.lng); maxLng = Math.max(maxLng, p.lng);
    }
    const pad = 20;
    const scaleX = (w - 2 * pad) / (maxLng - minLng || 0.001);
    const scaleY = (h - 2 * pad) / (maxLat - minLat || 0.001);
    ctx.beginPath();
    ctx.strokeStyle = '#00d4aa';
    ctx.lineWidth = 2;
    for (let i = 0; i < this._trail.length; i++) {
      const x = pad + (this._trail[i].lng - minLng) * scaleX;
      const y = h - pad - (this._trail[i].lat - minLat) * scaleY;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.stroke();
    const last = this._trail[this._trail.length - 1];
    const cx = pad + (last.lng - minLng) * scaleX;
    const cy = h - pad - (last.lat - minLat) * scaleY;
    ctx.beginPath(); ctx.arc(cx, cy, 5, 0, Math.PI * 2);
    ctx.fillStyle = '#ef4444'; ctx.fill();
  }
}

// ─── SensorStatus Widget ─────────────────────────────────────────────────────

/**
 * Displays status and health of vehicle sensors
 * @extends Widget
 */
export class SensorStatus extends Widget {
  /**
   * @param {string} id - Widget ID
   * @param {Object} [config={}]
   */
  constructor(id, config = {}) {
    super(id, { title: config.title || 'Sensor Status', size: '3x1', ...config });
    /** @private @type {Map<string, Object>} */ this._sensors = new Map();
  }

  /** @returns {string} */ get type() { return 'sensor_status'; }

  /** @protected */ _transformData(raw) {
    if (raw.sensors) {
      for (const [id, val] of Object.entries(raw.sensors)) {
        this._sensors.set(id, {
          id,
          status: val.status || (val.active ? 'active' : 'inactive'),
          lastReading: val.value ?? val.lastReading ?? null,
          health: val.health ?? 1.0,
          updatedAt: Date.now(),
        });
      }
    }
    const activeCount = Array.from(this._sensors.values()).filter((s) => s.status === 'active').length;
    return {
      sensors: Object.fromEntries(this._sensors),
      totalSensors: this._sensors.size,
      activeSensors: activeCount,
      healthScore: this._sensors.size > 0
        ? Array.from(this._sensors.values()).reduce((sum, s) => sum + s.health, 0) / this._sensors.size
        : 0,
    };
  }

  /** @protected */ _renderContent() {
    const content = document.createElement('div');
    content.className = 'av-widget__content av-sensor-status';
    content.innerHTML = `<div class="av-sensor-status__grid max-h-96 overflow-y-auto">Awaiting sensor data...</div>`;
    this.element.appendChild(content);
  }

  /** @protected */ _updateContent() {
    const grid = this.element?.querySelector('.av-sensor-status__grid');
    if (!grid) return;
    const rows = Object.values(this.data.sensors).map((s) => {
      const dotColor = s.status === 'active' ? '#22c55e' : s.status === 'warning' ? '#f59e0b' : '#ef4444';
      return `<div class="av-sensor-status__row">
        <span class="av-sensor-status__dot" style="background:${dotColor}"></span>
        <span class="av-sensor-status__name">${s.id}</span>
        <span class="av-sensor-status__health">${(s.health * 100).toFixed(0)}%</span>
      </div>`;
    }).join('');
    grid.innerHTML = rows;
    this._updateFooterTimestamp();
  }
}

// ─── ObjectDetectionFeed Widget ──────────────────────────────────────────────

/**
 * Displays real-time object detection results with bounding boxes
 * @extends Widget
 */
export class ObjectDetectionFeed extends Widget {
  /**
   * @param {string} id - Widget ID
   * @param {Object} [config={}]
   * @param {number} [config.maxObjects=50] - Maximum objects to track
   * @param {number} [config.confidenceThreshold=0.5] - Minimum confidence
   */
  constructor(id, config = {}) {
    super(id, { title: config.title || 'Object Detection', size: '2x2', ...config });
    /** @private */ this._maxObjects = config.maxObjects ?? 50;
    /** @private */ this._confidenceThreshold = config.confidenceThreshold ?? 0.5;
  }

  /** @returns {string} */ get type() { return 'object_detection'; }

  /** @protected */ _transformData(raw) {
    const objects = (raw.detectedObjects || []).filter((o) => (o.confidence ?? 1) >= this._confidenceThreshold).slice(0, this._maxObjects);
    const typeCounts = {};
    for (const obj of objects) {
      typeCounts[obj.type] = (typeCounts[obj.type] || 0) + 1;
    }
    return {
      objects,
      totalCount: objects.length,
      typeCounts,
      closestDistance: objects.length > 0 ? Math.min(...objects.map((o) => o.distance ?? Infinity)) : null,
      timestamp: raw.timestamp || Date.now(),
    };
  }

  /** @protected */ _renderContent() {
    const content = document.createElement('div');
    content.className = 'av-widget__content av-object-detection';
    content.innerHTML = `
      <canvas class="av-object-detection__canvas" width="400" height="300"></canvas>
      <div class="av-object-detection__summary">
        <span class="av-object-detection__count">Objects: 0</span>
      </div>
    `;
    this.element.appendChild(content);
  }

  /** @protected */ _updateContent() {
    const count = this.element?.querySelector('.av-object-detection__count');
    if (count) count.textContent = `Objects: ${this.data.totalCount}`;
    const canvas = this.element?.querySelector('.av-object-detection__canvas');
    if (canvas) {
      const ctx = canvas.getContext('2d');
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.fillStyle = '#0f1117';
      ctx.fillRect(0, 0, canvas.width, canvas.height);
      for (const obj of this.data.objects) {
        if (obj.bbox) {
          const [x, y, w, h] = obj.bbox;
          ctx.strokeStyle = obj.type === 'pedestrian' ? '#ef4444' : obj.type === 'vehicle' ? '#f59e0b' : '#3b82f6';
          ctx.lineWidth = 2;
          ctx.strokeRect(x, y, w, h);
          ctx.fillStyle = ctx.strokeStyle;
          ctx.font = '10px monospace';
          ctx.fillText(`${obj.type} ${(obj.confidence * 100).toFixed(0)}%`, x, y - 4);
        }
      }
    }
    this._updateFooterTimestamp();
  }
}

// ─── LaneDepartureIndicator Widget ───────────────────────────────────────────

/**
 * Displays lane departure warning status and lane position
 * @extends Widget
 */
export class LaneDepartureIndicator extends Widget {
  /**
   * @param {string} id - Widget ID
   * @param {Object} [config={}]
   * @param {number} [config.warningThreshold=0.3] - Lane offset for warning
   */
  constructor(id, config = {}) {
    super(id, { title: config.title || 'Lane Departure', size: '1x1', ...config });
    /** @private */ this._warningThreshold = config.warningThreshold ?? 0.3;
  }

  /** @returns {string} */ get type() { return 'lane_departure'; }

  /** @protected */ _transformData(raw) {
    const offset = raw.laneOffset ?? 0;
    const absOffset = Math.abs(offset);
    return {
      laneOffset: offset,
      lanePosition: offset < -this._warningThreshold ? 'left_departure' : offset > this._warningThreshold ? 'right_departure' : 'centered',
      isDeparting: absOffset > this._warningThreshold,
      confidence: raw.laneConfidence ?? 1.0,
      lanesDetected: raw.lanesDetected ?? 0,
      timestamp: raw.timestamp || Date.now(),
    };
  }

  /** @protected */ _renderContent() {
    const content = document.createElement('div');
    content.className = 'av-widget__content av-lane-departure';
    content.innerHTML = `
      <div class="av-lane-departure__visual">
        <div class="av-lane-departure__lane-left"></div>
        <div class="av-lane-departure__car"></div>
        <div class="av-lane-departure__lane-right"></div>
      </div>
      <div class="av-lane-departure__status">Centered</div>
    `;
    this.element.appendChild(content);
  }

  /** @protected */ _updateContent() {
    const car = this.element?.querySelector('.av-lane-departure__car');
    const status = this.element?.querySelector('.av-lane-departure__status');
    if (car) car.style.transform = `translateX(${this.data.laneOffset * 100}px)`;
    if (status) {
      status.textContent = this.data.isDeparting ? `⚠ ${this.data.lanePosition.replace('_', ' ')}` : 'Centered';
      status.style.color = this.data.isDeparting ? '#ef4444' : '#22c55e';
    }
    this._updateFooterTimestamp();
  }
}

// ─── WeatherWidget ───────────────────────────────────────────────────────────

/**
 * Displays current weather conditions affecting vehicle operation
 * @extends Widget
 */
export class WeatherWidget extends Widget {
  /**
   * @param {string} id - Widget ID
   * @param {Object} [config={}]
   */
  constructor(id, config = {}) {
    super(id, { title: config.title || 'Weather', size: '1x1', ...config });
  }

  /** @returns {string} */ get type() { return 'weather'; }

  /** @protected */ _transformData(raw) {
    return {
      temperature: raw.temperature ?? null,
      humidity: raw.humidity ?? null,
      windSpeed: raw.windSpeed ?? null,
      windDirection: raw.windDirection ?? null,
      condition: raw.condition || 'unknown',
      visibility: raw.visibility ?? null,
      precipitation: raw.precipitation ?? false,
      roadCondition: raw.roadCondition || 'dry',
      drivingRisk: this._computeRisk(raw),
      timestamp: raw.timestamp || Date.now(),
    };
  }

  /** @private */ _computeRisk(raw) {
    let risk = 0;
    if (raw.visibility && raw.visibility < 500) risk += 2;
    if (raw.precipitation) risk += 1;
    if (raw.roadCondition === 'icy') risk += 3;
    else if (raw.roadCondition === 'wet') risk += 1;
    if (raw.windSpeed && raw.windSpeed > 50) risk += 2;
    if (risk >= 4) return 'high';
    if (risk >= 2) return 'moderate';
    return 'low';
  }

  /** @protected */ _renderContent() {
    const content = document.createElement('div');
    content.className = 'av-widget__content av-weather';
    content.innerHTML = `
      <div class="av-weather__condition">--</div>
      <div class="av-weather__temp">--°C</div>
      <div class="av-weather__details">
        <span>Humidity: --%</span>
        <span>Wind: -- km/h</span>
        <span>Road: --</span>
      </div>
      <div class="av-weather__risk">Risk: --</div>
    `;
    this.element.appendChild(content);
  }

  /** @protected */ _updateContent() {
    const cond = this.element?.querySelector('.av-weather__condition');
    const temp = this.element?.querySelector('.av-weather__temp');
    const details = this.element?.querySelector('.av-weather__details');
    const risk = this.element?.querySelector('.av-weather__risk');
    if (cond) cond.textContent = this.data.condition;
    if (temp) temp.textContent = this.data.temperature !== null ? `${this.data.temperature.toFixed(1)}°C` : '--°C';
    if (details) details.innerHTML = `
      <span>Humidity: ${this.data.humidity ?? '--'}%</span>
      <span>Wind: ${this.data.windSpeed ?? '--'} km/h</span>
      <span>Road: ${this.data.roadCondition}</span>
    `;
    if (risk) {
      risk.textContent = `Risk: ${this.data.drivingRisk}`;
      risk.style.color = this.data.drivingRisk === 'high' ? '#ef4444' : this.data.drivingRisk === 'moderate' ? '#f59e0b' : '#22c55e';
    }
    this._updateFooterTimestamp();
  }
}

// ─── TripInfoWidget ──────────────────────────────────────────────────────────

/**
 * Displays trip information: distance, duration, avg speed, ETA
 * @extends Widget
 */
export class TripInfoWidget extends Widget {
  /**
   * @param {string} id - Widget ID
   * @param {Object} [config={}]
   */
  constructor(id, config = {}) {
    super(id, { title: config.title || 'Trip Info', size: '2x1', ...config });
    /** @private */ this._tripStart = null;
  }

  /** @returns {string} */ get type() { return 'trip_info'; }

  /** @protected */ _transformData(raw) {
    if (raw.tripActive && !this._tripStart) {
      this._tripStart = raw.tripStartTime || Date.now();
    }
    const duration = this._tripStart ? Date.now() - this._tripStart : 0;
    const distance = raw.tripDistance ?? 0;
    const avgSpeed = duration > 0 ? (distance / (duration / 3600000)) : 0;
    return {
      tripActive: raw.tripActive ?? false,
      distance: distance,
      duration: duration,
      avgSpeed: avgSpeed,
      currentSpeed: raw.speed ?? 0,
      eta: raw.eta ?? null,
      destination: raw.destination ?? null,
      remainingDistance: raw.remainingDistance ?? null,
      formatDuration: this._formatDuration(duration),
      timestamp: raw.timestamp || Date.now(),
    };
  }

  /** @private */ _formatDuration(ms) {
    const hours = Math.floor(ms / 3600000);
    const minutes = Math.floor((ms % 3600000) / 60000);
    const seconds = Math.floor((ms % 60000) / 1000);
    return `${hours}h ${minutes}m ${seconds}s`;
  }

  /** @protected */ _renderContent() {
    const content = document.createElement('div');
    content.className = 'av-widget__content av-trip-info';
    content.innerHTML = `
      <div class="av-trip-info__grid">
        <div class="av-trip-info__item"><span class="av-trip-info__label">Distance</span><span class="av-trip-info__value">0.0 km</span></div>
        <div class="av-trip-info__item"><span class="av-trip-info__label">Duration</span><span class="av-trip-info__value">0h 0m</span></div>
        <div class="av-trip-info__item"><span class="av-trip-info__label">Avg Speed</span><span class="av-trip-info__value">0 km/h</span></div>
        <div class="av-trip-info__item"><span class="av-trip-info__label">ETA</span><span class="av-trip-info__value">--</span></div>
      </div>
    `;
    this.element.appendChild(content);
  }

  /** @protected */ _updateContent() {
    const items = this.element?.querySelectorAll('.av-trip-info__value');
    if (items && items.length >= 4) {
      items[0].textContent = `${this.data.distance.toFixed(1)} km`;
      items[1].textContent = this.data.formatDuration;
      items[2].textContent = `${this.data.avgSpeed.toFixed(1)} km/h`;
      items[3].textContent = this.data.eta ? new Date(this.data.eta).toLocaleTimeString() : '--';
    }
    this._updateFooterTimestamp();
  }
}

// ─── Widget Factory ──────────────────────────────────────────────────────────

/** @type {Object<string, typeof Widget>} */
const WIDGET_CLASSES = {
  speed_gauge: SpeedGauge,
  battery_indicator: BatteryIndicator,
  gps_map: GpsMap,
  sensor_status: SensorStatus,
  object_detection: ObjectDetectionFeed,
  lane_departure: LaneDepartureIndicator,
  weather: WeatherWidget,
  trip_info: TripInfoWidget,
};

/**
 * Factory function to create widget instances
 * @param {string} type - Widget type
 * @param {string} id - Widget ID
 * @param {Object} [config={}] - Widget configuration
 * @returns {Widget} Widget instance
 * @throws {Error} If widget type is unknown
 */
export function createWidget(type, id, config = {}) {
  const WidgetClass = WIDGET_CLASSES[type];
  if (!WidgetClass) throw new Error(`Unknown widget type: ${type}`);
  return new WidgetClass(id, config);
}

/**
 * Get all available widget type names
 * @returns {string[]}
 */
export function getAvailableWidgetTypes() {
  return Object.keys(WIDGET_CLASSES);
}
