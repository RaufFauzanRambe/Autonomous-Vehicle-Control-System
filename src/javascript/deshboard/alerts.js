/**
 * @fileoverview Alert System
 * @description AlertManager, severity levels (critical/warning/info), alert rules engine,
 *   threshold monitoring, alert history, acknowledgment workflow, notification dispatch
 *   (console/websocket), and alert deduplication for the Autonomous Vehicle Dashboard.
 * @module alerts
 */

import { EventEmitter } from 'events';

// ─── Severity Levels ─────────────────────────────────────────────────────────

/** @enum {string} */
export const Severity = {
  CRITICAL: 'critical',
  WARNING: 'warning',
  INFO: 'info',
};

/** @type {Object<string, number>} */
export const SEVERITY_PRIORITY = {
  [Severity.CRITICAL]: 3,
  [Severity.WARNING]: 2,
  [Severity.INFO]: 1,
};

/** @type {Object<string, string>} */
export const SEVERITY_COLORS = {
  [Severity.CRITICAL]: '#ef4444',
  [Severity.WARNING]: '#f59e0b',
  [Severity.INFO]: '#3b82f6',
};

// ─── Alert Class ─────────────────────────────────────────────────────────────

/**
 * Represents a single alert instance
 */
export class Alert {
  /**
   * @param {Object} config
   * @param {string} config.id - Unique alert identifier
   * @param {string} config.ruleId - Rule that triggered the alert
   * @param {string} config.severity - Severity level
   * @param {string} config.message - Alert message
   * @param {Object} [config.data={}] - Additional alert data
   * @param {string} [config.source='system'] - Alert source
   */
  constructor(config) {
    this.id = config.id;
    this.ruleId = config.ruleId;
    this.severity = config.severity;
    this.message = config.message;
    this.data = config.data || {};
    this.source = config.source || 'system';
    this.createdAt = Date.now();
    this.updatedAt = Date.now();
    this.acknowledged = false;
    this.acknowledgedBy = null;
    this.acknowledgedAt = null;
    this.resolved = false;
    this.resolvedAt = null;
    this.notificationSent = false;
    this.duplicateCount = 0;
  }

  /**
   * Check if the alert is active (not resolved)
   * @returns {boolean}
   */
  get isActive() {
    return !this.resolved;
  }

  /**
   * Get the age of the alert in milliseconds
   * @returns {number}
   */
  get age() {
    return Date.now() - this.createdAt;
  }

  /**
   * Serialize the alert to a plain object
   * @returns {Object}
   */
  toJSON() {
    return {
      id: this.id,
      ruleId: this.ruleId,
      severity: this.severity,
      message: this.message,
      data: this.data,
      source: this.source,
      createdAt: this.createdAt,
      updatedAt: this.updatedAt,
      acknowledged: this.acknowledged,
      acknowledgedBy: this.acknowledgedBy,
      acknowledgedAt: this.acknowledgedAt,
      resolved: this.resolved,
      resolvedAt: this.resolvedAt,
      duplicateCount: this.duplicateCount,
      isActive: this.isActive,
      age: this.age,
    };
  }
}

// ─── Alert Rule ──────────────────────────────────────────────────────────────

/**
 * Defines a rule for automatic alert generation
 */
export class AlertRule {
  /**
   * @param {Object} config
   * @param {string} config.id - Rule identifier
   * @param {string} config.name - Rule display name
   * @param {string} config.severity - Default severity
   * @param {string} config.message - Alert message template
   * @param {Function} config.condition - Evaluation function (data) => boolean
   * @param {number} [config.cooldownMs=30000] - Minimum time between alerts
   * @param {number} [config.maxOccurrences=0] - Max alerts (0 = unlimited)
   * @param {boolean} [config.enabled=true] - Whether rule is active
   * @param {string} [config.category='general'] - Rule category
   */
  constructor(config) {
    this.id = config.id;
    this.name = config.name;
    this.severity = config.severity || Severity.WARNING;
    this.message = config.message;
    this.condition = config.condition;
    this.cooldownMs = config.cooldownMs ?? 30000;
    this.maxOccurrences = config.maxOccurrences ?? 0;
    this.enabled = config.enabled !== false;
    this.category = config.category || 'general';
    /** @private */ this._lastTriggered = 0;
    /** @private */ this._occurrenceCount = 0;
  }

  /**
   * Evaluate the rule against incoming data
   * @param {Object} data - Telemetry data to evaluate
   * @returns {boolean} Whether the rule triggers
   */
  evaluate(data) {
    if (!this.enabled) return false;
    if (this.maxOccurrences > 0 && this._occurrenceCount >= this.maxOccurrences) return false;
    if (Date.now() - this._lastTriggered < this.cooldownMs) return false;

    try {
      const triggered = this.condition(data);
      if (triggered) {
        this._lastTriggered = Date.now();
        this._occurrenceCount++;
      }
      return triggered;
    } catch (err) {
      console.error(`Alert rule ${this.id} evaluation error:`, err.message);
      return false;
    }
  }

  /** Reset the rule state */ reset() { this._lastTriggered = 0; this._occurrenceCount = 0; }
}

// ─── Notification Dispatchers ────────────────────────────────────────────────

/**
 * Console notification dispatcher
 */
class ConsoleNotifier {
  /**
   * Send an alert notification to console
   * @param {Alert} alert - Alert to notify about
   */
  dispatch(alert) {
    const color = SEVERITY_COLORS[alert.severity] || '#ffffff';
    const prefix = `[${alert.severity.toUpperCase()}]`;
    const timestamp = new Date(alert.createdAt).toISOString();
    console.log(`${prefix} ${timestamp} - ${alert.message} (rule: ${alert.ruleId}, id: ${alert.id})`);
  }
}

/**
 * WebSocket notification dispatcher
 * @extends EventEmitter
 */
class WebSocketNotifier extends EventEmitter {
  constructor() {
    super();
    /** @private @type {Set<Object>} */ this._clients = new Set();
  }

  /**
   * Register a WebSocket client for notifications
   * @param {Object} client - Socket.io client
   */
  addClient(client) {
    this._clients.add(client);
  }

  /**
   * Remove a WebSocket client
   * @param {Object} client - Socket.io client
   */
  removeClient(client) {
    this._clients.delete(client);
  }

  /**
   * Dispatch an alert to all connected WebSocket clients
   * @param {Alert} alert - Alert to dispatch
   */
  dispatch(alert) {
    const payload = alert.toJSON();
    for (const client of this._clients) {
      try {
        client.emit('alert:notification', payload);
      } catch (err) {
        this._clients.delete(client);
      }
    }
    this.emit('dispatched', payload);
  }
}

// ─── Alert Deduplication ─────────────────────────────────────────────────────

/**
 * Deduplicates alerts by rule ID with a configurable cooldown window
 */
class AlertDeduplicator {
  /**
   * @param {Object} [options={}]
   * @param {number} [options.windowMs=60000] - Deduplication window in ms
   * @param {number} [options.maxDuplicates=100] - Max tracked rules
   */
  constructor(options = {}) {
    /** @private @type {Map<string, { count: number, lastSeen: number, lastAlertId: string }>} */
    this._seen = new Map();
    /** @private */ this._windowMs = options.windowMs ?? 60000;
    /** @private */ this._maxDuplicates = options.maxDuplicates ?? 100;
  }

  /**
   * Check if an alert should be deduplicated (suppressed)
   * @param {string} ruleId - Rule identifier
   * @param {string} alertId - Alert identifier
   * @returns {boolean} True if alert should be suppressed
   */
  isDuplicate(ruleId, alertId) {
    const entry = this._seen.get(ruleId);
    if (!entry) return false;

    const withinWindow = Date.now() - entry.lastSeen < this._windowMs;
    if (withinWindow) {
      entry.count++;
      entry.lastSeen = Date.now();
      return true;
    }
    return false;
  }

  /**
   * Record that an alert was seen
   * @param {string} ruleId - Rule identifier
   * @param {string} alertId - Alert identifier
   */
  record(ruleId, alertId) {
    if (this._seen.size >= this._maxDuplicates) {
      // Evict oldest entries
      const entries = [...this._seen.entries()].sort((a, b) => a[1].lastSeen - b[1].lastSeen);
      for (let i = 0; i < Math.floor(this._maxDuplicates / 2); i++) {
        this._seen.delete(entries[i][0]);
      }
    }
    this._seen.set(ruleId, { count: 1, lastSeen: Date.now(), lastAlertId: alertId });
  }

  /**
   * Get deduplication statistics
   * @returns {Object}
   */
  getStats() {
    let totalDuplicates = 0;
    for (const entry of this._seen.values()) {
      totalDuplicates += entry.count - 1;
    }
    return { trackedRules: this._seen.size, totalDuplicatesSuppressed: totalDuplicates };
  }

  /** Clear all deduplication state */ clear() { this._seen.clear(); }
}

// ─── Alert Manager ───────────────────────────────────────────────────────────

/**
 * Central alert management system
 * @extends EventEmitter
 */
export class AlertManager extends EventEmitter {
  /**
   * @param {Object} [options={}]
   * @param {number} [options.maxHistory=1000] - Maximum alerts in history
   * @param {number} [options.autoResolveMs=0] - Auto-resolve after ms (0 = disabled)
   * @param {boolean} [options.consoleNotifications=true] - Enable console notifications
   * @param {boolean} [options.websocketNotifications=true] - Enable WebSocket notifications
   */
  constructor(options = {}) {
    super();
    /** @private @type {Map<string, Alert>} */ this._activeAlerts = new Map();
    /** @private @type {Alert[]} */ this._history = [];
    /** @private @type {Map<string, AlertRule>} */ this._rules = new Map();
    /** @private */ this._maxHistory = options.maxHistory ?? 1000;
    /** @private */ this._autoResolveMs = options.autoResolveMs ?? 0;
    /** @private @type {ConsoleNotifier|null} */ this._consoleNotifier = options.consoleNotifications !== false ? new ConsoleNotifier() : null;
    /** @private */ this._wsNotifier = options.websocketNotifications !== false ? new WebSocketNotifier() : null;
    /** @private */ this._deduplicator = new AlertDeduplicator();
    /** @private */ this._ready = true;
    /** @private */ this._alertCounter = 0;
    /** @private */ this._autoResolveInterval = null;

    this._registerBuiltinRules();

    if (this._autoResolveMs > 0) {
      this._startAutoResolve();
    }
  }

  /**
   * Register built-in alert rules for the vehicle dashboard
   * @private
   */
  _registerBuiltinRules() {
    const builtinRules = [
      new AlertRule({
        id: 'speed_exceeded',
        name: 'Speed Exceeded',
        severity: Severity.CRITICAL,
        message: 'Vehicle speed exceeds safety threshold',
        condition: (data) => data.speed !== undefined && data.speed > 180,
        cooldownMs: 10000,
        category: 'safety',
      }),
      new AlertRule({
        id: 'low_battery',
        name: 'Low Battery',
        severity: Severity.WARNING,
        message: 'Battery level is critically low',
        condition: (data) => data.batteryLevel !== undefined && data.batteryLevel < 15,
        cooldownMs: 30000,
        category: 'power',
      }),
      new AlertRule({
        id: 'battery_critical',
        name: 'Battery Critical',
        severity: Severity.CRITICAL,
        message: 'Battery level is dangerously low',
        condition: (data) => data.batteryLevel !== undefined && data.batteryLevel < 5,
        cooldownMs: 5000,
        category: 'power',
      }),
      new AlertRule({
        id: 'sensor_offline',
        name: 'Sensor Offline',
        severity: Severity.WARNING,
        message: 'A critical sensor has gone offline',
        condition: (data) => data.sensorOffline === true,
        cooldownMs: 15000,
        category: 'sensors',
      }),
      new AlertRule({
        id: 'lane_departure',
        name: 'Lane Departure Warning',
        severity: Severity.WARNING,
        message: 'Vehicle is departing from its lane',
        condition: (data) => data.laneDeparture === true || (data.laneOffset !== undefined && Math.abs(data.laneOffset) > 0.3),
        cooldownMs: 5000,
        category: 'safety',
      }),
      new AlertRule({
        id: 'object_proximity',
        name: 'Object Proximity Alert',
        severity: Severity.CRITICAL,
        message: 'Object detected in close proximity',
        condition: (data) => data.closestObjectDistance !== undefined && data.closestObjectDistance < 5,
        cooldownMs: 3000,
        category: 'safety',
      }),
      new AlertRule({
        id: 'high_temperature',
        name: 'High Temperature',
        severity: Severity.WARNING,
        message: 'System temperature exceeds threshold',
        condition: (data) => data.temperature !== undefined && data.temperature > 85,
        cooldownMs: 60000,
        category: 'system',
      }),
      new AlertRule({
        id: 'data_quality',
        name: 'Data Quality Degraded',
        severity: Severity.INFO,
        message: 'Telemetry data quality has degraded',
        condition: (data) => data._qualityScore !== undefined && data._qualityScore < 0.5,
        cooldownMs: 120000,
        category: 'system',
      }),
    ];

    for (const rule of builtinRules) {
      this._rules.set(rule.id, rule);
    }
  }

  /**
   * Manually fire an alert
   * @param {string} ruleId - Rule identifier or custom ID
   * @param {string} severity - Severity level
   * @param {string} message - Alert message
   * @param {Object} [data={}] - Additional data
   * @returns {Alert} The created alert
   */
  fireAlert(ruleId, severity, message, data = {}) {
    if (!SEVERITY_PRIORITY[severity]) {
      severity = Severity.WARNING;
    }

    const alertId = `alert_${++this._alertCounter}_${Date.now()}`;

    // Check deduplication
    if (this._deduplicator.isDuplicate(ruleId, alertId)) {
      // Update the existing alert's duplicate count
      const existingAlert = this._activeAlerts.get(ruleId);
      if (existingAlert) {
        existingAlert.duplicateCount++;
        existingAlert.updatedAt = Date.now();
        this.emit('alert:duplicate', existingAlert.toJSON());
      }
      return existingAlert;
    }

    this._deduplicator.record(ruleId, alertId);

    const alert = new Alert({
      id: alertId,
      ruleId,
      severity,
      message,
      data,
    });

    this._activeAlerts.set(alertId, alert);
    this._addToHistory(alert);

    // Dispatch notifications
    this._consoleNotifier?.dispatch(alert);
    this._wsNotifier?.dispatch(alert);

    this.emit('alertFired', alert.toJSON());
    return alert;
  }

  /**
   * Evaluate all rules against incoming data
   * @param {Object} data - Telemetry data to evaluate
   * @returns {Alert[]} Alerts triggered by the data
   */
  evaluateRules(data) {
    const triggered = [];
    for (const rule of this._rules.values()) {
      if (rule.evaluate(data)) {
        const alert = this.fireAlert(rule.id, rule.severity, rule.message, data);
        triggered.push(alert);
      }
    }
    return triggered;
  }

  /**
   * Acknowledge an alert
   * @param {string} alertId - Alert ID
   * @param {string} [userId='system'] - User who acknowledged
   * @returns {boolean} Whether acknowledgment succeeded
   */
  acknowledgeAlert(alertId, userId = 'system') {
    const alert = this._activeAlerts.get(alertId);
    if (!alert || alert.acknowledged) return false;

    alert.acknowledged = true;
    alert.acknowledgedBy = userId;
    alert.acknowledgedAt = Date.now();
    alert.updatedAt = Date.now();

    this.emit('alert:acknowledged', alert.toJSON());
    return true;
  }

  /**
   * Resolve an alert
   * @param {string} alertId - Alert ID
   * @returns {boolean} Whether resolution succeeded
   */
  resolveAlert(alertId) {
    const alert = this._activeAlerts.get(alertId);
    if (!alert || alert.resolved) return false;

    alert.resolved = true;
    alert.resolvedAt = Date.now();
    alert.updatedAt = Date.now();

    this._activeAlerts.delete(alertId);
    this._addToHistory(alert);

    this.emit('alert:resolved', alert.toJSON());
    return true;
  }

  /**
   * Resolve all active alerts
   * @returns {number} Number of alerts resolved
   */
  resolveAll() {
    const count = this._activeAlerts.size;
    for (const alertId of this._activeAlerts.keys()) {
      this.resolveAlert(alertId);
    }
    return count;
  }

  // ─── Query Methods ──────────────────────────────────────────────────────

  /**
   * Get all active alerts
   * @param {Object} [filter={}] - Filter options
   * @returns {Alert[]}
   */
  getActiveAlerts(filter = {}) {
    let alerts = Array.from(this._activeAlerts.values());
    if (filter.severity) alerts = alerts.filter((a) => a.severity === filter.severity);
    if (filter.category) alerts = alerts.filter((a) => a.data?.category === filter.category || a.ruleId.includes(filter.category));
    if (filter.acknowledged !== undefined) alerts = alerts.filter((a) => a.acknowledged === filter.acknowledged);
    return alerts.sort((a, b) => SEVERITY_PRIORITY[b.severity] - SEVERITY_PRIORITY[a.severity] || b.createdAt - a.createdAt);
  }

  /**
   * Get alert history
   * @param {Object} [filter={}] - Filter options
   * @param {number} [limit=50] - Max results
   * @returns {Object[]}
   */
  getAlertHistory(filter = {}, limit = 50) {
    let history = [...this._history];
    if (filter.severity) history = history.filter((a) => a.severity === filter.severity);
    if (filter.ruleId) history = history.filter((a) => a.ruleId === filter.ruleId);
    if (filter.since) history = history.filter((a) => a.createdAt >= filter.since);
    return history.slice(0, limit).map((a) => a.toJSON());
  }

  /**
   * Get a single alert by ID
   * @param {string} alertId - Alert ID
   * @returns {Alert|undefined}
   */
  getAlert(alertId) {
    return this._activeAlerts.get(alertId) || this._history.find((a) => a.id === alertId);
  }

  // ─── Rule Management ────────────────────────────────────────────────────

  /**
   * Add a custom alert rule
   * @param {AlertRule} rule - Rule to add
   */
  addRule(rule) {
    this._rules.set(rule.id, rule);
    this.emit('rule:added', { id: rule.id });
  }

  /**
   * Remove an alert rule
   * @param {string} ruleId - Rule ID
   */
  removeRule(ruleId) {
    this._rules.delete(ruleId);
    this.emit('rule:removed', { id: ruleId });
  }

  /**
   * Enable a rule
   * @param {string} ruleId - Rule ID
   */
  enableRule(ruleId) {
    const rule = this._rules.get(ruleId);
    if (rule) rule.enabled = true;
  }

  /**
   * Disable a rule
   * @param {string} ruleId - Rule ID
   */
  disableRule(ruleId) {
    const rule = this._rules.get(ruleId);
    if (rule) rule.enabled = false;
  }

  /**
   * Get all registered rules
   * @returns {Object[]}
   */
  getRules() {
    return Array.from(this._rules.values()).map((r) => ({
      id: r.id, name: r.name, severity: r.severity, enabled: r.enabled, category: r.category,
    }));
  }

  // ─── WebSocket Integration ──────────────────────────────────────────────

  /**
   * Register a WebSocket client for alert notifications
   * @param {Object} client - Socket.io client
   */
  registerWSClient(client) {
    this._wsNotifier?.addClient(client);
  }

  /**
   * Unregister a WebSocket client
   * @param {Object} client - Socket.io client
   */
  unregisterWSClient(client) {
    this._wsNotifier?.removeClient(client);
  }

  // ─── Status ─────────────────────────────────────────────────────────────

  /** @returns {boolean} */ isReady() { return this._ready; }

  /**
   * Get alert manager statistics
   * @returns {Object}
   */
  getStats() {
    const active = Array.from(this._activeAlerts.values());
    return {
      activeCount: active.length,
      criticalCount: active.filter((a) => a.severity === Severity.CRITICAL).length,
      warningCount: active.filter((a) => a.severity === Severity.WARNING).length,
      infoCount: active.filter((a) => a.severity === Severity.INFO).length,
      unacknowledgedCount: active.filter((a) => !a.acknowledged).length,
      historyCount: this._history.length,
      ruleCount: this._rules.size,
      enabledRuleCount: Array.from(this._rules.values()).filter((r) => r.enabled).length,
      deduplication: this._deduplicator.getStats(),
    };
  }

  // ─── Private Methods ────────────────────────────────────────────────────

  /**
   * Add an alert to history
   * @private
   * @param {Alert} alert
   */
  _addToHistory(alert) {
    this._history.push(alert);
    if (this._history.length > this._maxHistory) {
      this._history.shift();
    }
  }

  /**
   * Start auto-resolve timer
   * @private
   */
  _startAutoResolve() {
    this._autoResolveInterval = setInterval(() => {
      const now = Date.now();
      for (const [id, alert] of this._activeAlerts.entries()) {
        if (now - alert.createdAt > this._autoResolveMs) {
          this.resolveAlert(id);
        }
      }
    }, 10000);
  }

  /**
   * Stop auto-resolve timer
   */
  stopAutoResolve() {
    if (this._autoResolveInterval) {
      clearInterval(this._autoResolveInterval);
      this._autoResolveInterval = null;
    }
  }
}
