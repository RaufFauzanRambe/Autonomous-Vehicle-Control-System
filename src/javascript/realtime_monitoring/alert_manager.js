/**
 * @fileoverview AlertManager - Alert lifecycle management for the autonomous
 * vehicle control system. Handles alert creation with severity/source/message,
 * alert lifecycle transitions (active→acknowledged→resolved), escalation rules,
 * alert grouping/deduplication, notification channel dispatch, alert rules
 * engine with threshold and anomaly detection, and alert statistics.
 *
 * @module realtime_monitoring/alert_manager
 * @version 2.1.0
 * @author Autonomous Vehicle Control System Team
 */

import { EventEmitter } from 'events';

/** @typedef {'info'|'warning'|'error'|'critical'} AlertSeverity */

/** @typedef {'active'|'acknowledged'|'resolved'|'expired'|'escalated'} AlertState */

/** @typedef {'threshold'|'anomaly'|'composite'|'manual'} AlertRuleType */

/** @typedef {'email'|'sms'|'webhook'|'websocket'|'log'|'dashboard'} NotificationChannel */

/**
 * @typedef {Object} AlertManagerConfig
 * @property {number} [throttleMs=5000] - Min time between duplicate alerts
 * @property {number} [expiryMs=3600000] - Auto-expire alerts after 1 hour
 * @property {number} [maxActiveAlerts=500] - Maximum active alerts
 * @property {number} [escalationDelayMs=300000] - Auto-escalate after 5 min
 * @property {number} [dedupWindowMs=60000] - Window for deduplication
 * @property {Object<string, NotificationChannel[]>} [severityChannels] - Channels per severity
 * @property {NotificationChannel[]} [defaultChannels=['websocket', 'log']] - Default notification channels
 */

/**
 * @typedef {Object} Alert
 * @property {string} id - Unique alert ID
 * @property {AlertSeverity} severity - Alert severity
 * @property {string} source - Source module/component
 * @property {string} message - Alert message
 * @property {AlertState} state - Current alert state
 * @property {number} createdAt - Creation timestamp
 * @property {number} [updatedAt] - Last update timestamp
 * @property {number} [acknowledgedAt] - Acknowledgement timestamp
 * @property {number} [resolvedAt] - Resolution timestamp
 * @property {number} [escalatedAt] - Escalation timestamp
 * @property {string} [acknowledgedBy] - Who acknowledged
 * @property {string} [resolvedBy] - Who resolved
 * @property {string} [resolution] - Resolution description
 * @property {string} [groupId] - Deduplication group ID
 * @property {number} [occurrenceCount=1] - Number of occurrences
 * @property {Object} [metadata] - Additional metadata
 * @property {string[]} [tags] - Alert tags
 */

/**
 * @typedef {Object} AlertCreationParams
 * @property {AlertSeverity} severity - Alert severity
 * @property {string} source - Source module
 * @property {string} message - Alert message
 * @property {Object} [metadata] - Additional metadata
 * @property {string[]} [tags] - Alert tags
 * @property {string} [groupId] - Manual group ID
 */

/**
 * @typedef {Object} AlertRule
 * @property {string} id - Rule ID
 * @property {string} name - Rule name
 * @property {AlertRuleType} type - Rule type
 * @property {AlertSeverity} severity - Alert severity when triggered
 * @property {string} source - Source to monitor
 * @property {string} metric - Metric name to evaluate
 * @property {Object} condition - Rule condition
 * @property {number} [condition.threshold] - Threshold value (threshold type)
 * @property {string} [condition.operator] - Comparison operator (gt, lt, gte, lte, eq)
 * @property {number} [condition.duration] - Duration condition must persist (ms)
 * @property {number} [condition.zScore] - Z-score threshold (anomaly type)
 * @property {number} [condition.windowMs] - Anomaly detection window
 * @property {boolean} [enabled=true] - Whether rule is active
 * @property {number} [cooldownMs=60000] - Cooldown between triggers
 * @property {number} [lastTriggeredAt] - Last trigger timestamp
 */

/**
 * @typedef {Object} AlertStats
 * @property {number} totalCreated - Total alerts created
 * @property {number} activeCount - Currently active alerts
 * @property {number} acknowledgedCount - Acknowledged alerts
 * @property {number} resolvedCount - Resolved alerts
 * @property {number} escalatedCount - Escalated alerts
 * @property {number} expiredCount - Expired alerts
 * @property {Object<string, number>} countsBySeverity - Counts per severity
 * @property {Object<string, number>} countsBySource - Counts per source
 * @property {number} avgResolutionTimeMs - Average time to resolve
 */

/** @type {number} */
let alertIdCounter = 0;

/**
 * AlertManager provides comprehensive alert lifecycle management with
 * deduplication, escalation, and multi-channel notifications.
 *
 * @extends EventEmitter
 *
 * @example
 * const alerts = new AlertManager({ throttleMs: 5000 });
 *
 * alerts.on('alert:created', (alert) => {
 *   console.log(`New ${alert.severity} alert from ${alert.source}: ${alert.message}`);
 * });
 *
 * alerts.registerNotificationChannel('webhook', async (alert) => {
 *   await fetch('https://hooks.example.com/alert', {
 *     method: 'POST',
 *     body: JSON.stringify(alert),
 *   });
 * });
 *
 * const alert = alerts.createAlert({
 *   severity: 'critical',
 *   source: 'perception',
 *   message: 'LiDAR sensor failure detected',
 * });
 */
export class AlertManager extends EventEmitter {
  /** @type {AlertManagerConfig} */
  #config;

  /** @type {Map<string, Alert>} */
  #alerts = new Map();

  /** @type {Map<string, AlertRule>} */
  #rules = new Map();

  /** @type {Map<string, NotificationChannel>} */
  #notificationHandlers = new Map();

  /** @type {Map<string, number>} */
  #dedupTracker = new Map();

  /** @type {Map<string, number>} */
  #ruleCooldowns = new Map();

  /** @type {number[]} */
  #resolutionTimes = [];

  /** @type {NodeJS.Timeout|null} */
  #expiryTimer = null;

  /** @type {NodeJS.Timeout|null} */
  #escalationTimer = null;

  /** @type {Object<string, number>} */
  #stats = {
    totalCreated: 0,
    acknowledgedCount: 0,
    resolvedCount: 0,
    escalatedCount: 0,
    expiredCount: 0,
  };

  /**
   * Creates a new AlertManager.
   *
   * @param {AlertManagerConfig} [config={}] - Configuration
   */
  constructor(config = {}) {
    super();
    this.setMaxListeners(30);

    this.#config = {
      throttleMs: 5000,
      expiryMs: 3600000,
      maxActiveAlerts: 500,
      escalationDelayMs: 300000,
      dedupWindowMs: 60000,
      severityChannels: {
        info: ['log'],
        warning: ['websocket', 'log'],
        error: ['websocket', 'log', 'webhook'],
        critical: ['websocket', 'log', 'webhook', 'sms'],
      },
      defaultChannels: ['websocket', 'log'],
      ...config,
    };

    this.#startTimers();
  }

  /**
   * Number of currently active alerts.
   * @type {number}
   */
  get activeAlertCount() {
    let count = 0;
    for (const alert of this.#alerts.values()) {
      if (alert.state === 'active' || alert.state === 'escalated') count++;
    }
    return count;
  }

  /**
   * Total alerts in memory (all states).
   * @type {number}
   */
  get totalAlertCount() {
    return this.#alerts.size;
  }

  /**
   * Current alert statistics.
   * @type {AlertStats}
   */
  get stats() {
    return this.getStats();
  }

  /**
   * Creates a new alert.
   * Checks for deduplication before creating a new alert.
   *
   * @param {AlertCreationParams} params - Alert parameters
   * @returns {Alert|null} The created alert, or null if deduplicated/throttled
   *
   * @fires AlertManager#alert:created
   */
  createAlert(params) {
    if (!params.severity || !params.source || !params.message) {
      throw new TypeError('Alert requires severity, source, and message');
    }

    // Deduplication check
    const dedupKey = this.#computeDedupKey(params);
    const now = Date.now();

    if (dedupKey) {
      const lastCreated = this.#dedupTracker.get(dedupKey);
      if (lastCreated && (now - lastCreated) < this.#config.throttleMs) {
        // Throttled - increment existing alert occurrence count
        const existingAlert = this.#findAlertByDedupKey(dedupKey);
        if (existingAlert) {
          existingAlert.occurrenceCount = (existingAlert.occurrenceCount || 1) + 1;
          existingAlert.updatedAt = now;
          return null;
        }
      }
    }

    // Check max active alerts
    if (this.activeAlertCount >= this.#config.maxActiveAlerts) {
      this.#expireOldestAlert();
    }

    const alert = {
      id: `alert_${now}_${++alertIdCounter}`,
      severity: params.severity,
      source: params.source,
      message: params.message,
      state: 'active',
      createdAt: now,
      updatedAt: now,
      acknowledgedAt: null,
      resolvedAt: null,
      escalatedAt: null,
      acknowledgedBy: null,
      resolvedBy: null,
      resolution: null,
      groupId: params.groupId || dedupKey || null,
      occurrenceCount: 1,
      metadata: params.metadata || {},
      tags: params.tags || [],
    };

    this.#alerts.set(alert.id, alert);
    this.#stats.totalCreated++;

    if (dedupKey) {
      this.#dedupTracker.set(dedupKey, now);
    }

    // Dispatch notifications
    this.#dispatchNotifications(alert);

    /**
     * @event AlertManager#alert:created
     * @type {Alert}
     */
    this.emit('alert:created', alert);

    return alert;
  }

  /**
   * Acknowledges an alert.
   *
   * @param {string} alertId - Alert ID
   * @param {string} [acknowledgedBy] - Who acknowledged
   * @returns {Alert|null} Updated alert, or null if not found
   *
   * @fires AlertManager#alert:acknowledged
   */
  acknowledge(alertId, acknowledgedBy = 'system') {
    const alert = this.#alerts.get(alertId);
    if (!alert || (alert.state !== 'active' && alert.state !== 'escalated')) {
      return null;
    }

    alert.state = 'acknowledged';
    alert.acknowledgedAt = Date.now();
    alert.acknowledgedBy = acknowledgedBy;
    alert.updatedAt = Date.now();

    this.#stats.acknowledgedCount++;

    /**
     * @event AlertManager#alert:acknowledged
     * @type {Alert}
     */
    this.emit('alert:acknowledged', alert);

    return alert;
  }

  /**
   * Resolves an alert.
   *
   * @param {string} alertId - Alert ID
   * @param {Object} [options={}] - Resolution options
   * @param {string} [options.resolvedBy] - Who resolved
   * @param {string} [options.resolution] - Resolution description
   * @returns {Alert|null} Updated alert, or null if not found
   *
   * @fires AlertManager#alert:resolved
   */
  resolve(alertId, options = {}) {
    const alert = this.#alerts.get(alertId);
    if (!alert || alert.state === 'resolved' || alert.state === 'expired') {
      return null;
    }

    const now = Date.now();
    alert.state = 'resolved';
    alert.resolvedAt = now;
    alert.resolvedBy = options.resolvedBy || 'system';
    alert.resolution = options.resolution || '';
    alert.updatedAt = now;

    // Track resolution time
    if (alert.createdAt) {
      this.#resolutionTimes.push(now - alert.createdAt);
      if (this.#resolutionTimes.length > 1000) {
        this.#resolutionTimes.shift();
      }
    }

    this.#stats.resolvedCount++;

    /**
     * @event AlertManager#alert:resolved
     * @type {Alert}
     */
    this.emit('alert:resolved', alert);

    return alert;
  }

  /**
   * Escalates an alert to a higher severity or notification level.
   *
   * @param {string} alertId - Alert ID
   * @returns {Alert|null} Updated alert, or null if not found
   *
   * @fires AlertManager#alert:escalated
   */
  escalate(alertId) {
    const alert = this.#alerts.get(alertId);
    if (!alert || alert.state === 'resolved' || alert.state === 'expired') {
      return null;
    }

    alert.state = 'escalated';
    alert.escalatedAt = Date.now();
    alert.updatedAt = Date.now();

    // Escalate severity if possible
    const severityOrder = ['info', 'warning', 'error', 'critical'];
    const currentIndex = severityOrder.indexOf(alert.severity);
    if (currentIndex < severityOrder.length - 1) {
      alert.severity = severityOrder[currentIndex + 1];
    }

    this.#stats.escalatedCount++;

    // Re-dispatch notifications with escalated severity
    this.#dispatchNotifications(alert);

    /**
     * @event AlertManager#alert:escalated
     * @type {Alert}
     */
    this.emit('alert:escalated', alert);

    return alert;
  }

  /**
   * Gets all active (non-resolved, non-expired) alerts.
   *
   * @returns {Alert[]} Active alerts
   */
  getActiveAlerts() {
    return Array.from(this.#alerts.values()).filter(
      (a) => a.state === 'active' || a.state === 'escalated'
    );
  }

  /**
   * Gets an alert by ID.
   *
   * @param {string} alertId - Alert ID
   * @returns {Alert|undefined} The alert
   */
  getAlert(alertId) {
    return this.#alerts.get(alertId);
  }

  /**
   * Gets alerts matching filter criteria.
   *
   * @param {Object} [filter={}] - Filter criteria
   * @param {AlertSeverity} [filter.severity] - Filter by severity
   * @param {string} [filter.source] - Filter by source
   * @param {AlertState} [filter.state] - Filter by state
   * @param {string} [filter.groupId] - Filter by group
   * @param {number} [filter.since] - Filter by creation time
   * @param {number} [filter.limit=100] - Max results
   * @returns {Alert[]} Filtered alerts
   */
  getAlerts(filter = {}) {
    let results = Array.from(this.#alerts.values());

    if (filter.severity) results = results.filter((a) => a.severity === filter.severity);
    if (filter.source) results = results.filter((a) => a.source === filter.source);
    if (filter.state) results = results.filter((a) => a.state === filter.state);
    if (filter.groupId) results = results.filter((a) => a.groupId === filter.groupId);
    if (filter.since) results = results.filter((a) => a.createdAt >= filter.since);

    return results.slice(0, filter.limit ?? 100);
  }

  /**
   * Gets alert statistics.
   *
   * @returns {AlertStats} Alert statistics
   */
  getStats() {
    const countsBySeverity = { info: 0, warning: 0, error: 0, critical: 0 };
    const countsBySource = {};

    for (const alert of this.#alerts.values()) {
      countsBySeverity[alert.severity] = (countsBySeverity[alert.severity] || 0) + 1;
      countsBySource[alert.source] = (countsBySource[alert.source] || 0) + 1;
    }

    const avgResolutionTimeMs = this.#resolutionTimes.length > 0
      ? this.#resolutionTimes.reduce((a, b) => a + b, 0) / this.#resolutionTimes.length
      : 0;

    return {
      totalCreated: this.#stats.totalCreated,
      activeCount: this.activeAlertCount,
      acknowledgedCount: this.#stats.acknowledgedCount,
      resolvedCount: this.#stats.resolvedCount,
      escalatedCount: this.#stats.escalatedCount,
      expiredCount: this.#stats.expiredCount,
      countsBySeverity,
      countsBySource,
      avgResolutionTimeMs: Math.round(avgResolutionTimeMs),
    };
  }

  /**
   * Adds an alert rule to the rules engine.
   *
   * @param {AlertRule} rule - Alert rule definition
   * @returns {string} Rule ID
   *
   * @example
   * alerts.addRule({
   *   name: 'High CPU Usage',
   *   type: 'threshold',
   *   severity: 'warning',
   *   source: 'system',
   *   metric: 'cpu_percent',
   *   condition: { threshold: 90, operator: 'gt', duration: 30000 },
   * });
   */
  addRule(rule) {
    const id = rule.id || `rule_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    this.#rules.set(id, {
      ...rule,
      id,
      enabled: rule.enabled !== false,
      cooldownMs: rule.cooldownMs || 60000,
      lastTriggeredAt: 0,
    });
    return id;
  }

  /**
   * Removes an alert rule.
   *
   * @param {string} ruleId - Rule ID
   * @returns {boolean} Whether the rule was removed
   */
  removeRule(ruleId) {
    return this.#rules.delete(ruleId);
  }

  /**
   * Evaluates a metric value against all registered rules.
   *
   * @param {string} source - Metric source
   * @param {string} metric - Metric name
   * @param {number} value - Current metric value
   * @param {Object} [context={}] - Additional context
   * @returns {Alert[]} Alerts created by triggered rules
   */
  evaluateRules(source, metric, value, context = {}) {
    const triggered = [];

    for (const [ruleId, rule] of this.#rules) {
      if (!rule.enabled) continue;
      if (rule.source !== source || rule.metric !== metric) continue;

      // Check cooldown
      const now = Date.now();
      const lastTriggered = this.#ruleCooldowns.get(ruleId) || 0;
      if (now - lastTriggered < rule.cooldownMs) continue;

      let isTriggered = false;

      switch (rule.type) {
        case 'threshold':
          isTriggered = this.#evaluateThreshold(rule.condition, value);
          break;
        case 'anomaly':
          isTriggered = this.#evaluateAnomaly(rule.condition, value, context);
          break;
        case 'composite':
          isTriggered = this.#evaluateComposite(rule.condition, value, context);
          break;
      }

      if (isTriggered) {
        this.#ruleCooldowns.set(ruleId, now);
        rule.lastTriggeredAt = now;

        const alert = this.createAlert({
          severity: rule.severity,
          source: rule.source,
          message: `Rule "${rule.name}" triggered: ${metric}=${value}`,
          metadata: { ruleId, metric, value, ...context },
          tags: ['rule-based', rule.type],
        });

        if (alert) triggered.push(alert);
      }
    }

    return triggered;
  }

  /**
   * Registers a notification channel handler.
   *
   * @param {NotificationChannel} channel - Channel type
   * @param {Function} handler - Handler function (alert) => void | Promise<void>
   * @returns {void}
   */
  registerNotificationChannel(channel, handler) {
    if (typeof handler !== 'function') {
      throw new TypeError('Notification handler must be a function');
    }
    this.#notificationHandlers.set(channel, handler);
  }

  /**
   * Clears all alerts (typically used during shutdown).
   *
   * @returns {void}
   */
  clearAll() {
    this.#alerts.clear();
    this.#dedupTracker.clear();
    this.#ruleCooldowns.clear();

    if (this.#expiryTimer) {
      clearInterval(this.#expiryTimer);
      this.#expiryTimer = null;
    }
    if (this.#escalationTimer) {
      clearInterval(this.#escalationTimer);
      this.#escalationTimer = null;
    }
  }

  // ─── Private Methods ───────────────────────────────────────────────

  /**
   * Starts the expiry and escalation check timers.
   * @private
   */
  #startTimers() {
    this.#expiryTimer = setInterval(() => {
      this.#checkExpired();
    }, 60000);

    this.#escalationTimer = setInterval(() => {
      this.#checkEscalation();
    }, 30000);
  }

  /**
   * Computes a deduplication key from alert parameters.
   * @private
   * @param {AlertCreationParams} params - Alert parameters
   * @returns {string} Dedup key
   */
  #computeDedupKey(params) {
    return `${params.severity}:${params.source}:${params.message}`;
  }

  /**
   * Finds an existing alert by its dedup key.
   * @private
   * @param {string} dedupKey - Dedup key
   * @returns {Alert|undefined}
   */
  #findAlertByDedupKey(dedupKey) {
    for (const alert of this.#alerts.values()) {
      if (alert.groupId === dedupKey && (alert.state === 'active' || alert.state === 'escalated')) {
        return alert;
      }
    }
    return undefined;
  }

  /**
   * Expires old alerts past the expiry threshold.
   * @private
   */
  #checkExpired() {
    const now = Date.now();

    for (const [id, alert] of this.#alerts) {
      if (alert.state === 'resolved' || alert.state === 'expired') continue;

      if (now - alert.createdAt > this.#config.expiryMs) {
        alert.state = 'expired';
        alert.updatedAt = now;
        this.#stats.expiredCount++;
        this.emit('alert:expired', alert);
      }
    }

    // Cleanup very old resolved/expired alerts
    const cleanupThreshold = now - this.#config.expiryMs * 2;
    for (const [id, alert] of this.#alerts) {
      if ((alert.state === 'resolved' || alert.state === 'expired') && alert.updatedAt < cleanupThreshold) {
        this.#alerts.delete(id);
      }
    }
  }

  /**
   * Checks for alerts that should be auto-escalated.
   * @private
   */
  #checkEscalation() {
    const now = Date.now();

    for (const alert of this.#alerts.values()) {
      if (alert.state !== 'active') continue;

      if (now - alert.createdAt > this.#config.escalationDelayMs) {
        this.escalate(alert.id);
      }
    }
  }

  /**
   * Expires the oldest active alert to make room.
   * @private
   */
  #expireOldestAlert() {
    let oldest = null;
    let oldestTime = Infinity;

    for (const alert of this.#alerts.values()) {
      if (alert.state === 'active' && alert.createdAt < oldestTime) {
        oldest = alert;
        oldestTime = alert.createdAt;
      }
    }

    if (oldest) {
      oldest.state = 'expired';
      oldest.updatedAt = Date.now();
      this.#stats.expiredCount++;
    }
  }

  /**
   * Dispatches notifications for an alert through configured channels.
   * @private
   * @param {Alert} alert - Alert to notify about
   */
  async #dispatchNotifications(alert) {
    const channels = this.#config.severityChannels[alert.severity]
      || this.#config.defaultChannels;

    for (const channel of channels) {
      const handler = this.#notificationHandlers.get(channel);
      if (handler) {
        try {
          await handler(alert);
        } catch (error) {
          this.emit('notification:error', {
            channel,
            alertId: alert.id,
            error: error.message,
          });
        }
      }
    }
  }

  /**
   * Evaluates a threshold condition.
   * @private
   * @param {Object} condition - Threshold condition
   * @param {number} value - Current value
   * @returns {boolean}
   */
  #evaluateThreshold(condition, value) {
    const threshold = condition.threshold;
    switch (condition.operator) {
      case 'gt': return value > threshold;
      case 'lt': return value < threshold;
      case 'gte': return value >= threshold;
      case 'lte': return value <= threshold;
      case 'eq': return value === threshold;
      default: return value > threshold;
    }
  }

  /**
   * Evaluates an anomaly condition using z-score.
   * @private
   * @param {Object} condition - Anomaly condition
   * @param {number} value - Current value
   * @param {Object} context - Context with historical values
   * @returns {boolean}
   */
  #evaluateAnomaly(condition, value, context) {
    const history = context.history || [];
    if (history.length < 5) return false;

    const mean = history.reduce((a, b) => a + b, 0) / history.length;
    const variance = history.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / history.length;
    const stddev = Math.sqrt(variance);

    if (stddev === 0) return false;

    const zScore = Math.abs((value - mean) / stddev);
    return zScore > (condition.zScore || 2.5);
  }

  /**
   * Evaluates a composite condition (multiple conditions).
   * @private
   * @param {Object} condition - Composite condition
   * @param {number} value - Current value
   * @param {Object} context - Context
   * @returns {boolean}
   */
  #evaluateComposite(condition, value, context) {
    if (!condition.conditions || !Array.isArray(condition.conditions)) {
      return false;
    }

    const results = condition.conditions.map((c) => this.#evaluateThreshold(c, value));
    return condition.logic === 'and' ? results.every(Boolean) : results.some(Boolean);
  }
}

export default AlertManager;
