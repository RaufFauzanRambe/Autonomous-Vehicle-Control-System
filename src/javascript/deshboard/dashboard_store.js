/**
 * @fileoverview Dashboard Data Store
 * @description EventEmitter-based state management, dashboard state tree,
 *   subscribe/notify pattern, immutable state updates, snapshot/restore,
 *   and state history for the Autonomous Vehicle Dashboard.
 * @module dashboard_store
 */

import { EventEmitter } from 'events';
import { DashboardController, THEMES, WIDGET_SIZES } from './dashboard.js';

// ─── State Action Types ──────────────────────────────────────────────────────

/** @enum {string} */
export const ActionTypes = {
  DASHBOARD_CREATED: 'DASHBOARD_CREATED',
  DASHBOARD_UPDATED: 'DASHBOARD_UPDATED',
  DASHBOARD_DELETED: 'DASHBOARD_DELETED',
  WIDGET_ADDED: 'WIDGET_ADDED',
  WIDGET_REMOVED: 'WIDGET_REMOVED',
  WIDGET_UPDATED: 'WIDGET_UPDATED',
  THEME_CHANGED: 'THEME_CHANGED',
  TELEMETRY_UPDATED: 'TELEMETRY_UPDATED',
  STATE_RESTORED: 'STATE_RESTORED',
  STATE_RESET: 'STATE_RESET',
};

// ─── Immutable State Helpers ─────────────────────────────────────────────────

/**
 * Deep freeze an object to make it immutable
 * @param {Object} obj - Object to freeze
 * @returns {Object} Frozen object
 */
const deepFreeze = (obj) => {
  if (obj === null || typeof obj !== 'object') return obj;
  Object.freeze(obj);
  for (const value of Object.values(obj)) {
    if (typeof value === 'object' && value !== null && !Object.isFrozen(value)) {
      deepFreeze(value);
    }
  }
  return obj;
};

/**
 * Deep clone an object
 * @param {Object} obj - Object to clone
 * @returns {Object} Cloned object
 */
const deepClone = (obj) => {
  if (obj === null || typeof obj !== 'object') return obj;
  return JSON.parse(JSON.stringify(obj));
};

// ─── State History ───────────────────────────────────────────────────────────

/**
 * Maintains a history of state changes for undo/redo functionality
 */
class StateHistory {
  /**
   * @param {number} [maxSize=50] - Maximum history entries
   */
  constructor(maxSize = 50) {
    /** @private @type {Object[]} */ this._history = [];
    /** @private */ this._currentIndex = -1;
    /** @private */ this._maxSize = maxSize;
  }

  /**
   * Push a new state onto the history
   * @param {Object} state - State snapshot
   * @param {string} actionType - Action that caused the change
   */
  push(state, actionType) {
    // Remove any future states if we're not at the end
    if (this._currentIndex < this._history.length - 1) {
      this._history = this._history.slice(0, this._currentIndex + 1);
    }

    this._history.push({
      state: deepClone(state),
      actionType,
      timestamp: Date.now(),
    });

    // Enforce max size
    if (this._history.length > this._maxSize) {
      this._history.shift();
    }

    this._currentIndex = this._history.length - 1;
  }

  /**
   * Undo: move back in history
   * @returns {Object|null} Previous state or null
   */
  undo() {
    if (this._currentIndex <= 0) return null;
    this._currentIndex--;
    return deepClone(this._history[this._currentIndex].state);
  }

  /**
   * Redo: move forward in history
   * @returns {Object|null} Next state or null
   */
  redo() {
    if (this._currentIndex >= this._history.length - 1) return null;
    this._currentIndex++;
    return deepClone(this._history[this._currentIndex].state);
  }

  /**
   * Check if undo is available
   * @returns {boolean}
   */
  get canUndo() {
    return this._currentIndex > 0;
  }

  /**
   * Check if redo is available
   * @returns {boolean}
   */
  get canRedo() {
    return this._currentIndex < this._history.length - 1;
  }

  /**
   * Get history metadata
   * @returns {Object[]}
   */
  getEntries() {
    return this._history.map((entry, idx) => ({
      index: idx,
      actionType: entry.actionType,
      timestamp: entry.timestamp,
      isCurrent: idx === this._currentIndex,
    }));
  }

  /** @returns {number} */ get length() { return this._history.length; }
}

// ─── Dashboard Store ─────────────────────────────────────────────────────────

/**
 * Centralized state store for dashboard data
 * @extends EventEmitter
 */
export class DashboardStore extends EventEmitter {
  constructor() {
    super();

    /** @private @type {Map<string, Object>} */ this._dashboards = new Map();
    /** @private */ this._controller = new DashboardController();
    /** @private */ this._history = new StateHistory(50);
    /** @private @type {Map<string, Object>} */ this._telemetry = new Map();
    /** @private */ this._ready = false;
    /** @private @type {Map<string, Set<Function>>} */ this._watchers = new Map();
    /** @private */ this._version = 0;

    // Bridge controller events to store events
    this._controller.on('dashboard:created', (d) => this._emitChange(ActionTypes.DASHBOARD_CREATED, d));
    this._controller.on('dashboard:updated', (d) => this._emitChange(ActionTypes.DASHBOARD_UPDATED, d));
    this._controller.on('dashboard:deleted', (d) => this._emitChange(ActionTypes.DASHBOARD_DELETED, d));
    this._controller.on('widget:added', (d) => this._emitChange(ActionTypes.WIDGET_ADDED, d));
    this._controller.on('widget:removed', (d) => this._emitChange(ActionTypes.WIDGET_REMOVED, d));
    this._controller.on('theme:changed', (d) => this._emitChange(ActionTypes.THEME_CHANGED, d));

    this._ready = true;
  }

  // ─── Dashboard Operations ───────────────────────────────────────────────

  /**
   * Create a new dashboard
   * @param {Object} config - Dashboard configuration
   * @returns {Object} Created dashboard
   */
  createDashboard(config) {
    const dashboard = this._controller.createDashboard(config);
    this._dashboards.set(dashboard.id, dashboard);
    this._saveSnapshot(ActionTypes.DASHBOARD_CREATED);
    return this._getState().dashboards[dashboard.id];
  }

  /**
   * Get a dashboard by ID
   * @param {string} id - Dashboard ID
   * @returns {Object|null}
   */
  getDashboard(id) {
    return this._dashboards.has(id) ? deepClone(this._dashboards.get(id)) : null;
  }

  /**
   * List all dashboards
   * @param {Object} [filter={}] - Filter options
   * @returns {Object[]}
   */
  listDashboards(filter = {}) {
    return this._controller.listDashboards(filter);
  }

  /**
   * Update a dashboard
   * @param {string} id - Dashboard ID
   * @param {Object} updates - Fields to update
   * @returns {Object} Updated dashboard
   */
  updateDashboard(id, updates) {
    const dashboard = this._controller.updateDashboard(id, updates);
    this._dashboards.set(id, dashboard);
    this._saveSnapshot(ActionTypes.DASHBOARD_UPDATED);
    return this._getState().dashboards[id];
  }

  /**
   * Delete a dashboard
   * @param {string} id - Dashboard ID
   * @returns {boolean}
   */
  deleteDashboard(id) {
    const result = this._controller.deleteDashboard(id);
    if (result) {
      this._dashboards.delete(id);
      this._saveSnapshot(ActionTypes.DASHBOARD_DELETED);
    }
    return result;
  }

  // ─── Widget Operations ──────────────────────────────────────────────────

  /**
   * Add a widget to a dashboard
   * @param {string} dashboardId - Dashboard ID
   * @param {Object} widgetConfig - Widget configuration
   * @returns {Object} Added widget
   */
  addWidget(dashboardId, widgetConfig) {
    const widget = this._controller.addWidget(dashboardId, widgetConfig);
    const dashboard = this._controller.getDashboard(dashboardId);
    this._dashboards.set(dashboardId, dashboard);
    this._saveSnapshot(ActionTypes.WIDGET_ADDED);
    return widget;
  }

  /**
   * Remove a widget from a dashboard
   * @param {string} dashboardId - Dashboard ID
   * @param {string} widgetId - Widget ID
   * @returns {boolean}
   */
  removeWidget(dashboardId, widgetId) {
    const result = this._controller.removeWidget(dashboardId, widgetId);
    if (result) {
      const dashboard = this._controller.getDashboard(dashboardId);
      if (dashboard) this._dashboards.set(dashboardId, dashboard);
      this._saveSnapshot(ActionTypes.WIDGET_REMOVED);
    }
    return result;
  }

  // ─── Layout ─────────────────────────────────────────────────────────────

  /**
   * Get computed layout for a dashboard
   * @param {string} dashboardId - Dashboard ID
   * @param {number} [viewportWidth] - Viewport width
   * @returns {Object[]}
   */
  getComputedLayout(dashboardId, viewportWidth) {
    return this._controller.getComputedLayout(dashboardId, viewportWidth);
  }

  // ─── Export / Import ────────────────────────────────────────────────────

  /**
   * Export dashboard configuration
   * @param {string} dashboardId - Dashboard ID
   * @returns {string} JSON string
   */
  exportDashboard(dashboardId) {
    return this._controller.exportDashboard(dashboardId);
  }

  /**
   * Import dashboard configuration
   * @param {string} jsonStr - JSON configuration
   * @returns {Object} Imported dashboard
   */
  importDashboard(jsonStr) {
    const dashboard = this._controller.importDashboard(jsonStr);
    this._dashboards.set(dashboard.id, dashboard);
    this._saveSnapshot(ActionTypes.DASHBOARD_CREATED);
    return dashboard;
  }

  // ─── Telemetry ──────────────────────────────────────────────────────────

  /**
   * Update telemetry data
   * @param {Object} data - Telemetry data
   */
  updateTelemetry(data) {
    const source = data.source || data.sensorId || 'default';
    this._telemetry.set(source, {
      ...data,
      _updatedAt: Date.now(),
    });
    this._emitChange(ActionTypes.TELEMETRY_UPDATED, { source, data });
  }

  /**
   * Get current telemetry for a source
   * @param {string} [source='default'] - Data source
   * @returns {Object|null}
   */
  getTelemetry(source = 'default') {
    return this._telemetry.get(source) || null;
  }

  /**
   * Get all current telemetry data
   * @returns {Object<string, Object>}
   */
  getAllTelemetry() {
    const result = {};
    for (const [key, val] of this._telemetry.entries()) {
      result[key] = deepClone(val);
    }
    return result;
  }

  // ─── State History ──────────────────────────────────────────────────────

  /**
   * Undo the last state change
   * @returns {boolean} Whether undo was successful
   */
  undo() {
    const prevState = this._history.undo();
    if (!prevState) return false;
    this._restoreState(prevState);
    this._emitChange(ActionTypes.STATE_RESTORED, { action: 'undo' });
    return true;
  }

  /**
   * Redo the last undone state change
   * @returns {boolean} Whether redo was successful
   */
  redo() {
    const nextState = this._history.redo();
    if (!nextState) return false;
    this._restoreState(nextState);
    this._emitChange(ActionTypes.STATE_RESTORED, { action: 'redo' });
    return true;
  }

  /** @returns {boolean} */ get canUndo() { return this._history.canUndo; }
  /** @returns {boolean} */ get canRedo() { return this._history.canRedo; }

  // ─── Watchers / Subscriptions ───────────────────────────────────────────

  /**
   * Watch a specific path in the state tree
   * @param {string} path - Dot-separated path (e.g., "dashboards.dash_123")
   * @param {Function} callback - Called with (newValue, oldValue, path)
   * @returns {Function} Unwatch function
   */
  watch(path, callback) {
    if (!this._watchers.has(path)) {
      this._watchers.set(path, new Set());
    }
    this._watchers.get(path).add(callback);
    return () => {
      this._watchers.get(path)?.delete(callback);
    };
  }

  // ─── Snapshot / Restore ─────────────────────────────────────────────────

  /**
   * Take a snapshot of the current state
   * @returns {string} JSON string of the state
   */
  takeSnapshot() {
    return JSON.stringify(this._getState());
  }

  /**
   * Restore state from a snapshot
   * @param {string} snapshot - JSON state snapshot
   */
  restoreSnapshot(snapshot) {
    try {
      const state = JSON.parse(snapshot);
      this._restoreState(state);
      this._emitChange(ActionTypes.STATE_RESTORED, { action: 'restore' });
    } catch {
      throw new Error('Failed to restore snapshot: invalid JSON');
    }
  }

  /**
   * Flush current state to persistent storage (simulated)
   * @returns {Promise<void>}
   */
  async flush() {
    const snapshot = this.takeSnapshot();
    // In production, write to database or file system
    this.emit('state:flushed', { size: snapshot.length, version: this._version });
  }

  // ─── Status ─────────────────────────────────────────────────────────────

  /** @returns {boolean} */ isReady() { return this._ready; }
  /** @returns {number} */ get version() { return this._version; }
  /** @returns {number} */ getDashboardCount() { return this._dashboards.size; }
  /** @returns {number} */ getWidgetCount() { return Array.from(this._dashboards.values()).reduce((sum, d) => sum + d.widgets.length, 0); }

  // ─── Private Helpers ────────────────────────────────────────────────────

  /**
   * Get the current state tree
   * @private
   * @returns {Object}
   */
  _getState() {
    const dashboards = {};
    for (const [id, d] of this._dashboards.entries()) {
      dashboards[id] = deepClone(d);
    }
    return {
      dashboards,
      telemetry: this.getAllTelemetry(),
      theme: this._controller.getTheme().theme,
      version: this._version,
      timestamp: Date.now(),
    };
  }

  /**
   * Save a snapshot to history
   * @private
   * @param {string} actionType - Action that triggered the change
   */
  _saveSnapshot(actionType) {
    this._version++;
    this._history.push(this._getState(), actionType);
  }

  /**
   * Restore a previous state
   * @private
   * @param {Object} state - State to restore
   */
  _restoreState(state) {
    this._dashboards.clear();
    if (state.dashboards) {
      for (const [id, d] of Object.entries(state.dashboards)) {
        this._dashboards.set(id, deepClone(d));
      }
    }
    this._version = state.version || this._version + 1;
  }

  /**
   * Emit a state change event and notify watchers
   * @private
   * @param {string} actionType - Action type
   * @param {Object} [payload={}] - Change payload
   */
  _emitChange(actionType, payload = {}) {
    this._version++;
    const change = {
      type: actionType,
      payload,
      version: this._version,
      timestamp: Date.now(),
    };

    this.emit('stateChanged', change);

    // Notify path-based watchers
    for (const [path, watchers] of this._watchers.entries()) {
      const pathParts = path.split('.');
      let value = this._getState();
      for (const part of pathParts) {
        value = value?.[part];
      }
      for (const watcher of watchers) {
        try {
          watcher(value, undefined, path);
        } catch (err) {
          console.error(`Watcher error for path ${path}:`, err.message);
        }
      }
    }
  }
}

export { StateHistory, deepFreeze, deepClone };
