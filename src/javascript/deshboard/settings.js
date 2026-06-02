/**
 * @fileoverview Settings Management
 * @description User preferences, dashboard configuration, notification preferences,
 *   theme settings, unit system (metric/imperial), save/load from localStorage,
 *   defaults, and validation for the Autonomous Vehicle Dashboard.
 * @module settings
 */

import { EventEmitter } from 'events';

// ─── Constants ───────────────────────────────────────────────────────────────

/** @enum {string} */
export const UnitSystem = {
  METRIC: 'metric',
  IMPERIAL: 'imperial',
};

/** @enum {string} */
export const ThemeMode = {
  DARK: 'dark',
  LIGHT: 'light',
  HIGH_CONTRAST: 'high_contrast',
  SYSTEM: 'system',
};

/** @enum {string} */
export const NotificationLevel = {
  ALL: 'all',
  CRITICAL_ONLY: 'critical_only',
  NONE: 'none',
};

/** @enum {string} */
export const RefreshRate = {
  REALTIME: 250,
  FAST: 500,
  NORMAL: 1000,
  SLOW: 2000,
  MANUAL: 0,
};

// ─── Default Settings ────────────────────────────────────────────────────────

/**
 * @typedef {Object} UserSettings
 * @property {string} unitSystem - Metric or Imperial
 * @property {string} theme - Theme mode
 * @property {number} refreshRate - Data refresh rate in ms
 * @property {string} notificationLevel - Notification filter level
 * @property {boolean} soundEnabled - Enable sound alerts
 * @property {number} soundVolume - Sound volume (0-1)
 * @property {boolean} autoScroll - Auto-scroll logs
 * @property {number} maxDataPoints - Max points on charts
 * @property {string} language - UI language
 * @property {string} timezone - Timezone
 * @property {boolean} showGridLines - Show chart grid lines
 * @property {boolean} animateCharts - Enable chart animations
 * @property {number} chartAnimationDuration - Chart animation ms
 * @property {Object} dashboard - Dashboard-specific settings
 * @property {Object} notifications - Notification channel settings
 * @property {Object} vehicle - Vehicle-specific settings
 */

/** @type {UserSettings} */
const DEFAULT_SETTINGS = {
  unitSystem: UnitSystem.METRIC,
  theme: ThemeMode.DARK,
  refreshRate: RefreshRate.NORMAL,
  notificationLevel: NotificationLevel.ALL,
  soundEnabled: true,
  soundVolume: 0.7,
  autoScroll: true,
  maxDataPoints: 300,
  language: 'en',
  timezone: Intl.DateTimeFormat().resolvedOptions().timeZone || 'UTC',
  showGridLines: true,
  animateCharts: true,
  chartAnimationDuration: 300,
  dashboard: {
    defaultLayout: 'grid',
    defaultColumns: 3,
    widgetGap: 16,
    snapToGrid: true,
    persistLayout: true,
    showWidgetBorders: true,
    compactMode: false,
  },
  notifications: {
    console: true,
    websocket: true,
    email: false,
    emailCriticalOnly: true,
    push: false,
    alertSound: true,
    alertSoundCritical: 'alarm',
    alertSoundWarning: 'beep',
    alertSoundInfo: 'chime',
    quietHoursEnabled: false,
    quietHoursStart: '22:00',
    quietHoursEnd: '07:00',
  },
  vehicle: {
    maxSpeedKmh: 220,
    speedWarningThreshold: 0.6,
    speedCriticalThreshold: 0.8,
    batteryWarningPercent: 25,
    batteryCriticalPercent: 10,
    laneDepartureThreshold: 0.3,
    objectProximityWarningM: 10,
    objectProximityCriticalM: 5,
    temperatureWarningC: 75,
    temperatureCriticalC: 90,
  },
};

// ─── Validation Schema ───────────────────────────────────────────────────────

/**
 * Validation rules for settings fields
 * @type {Object<string, { type: string, values?: any[], min?: number, max?: number, pattern?: RegExp }>}
 */
const VALIDATION_SCHEMA = {
  unitSystem: { type: 'string', values: Object.values(UnitSystem) },
  theme: { type: 'string', values: Object.values(ThemeMode) },
  refreshRate: { type: 'number', values: Object.values(RefreshRate) },
  notificationLevel: { type: 'string', values: Object.values(NotificationLevel) },
  soundEnabled: { type: 'boolean' },
  soundVolume: { type: 'number', min: 0, max: 1 },
  autoScroll: { type: 'boolean' },
  maxDataPoints: { type: 'number', min: 10, max: 10000 },
  language: { type: 'string', pattern: /^[a-z]{2}(-[A-Z]{2})?$/ },
  showGridLines: { type: 'boolean' },
  animateCharts: { type: 'boolean' },
  chartAnimationDuration: { type: 'number', min: 0, max: 5000 },
};

/**
 * Validate a settings value against the schema
 * @param {string} key - Settings key
 * @param {*} value - Value to validate
 * @returns {{ valid: boolean, error?: string }}
 */
function validateField(key, value) {
  const schema = VALIDATION_SCHEMA[key];
  if (!schema) return { valid: true }; // Allow unknown keys

  if (typeof value !== schema.type) {
    return { valid: false, error: `${key} must be of type ${schema.type}, got ${typeof value}` };
  }

  if (schema.values && !schema.values.includes(value)) {
    return { valid: false, error: `${key} must be one of: ${schema.values.join(', ')}` };
  }

  if (schema.min !== undefined && value < schema.min) {
    return { valid: false, error: `${key} must be at least ${schema.min}` };
  }

  if (schema.max !== undefined && value > schema.max) {
    return { valid: false, error: `${key} must be at most ${schema.max}` };
  }

  if (schema.pattern && !schema.pattern.test(String(value))) {
    return { valid: false, error: `${key} format is invalid` };
  }

  return { valid: true };
}

// ─── Settings Manager ────────────────────────────────────────────────────────

/**
 * Manages application settings with persistence and validation
 * @extends EventEmitter
 */
export class SettingsManager extends EventEmitter {
  /**
   * @param {Object} [options={}]
   * @param {string} [options.storageKey='av_dashboard_settings'] - localStorage key
   * @param {boolean} [options.persist=true] - Enable persistence
   * @param {UserSettings} [options.defaults] - Custom defaults
   */
  constructor(options = {}) {
    super();
    /** @private */ this._storageKey = options.storageKey || 'av_dashboard_settings';
    /** @private */ this._persist = options.persist !== false;
    /** @private */ this._settings = this._deepMerge(
      {},
      options.defaults || DEFAULT_SETTINGS
    );
    /** @private @type {Set<(settings: UserSettings) => void>} */ this._listeners = new Set();

    // Load persisted settings
    if (this._persist) {
      this._loadFromStorage();
    }
  }

  // ─── Getters / Setters ──────────────────────────────────────────────────

  /**
   * Get a setting value
   * @param {string} key - Setting key (dot notation supported, e.g., 'vehicle.maxSpeedKmh')
   * @returns {*} Setting value
   */
  get(key) {
    if (!key.includes('.')) return this._settings[key];
    return key.split('.').reduce((obj, part) => obj?.[part], this._settings);
  }

  /**
   * Set a setting value with validation
   * @param {string} key - Setting key (dot notation supported)
   * @param {*} value - New value
   * @returns {{ success: boolean, error?: string }}
   */
  set(key, value) {
    const simpleKey = key.includes('.') ? key.split('.').pop() : key;
    const validation = validateField(simpleKey, value);
    if (!validation.valid) {
      return { success: false, error: validation.error };
    }

    const oldValue = this.get(key);
    if (key.includes('.')) {
      const parts = key.split('.');
      let target = this._settings;
      for (let i = 0; i < parts.length - 1; i++) {
        if (!target[parts[i]]) target[parts[i]] = {};
        target = target[parts[i]];
      }
      target[parts[parts.length - 1]] = value;
    } else {
      this._settings[key] = value;
    }

    if (this._persist) this._saveToStorage();

    this.emit('setting:changed', { key, oldValue, newValue: value });
    this._notifyListeners();

    return { success: true };
  }

  /**
   * Update multiple settings at once
   * @param {Object} updates - Key-value pairs to update
   * @returns {{ success: boolean, errors: Object<string, string> }}
   */
  batchUpdate(updates) {
    const errors = {};
    const originalSettings = this._deepMerge({}, this._settings);

    for (const [key, value] of Object.entries(updates)) {
      const result = this.set(key, value);
      if (!result.success) {
        errors[key] = result.error;
        // Restore original value
        const parts = key.split('.');
        let target = this._settings;
        for (let i = 0; i < parts.length - 1; i++) target = target[parts[i]];
        const lastPart = parts[parts.length - 1];
        const origValue = parts.length > 1
          ? parts.slice(0, -1).reduce((o, p) => o?.[p], originalSettings)
          : originalSettings;
        if (target && origValue) target[lastPart] = origValue[lastPart];
      }
    }

    if (Object.keys(errors).length > 0) {
      return { success: false, errors };
    }

    return { success: true, errors: {} };
  }

  /**
   * Get all settings as a plain object
   * @returns {UserSettings}
   */
  getAll() {
    return this._deepMerge({}, this._settings);
  }

  /**
   * Reset a specific setting to its default value
   * @param {string} key - Setting key
   */
  reset(key) {
    const defaultValue = key.includes('.')
      ? key.split('.').reduce((obj, part) => obj?.[part], DEFAULT_SETTINGS)
      : DEFAULT_SETTINGS[key];
    if (defaultValue !== undefined) {
      this.set(key, defaultValue);
    }
  }

  /**
   * Reset all settings to defaults
   */
  resetAll() {
    this._settings = this._deepMerge({}, DEFAULT_SETTINGS);
    if (this._persist) this._saveToStorage();
    this.emit('settings:reset');
    this._notifyListeners();
  }

  // ─── Unit Conversion Helpers ────────────────────────────────────────────

  /**
   * Convert speed based on current unit system
   * @param {number} speedKmh - Speed in km/h
   * @returns {{ value: number, unit: string }}
   */
  convertSpeed(speedKmh) {
    if (this._settings.unitSystem === UnitSystem.IMPERIAL) {
      return { value: speedKmh * 0.621371, unit: 'mph' };
    }
    return { value: speedKmh, unit: 'km/h' };
  }

  /**
   * Convert distance based on current unit system
   * @param {number} distanceKm - Distance in km
   * @returns {{ value: number, unit: string }}
   */
  convertDistance(distanceKm) {
    if (this._settings.unitSystem === UnitSystem.IMPERIAL) {
      return { value: distanceKm * 0.621371, unit: 'mi' };
    }
    return { value: distanceKm, unit: 'km' };
  }

  /**
   * Convert temperature based on current unit system
   * @param {number} tempC - Temperature in Celsius
   * @returns {{ value: number, unit: string }}
   */
  convertTemperature(tempC) {
    if (this._settings.unitSystem === UnitSystem.IMPERIAL) {
      return { value: (tempC * 9 / 5) + 32, unit: '°F' };
    }
    return { value: tempC, unit: '°C' };
  }

  // ─── Subscription ───────────────────────────────────────────────────────

  /**
   * Subscribe to settings changes
   * @param {(settings: UserSettings) => void} callback - Called on any change
   * @returns {Function} Unsubscribe function
   */
  onChange(callback) {
    this._listeners.add(callback);
    return () => this._listeners.delete(callback);
  }

  // ─── Import / Export ────────────────────────────────────────────────────

  /**
   * Export settings as JSON string
   * @returns {string}
   */
  exportSettings() {
    return JSON.stringify({
      version: '1.0.0',
      exportedAt: new Date().toISOString(),
      settings: this._settings,
    }, null, 2);
  }

  /**
   * Import settings from a JSON string
   * @param {string} jsonStr - JSON settings string
   * @returns {{ success: boolean, error?: string }}
   */
  importSettings(jsonStr) {
    try {
      const imported = JSON.parse(jsonStr);
      if (!imported.settings || typeof imported.settings !== 'object') {
        return { success: false, error: 'Invalid settings format: missing settings object' };
      }
      const merged = this._deepMerge({}, DEFAULT_SETTINGS, imported.settings);
      this._settings = merged;
      if (this._persist) this._saveToStorage();
      this.emit('settings:imported', { settings: merged });
      this._notifyListeners();
      return { success: true };
    } catch (err) {
      return { success: false, error: `Failed to parse settings: ${err.message}` };
    }
  }

  // ─── Private Methods ────────────────────────────────────────────────────

  /**
   * Deep merge objects
   * @private
   * @param {Object} target - Target object
   * @param {...Object} sources - Source objects
   * @returns {Object} Merged object
   */
  _deepMerge(target, ...sources) {
    for (const source of sources) {
      if (!source || typeof source !== 'object') continue;
      for (const [key, value] of Object.entries(source)) {
        if (value && typeof value === 'object' && !Array.isArray(value)) {
          if (!target[key]) target[key] = {};
          this._deepMerge(target[key], value);
        } else {
          target[key] = value;
        }
      }
    }
    return target;
  }

  /**
   * Save settings to localStorage
   * @private
   */
  _saveToStorage() {
    try {
      if (typeof localStorage !== 'undefined') {
        localStorage.setItem(this._storageKey, JSON.stringify(this._settings));
      }
    } catch (err) {
      console.error('Failed to save settings to localStorage:', err.message);
    }
  }

  /**
   * Load settings from localStorage
   * @private
   */
  _loadFromStorage() {
    try {
      if (typeof localStorage !== 'undefined') {
        const stored = localStorage.getItem(this._storageKey);
        if (stored) {
          const parsed = JSON.parse(stored);
          this._settings = this._deepMerge({}, DEFAULT_SETTINGS, parsed);
        }
      }
    } catch (err) {
      console.error('Failed to load settings from localStorage:', err.message);
      this._settings = this._deepMerge({}, DEFAULT_SETTINGS);
    }
  }

  /**
   * Notify all registered listeners
   * @private
   */
  _notifyListeners() {
    const settings = this.getAll();
    for (const listener of this._listeners) {
      try {
        listener(settings);
      } catch (err) {
        console.error('Settings listener error:', err.message);
      }
    }
  }
}

export { DEFAULT_SETTINGS, VALIDATION_SCHEMA };
