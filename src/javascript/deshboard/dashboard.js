/**
 * @fileoverview Dashboard Controller
 * @description Dashboard layout management, widget registry, theme management,
 *   responsive grid system, dashboard state persistence, and export/import
 *   configuration for the Autonomous Vehicle Dashboard.
 * @module dashboard
 */

import { EventEmitter } from 'events';

// ─── Constants ───────────────────────────────────────────────────────────────

/** @enum {string} */
export const THEMES = {
  DARK: 'dark',
  LIGHT: 'light',
  HIGH_CONTRAST: 'high_contrast',
};

/** @enum {string} */
export const LAYOUT_TYPES = {
  GRID: 'grid',
  FLEX: 'flex',
  FREEFORM: 'freeform',
};

/** @enum {string} */
export const WIDGET_SIZES = {
  SMALL: '1x1',
  MEDIUM: '2x1',
  LARGE: '2x2',
  WIDE: '3x1',
  TALL: '1x2',
  FULL: '3x2',
};

/** @type {Object<string, Object>} */
const THEME_CONFIGS = {
  [THEMES.DARK]: {
    background: '#0f1117',
    surface: '#1a1d27',
    surfaceHover: '#242836',
    primary: '#00d4aa',
    primaryDim: '#00d4aa66',
    text: '#e4e6eb',
    textSecondary: '#9ca3af',
    border: '#2d3348',
    error: '#ef4444',
    warning: '#f59e0b',
    info: '#3b82f6',
    success: '#22c55e',
    chartColors: ['#00d4aa', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'],
  },
  [THEMES.LIGHT]: {
    background: '#f8fafc',
    surface: '#ffffff',
    surfaceHover: '#f1f5f9',
    primary: '#059669',
    primaryDim: '#05966966',
    text: '#1e293b',
    textSecondary: '#64748b',
    border: '#e2e8f0',
    error: '#dc2626',
    warning: '#d97706',
    info: '#2563eb',
    success: '#16a34a',
    chartColors: ['#059669', '#2563eb', '#d97706', '#dc2626', '#7c3aed', '#0891b2'],
  },
  [THEMES.HIGH_CONTRAST]: {
    background: '#000000',
    surface: '#1a1a1a',
    surfaceHover: '#333333',
    primary: '#00ff88',
    primaryDim: '#00ff8888',
    text: '#ffffff',
    textSecondary: '#cccccc',
    border: '#555555',
    error: '#ff4444',
    warning: '#ffaa00',
    info: '#4488ff',
    success: '#44ff44',
    chartColors: ['#00ff88', '#4488ff', '#ffaa00', '#ff4444', '#cc44ff', '#00cccc'],
  },
};

/** @type {Object<string, {minWidth: number, columns: number, rowHeight: number}>} */
const BREAKPOINTS = {
  mobile: { minWidth: 0, columns: 1, rowHeight: 200 },
  tablet: { minWidth: 768, columns: 2, rowHeight: 180 },
  desktop: { minWidth: 1200, columns: 3, rowHeight: 160 },
  ultrawide: { minWidth: 1920, columns: 4, rowHeight: 150 },
};

// ─── Widget Registry ─────────────────────────────────────────────────────────

/**
 * Registry for dashboard widget types
 * @extends EventEmitter
 */
export class WidgetRegistry extends EventEmitter {
  constructor() {
    super();
    /** @private @type {Map<string, Object>} */
    this._widgets = new Map();
    this._registerBuiltinWidgets();
  }

  /**
   * Register built-in vehicle dashboard widgets
   * @private
   */
  _registerBuiltinWidgets() {
    const builtins = [
      { type: 'speed_gauge', name: 'Speed Gauge', category: 'vehicle', size: WIDGET_SIZES.MEDIUM, icon: 'gauge' },
      { type: 'battery_indicator', name: 'Battery Indicator', category: 'vehicle', size: WIDGET_SIZES.SMALL, icon: 'battery' },
      { type: 'gps_map', name: 'GPS Map', category: 'navigation', size: WIDGET_SIZES.LARGE, icon: 'map' },
      { type: 'sensor_status', name: 'Sensor Status', category: 'telemetry', size: WIDGET_SIZES.WIDE, icon: 'activity' },
      { type: 'object_detection', name: 'Object Detection Feed', category: 'perception', size: WIDGET_SIZES.LARGE, icon: 'eye' },
      { type: 'lane_departure', name: 'Lane Departure Indicator', category: 'safety', size: WIDGET_SIZES.SMALL, icon: 'road' },
      { type: 'weather', name: 'Weather Widget', category: 'environment', size: WIDGET_SIZES.SMALL, icon: 'cloud' },
      { type: 'trip_info', name: 'Trip Information', category: 'trip', size: WIDGET_SIZES.MEDIUM, icon: 'navigation' },
      { type: 'speed_chart', name: 'Speed Over Time', category: 'charts', size: WIDGET_SIZES.WIDE, icon: 'trending-up' },
      { type: 'sensor_distribution', name: 'Sensor Distribution', category: 'charts', size: WIDGET_SIZES.MEDIUM, icon: 'pie-chart' },
    ];

    builtins.forEach((widget) => {
      this._widgets.set(widget.type, { ...widget, registered: new Date().toISOString() });
    });
  }

  /**
   * Register a custom widget type
   * @param {string} type - Widget type identifier
   * @param {Object} config - Widget configuration
   * @param {string} config.name - Display name
   * @param {string} config.category - Widget category
   * @param {string} config.size - Default widget size
   * @param {string} config.icon - Icon identifier
   * @param {Function} [config.render] - Custom render function
   * @param {Function} [config.update] - Custom update function
   * @returns {boolean} Whether registration succeeded
   */
  register(type, config) {
    if (!type || typeof type !== 'string') {
      throw new Error('Widget type must be a non-empty string');
    }
    if (this._widgets.has(type)) {
      return false;
    }
    const entry = {
      type,
      name: config.name || type,
      category: config.category || 'custom',
      size: config.size || WIDGET_SIZES.MEDIUM,
      icon: config.icon || 'box',
      render: config.render || null,
      update: config.update || null,
      registered: new Date().toISOString(),
    };
    this._widgets.set(type, entry);
    this.emit('widget:registered', entry);
    return true;
  }

  /**
   * Unregister a widget type
   * @param {string} type - Widget type to remove
   * @returns {boolean} Whether unregistration succeeded
   */
  unregister(type) {
    const deleted = this._widgets.delete(type);
    if (deleted) this.emit('widget:unregistered', { type });
    return deleted;
  }

  /**
   * Get widget configuration by type
   * @param {string} type - Widget type identifier
   * @returns {Object|undefined}
   */
  get(type) {
    return this._widgets.get(type);
  }

  /**
   * Get all registered widget types
   * @returns {Object[]}
   */
  getAll() {
    return Array.from(this._widgets.values());
  }

  /**
   * Get widgets filtered by category
   * @param {string} category - Category to filter by
   * @returns {Object[]}
   */
  getByCategory(category) {
    return this.getAll().filter((w) => w.category === category);
  }

  /** @returns {number} */
  get count() {
    return this._widgets.size;
  }
}

// ─── Grid Layout Engine ──────────────────────────────────────────────────────

/**
 * Responsive grid layout engine for dashboard widgets
 */
export class GridLayoutEngine {
  /**
   * @param {Object} [options={}]
   * @param {number} [options.columns=3] - Grid columns
   * @param {number} [options.rowHeight=160] - Row height in px
   * @param {number} [options.gap=16] - Gap between widgets in px
   * @param {string} [options.layoutType='grid'] - Layout type
   */
  constructor(options = {}) {
    this.columns = options.columns ?? 3;
    this.rowHeight = options.rowHeight ?? 160;
    this.gap = options.gap ?? 16;
    this.layoutType = options.layoutType ?? LAYOUT_TYPES.GRID;
  }

  /**
   * Resolve layout breakpoints for a given viewport width
   * @param {number} viewportWidth - Viewport width in pixels
   * @returns {{ columns: number, rowHeight: number, breakpoint: string }}
   */
  resolveBreakpoint(viewportWidth) {
    let matched = BREAKPOINTS.mobile;
    let breakpointName = 'mobile';

    for (const [name, bp] of Object.entries(BREAKPOINTS)) {
      if (viewportWidth >= bp.minWidth) {
        matched = bp;
        breakpointName = name;
      }
    }
    return { columns: matched.columns, rowHeight: matched.rowHeight, breakpoint: breakpointName };
  }

  /**
   * Compute grid positions for a set of widgets
   * @param {Object[]} widgets - Widget instances with position/size data
   * @param {number} viewportWidth - Current viewport width
   * @returns {Object[]} Widgets with computed layout positions
   */
  computeLayout(widgets, viewportWidth) {
    const { columns, rowHeight } = this.resolveBreakpoint(viewportWidth);
    const gap = this.gap;

    const sorted = [...widgets].sort((a, b) => {
      const aPos = a.position ?? { row: 0, col: 0 };
      const bPos = b.position ?? { row: 0, col: 0 };
      return aPos.row - bPos.row || aPos.col - bPos.col;
    });

    /** @type {Set<string>} */
    const occupied = new Set();
    const result = [];

    for (const widget of sorted) {
      const size = this._parseSize(widget.size || WIDGET_SIZES.MEDIUM);
      const pos = this._findPosition(occupied, columns, size);

      const layoutData = {
        id: widget.id,
        type: widget.type,
        row: pos.row,
        col: pos.col,
        width: size.width,
        height: size.height,
        pixelX: pos.col * (100 / columns) + (pos.col > 0 ? gap / 2 : 0),
        pixelY: pos.row * (rowHeight + gap),
        pixelWidth: (size.width / columns) * 100 - gap,
        pixelHeight: size.height * rowHeight + (size.height - 1) * gap,
      };

      // Mark cells as occupied
      for (let r = pos.row; r < pos.row + size.height; r++) {
        for (let c = pos.col; c < pos.col + size.width; c++) {
          occupied.add(`${r},${c}`);
        }
      }
      result.push({ ...widget, layout: layoutData });
    }
    return result;
  }

  /**
   * Find the next available grid position
   * @private
   * @param {Set<string>} occupied - Occupied cells
   * @param {number} columns - Grid columns
   * @param {{ width: number, height: number }} size - Widget size
   * @returns {{ row: number, col: number }}
   */
  _findPosition(occupied, columns, size) {
    for (let row = 0; row < 100; row++) {
      for (let col = 0; col <= columns - size.width; col++) {
        let fits = true;
        for (let r = row; r < row + size.height && fits; r++) {
          for (let c = col; c < col + size.width && fits; c++) {
            if (occupied.has(`${r},${c}`)) fits = false;
          }
        }
        if (fits) return { row, col };
      }
    }
    return { row: 0, col: 0 };
  }

  /**
   * Parse a size string into width/height grid units
   * @private
   * @param {string} sizeStr - Size string like "2x1"
   * @returns {{ width: number, height: number }}
   */
  _parseSize(sizeStr) {
    const [w, h] = sizeStr.split('x').map(Number);
    return { width: w || 1, height: h || 1 };
  }
}

// ─── Dashboard Controller ────────────────────────────────────────────────────

/**
 * Main dashboard controller managing layout, themes, and state
 * @extends EventEmitter
 */
export class DashboardController extends EventEmitter {
  /**
   * @param {WidgetRegistry} [widgetRegistry] - Widget registry instance
   * @param {GridLayoutEngine} [layoutEngine] - Layout engine instance
   */
  constructor(widgetRegistry, layoutEngine) {
    super();
    /** @private */ this._widgetRegistry = widgetRegistry || new WidgetRegistry();
    /** @private */ this._layoutEngine = layoutEngine || new GridLayoutEngine();
    /** @private @type {Map<string, Object>} */ this._dashboards = new Map();
    /** @private */ this._currentTheme = THEMES.DARK;
    /** @private */ this._currentViewportWidth = 1920;
    /** @private @type {Map<string, Function[]>} */ this._subscribers = new Map();
  }

  /**
   * Create a new dashboard
   * @param {Object} config - Dashboard configuration
   * @param {string} config.name - Dashboard name
   * @param {string} [config.description] - Dashboard description
   * @param {string} [config.layoutType] - Layout type
   * @param {Object[]} [config.widgets=[]] - Initial widgets
   * @returns {Object} Created dashboard with ID
   */
  createDashboard(config) {
    if (!config.name || typeof config.name !== 'string') {
      throw new Error('Dashboard name is required');
    }

    const id = `dash_${Date.now()}_${Math.random().toString(36).slice(2, 8)}`;
    const dashboard = {
      id,
      name: config.name,
      description: config.description || '',
      layoutType: config.layoutType || LAYOUT_TYPES.GRID,
      theme: this._currentTheme,
      widgets: (config.widgets || []).map((w, idx) => ({
        id: `widget_${id}_${idx}`,
        type: w.type,
        title: w.title || w.type,
        size: w.size || WIDGET_SIZES.MEDIUM,
        position: w.position || null,
        config: w.config || {},
        enabled: w.enabled !== false,
      })),
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      metadata: {
        author: config.author || 'system',
        version: 1,
        tags: config.tags || [],
      },
    };

    this._dashboards.set(id, dashboard);
    this.emit('dashboard:created', dashboard);
    return dashboard;
  }

  /**
   * Retrieve a dashboard by ID
   * @param {string} id - Dashboard identifier
   * @returns {Object|null} Dashboard data or null
   */
  getDashboard(id) {
    return this._dashboards.get(id) || null;
  }

  /**
   * List all dashboards with optional filtering
   * @param {Object} [filter={}] - Filter criteria
   * @returns {Object[]} Array of dashboard summaries
   */
  listDashboards(filter = {}) {
    let dashboards = Array.from(this._dashboards.values());
    if (filter.tag) {
      dashboards = dashboards.filter((d) => d.metadata.tags.includes(filter.tag));
    }
    if (filter.name) {
      const nameFilter = filter.name.toLowerCase();
      dashboards = dashboards.filter((d) => d.name.toLowerCase().includes(nameFilter));
    }
    return dashboards.map((d) => ({
      id: d.id,
      name: d.name,
      description: d.description,
      widgetCount: d.widgets.length,
      theme: d.theme,
      updatedAt: d.updatedAt,
    }));
  }

  /**
   * Update a dashboard's configuration
   * @param {string} id - Dashboard ID
   * @param {Object} updates - Fields to update
   * @returns {Object} Updated dashboard
   */
  updateDashboard(id, updates) {
    const dashboard = this._dashboards.get(id);
    if (!dashboard) throw new Error(`Dashboard not found: ${id}`);

    const allowedFields = ['name', 'description', 'layoutType', 'theme', 'widgets', 'metadata'];
    for (const [key, value] of Object.entries(updates)) {
      if (allowedFields.includes(key)) {
        dashboard[key] = value;
      }
    }
    dashboard.updatedAt = new Date().toISOString();
    this.emit('dashboard:updated', dashboard);
    return dashboard;
  }

  /**
   * Delete a dashboard
   * @param {string} id - Dashboard ID
   * @returns {boolean} Whether deletion succeeded
   */
  deleteDashboard(id) {
    const deleted = this._dashboards.delete(id);
    if (deleted) this.emit('dashboard:deleted', { id });
    return deleted;
  }

  /**
   * Add a widget to an existing dashboard
   * @param {string} dashboardId - Dashboard ID
   * @param {Object} widgetConfig - Widget configuration
   * @returns {Object} Created widget
   */
  addWidget(dashboardId, widgetConfig) {
    const dashboard = this._dashboards.get(dashboardId);
    if (!dashboard) throw new Error(`Dashboard not found: ${dashboardId}`);

    const widgetType = this._widgetRegistry.get(widgetConfig.type);
    if (!widgetType) throw new Error(`Unknown widget type: ${widgetConfig.type}`);

    const widget = {
      id: `widget_${dashboardId}_${dashboard.widgets.length}`,
      type: widgetConfig.type,
      title: widgetConfig.title || widgetType.name,
      size: widgetConfig.size || widgetType.size,
      position: widgetConfig.position || null,
      config: widgetConfig.config || {},
      enabled: widgetConfig.enabled !== false,
    };

    dashboard.widgets.push(widget);
    dashboard.updatedAt = new Date().toISOString();
    this.emit('widget:added', { dashboardId, widget });
    return widget;
  }

  /**
   * Remove a widget from a dashboard
   * @param {string} dashboardId - Dashboard ID
   * @param {string} widgetId - Widget ID
   * @returns {boolean} Whether removal succeeded
   */
  removeWidget(dashboardId, widgetId) {
    const dashboard = this._dashboards.get(dashboardId);
    if (!dashboard) throw new Error(`Dashboard not found: ${dashboardId}`);

    const idx = dashboard.widgets.findIndex((w) => w.id === widgetId);
    if (idx === -1) return false;
    const [removed] = dashboard.widgets.splice(idx, 1);
    dashboard.updatedAt = new Date().toISOString();
    this.emit('widget:removed', { dashboardId, widget: removed });
    return true;
  }

  /**
   * Get the computed layout for a dashboard
   * @param {string} dashboardId - Dashboard ID
   * @param {number} [viewportWidth] - Viewport width for responsive layout
   * @returns {Object[]} Computed layout with positions
   */
  getComputedLayout(dashboardId, viewportWidth) {
    const dashboard = this._dashboards.get(dashboardId);
    if (!dashboard) throw new Error(`Dashboard not found: ${dashboardId}`);

    const width = viewportWidth || this._currentViewportWidth;
    const enabledWidgets = dashboard.widgets.filter((w) => w.enabled);
    return this._layoutEngine.computeLayout(enabledWidgets, width);
  }

  // ─── Theme Management ────────────────────────────────────────────────────

  /**
   * Set the active theme
   * @param {string} theme - Theme identifier from THEMES enum
   */
  setTheme(theme) {
    if (!THEME_CONFIGS[theme]) throw new Error(`Unknown theme: ${theme}`);
    this._currentTheme = theme;
    this.emit('theme:changed', { theme, config: THEME_CONFIGS[theme] });
  }

  /**
   * Get the current theme configuration
   * @returns {{ theme: string, config: Object }}
   */
  getTheme() {
    return { theme: this._currentTheme, config: THEME_CONFIGS[this._currentTheme] };
  }

  /**
   * Get all available themes
   * @returns {Object<string, Object>}
   */
  getAllThemes() {
    return { ...THEME_CONFIGS };
  }

  // ─── Export / Import ─────────────────────────────────────────────────────

  /**
   * Export a dashboard configuration as JSON
   * @param {string} dashboardId - Dashboard ID
   * @returns {string} JSON string of dashboard config
   */
  exportDashboard(dashboardId) {
    const dashboard = this._dashboards.get(dashboardId);
    if (!dashboard) throw new Error(`Dashboard not found: ${dashboardId}`);

    const exportData = {
      version: '1.0.0',
      exportedAt: new Date().toISOString(),
      dashboard: { ...dashboard },
      theme: this._currentTheme,
    };
    return JSON.stringify(exportData, null, 2);
  }

  /**
   * Import a dashboard from a JSON configuration
   * @param {string} jsonStr - JSON string of dashboard config
   * @returns {Object} Imported dashboard
   */
  importDashboard(jsonStr) {
    let importData;
    try {
      importData = JSON.parse(jsonStr);
    } catch {
      throw new Error('Invalid JSON: unable to parse dashboard configuration');
    }
    if (!importData.dashboard || !importData.dashboard.name) {
      throw new Error('Invalid dashboard configuration: missing name');
    }
    const dashboard = this.createDashboard({
      ...importData.dashboard,
      name: `${importData.dashboard.name} (imported)`,
    });
    this.emit('dashboard:imported', dashboard);
    return dashboard;
  }

  // ─── Subscriptions ───────────────────────────────────────────────────────

  /**
   * Subscribe to dashboard events
   * @param {string} eventType - Event type to subscribe to
   * @param {Function} callback - Event handler
   * @returns {Function} Unsubscribe function
   */
  subscribe(eventType, callback) {
    if (!this._subscribers.has(eventType)) {
      this._subscribers.set(eventType, []);
    }
    this._subscribers.get(eventType).push(callback);
    this.on(eventType, callback);
    return () => {
      const subs = this._subscribers.get(eventType);
      const idx = subs?.indexOf(callback) ?? -1;
      if (idx > -1) subs.splice(idx, 1);
      this.off(eventType, callback);
    };
  }

  /**
   * Get the widget registry instance
   * @returns {WidgetRegistry}
   */
  get registry() {
    return this._widgetRegistry;
  }

  /**
   * Get the layout engine instance
   * @returns {GridLayoutEngine}
   */
  get layout() {
    return this._layoutEngine;
  }

  /** @returns {number} */
  get dashboardCount() {
    return this._dashboards.size;
  }
}
