/**
 * @fileoverview SystemStatus - Overall system status aggregation for the
 * autonomous vehicle control system. Tracks per-module status (perception,
 * planning, control, communication), dependency status, mode indicators
 * (autonomous/manual/emergency), status change events, and provides
 * dashboard-ready status data.
 *
 * @module realtime_monitoring/system_status
 * @version 2.1.0
 * @author Autonomous Vehicle Control System Team
 */

import { EventEmitter } from 'events';

/** @typedef {'autonomous'|'manual'|'emergency'|'maintenance'|'initializing'} OperationMode */

/** @typedef {'online'|'degraded'|'offline'|'error'|'unknown'} ModuleStatus */

/** @typedef {'connected'|'degraded'|'disconnected'|'unknown'} DependencyStatus */

/**
 * @typedef {Object} SystemStatusConfig
 * @property {number} [updateIntervalMs=500] - Status update interval
 * @property {number} [statusTimeoutMs=5000] - Module status timeout
 * @property {number} [maxHistoryEntries=1000] - Max status history entries
 * @property {Object} [dependencies] - External dependency definitions
 */

/**
 * @typedef {Object} ModuleInfo
 * @property {string} name - Module name
 * @property {ModuleStatus} status - Current status
 * @property {number} lastUpdate - Last status update timestamp
 * @property {number} lastHealthScore - Last known health score (0-1)
 * @property {string} version - Module version
 * @property {number} [uptime] - Module uptime in ms
 * @property {Object} [metadata] - Additional metadata
 * @property {string[]} [dependencies] - Module's dependencies
 */

/**
 * @typedef {Object} DependencyInfo
 * @property {string} name - Dependency name
 * @property {string} type - Dependency type (service, hardware, network)
 * @property {DependencyStatus} status - Current status
 * @property {string} [endpoint] - Dependency endpoint
 * @property {number} [lastCheck] - Last check timestamp
 * @property {string} [message] - Status message
 */

/**
 * @typedef {Object} SystemStatusReport
 * @property {number} timestamp - Report timestamp
 * @property {OperationMode} mode - Current operation mode
 * @property {'operational'|'degraded'|'critical'|'offline'} overallStatus - Overall status
 * @property {Object<string, ModuleInfo>} modules - Per-module status
 * @property {Object<string, DependencyInfo>} dependencies - Dependency status
 * @property {Object} modeHistory - Recent mode changes
 * @property {Object} summary - Status summary counts
 */

/**
 * @typedef {Object} ModeChangeEvent
 * @property {OperationMode} from - Previous mode
 * @property {OperationMode} to - New mode
 * @property {number} timestamp - Change timestamp
 * @property {string} [reason] - Reason for change
 * @property {string} [triggeredBy] - What triggered the change
 */

/**
 * SystemStatus provides a unified view of the autonomous vehicle system's
 * operational state, aggregating module status, dependency health, and
 * operation mode.
 *
 * @extends EventEmitter
 *
 * @example
 * const status = new SystemStatus();
 *
 * status.on('mode:change', (event) => {
 *   if (event.to === 'emergency') {
 *     triggerEmergencyProtocol();
 *   }
 * });
 *
 * status.registerModule('perception', { version: '2.1.0', dependencies: ['lidar', 'camera'] });
 * status.updateModuleStatus('perception', 'online');
 * status.setMode('autonomous');
 */
export class SystemStatus extends EventEmitter {
  /** @type {SystemStatusConfig} */
  #config;

  /** @type {OperationMode} */
  #mode = 'initializing';

  /** @type {Map<string, ModuleInfo>} */
  #modules = new Map();

  /** @type {Map<string, DependencyInfo>} */
  #dependencies = new Map();

  /** @type {ModeChangeEvent[]} */
  #modeHistory = [];

  /** @type {Map<string, ModuleStatus>} */
  #previousModuleStatus = new Map();

  /** @type {NodeJS.Timeout|null} */
  #updateTimer = null;

  /** @type {boolean} */
  #running = false;

  /**
   * Creates a new SystemStatus.
   *
   * @param {SystemStatusConfig} [config={}] - Configuration
   */
  constructor(config = {}) {
    super();
    this.setMaxListeners(20);

    this.#config = {
      updateIntervalMs: 500,
      statusTimeoutMs: 5000,
      maxHistoryEntries: 1000,
      dependencies: {},
      ...config,
    };

    // Register built-in modules
    this.#registerBuiltinModules();
    this.#registerBuiltinDependencies();
  }

  /**
   * Current operation mode.
   * @type {OperationMode}
   */
  get mode() {
    return this.#mode;
  }

  /**
   * Whether the system is in autonomous mode.
   * @type {boolean}
   */
  get isAutonomous() {
    return this.#mode === 'autonomous';
  }

  /**
   * Whether the system is in emergency mode.
   * @type {boolean}
   */
  get isEmergency() {
    return this.#mode === 'emergency';
  }

  /**
   * Number of registered modules.
   * @type {number}
   */
  get moduleCount() {
    return this.#modules.size;
  }

  /**
   * Number of registered dependencies.
   * @type {number}
   */
  get dependencyCount() {
    return this.#dependencies.size;
  }

  /**
   * Starts the status monitoring loop.
   *
   * @returns {void}
   */
  start() {
    if (this.#running) return;
    this.#running = true;

    this.#updateTimer = setInterval(() => {
      this.#checkModuleTimeouts();
      this.#checkDependencyStatuses();
      this.#emitStatusUpdate();
    }, this.#config.updateIntervalMs);
  }

  /**
   * Stops the status monitoring loop.
   *
   * @returns {void}
   */
  stop() {
    this.#running = false;

    if (this.#updateTimer) {
      clearInterval(this.#updateTimer);
      this.#updateTimer = null;
    }
  }

  /**
   * Registers a module for status tracking.
   *
   * @param {string} name - Module name
   * @param {Object} [options={}] - Module options
   * @param {string} [options.version] - Module version
   * @param {string[]} [options.dependencies] - Module dependencies
   * @param {Object} [options.metadata] - Additional metadata
   * @returns {void}
   */
  registerModule(name, options = {}) {
    this.#modules.set(name, {
      name,
      status: 'unknown',
      lastUpdate: Date.now(),
      lastHealthScore: 1.0,
      version: options.version || '0.0.0',
      uptime: 0,
      metadata: options.metadata || {},
      dependencies: options.dependencies || [],
    });
  }

  /**
   * Updates the status of a registered module.
   *
   * @param {string} name - Module name
   * @param {ModuleStatus} status - New status
   * @param {Object} [data={}] - Additional update data
   * @returns {boolean} Whether the update was accepted
   *
   * @fires SystemStatus#module:status_change
   */
  updateModuleStatus(name, status, data = {}) {
    const module = this.#modules.get(name);
    if (!module) return false;

    const previousStatus = module.status;
    module.status = status;
    module.lastUpdate = Date.now();

    if (data.healthScore !== undefined) module.lastHealthScore = data.healthScore;
    if (data.uptime !== undefined) module.uptime = data.uptime;
    if (data.metadata) module.metadata = { ...module.metadata, ...data.metadata };

    if (previousStatus !== status) {
      this.#previousModuleStatus.set(name, previousStatus);

      /**
       * @event SystemStatus#module:status_change
       * @type {Object}
       * @property {string} module - Module name
       * @property {ModuleStatus} from - Previous status
       * @property {ModuleStatus} to - New status
       * @property {number} timestamp
       */
      this.emit('module:status_change', {
        module: name,
        from: previousStatus,
        to: status,
        timestamp: Date.now(),
      });

      // Auto-mode adjustment based on critical module failures
      this.#adjustModeForModuleChange(name, status);
    }

    return true;
  }

  /**
   * Updates module health information.
   *
   * @param {string} name - Module name
   * @param {Object} healthData - Health data from HealthMonitor
   * @returns {void}
   */
  updateModuleHealth(name, healthData) {
    const module = this.#modules.get(name);
    if (!module) return;

    if (healthData.status) {
      this.updateModuleStatus(name, this.#healthToModuleStatus(healthData.status), {
        healthScore: healthData.score,
      });
    } else if (healthData.score !== undefined) {
      module.lastHealthScore = healthData.score;
    }
  }

  /**
   * Sets the system operation mode.
   *
   * @param {OperationMode} mode - New operation mode
   * @param {string} [reason] - Reason for mode change
   * @param {string} [triggeredBy] - What triggered the change
   * @returns {void}
   *
   * @fires SystemStatus#mode:change
   */
  setMode(mode, reason = '', triggeredBy = 'system') {
    if (!['autonomous', 'manual', 'emergency', 'maintenance', 'initializing'].includes(mode)) {
      throw new TypeError(`Invalid operation mode: ${mode}`);
    }

    const previousMode = this.#mode;
    if (previousMode === mode) return;

    this.#mode = mode;

    const event = {
      from: previousMode,
      to: mode,
      timestamp: Date.now(),
      reason,
      triggeredBy,
    };

    this.#modeHistory.push(event);
    while (this.#modeHistory.length > this.#config.maxHistoryEntries) {
      this.#modeHistory.shift();
    }

    /**
     * @event SystemStatus#mode:change
     * @type {ModeChangeEvent}
     */
    this.emit('mode:change', event);
  }

  /**
   * Registers an external dependency for tracking.
   *
   * @param {string} name - Dependency name
   * @param {Object} [options={}] - Dependency options
   * @param {string} [options.type] - Dependency type
   * @param {string} [options.endpoint] - Endpoint URL
   * @returns {void}
   */
  registerDependency(name, options = {}) {
    this.#dependencies.set(name, {
      name,
      type: options.type || 'service',
      status: 'unknown',
      endpoint: options.endpoint || '',
      lastCheck: Date.now(),
      message: '',
    });
  }

  /**
   * Updates a dependency's status.
   *
   * @param {string} name - Dependency name
   * @param {DependencyStatus} status - New status
   * @param {string} [message=''] - Status message
   * @returns {void}
   */
  updateDependencyStatus(name, status, message = '') {
    const dep = this.#dependencies.get(name);
    if (!dep) return;

    const previousStatus = dep.status;
    dep.status = status;
    dep.lastCheck = Date.now();
    dep.message = message;

    if (previousStatus !== status) {
      this.emit('dependency:status_change', {
        dependency: name,
        from: previousStatus,
        to: status,
        timestamp: Date.now(),
      });
    }
  }

  /**
   * Gets the complete system status report.
   *
   * @returns {SystemStatusReport} Current status report
   */
  getStatus() {
    const timestamp = Date.now();
    const modules = {};
    const dependencies = {};

    for (const [name, module] of this.#modules) {
      modules[name] = { ...module };
    }

    for (const [name, dep] of this.#dependencies) {
      dependencies[name] = { ...dep };
    }

    return {
      timestamp,
      mode: this.#mode,
      overallStatus: this.#computeOverallStatus(),
      modules,
      dependencies,
      modeHistory: this.#modeHistory.slice(-10),
      summary: this.#computeSummary(),
    };
  }

  /**
   * Gets the overall system status without full report.
   *
   * @returns {'operational'|'degraded'|'critical'|'offline'} Overall status
   */
  getOverallStatus() {
    return this.#computeOverallStatus();
  }

  /**
   * Gets mode change history.
   *
   * @param {number} [limit=20] - Max entries
   * @returns {ModeChangeEvent[]} Mode change history
   */
  getModeHistory(limit = 20) {
    return this.#modeHistory.slice(-limit);
  }

  /**
   * Gets a specific module's status.
   *
   * @param {string} name - Module name
   * @returns {ModuleInfo|null} Module info or null
   */
  getModuleStatus(name) {
    const module = this.#modules.get(name);
    return module ? { ...module } : null;
  }

  // ─── Private Methods ───────────────────────────────────────────────

  /**
   * Registers built-in AV modules.
   * @private
   */
  #registerBuiltinModules() {
    const builtins = [
      { name: 'perception', version: '2.1.0', dependencies: ['lidar', 'camera', 'radar'] },
      { name: 'planning', version: '2.1.0', dependencies: ['perception', 'hd_map'] },
      { name: 'control', version: '2.1.0', dependencies: ['planning', 'can_bus'] },
      { name: 'communication', version: '2.1.0', dependencies: ['v2x', 'cloud'] },
    ];

    for (const mod of builtins) {
      this.registerModule(mod.name, {
        version: mod.version,
        dependencies: mod.dependencies,
      });
    }
  }

  /**
   * Registers built-in dependencies.
   * @private
   */
  #registerBuiltinDependencies() {
    const deps = {
      lidar: { type: 'hardware' },
      camera: { type: 'hardware' },
      radar: { type: 'hardware' },
      can_bus: { type: 'hardware' },
      hd_map: { type: 'service' },
      v2x: { type: 'network' },
      cloud: { type: 'service' },
      gps: { type: 'hardware' },
      imu: { type: 'hardware' },
    };

    for (const [name, options] of Object.entries(deps)) {
      this.registerDependency(name, options);
    }

    // Also register any configured dependencies
    for (const [name, options] of Object.entries(this.#config.dependencies)) {
      this.registerDependency(name, options);
    }
  }

  /**
   * Checks for module status timeouts.
   * @private
   */
  #checkModuleTimeouts() {
    const now = Date.now();

    for (const [name, module] of this.#modules) {
      if (module.status === 'offline' || module.status === 'error') continue;

      if (now - module.lastUpdate > this.#config.statusTimeoutMs) {
        this.updateModuleStatus(name, 'degraded');
      }
    }
  }

  /**
   * Checks dependency statuses (simulated check).
   * @private
   */
  #checkDependencyStatuses() {
    for (const [name, dep] of this.#dependencies) {
      dep.lastCheck = Date.now();
      // In a real system, this would ping or check the actual dependency
    }
  }

  /**
   * Computes the overall system status based on module and dependency states.
   * @private
   * @returns {'operational'|'degraded'|'critical'|'offline'}
   */
  #computeOverallStatus() {
    if (this.#mode === 'emergency') return 'critical';
    if (this.#mode === 'maintenance' || this.#mode === 'initializing') return 'offline';

    let hasDegraded = false;
    let hasCritical = false;

    for (const module of this.#modules.values()) {
      if (module.status === 'error' || module.status === 'offline') {
        hasCritical = true;
      } else if (module.status === 'degraded') {
        hasDegraded = true;
      }
    }

    for (const dep of this.#dependencies.values()) {
      if (dep.status === 'disconnected') {
        hasCritical = true;
      } else if (dep.status === 'degraded') {
        hasDegraded = true;
      }
    }

    if (hasCritical) return 'critical';
    if (hasDegraded) return 'degraded';
    return 'operational';
  }

  /**
   * Computes a status summary.
   * @private
   * @returns {Object} Summary counts
   */
  #computeSummary() {
    const moduleSummary = { online: 0, degraded: 0, offline: 0, error: 0, unknown: 0 };
    const depSummary = { connected: 0, degraded: 0, disconnected: 0, unknown: 0 };

    for (const module of this.#modules.values()) {
      moduleSummary[module.status] = (moduleSummary[module.status] || 0) + 1;
    }

    for (const dep of this.#dependencies.values()) {
      depSummary[dep.status] = (depSummary[dep.status] || 0) + 1;
    }

    return {
      totalModules: this.#modules.size,
      moduleStatus: moduleSummary,
      totalDependencies: this.#dependencies.size,
      dependencyStatus: depSummary,
      operationMode: this.#mode,
    };
  }

  /**
   * Adjusts operation mode based on critical module failures.
   * @private
   * @param {string} moduleName - Module that changed
   * @param {ModuleStatus} newStatus - New module status
   */
  #adjustModeForModuleChange(moduleName, newStatus) {
    const criticalModules = ['perception', 'control'];

    if (criticalModules.includes(moduleName) && (newStatus === 'error' || newStatus === 'offline')) {
      if (this.#mode === 'autonomous') {
        this.setMode('manual', `${moduleName} module ${newStatus}`, 'auto_fallback');
      }
    }
  }

  /**
   * Converts a health status to a module status.
   * @private
   * @param {string} healthStatus - Health status string
   * @returns {ModuleStatus}
   */
  #healthToModuleStatus(healthStatus) {
    switch (healthStatus) {
      case 'healthy': return 'online';
      case 'degraded': return 'degraded';
      case 'unhealthy': return 'error';
      case 'critical': return 'offline';
      default: return 'unknown';
    }
  }

  /**
   * Emits a periodic status update event.
   * @private
   */
  #emitStatusUpdate() {
    /**
     * @event SystemStatus#status:update
     * @type {SystemStatusReport}
     */
    this.emit('status:update', this.getStatus());
  }
}

export default SystemStatus;
