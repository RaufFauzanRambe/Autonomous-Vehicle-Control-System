/**
 * @fileoverview Main Application Entry Point
 * @description Express server setup, middleware chain, static file serving,
 *   WebSocket initialization, graceful shutdown, health check endpoints,
 *   and environment configuration for the Autonomous Vehicle Dashboard.
 * @module app
 */

import express from 'express';
import http from 'http';
import { Server as SocketIO } from 'socket.io';
import cors from 'cors';
import helmet from 'helmet';
import compression from 'compression';
import { DashboardRouter } from './dashboard_router.js';
import { DashboardService } from './dashboard_service.js';
import { DashboardStore } from './dashboard_store.js';
import { AlertManager } from './alerts.js';
import { SettingsManager } from './settings.js';

// ─── Environment Configuration ───────────────────────────────────────────────

/**
 * @typedef {Object} EnvironmentConfig
 * @property {number} port - Server port
 * @property {string} host - Server host
 * @property {string} nodeEnv - Node environment (development/production/test)
 * @property {number} wsPingInterval - WebSocket ping interval in ms
 * @property {number} wsPingTimeout - WebSocket ping timeout in ms
 * @property {number} maxConnections - Maximum concurrent connections
 * @property {number} rateLimitWindowMs - Rate limit window in ms
 * @property {number} rateLimitMax - Max requests per window
 * @property {string} corsOrigin - Allowed CORS origin
 * @property {string} logLevel - Logging level
 */

/** @type {EnvironmentConfig} */
const ENV_CONFIG = {
  port: parseInt(process.env.PORT, 10) || 8080,
  host: process.env.HOST || '0.0.0.0',
  nodeEnv: process.env.NODE_ENV || 'development',
  wsPingInterval: parseInt(process.env.WS_PING_INTERVAL, 10) || 25000,
  wsPingTimeout: parseInt(process.env.WS_PING_TIMEOUT, 10) || 60000,
  maxConnections: parseInt(process.env.MAX_CONNECTIONS, 10) || 1000,
  rateLimitWindowMs: parseInt(process.env.RATE_LIMIT_WINDOW, 10) || 900000,
  rateLimitMax: parseInt(process.env.RATE_LIMIT_MAX, 10) || 100,
  corsOrigin: process.env.CORS_ORIGIN || '*',
  logLevel: process.env.LOG_LEVEL || 'info',
};

// ─── Custom Error Classes ────────────────────────────────────────────────────

/**
 * Application-specific error with HTTP status code
 * @extends Error
 */
class AppError extends Error {
  /**
   * @param {string} message - Error description
   * @param {number} statusCode - HTTP status code
   * @param {boolean} [isOperational=true] - Whether error is operational
   */
  constructor(message, statusCode, isOperational = true) {
    super(message);
    this.name = 'AppError';
    this.statusCode = statusCode;
    this.isOperational = isOperational;
    this.timestamp = new Date().toISOString();
    Error.captureStackTrace(this, this.constructor);
  }
}

/**
 * Error for validation failures
 * @extends AppError
 */
class ValidationError extends AppError {
  /**
   * @param {string} message - Validation error description
   * @param {Object<string, string>} [errors={}] - Field-specific errors
   */
  constructor(message, errors = {}) {
    super(message, 400);
    this.name = 'ValidationError';
    this.errors = errors;
  }
}

// ─── Logger Utility ──────────────────────────────────────────────────────────

/** @enum {string} */
const LOG_LEVELS = { debug: 0, info: 1, warn: 2, error: 3, silent: 4 };

/**
 * Structured logger with level filtering
 */
class Logger {
  /** @param {string} [level='info'] - Minimum log level */
  constructor(level = 'info') {
    /** @private */ this._level = LOG_LEVELS[level] ?? LOG_LEVELS.info;
  }

  /**
   * Format and emit a log entry
   * @param {string} level - Log level
   * @param {string} message - Log message
   * @param {Object} [meta={}] - Additional metadata
   */
  _log(level, message, meta = {}) {
    if (LOG_LEVELS[level] < this._level) return;
    const entry = {
      timestamp: new Date().toISOString(),
      level,
      message,
      ...meta,
    };
    const output = JSON.stringify(entry);
    if (level === 'error') {
      console.error(output);
    } else if (level === 'warn') {
      console.warn(output);
    } else {
      console.log(output);
    }
  }

  /** @param {string} msg @param {Object} [m] */ debug(msg, m) { this._log('debug', msg, m); }
  /** @param {string} msg @param {Object} [m] */ info(msg, m) { this._log('info', msg, m); }
  /** @param {string} msg @param {Object} [m] */ warn(msg, m) { this._log('warn', msg, m); }
  /** @param {string} msg @param {Object} [m] */ error(msg, m) { this._log('error', msg, m); }
}

const logger = new Logger(ENV_CONFIG.logLevel);

// ─── Application Class ───────────────────────────────────────────────────────

/**
 * Main application class managing server lifecycle
 */
class Application {
  constructor() {
    /** @private */ this.app = express();
    /** @private */ this.server = null;
    /** @private */ this.io = null;
    /** @private */ this.store = new DashboardStore();
    /** @private */ this.alertManager = new AlertManager();
    /** @private */ this.settingsManager = new SettingsManager();
    /** @private */ this.dashboardService = new DashboardService(this.store, this.alertManager, this.settingsManager);
    /** @private */ this.isShuttingDown = false;
    /** @private */ this.activeConnections = new Set();
    /** @private */ this.startTime = Date.now();
    /** @private */ this.requestCount = 0;
    /** @private */ this.errorCount = 0;
  }

  /**
   * Initialize and configure all middleware
   * @private
   */
  _configureMiddleware() {
    // Security headers
    this.app.use(helmet({
      contentSecurityPolicy: {
        directives: {
          defaultSrc: ["'self'"],
          scriptSrc: ["'self'", "'unsafe-inline'"],
          styleSrc: ["'self'", "'unsafe-inline'"],
          imgSrc: ["'self'", 'data:', 'blob:'],
          connectSrc: ["'self'", 'ws:', 'wss:'],
        },
      },
      crossOriginEmbedderPolicy: false,
    }));

    // CORS
    this.app.use(cors({
      origin: ENV_CONFIG.corsOrigin,
      methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH', 'OPTIONS'],
      allowedHeaders: ['Content-Type', 'Authorization', 'X-Request-ID'],
      credentials: true,
      maxAge: 86400,
    }));

    // Compression
    this.app.use(compression({ threshold: 1024 }));

    // Body parsing
    this.app.use(express.json({ limit: '10mb' }));
    this.app.use(express.urlencoded({ extended: true, limit: '10mb' }));

    // Request tracking
    this.app.use((req, res, next) => {
      req.id = `req_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
      req.startTime = process.hrtime.bigint();
      this.requestCount++;
      res.setHeader('X-Request-ID', req.id);
      next();
    });

    // Static files
    this.app.use(express.static('public', {
      maxAge: ENV_CONFIG.nodeEnv === 'production' ? '1d' : 0,
      etag: true,
      lastModified: true,
    }));

    logger.info('Middleware configured successfully');
  }

  /**
   * Set up API routes
   * @private
   */
  _configureRoutes() {
    const dashboardRouter = new DashboardRouter(this.dashboardService, this.store);
    this.app.use('/api/dashboards', dashboardRouter.getRouter());

    // Health check endpoint
    this.app.get('/health', (req, res) => {
      const uptime = Date.now() - this.startTime;
      const memUsage = process.memoryUsage();
      const healthData = {
        status: this.isShuttingDown ? 'shutting_down' : 'healthy',
        uptime_ms: uptime,
        uptime_human: this._formatUptime(uptime),
        version: process.env.npm_package_version || '1.0.0',
        environment: ENV_CONFIG.nodeEnv,
        connections: this.activeConnections.size,
        requests_total: this.requestCount,
        errors_total: this.errorCount,
        memory: {
          rss_mb: (memUsage.rss / 1048576).toFixed(2),
          heap_used_mb: (memUsage.heapUsed / 1048576).toFixed(2),
          heap_total_mb: (memUsage.heapTotal / 1048576).toFixed(2),
          external_mb: (memUsage.external / 1048576).toFixed(2),
        },
        store: {
          dashboard_count: this.store.getDashboardCount(),
          widget_count: this.store.getWidgetCount(),
        },
        timestamp: new Date().toISOString(),
      };
      const statusCode = this.isShuttingDown ? 503 : 200;
      res.status(statusCode).json(healthData);
    });

    // Readiness probe
    this.app.get('/ready', (req, res) => {
      const checks = {
        store: this.store.isReady(),
        alertManager: this.alertManager.isReady(),
      };
      const allReady = Object.values(checks).every(Boolean);
      res.status(allReady ? 200 : 503).json({ ready: allReady, checks });
    });

    // Liveness probe
    this.app.get('/live', (req, res) => {
      res.status(200).json({ alive: true, pid: process.pid });
    });

    // API version info
    this.app.get('/api', (req, res) => {
      res.json({
        name: 'Autonomous Vehicle Dashboard API',
        version: '1.0.0',
        endpoints: {
          dashboards: '/api/dashboards',
          widgets: '/api/dashboards/:id/widgets',
          health: '/health',
          readiness: '/ready',
          liveness: '/live',
        },
      });
    });

    // 404 handler
    this.app.use((req, res) => {
      res.status(404).json({
        error: 'Not Found',
        message: `Route ${req.method} ${req.path} does not exist`,
        requestId: req.id,
        timestamp: new Date().toISOString(),
      });
    });

    logger.info('Routes configured successfully');
  }

  /**
   * Configure global error handler
   * @private
   */
  _configureErrorHandler() {
    this.app.use((err, req, res, _next) => {
      this.errorCount++;
      const statusCode = err.statusCode || 500;
      const isOperational = err.isOperational ?? false;

      logger.error('Unhandled error', {
        error: err.message,
        stack: err.stack,
        requestId: req.id,
        path: req.path,
        method: req.method,
        statusCode,
      });

      const response = {
        error: err.name || 'InternalServerError',
        message: ENV_CONFIG.nodeEnv === 'production' && !isOperational
          ? 'An internal error occurred'
          : err.message,
        requestId: req.id,
        timestamp: new Date().toISOString(),
      };

      if (err instanceof ValidationError) {
        response.errors = err.errors;
      }

      res.status(statusCode).json(response);
    });
  }

  /**
   * Initialize WebSocket server
   * @private
   */
  _configureWebSocket() {
    this.io = new SocketIO(this.server, {
      pingInterval: ENV_CONFIG.wsPingInterval,
      pingTimeout: ENV_CONFIG.wsPingTimeout,
      maxHttpBufferSize: 1e6,
      cors: { origin: ENV_CONFIG.corsOrigin, methods: ['GET', 'POST'] },
    });

    this.io.use((socket, next) => {
      const token = socket.handshake.auth?.token;
      if (!token && ENV_CONFIG.nodeEnv === 'production') {
        return next(new AppError('Authentication required', 401));
      }
      socket.userId = token ? `user_${token.slice(0, 8)}` : `anon_${socket.id}`;
      next();
    });

    this.io.on('connection', (socket) => {
      if (this.activeConnections.size >= ENV_CONFIG.maxConnections) {
        logger.warn('Max connections reached, rejecting', { socketId: socket.id });
        socket.disconnect(true);
        return;
      }

      this.activeConnections.add(socket.id);
      logger.info('Client connected', { socketId: socket.id, userId: socket.userId, total: this.activeConnections.size });

      // Subscribe to dashboard updates
      socket.on('dashboard:subscribe', (dashboardId) => {
        socket.join(`dashboard:${dashboardId}`);
        logger.debug('Client subscribed to dashboard', { socketId: socket.id, dashboardId });
      });

      socket.on('dashboard:unsubscribe', (dashboardId) => {
        socket.leave(`dashboard:${dashboardId}`);
        logger.debug('Client unsubscribed from dashboard', { socketId: socket.id, dashboardId });
      });

      // Vehicle telemetry real-time streaming
      socket.on('telemetry:stream', (data) => {
        try {
          this.store.updateTelemetry(data);
          this.io.emit('telemetry:update', data);
        } catch (err) {
          logger.error('Telemetry processing error', { error: err.message });
          socket.emit('error', { message: 'Failed to process telemetry data' });
        }
      });

      // Alert acknowledgment
      socket.on('alert:acknowledge', (alertId) => {
        this.alertManager.acknowledgeAlert(alertId, socket.userId);
        this.io.emit('alert:updated', { id: alertId, acknowledgedBy: socket.userId });
      });

      socket.on('disconnect', (reason) => {
        this.activeConnections.delete(socket.id);
        logger.info('Client disconnected', { socketId: socket.id, reason, total: this.activeConnections.size });
      });

      socket.on('error', (err) => {
        logger.error('Socket error', { socketId: socket.id, error: err.message });
      });
    });

    // Bridge store events to WebSocket
    this.store.on('stateChanged', (change) => {
      if (this.io) {
        this.io.emit('state:changed', change);
      }
    });

    this.alertManager.on('alertFired', (alert) => {
      if (this.io) {
        this.io.emit('alert:new', alert);
      }
    });

    logger.info('WebSocket server configured');
  }

  /**
   * Set up periodic tasks
   * @private
   */
  _configurePeriodicTasks() {
    // Memory monitoring every 30s
    this._memoryInterval = setInterval(() => {
      const mem = process.memoryUsage();
      if (mem.heapUsed / mem.heapTotal > 0.9) {
        logger.warn('High memory usage detected', {
          heapUsedMB: (mem.heapUsed / 1048576).toFixed(2),
          heapTotalMB: (mem.heapTotal / 1048576).toFixed(2),
          ratio: (mem.heapUsed / mem.heapTotal).toFixed(3),
        });
        this.alertManager.fireAlert('high_memory', 'warning', 'Memory usage exceeds 90% threshold');
      }
    }, 30000);

    // Telemetry data cleanup every 5 minutes
    this._cleanupInterval = setInterval(() => {
      try {
        this.dashboardService.cleanupStaleData();
        logger.debug('Stale data cleanup completed');
      } catch (err) {
        logger.error('Cleanup task failed', { error: err.message });
      }
    }, 300000);

    // Connection health check every 60s
    this._healthInterval = setInterval(() => {
      if (this.io) {
        const connectedSockets = this.io.sockets.sockets.size;
        logger.info('Connection health check', { activeSockets: connectedSockets, activeConnections: this.activeConnections.size });
      }
    }, 60000);
  }

  /**
   * Register graceful shutdown handlers
   * @private
   */
  _registerShutdownHandlers() {
    const shutdown = async (signal) => {
      if (this.isShuttingDown) return;
      this.isShuttingDown = true;
      logger.info(`Received ${signal}, starting graceful shutdown...`);

      // Stop accepting new connections
      if (this.io) {
        this.io.disconnectSockets(true);
        this.io.close();
      }

      // Clear periodic tasks
      clearInterval(this._memoryInterval);
      clearInterval(this._cleanupInterval);
      clearInterval(this._healthInterval);

      // Close HTTP server
      if (this.server) {
        this.server.close(() => {
          logger.info('HTTP server closed');
        });
      }

      // Flush remaining store state
      try {
        await this.store.flush();
        logger.info('Store state flushed');
      } catch (err) {
        logger.error('Store flush failed', { error: err.message });
      }

      logger.info('Graceful shutdown completed');
      process.exit(0);
    };

    process.on('SIGTERM', () => shutdown('SIGTERM'));
    process.on('SIGINT', () => shutdown('SIGINT'));
    process.on('SIGUSR2', () => shutdown('SIGUSR2')); // nodemon restart

    process.on('uncaughtException', (err) => {
      logger.error('Uncaught exception', { error: err.message, stack: err.stack });
      shutdown('uncaughtException');
    });

    process.on('unhandledRejection', (reason) => {
      logger.error('Unhandled rejection', { reason: String(reason) });
      shutdown('unhandledRejection');
    });
  }

  /**
   * Format uptime in human-readable form
   * @private
   * @param {number} ms - Uptime in milliseconds
   * @returns {string} Formatted uptime string
   */
  _formatUptime(ms) {
    const seconds = Math.floor(ms / 1000);
    const days = Math.floor(seconds / 86400);
    const hours = Math.floor((seconds % 86400) / 3600);
    const minutes = Math.floor((seconds % 3600) / 60);
    const parts = [];
    if (days > 0) parts.push(`${days}d`);
    if (hours > 0) parts.push(`${hours}h`);
    if (minutes > 0) parts.push(`${minutes}m`);
    parts.push(`${seconds % 60}s`);
    return parts.join(' ');
  }

  /**
   * Start the application server
   * @returns {Promise<http.Server>}
   */
  async start() {
    try {
      this._configureMiddleware();
      this._configureRoutes();
      this._configureErrorHandler();

      this.server = http.createServer(this.app);
      this._configureWebSocket();
      this._configurePeriodicTasks();
      this._registerShutdownHandlers();

      await new Promise((resolve, reject) => {
        this.server.listen(ENV_CONFIG.port, ENV_CONFIG.host, () => {
          resolve();
        });
        this.server.on('error', reject);
      });

      logger.info(`Autonomous Vehicle Dashboard server started`, {
        host: ENV_CONFIG.host,
        port: ENV_CONFIG.port,
        environment: ENV_CONFIG.nodeEnv,
        pid: process.pid,
      });

      return this.server;
    } catch (err) {
      logger.error('Failed to start server', { error: err.message });
      throw err;
    }
  }

  /**
   * Stop the application server gracefully
   * @returns {Promise<void>}
   */
  async stop() {
    logger.info('Stopping application server...');
    if (this.server) {
      await new Promise((resolve) => this.server.close(resolve));
    }
    logger.info('Application server stopped');
  }
}

// ─── Entry Point ─────────────────────────────────────────────────────────────

const application = new Application();

application.start().catch((err) => {
  console.error('Fatal error starting application:', err);
  process.exit(1);
});

export { Application, AppError, ValidationError, ENV_CONFIG, logger };
