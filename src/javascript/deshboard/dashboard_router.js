/**
 * @fileoverview Dashboard Express Router
 * @description REST API endpoints for dashboard CRUD operations, widget management,
 *   layout endpoints, and authentication middleware for the Autonomous Vehicle Dashboard.
 * @module dashboard_router
 */

import express from 'express';
import crypto from 'crypto';

// ─── Authentication Middleware ────────────────────────────────────────────────

/**
 * Authentication middleware for API endpoints
 * @param {Object} options - Auth options
 * @param {boolean} [options.required=true] - Whether auth is required
 * @param {string[]} [options.roles=[]] - Required roles
 * @returns {express.RequestHandler}
 */
const authenticate = (options = {}) => {
  const { required = true, roles = [] } = options;

  return (req, res, next) => {
    const authHeader = req.headers.authorization;
    const apiKey = req.headers['x-api-key'];

    // API key authentication
    if (apiKey) {
      const keyHash = crypto.createHash('sha256').update(apiKey).digest('hex');
      // In production, validate against stored API keys
      req.user = {
        id: `apikey_${keyHash.slice(0, 12)}`,
        type: 'apikey',
        roles: ['admin', 'user'],
        authenticatedAt: new Date().toISOString(),
      };
      return next();
    }

    // Bearer token authentication
    if (authHeader?.startsWith('Bearer ')) {
      const token = authHeader.slice(7);
      if (token.length < 1) {
        return res.status(401).json({
          error: 'InvalidToken',
          message: 'Bearer token is empty',
          timestamp: new Date().toISOString(),
        });
      }
      try {
        const tokenHash = crypto.createHash('sha256').update(token).digest('hex');
        req.user = {
          id: `user_${tokenHash.slice(0, 12)}`,
          type: 'bearer',
          roles: ['user'],
          authenticatedAt: new Date().toISOString(),
        };
        // Role check
        if (roles.length > 0 && !roles.some((r) => req.user.roles.includes(r))) {
          return res.status(403).json({
            error: 'Forbidden',
            message: `Requires one of roles: ${roles.join(', ')}`,
            timestamp: new Date().toISOString(),
          });
        }
        return next();
      } catch {
        return res.status(401).json({
          error: 'InvalidToken',
          message: 'Failed to validate token',
          timestamp: new Date().toISOString(),
        });
      }
    }

    // No auth provided
    if (required) {
      return res.status(401).json({
        error: 'AuthenticationRequired',
        message: 'Provide Authorization header or X-API-Key',
        timestamp: new Date().toISOString(),
      });
    }

    // Anonymous access
    req.user = { id: 'anonymous', type: 'anonymous', roles: ['viewer'] };
    next();
  };
};

// ─── Validation Middleware ────────────────────────────────────────────────────

/**
 * Validate request body against a schema
 * @param {Object} schema - Validation schema
 * @param {Object} schema.name - Name field rules
 * @param {boolean} schema.name.required - Whether field is required
 * @param {string} schema.name.type - Expected type
 * @returns {express.RequestHandler}
 */
const validateBody = (schema) => {
  return (req, res, next) => {
    const errors = {};

    for (const [field, rules] of Object.entries(schema)) {
      const value = req.body[field];

      if (rules.required && (value === undefined || value === null || value === '')) {
        errors[field] = `${field} is required`;
        continue;
      }

      if (value !== undefined && rules.type && typeof value !== rules.type) {
        errors[field] = `${field} must be of type ${rules.type}`;
      }

      if (value !== undefined && rules.minLength && String(value).length < rules.minLength) {
        errors[field] = `${field} must be at least ${rules.minLength} characters`;
      }

      if (value !== undefined && rules.maxLength && String(value).length > rules.maxLength) {
        errors[field] = `${field} must not exceed ${rules.maxLength} characters`;
      }

      if (value !== undefined && rules.pattern && !rules.pattern.test(String(value))) {
        errors[field] = `${field} format is invalid`;
      }
    }

    if (Object.keys(errors).length > 0) {
      return res.status(400).json({
        error: 'ValidationError',
        message: 'Request body validation failed',
        errors,
        timestamp: new Date().toISOString(),
      });
    }
    next();
  };
};

// ─── Request Logger Middleware ────────────────────────────────────────────────

/**
 * Log API request details
 * @returns {express.RequestHandler}
 */
const requestLogger = () => {
  return (req, res, next) => {
    const start = process.hrtime.bigint();
    res.on('finish', () => {
      const duration = Number(process.hrtime.bigint() - start) / 1e6;
      const logEntry = {
        method: req.method,
        path: req.path,
        status: res.statusCode,
        durationMs: duration.toFixed(2),
        userId: req.user?.id || 'anonymous',
        requestId: req.id,
        ip: req.ip,
      };
      if (res.statusCode >= 500) {
        console.error(JSON.stringify(logEntry));
      } else if (res.statusCode >= 400) {
        console.warn(JSON.stringify(logEntry));
      } else {
        console.log(JSON.stringify(logEntry));
      }
    });
    next();
  };
};

// ─── Dashboard Router Class ──────────────────────────────────────────────────

/**
 * Express router for dashboard API endpoints
 */
export class DashboardRouter {
  /**
   * @param {Object} dashboardService - Dashboard service instance
   * @param {Object} dashboardStore - Dashboard store instance
   */
  constructor(dashboardService, dashboardStore) {
    /** @private */ this.service = dashboardService;
    /** @private */ this.store = dashboardStore;
    /** @private */ this.router = express.Router();
    this._registerRoutes();
  }

  /**
   * Register all API routes
   * @private
   */
  _registerRoutes() {
    const r = this.router;

    // Apply request logger to all routes
    r.use(requestLogger());

    // ─── Dashboard CRUD ────────────────────────────────────────────────

    /**
     * GET / - List all dashboards
     * @route GET /
     * @query {string} [name] - Filter by name (partial match)
     * @query {string} [tag] - Filter by tag
     * @query {number} [limit=50] - Max results
     * @query {number} [offset=0] - Offset for pagination
     */
    r.get('/', authenticate({ required: false }), (req, res) => {
      try {
        const { name, tag, limit = 50, offset = 0 } = req.query;
        const filter = {};
        if (name) filter.name = String(name);
        if (tag) filter.tag = String(tag);

        const dashboards = this.service.listDashboards(filter);
        const total = dashboards.length;
        const paginated = dashboards.slice(Number(offset), Number(offset) + Number(limit));

        res.json({
          data: paginated,
          pagination: {
            total,
            limit: Number(limit),
            offset: Number(offset),
            hasMore: Number(offset) + Number(limit) < total,
          },
        });
      } catch (err) {
        res.status(500).json({ error: 'InternalServerError', message: err.message });
      }
    });

    /**
     * POST / - Create a new dashboard
     * @route POST /
     * @body {string} name - Dashboard name (required)
     * @body {string} [description] - Description
     * @body {string} [layoutType] - Layout type
     * @body {Object[]} [widgets] - Initial widgets
     */
    r.post('/',
      authenticate({ required: true }),
      validateBody({ name: { required: true, type: 'string', minLength: 1, maxLength: 100 } }),
      (req, res) => {
        try {
          const dashboard = this.service.createDashboard({
            ...req.body,
            author: req.user.id,
          });
          res.status(201).json({ data: dashboard });
        } catch (err) {
          res.status(400).json({ error: 'BadRequest', message: err.message });
        }
      }
    );

    /**
     * GET /:id - Get a specific dashboard
     * @route GET /:id
     * @param {string} id - Dashboard ID
     */
    r.get('/:id', authenticate({ required: false }), (req, res) => {
      try {
        const dashboard = this.service.getDashboard(req.params.id);
        if (!dashboard) {
          return res.status(404).json({
            error: 'NotFound',
            message: `Dashboard not found: ${req.params.id}`,
          });
        }
        res.json({ data: dashboard });
      } catch (err) {
        res.status(500).json({ error: 'InternalServerError', message: err.message });
      }
    });

    /**
     * PUT /:id - Update a dashboard
     * @route PUT /:id
     * @param {string} id - Dashboard ID
     * @body {string} [name] - Updated name
     * @body {string} [description] - Updated description
     * @body {string} [layoutType] - Updated layout type
     * @body {string} [theme] - Updated theme
     */
    r.put('/:id', authenticate({ required: true, roles: ['admin', 'user'] }), (req, res) => {
      try {
        const dashboard = this.service.updateDashboard(req.params.id, req.body);
        res.json({ data: dashboard });
      } catch (err) {
        const status = err.message.includes('not found') ? 404 : 400;
        res.status(status).json({ error: status === 404 ? 'NotFound' : 'BadRequest', message: err.message });
      }
    });

    /**
     * DELETE /:id - Delete a dashboard
     * @route DELETE /:id
     * @param {string} id - Dashboard ID
     */
    r.delete('/:id', authenticate({ required: true, roles: ['admin'] }), (req, res) => {
      try {
        const deleted = this.service.deleteDashboard(req.params.id);
        if (!deleted) {
          return res.status(404).json({
            error: 'NotFound',
            message: `Dashboard not found: ${req.params.id}`,
          });
        }
        res.json({ data: { id: req.params.id, deleted: true } });
      } catch (err) {
        res.status(500).json({ error: 'InternalServerError', message: err.message });
      }
    });

    // ─── Widget Endpoints ──────────────────────────────────────────────

    /**
     * GET /:id/widgets - List widgets for a dashboard
     * @route GET /:id/widgets
     */
    r.get('/:id/widgets', authenticate({ required: false }), (req, res) => {
      try {
        const widgets = this.service.getWidgets(req.params.id);
        if (widgets === null) {
          return res.status(404).json({ error: 'NotFound', message: `Dashboard not found: ${req.params.id}` });
        }
        res.json({ data: widgets });
      } catch (err) {
        res.status(500).json({ error: 'InternalServerError', message: err.message });
      }
    });

    /**
     * POST /:id/widgets - Add a widget to a dashboard
     * @route POST /:id/widgets
     * @body {string} type - Widget type (required)
     * @body {string} [title] - Widget title
     * @body {string} [size] - Widget size
     * @body {Object} [config] - Widget configuration
     */
    r.post('/:id/widgets',
      authenticate({ required: true }),
      validateBody({ type: { required: true, type: 'string' } }),
      (req, res) => {
        try {
          const widget = this.service.addWidget(req.params.id, req.body);
          res.status(201).json({ data: widget });
        } catch (err) {
          const status = err.message.includes('not found') ? 404 : 400;
          res.status(status).json({ error: status === 404 ? 'NotFound' : 'BadRequest', message: err.message });
        }
      }
    );

    /**
     * DELETE /:id/widgets/:widgetId - Remove a widget
     * @route DELETE /:id/widgets/:widgetId
     */
    r.delete('/:id/widgets/:widgetId', authenticate({ required: true }), (req, res) => {
      try {
        const removed = this.service.removeWidget(req.params.id, req.params.widgetId);
        if (!removed) {
          return res.status(404).json({ error: 'NotFound', message: 'Widget not found' });
        }
        res.json({ data: { removed: true } });
      } catch (err) {
        const status = err.message.includes('not found') ? 404 : 500;
        res.status(status).json({ error: status === 404 ? 'NotFound' : 'InternalServerError', message: err.message });
      }
    });

    // ─── Layout Endpoints ──────────────────────────────────────────────

    /**
     * GET /:id/layout - Get computed layout for a dashboard
     * @route GET /:id/layout
     * @query {number} [viewportWidth] - Viewport width for responsive layout
     */
    r.get('/:id/layout', authenticate({ required: false }), (req, res) => {
      try {
        const viewportWidth = req.query.viewportWidth ? Number(req.query.viewportWidth) : undefined;
        const layout = this.service.getComputedLayout(req.params.id, viewportWidth);
        res.json({ data: layout });
      } catch (err) {
        const status = err.message.includes('not found') ? 404 : 500;
        res.status(status).json({ error: status === 404 ? 'NotFound' : 'InternalServerError', message: err.message });
      }
    });

    // ─── Export / Import ───────────────────────────────────────────────

    /**
     * GET /:id/export - Export dashboard configuration
     * @route GET /:id/export
     */
    r.get('/:id/export', authenticate({ required: true }), (req, res) => {
      try {
        const json = this.service.exportDashboard(req.params.id);
        res.setHeader('Content-Type', 'application/json');
        res.setHeader('Content-Disposition', `attachment; filename="dashboard-${req.params.id}.json"`);
        res.send(json);
      } catch (err) {
        const status = err.message.includes('not found') ? 404 : 500;
        res.status(status).json({ error: status === 404 ? 'NotFound' : 'InternalServerError', message: err.message });
      }
    });

    /**
     * POST /import - Import a dashboard configuration
     * @route POST /import
     * @body {string} config - JSON string of dashboard configuration
     */
    r.post('/import', authenticate({ required: true }), (req, res) => {
      try {
        const configStr = typeof req.body === 'string' ? req.body : JSON.stringify(req.body);
        const dashboard = this.service.importDashboard(configStr);
        res.status(201).json({ data: dashboard });
      } catch (err) {
        res.status(400).json({ error: 'BadRequest', message: err.message });
      }
    });
  }

  /**
   * Get the configured Express router
   * @returns {express.Router}
   */
  getRouter() {
    return this.router;
  }
}

export { authenticate, validateBody, requestLogger };
