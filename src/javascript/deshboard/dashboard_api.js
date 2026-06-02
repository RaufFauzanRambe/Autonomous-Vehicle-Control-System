/**
 * @fileoverview Dashboard API Client
 * @description Fetch/axios wrapper, request interceptors, response transformers,
 *   retry logic, timeout handling, API endpoint constants, and WebSocket message
 *   handlers for the Autonomous Vehicle Dashboard.
 * @module dashboard_api
 */

import { EventEmitter } from 'events';

// ─── API Endpoint Constants ──────────────────────────────────────────────────

/** @enum {string} */
export const API_ENDPOINTS = {
  // Dashboard CRUD
  DASHBOARDS: '/api/dashboards',
  DASHBOARD_BY_ID: (id) => `/api/dashboards/${id}`,
  DASHBOARD_WIDGETS: (id) => `/api/dashboards/${id}/widgets`,
  DASHBOARD_WIDGET_BY_ID: (dashId, widgetId) => `/api/dashboards/${dashId}/widgets/${widgetId}`,
  DASHBOARD_LAYOUT: (id) => `/api/dashboards/${id}/layout`,
  DASHBOARD_EXPORT: (id) => `/api/dashboards/${id}/export`,
  DASHBOARD_IMPORT: '/api/dashboards/import',

  // Health & System
  HEALTH: '/health',
  READY: '/ready',
  LIVE: '/live',
  API_INFO: '/api',

  // Telemetry (WebSocket)
  WS_TELEMETRY_STREAM: 'telemetry:stream',
  WS_TELEMETRY_UPDATE: 'telemetry:update',
  WS_DASHBOARD_SUBSCRIBE: 'dashboard:subscribe',
  WS_DASHBOARD_UNSUBSCRIBE: 'dashboard:unsubscribe',
  WS_ALERT_NEW: 'alert:new',
  WS_ALERT_ACK: 'alert:acknowledge',
  WS_STATE_CHANGED: 'state:changed',
};

// ─── Custom Error Classes ────────────────────────────────────────────────────

/**
 * API request error with status code and response data
 * @extends Error
 */
class APIError extends Error {
  /**
   * @param {string} message - Error message
   * @param {number} [status=0] - HTTP status code
   * @param {Object} [data=null] - Response body data
   * @param {string} [endpoint=''] - API endpoint
   */
  constructor(message, status = 0, data = null, endpoint = '') {
    super(message);
    this.name = 'APIError';
    this.status = status;
    this.data = data;
    this.endpoint = endpoint;
    this.timestamp = new Date().toISOString();
  }

  /** @returns {boolean} */ get isNetworkError() { return this.status === 0; }
  /** @returns {boolean} */ get isClientError() { return this.status >= 400 && this.status < 500; }
  /** @returns {boolean} */ get isServerError() { return this.status >= 500; }
  /** @returns {boolean} */ get isRetryable() { return this.isNetworkError || this.status >= 500 || this.status === 429; }
}

/**
 * Timeout error for requests that exceed their deadline
 * @extends Error
 */
class TimeoutError extends Error {
  /**
   * @param {number} timeout - Timeout in ms
   * @param {string} endpoint - API endpoint
   */
  constructor(timeout, endpoint) {
    super(`Request timed out after ${timeout}ms for ${endpoint}`);
    this.name = 'TimeoutError';
    this.timeout = timeout;
    this.endpoint = endpoint;
  }
}

// ─── Request Interceptor Chain ───────────────────────────────────────────────

/**
 * Manages request/response interceptors
 */
class InterceptorManager {
  constructor() {
    /** @private @type {Function[]} */ this._requestInterceptors = [];
    /** @private @type {Function[]} */ this._responseInterceptors = [];
  }

  /**
   * Add a request interceptor
   * @param {Function} onFulfilled - Called with request config
   * @param {Function} [onRejected] - Called on error
   * @returns {number} Interceptor ID
   */
  addRequestInterceptor(onFulfilled, onRejected) {
    this._requestInterceptors.push({ onFulfilled, onRejected });
    return this._requestInterceptors.length - 1;
  }

  /**
   * Add a response interceptor
   * @param {Function} onFulfilled - Called with response
   * @param {Function} [onRejected] - Called on error
   * @returns {number} Interceptor ID
   */
  addResponseInterceptor(onFulfilled, onRejected) {
    this._responseInterceptors.push({ onFulfilled, onRejected });
    return this._responseInterceptors.length - 1;
  }

  /**
   * Apply request interceptors sequentially
   * @param {Object} config - Request configuration
   * @returns {Promise<Object>} Transformed config
   */
  async applyRequestInterceptors(config) {
    let result = config;
    for (const interceptor of this._requestInterceptors) {
      try {
        result = await interceptor.onFulfilled(result);
      } catch (err) {
        if (interceptor.onRejected) {
          interceptor.onRejected(err);
        }
        throw err;
      }
    }
    return result;
  }

  /**
   * Apply response interceptors sequentially
   * @param {Object} response - Response object
   * @returns {Promise<Object>} Transformed response
   */
  async applyResponseInterceptors(response) {
    let result = response;
    for (const interceptor of this._responseInterceptors) {
      try {
        result = await interceptor.onFulfilled(result);
      } catch (err) {
        if (interceptor.onRejected) {
          interceptor.onRejected(err);
        }
        throw err;
      }
    }
    return result;
  }

  /**
   * Remove a request interceptor by ID
   * @param {number} id - Interceptor ID
   */
  removeRequestInterceptor(id) {
    if (this._requestInterceptors[id]) {
      this._requestInterceptors[id] = null;
    }
  }

  /**
   * Remove a response interceptor by ID
   * @param {number} id - Interceptor ID
   */
  removeResponseInterceptor(id) {
    if (this._responseInterceptors[id]) {
      this._responseInterceptors[id] = null;
    }
  }
}

// ─── API Client ──────────────────────────────────────────────────────────────

/**
 * HTTP API client with retry, timeout, and interceptor support
 * @extends EventEmitter
 */
export class DashboardAPIClient extends EventEmitter {
  /**
   * @param {Object} [options={}]
   * @param {string} [options.baseURL=''] - Base URL for API requests
   * @param {number} [options.timeout=10000] - Default request timeout in ms
   * @param {number} [options.maxRetries=3] - Maximum retry attempts
   * @param {number} [options.retryDelay=1000] - Base retry delay in ms
   * @param {number} [options.retryBackoff=2] - Exponential backoff multiplier
   * @param {Object} [options.defaultHeaders={}] - Default headers
   * @param {string} [options.authToken=''] - Bearer token
   */
  constructor(options = {}) {
    super();
    /** @private */ this._baseURL = options.baseURL || '';
    /** @private */ this._timeout = options.timeout ?? 10000;
    /** @private */ this._maxRetries = options.maxRetries ?? 3;
    /** @private */ this._retryDelay = options.retryDelay ?? 1000;
    /** @private */ this._retryBackoff = options.retryBackoff ?? 2;
    /** @private @type {Object<string, string>} */ this._defaultHeaders = {
      'Content-Type': 'application/json',
      ...options.defaultHeaders,
    };
    /** @private */ this._authToken = options.authToken || '';
    /** @private */ this._interceptors = new InterceptorManager();
    /** @private */ this._requestCount = 0;
    /** @private */ this._errorCount = 0;
    /** @private @type {AbortController|null} */ this._activeRequests = new Map();

    this._setupDefaultInterceptors();
  }

  /**
   * Set up default interceptors for auth and logging
   * @private
   */
  _setupDefaultInterceptors() {
    // Auth interceptor - attach token to requests
    this._interceptors.addRequestInterceptor((config) => {
      if (this._authToken) {
        config.headers = {
          ...config.headers,
          Authorization: `Bearer ${this._authToken}`,
        };
      }
      return config;
    });

    // Response data extractor
    this._interceptors.addResponseInterceptor((response) => {
      if (response.data && typeof response.data === 'object' && response.data.data) {
        response.extractedData = response.data.data;
        response.pagination = response.data.pagination || null;
      }
      return response;
    });
  }

  /**
   * Set the authentication token
   * @param {string} token - Bearer token
   */
  setAuthToken(token) {
    this._authToken = token;
    this.emit('auth:changed', { hasToken: !!token });
  }

  /**
   * Make an HTTP request with retry and timeout
   * @param {string} method - HTTP method
   * @param {string} endpoint - API endpoint
   * @param {Object} [options={}]
   * @param {Object} [options.body] - Request body
   * @param {Object} [options.params] - Query parameters
   * @param {Object} [options.headers] - Additional headers
   * @param {number} [options.timeout] - Request-specific timeout
   * @param {number} [options.retries] - Request-specific retry count
   * @returns {Promise<Object>} Response object
   */
  async request(method, endpoint, options = {}) {
    const requestId = `req_${++this._requestCount}`;
    const timeout = options.timeout ?? this._timeout;
    const maxRetries = options.retries ?? this._maxRetries;

    let config = {
      method,
      url: `${this._baseURL}${endpoint}`,
      headers: { ...this._defaultHeaders, ...options.headers },
      body: options.body ? JSON.stringify(options.body) : undefined,
      params: options.params || {},
      requestId,
      timeout,
    };

    // Apply request interceptors
    config = await this._interceptors.applyRequestInterceptors(config);

    // Build URL with query params
    let url = config.url;
    const paramEntries = Object.entries(config.params).filter(([, v]) => v !== undefined && v !== null);
    if (paramEntries.length > 0) {
      const qs = paramEntries.map(([k, v]) => `${encodeURIComponent(k)}=${encodeURIComponent(v)}`).join('&');
      url += `?${qs}`;
    }

    // Retry loop
    let lastError = null;
    for (let attempt = 0; attempt <= maxRetries; attempt++) {
      const abortController = new AbortController();
      this._activeRequests.set(requestId, abortController);

      // Set up timeout
      const timeoutId = setTimeout(() => abortController.abort(), timeout);

      try {
        const fetchOptions = {
          method: config.method,
          headers: config.headers,
          body: config.body,
          signal: abortController.signal,
        };

        this.emit('request:start', { requestId, method, endpoint, attempt });

        const response = await fetch(url, fetchOptions);
        clearTimeout(timeoutId);
        this._activeRequests.delete(requestId);

        let data;
        const contentType = response.headers.get('content-type') || '';
        if (contentType.includes('application/json')) {
          data = await response.json();
        } else {
          data = await response.text();
        }

        const result = {
          status: response.status,
          statusText: response.statusText,
          headers: Object.fromEntries(response.headers.entries()),
          data,
          requestId,
          endpoint,
        };

        if (!response.ok) {
          const apiError = new APIError(
            data?.error || data?.message || `HTTP ${response.status}`,
            response.status,
            data,
            endpoint,
          );

          // Retry on retryable errors
          if (apiError.isRetryable && attempt < maxRetries) {
            const delay = this._retryDelay * Math.pow(this._retryBackoff, attempt);
            this.emit('request:retry', { requestId, attempt, delay, error: apiError.message });
            await this._sleep(delay);
            continue;
          }

          this._errorCount++;
          this.emit('request:error', { requestId, error: apiError });
          throw apiError;
        }

        // Apply response interceptors
        const transformedResult = await this._interceptors.applyResponseInterceptors(result);

        this.emit('request:success', { requestId, status: response.status });
        return transformedResult;

      } catch (err) {
        clearTimeout(timeoutId);
        this._activeRequests.delete(requestId);

        if (err.name === 'AbortError') {
          const timeoutErr = new TimeoutError(timeout, endpoint);
          if (attempt < maxRetries) {
            const delay = this._retryDelay * Math.pow(this._retryBackoff, attempt);
            this.emit('request:retry', { requestId, attempt, delay, error: 'timeout' });
            await this._sleep(delay);
            continue;
          }
          this._errorCount++;
          this.emit('request:timeout', { requestId, timeout });
          throw timeoutErr;
        }

        if (err instanceof APIError) throw err;

        // Network error
        const netError = new APIError(err.message, 0, null, endpoint);
        if (attempt < maxRetries && netError.isRetryable) {
          const delay = this._retryDelay * Math.pow(this._retryBackoff, attempt);
          this.emit('request:retry', { requestId, attempt, delay, error: err.message });
          await this._sleep(delay);
          continue;
        }

        this._errorCount++;
        throw netError;
      }
    }

    throw lastError || new APIError('Max retries exceeded', 0, null, endpoint);
  }

  // ─── Convenience Methods ────────────────────────────────────────────────

  /**
   * GET request
   * @param {string} endpoint - API endpoint
   * @param {Object} [params] - Query parameters
   * @param {Object} [options] - Additional options
   * @returns {Promise<Object>}
   */
  async get(endpoint, params, options) {
    return this.request('GET', endpoint, { ...options, params });
  }

  /**
   * POST request
   * @param {string} endpoint - API endpoint
   * @param {Object} body - Request body
   * @param {Object} [options] - Additional options
   * @returns {Promise<Object>}
   */
  async post(endpoint, body, options) {
    return this.request('POST', endpoint, { ...options, body });
  }

  /**
   * PUT request
   * @param {string} endpoint - API endpoint
   * @param {Object} body - Request body
   * @param {Object} [options] - Additional options
   * @returns {Promise<Object>}
   */
  async put(endpoint, body, options) {
    return this.request('PUT', endpoint, { ...options, body });
  }

  /**
   * DELETE request
   * @param {string} endpoint - API endpoint
   * @param {Object} [options] - Additional options
   * @returns {Promise<Object>}
   */
  async delete(endpoint, options) {
    return this.request('DELETE', endpoint, options);
  }

  // ─── Dashboard-Specific Methods ─────────────────────────────────────────

  /**
   * Fetch all dashboards
   * @param {Object} [filter={}] - Filter options
   * @returns {Promise<Object[]>}
   */
  async fetchDashboards(filter = {}) {
    const response = await this.get(API_ENDPOINTS.DASHBOARDS, filter);
    return response.extractedData || response.data;
  }

  /**
   * Fetch a single dashboard
   * @param {string} id - Dashboard ID
   * @returns {Promise<Object>}
   */
  async fetchDashboard(id) {
    const response = await this.get(API_ENDPOINTS.DASHBOARD_BY_ID(id));
    return response.extractedData || response.data;
  }

  /**
   * Create a new dashboard
   * @param {Object} config - Dashboard configuration
   * @returns {Promise<Object>}
   */
  async createDashboard(config) {
    const response = await this.post(API_ENDPOINTS.DASHBOARDS, config);
    return response.extractedData || response.data;
  }

  /**
   * Update a dashboard
   * @param {string} id - Dashboard ID
   * @param {Object} updates - Fields to update
   * @returns {Promise<Object>}
   */
  async updateDashboard(id, updates) {
    const response = await this.put(API_ENDPOINTS.DASHBOARD_BY_ID(id), updates);
    return response.extractedData || response.data;
  }

  /**
   * Delete a dashboard
   * @param {string} id - Dashboard ID
   * @returns {Promise<Object>}
   */
  async deleteDashboard(id) {
    const response = await this.delete(API_ENDPOINTS.DASHBOARD_BY_ID(id));
    return response.extractedData || response.data;
  }

  /**
   * Fetch dashboard layout
   * @param {string} id - Dashboard ID
   * @param {number} [viewportWidth] - Viewport width
   * @returns {Promise<Object[]>}
   */
  async fetchLayout(id, viewportWidth) {
    const params = viewportWidth ? { viewportWidth } : {};
    const response = await this.get(API_ENDPOINTS.DASHBOARD_LAYOUT(id), params);
    return response.extractedData || response.data;
  }

  /**
   * Export a dashboard configuration
   * @param {string} id - Dashboard ID
   * @returns {Promise<string>}
   */
  async exportDashboard(id) {
    const response = await this.get(API_ENDPOINTS.DASHBOARD_EXPORT(id));
    return typeof response.data === 'string' ? response.data : JSON.stringify(response.data);
  }

  /**
   * Import a dashboard configuration
   * @param {string} configJson - JSON configuration
   * @returns {Promise<Object>}
   */
  async importDashboard(configJson) {
    const response = await this.post(API_ENDPOINTS.DASHBOARD_IMPORT, { config: configJson });
    return response.extractedData || response.data;
  }

  // ─── Health Check ───────────────────────────────────────────────────────

  /**
   * Check server health
   * @returns {Promise<Object>}
   */
  async checkHealth() {
    return this.get(API_ENDPOINTS.HEALTH);
  }

  // ─── WebSocket Message Handlers ─────────────────────────────────────────

  /**
   * Create a WebSocket message handler map
   * @param {Object} callbacks - Event callbacks
   * @param {Function} [callbacks.onTelemetryUpdate] - Telemetry data callback
   * @param {Function} [callbacks.onStateChanged] - State change callback
   * @param {Function} [callbacks.onAlertNew] - New alert callback
   * @param {Function} [callbacks.onAlertUpdated] - Alert update callback
   * @returns {Object<string, Function>} Handler map for socket.io events
   */
  createWSHandlers(callbacks = {}) {
    return {
      [API_ENDPOINTS.WS_TELEMETRY_UPDATE]: (data) => {
        this.emit('ws:telemetry', data);
        callbacks.onTelemetryUpdate?.(data);
      },
      [API_ENDPOINTS.WS_STATE_CHANGED]: (data) => {
        this.emit('ws:stateChanged', data);
        callbacks.onStateChanged?.(data);
      },
      [API_ENDPOINTS.WS_ALERT_NEW]: (data) => {
        this.emit('ws:alert', data);
        callbacks.onAlertNew?.(data);
      },
      'alert:updated': (data) => {
        this.emit('ws:alertUpdated', data);
        callbacks.onAlertUpdated?.(data);
      },
    };
  }

  // ─── Utilities ──────────────────────────────────────────────────────────

  /**
   * Cancel all active requests
   */
  cancelAll() {
    for (const [id, controller] of this._activeRequests.entries()) {
      controller.abort();
      this._activeRequests.delete(id);
    }
    this.emit('requests:cancelled');
  }

  /**
   * Get client statistics
   * @returns {Object}
   */
  getStats() {
    return {
      totalRequests: this._requestCount,
      totalErrors: this._errorCount,
      activeRequests: this._activeRequests.size,
      errorRate: this._requestCount > 0 ? (this._errorCount / this._requestCount * 100).toFixed(2) + '%' : '0%',
      hasAuthToken: !!this._authToken,
    };
  }

  /**
   * Sleep utility for retry delays
   * @private
   * @param {number} ms - Milliseconds to sleep
   * @returns {Promise<void>}
   */
  _sleep(ms) {
    return new Promise((resolve) => setTimeout(resolve, ms));
  }

  /**
   * Get the interceptor manager
   * @returns {InterceptorManager}
   */
  get interceptors() {
    return this._interceptors;
  }
}

export { APIError, TimeoutError, InterceptorManager };
