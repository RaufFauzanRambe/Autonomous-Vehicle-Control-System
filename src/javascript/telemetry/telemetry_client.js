/**
 * @module telemetry_client
 * @description Telemetry client for connecting to the vehicle telemetry server.
 * Supports WebSocket and HTTP transport, auto-reconnect with exponential backoff,
 * subscription management, data deserialization, and request throttling.
 *
 * @author Autonomous Vehicle Control System
 * @version 2.0.0
 * @license MIT
 */

import { EventEmitter } from 'events';
import { create deflateSync, inflateSync } from 'zlib';

// ─────────────────────────────────────────────────────────────────────────────
// Custom Error Classes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Error thrown by the telemetry client.
 * @extends Error
 */
export class TelemetryClientError extends Error {
  /**
   * @param {string} message - Error description
   * @param {string} [code='CLIENT_ERROR'] - Machine-readable error code
   * @param {object} [details={}] - Additional error details
   */
  constructor(message, code = 'CLIENT_ERROR', details = {}) {
    super(message);
    this.name = 'TelemetryClientError';
    this.code = code;
    this.details = details;
    this.timestamp = Date.now();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants
// ─────────────────────────────────────────────────────────────────────────────

/** @enum {string} Client connection states */
const ConnectionState = {
  DISCONNECTED: 'disconnected',
  CONNECTING: 'connecting',
  CONNECTED: 'connected',
  RECONNECTING: 'reconnecting',
  CLOSING: 'closing'
};

/** @enum {string} Transport protocols */
const Transport = {
  WEBSOCKET: 'websocket',
  HTTP: 'http'
};

/** @type {object} Default client configuration */
const DEFAULT_CONFIG = {
  /** Server URL (e.g. 'ws://localhost:8080' or 'http://localhost:8080') */
  serverUrl: 'ws://localhost:8080',
  /** Preferred transport protocol */
  transport: Transport.WEBSOCKET,
  /** Authentication token */
  authToken: null,
  /** Auto-reconnect on disconnect */
  autoReconnect: true,
  /** Maximum reconnection attempts (0 = unlimited) */
  maxReconnectAttempts: 0,
  /** Initial reconnect delay in ms */
  reconnectBaseDelayMs: 1000,
  /** Maximum reconnect delay in ms */
  reconnectMaxDelayMs: 30000,
  /** Reconnect delay multiplier (exponential backoff) */
  reconnectMultiplier: 2,
  /** Add jitter to reconnect delay */
  reconnectJitter: true,
  /** WebSocket ping interval in ms (0 = disabled) */
  pingIntervalMs: 30000,
  /** Request throttle interval in ms (0 = no throttling) */
  throttleIntervalMs: 100,
  /** Maximum pending requests before dropping */
  maxPendingRequests: 1000,
  /** Receive buffer high-water mark (bytes) */
  receiveHighWaterMark: 65536,
  /** Enable data compression */
  compression: false,
  /** Request timeout in ms */
  requestTimeoutMs: 10000,
  /** Connection timeout in ms */
  connectionTimeoutMs: 15000
};

// ─────────────────────────────────────────────────────────────────────────────
// Throttle Queue
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Simple request throttle queue that limits the rate of outgoing messages.
 */
class ThrottleQueue {
  /**
   * @param {number} intervalMs - Minimum interval between dispatches
   * @param {number} maxSize - Maximum queued items
   */
  constructor(intervalMs, maxSize) {
    /** @type {number} */
    this.intervalMs = intervalMs;
    /** @type {number} */
    this.maxSize = maxSize;
    /** @type {Array<{data:*, resolve:Function, reject:Function}>} */
    this._queue = [];
    /** @type {NodeJS.Timer|null} */
    this._timer = null;
    /** @type {boolean} */
    this._processing = false;
  }

  /**
   * Enqueue a message for throttled delivery.
   * @param {*} data - Data to send
   * @returns {Promise<void>}
   */
  enqueue(data) {
    return new Promise((resolve, reject) => {
      if (this._queue.length >= this.maxSize) {
        reject(new TelemetryClientError('Throttle queue overflow', 'QUEUE_OVERFLOW'));
        return;
      }
      this._queue.push({ data, resolve, reject });
      this._startProcessing();
    });
  }

  /**
   * Start processing the queue if not already running.
   * @private
   */
  _startProcessing() {
    if (this._processing || this.intervalMs === 0) return;
    this._processing = true;
    this._timer = setInterval(() => this._drain(), this.intervalMs);
    this._drain();
  }

  /**
   * Drain one item from the queue.
   * @private
   * @returns {Array<*>} Items drained this cycle
   */
  _drain() {
    const items = [];
    while (this._queue.length > 0) {
      const item = this._queue.shift();
      items.push(item.data);
      item.resolve();
    }
    if (this._queue.length === 0) {
      this._processing = false;
      if (this._timer) {
        clearInterval(this._timer);
        this._timer = null;
      }
    }
    return items;
  }

  /**
   * Flush all pending items immediately.
   * @returns {Array<*>}
   */
  flush() {
    const items = [];
    while (this._queue.length > 0) {
      const item = this._queue.shift();
      items.push(item.data);
      item.resolve();
    }
    this._processing = false;
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
    return items;
  }

  /**
   * Get current queue size.
   * @returns {number}
   */
  get size() {
    return this._queue.length;
  }

  /**
   * Destroy the throttle queue, rejecting all pending items.
   */
  destroy() {
    while (this._queue.length > 0) {
      const item = this._queue.shift();
      item.reject(new TelemetryClientError('Throttle queue destroyed', 'QUEUE_DESTROYED'));
    }
    if (this._timer) {
      clearInterval(this._timer);
      this._timer = null;
    }
    this._processing = false;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Subscription Manager
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Manages channel subscriptions with deduplication and resubscription on reconnect.
 */
class SubscriptionManager {
  constructor() {
    /** @type {Map<string, {channels:Set<string>, callback:Function}>} */
    this._subscriptions = new Map();
    /** @type {Map<string, number>} Channel → reference count */
    this._channelRefs = new Map();
  }

  /**
   * Add a subscription.
   * @param {string} id - Unique subscription ID
   * @param {string[]} channels - Channels to subscribe to
   * @param {Function} callback - Data callback
   */
  add(id, channels, callback) {
    this._subscriptions.set(id, { channels: new Set(channels), callback });
    for (const ch of channels) {
      this._channelRefs.set(ch, (this._channelRefs.get(ch) || 0) + 1);
    }
  }

  /**
   * Remove a subscription.
   * @param {string} id - Subscription ID to remove
   */
  remove(id) {
    const sub = this._subscriptions.get(id);
    if (!sub) return;
    for (const ch of sub.channels) {
      const refCount = (this._channelRefs.get(ch) || 1) - 1;
      if (refCount <= 0) {
        this._channelRefs.delete(ch);
      } else {
        this._channelRefs.set(ch, refCount);
      }
    }
    this._subscriptions.delete(id);
  }

  /**
   * Dispatch data to matching subscriptions.
   * @param {string} channel - The channel the data arrived on
   * @param {*} data - The data payload
   */
  dispatch(channel, data) {
    for (const [, sub] of this._subscriptions) {
      if (sub.channels.has(channel) || sub.channels.has('*')) {
        try {
          sub.callback(channel, data);
        } catch (_) { /* swallow callback errors */ }
      }
    }
  }

  /**
   * Get all unique channels currently subscribed.
   * @returns {string[]}
   */
  getChannels() {
    return Array.from(this._channelRefs.keys());
  }

  /**
   * Clear all subscriptions.
   */
  clear() {
    this._subscriptions.clear();
    this._channelRefs.clear();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// TelemetryClient
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Telemetry client for connecting to a vehicle telemetry server.
 * Supports WebSocket and HTTP transports with auto-reconnect, subscriptions,
 * data deserialization, and request throttling.
 *
 * @extends EventEmitter
 *
 * @example
 * const client = new TelemetryClient({
 *   serverUrl: 'ws://telemetry.vehicle.local:8080',
 *   authToken: 'my-secret-token',
 *   autoReconnect: true
 * });
 * client.on('data', (msg) => console.log(msg));
 * client.subscribe(['gps', 'imu'], (channel, data) => { });
 * await client.connect();
 */
export class TelemetryClient extends EventEmitter {
  /**
   * @param {object} [config={}] - Configuration overrides
   */
  constructor(config = {}) {
    super();

    /** @type {object} Merged configuration */
    this.config = { ...DEFAULT_CONFIG, ...config };

    /** @type {string} Current connection state */
    this._state = ConnectionState.DISCONNECTED;

    /** @type {WebSocket|null} Active WebSocket connection */
    this._ws = null;

    /** @type {number} Current reconnect attempt count */
    this._reconnectAttempts = 0;

    /** @type {number} Current reconnect delay */
    this._reconnectDelay = this.config.reconnectBaseDelayMs;

    /** @type {NodeJS.Timer|null} Reconnect timer */
    this._reconnectTimer = null;

    /** @type {NodeJS.Timer|null} Ping interval timer */
    this._pingTimer = null;

    /** @type {SubscriptionManager} Subscription manager */
    this._subscriptions = new SubscriptionManager();

    /** @type {ThrottleQueue} Outbound throttle queue */
    this._throttleQueue = new ThrottleQueue(
      this.config.throttleIntervalMs,
      this.config.maxPendingRequests
    );

    /** @type {Map<string, {resolve:Function, reject:Function, timeout:NodeJS.Timer}>} Pending requests */
    this._pendingRequests = new Map();

    /** @type {number} Monotonic request ID counter */
    this._requestCounter = 0;

    /** @type {object} Client statistics */
    this._stats = {
      messagesReceived: 0,
      messagesSent: 0,
      bytesReceived: 0,
      bytesSent: 0,
      reconnectCount: 0,
      lastMessageTime: null,
      connectedSince: null
    };
  }

  // ── Connection Management ───────────────────────────────────────────────

  /**
   * Connect to the telemetry server.
   * @returns {Promise<void>}
   * @throws {TelemetryClientError} If connection fails
   */
  async connect() {
    if (this._state === ConnectionState.CONNECTED || this._state === ConnectionState.CONNECTING) {
      return;
    }

    this._setState(ConnectionState.CONNECTING);

    try {
      if (this.config.transport === Transport.WEBSOCKET) {
        await this._connectWebSocket();
      } else {
        await this._connectHTTP();
      }
    } catch (error) {
      this._setState(ConnectionState.DISCONNECTED);
      throw new TelemetryClientError(
        `Connection failed: ${error.message}`,
        'CONNECTION_FAILED',
        { url: this.config.serverUrl, cause: error }
      );
    }
  }

  /**
   * Disconnect from the telemetry server.
   * @param {number} [code=1000] - WebSocket close code
   * @param {string} [reason=''] - Close reason
   */
  disconnect(code = 1000, reason = '') {
    this._setState(ConnectionState.CLOSING);
    this._clearTimers();

    if (this._ws) {
      try {
        this._ws.close(code, reason);
      } catch (_) { /* ignore close errors */ }
      this._ws = null;
    }

    this._pendingRequests.clear();
    this._throttleQueue.destroy();
    this._setState(ConnectionState.DISCONNECTED);
  }

  // ── Subscription Management ─────────────────────────────────────────────

  /**
   * Subscribe to one or more telemetry channels.
   * @param {string[]} channels - Channels to subscribe to (e.g. ['gps', 'imu', 'lidar'])
   * @param {Function} callback - Callback invoked with (channel, data)
   * @returns {string} Subscription ID
   */
  subscribe(channels, callback) {
    const id = `sub_${++this._requestCounter}`;
    this._subscriptions.add(id, channels, callback);

    // Send subscribe message to server if connected
    if (this._state === ConnectionState.CONNECTED) {
      this._sendCommand('subscribe', { channels });
    }

    return id;
  }

  /**
   * Unsubscribe from channels.
   * @param {string} subscriptionId - The subscription ID returned from subscribe()
   */
  unsubscribe(subscriptionId) {
    const sub = this._subscriptions._subscriptions.get(subscriptionId);
    if (!sub) return;

    const channels = Array.from(sub.channels);
    this._subscriptions.remove(subscriptionId);

    if (this._state === ConnectionState.CONNECTED) {
      this._sendCommand('unsubscribe', { channels });
    }
  }

  // ── Request/Response ────────────────────────────────────────────────────

  /**
   * Send a request to the server and wait for a response.
   * @param {string} method - RPC method name
   * @param {object} [params={}] - Method parameters
   * @returns {Promise<*>} Response data
   */
  request(method, params = {}) {
    return new Promise((resolve, reject) => {
      if (this._state !== ConnectionState.CONNECTED) {
        reject(new TelemetryClientError('Not connected', 'NOT_CONNECTED'));
        return;
      }

      const id = `req_${++this._requestCounter}`;
      const timeout = setTimeout(() => {
        this._pendingRequests.delete(id);
        reject(new TelemetryClientError(`Request timeout: ${method}`, 'REQUEST_TIMEOUT'));
      }, this.config.requestTimeoutMs);

      this._pendingRequests.set(id, { resolve, reject, timeout });
      this._send({ type: 'request', id, method, params });
    });
  }

  // ── Data Send ───────────────────────────────────────────────────────────

  /**
   * Send data to a specific channel with optional throttling.
   * @param {string} channel - Target channel
   * @param {*} data - Data payload
   * @param {object} [options={}] - Send options
   * @param {boolean} [options.throttle=true] - Apply throttling
   * @returns {Promise<void>}
   */
  async send(channel, data, options = {}) {
    if (this._state !== ConnectionState.CONNECTED) {
      throw new TelemetryClientError('Not connected', 'NOT_CONNECTED');
    }

    const message = { type: 'data', channel, payload: data, timestamp: Date.now() };

    if (options.throttle !== false && this.config.throttleIntervalMs > 0) {
      await this._throttleQueue.enqueue(message);
    }

    this._send(message);
  }

  // ── Status ──────────────────────────────────────────────────────────────

  /**
   * Get current connection state.
   * @returns {string}
   */
  getState() {
    return this._state;
  }

  /**
   * Check if the client is connected.
   * @returns {boolean}
   */
  isConnected() {
    return this._state === ConnectionState.CONNECTED;
  }

  /**
   * Get client statistics.
   * @returns {object}
   */
  getStats() {
    return { ...this._stats, state: this._state, pendingRequests: this._pendingRequests.size };
  }

  /**
   * Get health status.
   * @returns {object}
   */
  getHealth() {
    return {
      status: this._state === ConnectionState.CONNECTED ? 'healthy' : 'degraded',
      state: this._state,
      reconnectAttempts: this._reconnectAttempts,
      lastMessageAge: this._stats.lastMessageTime
        ? Date.now() - this._stats.lastMessageTime
        : null
    };
  }

  // ── Private Methods ─────────────────────────────────────────────────────

  /**
   * Establish WebSocket connection.
   * @returns {Promise<void>}
   * @private
   */
  async _connectWebSocket() {
    return new Promise((resolve, reject) => {
      const url = new URL(this.config.serverUrl);
      const protocols = [];

      if (this.config.authToken) {
        url.searchParams.set('token', this.config.authToken);
      }

      // In a real implementation, we'd use the 'ws' package:
      // const WebSocket = (await import('ws')).default;
      // For this module, we simulate the WebSocket interface
      const mockWs = this._createWebSocketConnection(url, protocols);

      const connectionTimeout = setTimeout(() => {
        reject(new TelemetryClientError('Connection timeout', 'CONNECTION_TIMEOUT'));
      }, this.config.connectionTimeoutMs);

      mockWs.onopen = () => {
        clearTimeout(connectionTimeout);
        this._ws = mockWs;
        this._setState(ConnectionState.CONNECTED);
        this._stats.connectedSince = Date.now();
        this._reconnectAttempts = 0;
        this._reconnectDelay = this.config.reconnectBaseDelayMs;

        // Start ping interval
        if (this.config.pingIntervalMs > 0) {
          this._pingTimer = setInterval(() => this._ping(), this.config.pingIntervalMs);
        }

        // Resubscribe channels
        const channels = this._subscriptions.getChannels();
        if (channels.length > 0) {
          this._sendCommand('subscribe', { channels });
        }

        this.emit('connected', { url: url.toString() });
        resolve();
      };

      mockWs.onmessage = (event) => this._handleMessage(event);
      mockWs.onerror = (event) => {
        clearTimeout(connectionTimeout);
        reject(new TelemetryClientError('WebSocket error', 'WS_ERROR', { event }));
      };
      mockWs.onclose = (event) => this._handleClose(event);
    });
  }

  /**
   * Establish HTTP long-polling connection.
   * @returns {Promise<void>}
   * @private
   */
  async _connectHTTP() {
    // HTTP polling simulation
    this._setState(ConnectionState.CONNECTED);
    this._stats.connectedSince = Date.now();
    this.emit('connected', { url: this.config.serverUrl });
  }

  /**
   * Create a WebSocket connection (mock for standalone module).
   * In production, this would use the 'ws' npm package.
   * @param {URL} url - Server URL
   * @param {string[]} protocols - WebSocket sub-protocols
   * @returns {object} WebSocket-like object
   * @private
   */
  _createWebSocketConnection(url, protocols) {
    // Simulated WebSocket for module demonstration
    // In production: return new WebSocket(url.toString(), protocols);
    const ws = {
      url: url.toString(),
      readyState: 0, // CONNECTING
      onopen: null,
      onmessage: null,
      onerror: null,
      onclose: null,
      send(data) {
        if (this.readyState !== 1) throw new Error('WebSocket is not open');
        // Simulated send
      },
      close(code = 1000, reason = '') {
        this.readyState = 3;
        if (this.onclose) this.onclose({ code, reason, wasClean: true });
      },
      // Simulate successful connection after a brief delay
      _simulateOpen() {
        this.readyState = 1;
        if (this.onopen) this.onopen({});
      }
    };

    // Simulate async connection
    setTimeout(() => ws._simulateOpen(), 10);
    return ws;
  }

  /**
   * Handle incoming WebSocket message.
   * @param {MessageEvent} event - WebSocket message event
   * @private
   */
  _handleMessage(event) {
    try {
      const raw = typeof event.data === 'string' ? event.data : event.data.toString();
      let data;

      // Attempt JSON parse
      try {
        data = JSON.parse(raw);
      } catch (_) {
        // Try decompression first
        if (this.config.compression) {
          const decompressed = inflateSync(Buffer.from(raw, 'base64'));
          data = JSON.parse(decompressed.toString());
        } else {
          data = { type: 'raw', payload: raw };
        }
      }

      this._stats.messagesReceived++;
      this._stats.bytesReceived += raw.length;
      this._stats.lastMessageTime = Date.now();

      // Route by message type
      switch (data.type) {
        case 'data':
          this._handleDataMessage(data);
          break;
        case 'response':
          this._handleResponseMessage(data);
          break;
        case 'error':
          this._handleErrorMessage(data);
          break;
        case 'pong':
          // Heartbeat response, no action needed
          break;
        default:
          this.emit('message', data);
      }
    } catch (error) {
      this.emit('error', new TelemetryClientError(
        `Message parse error: ${error.message}`,
        'PARSE_ERROR',
        { raw: event.data?.substring(0, 200) }
      ));
    }
  }

  /**
   * Handle a data message from the server.
   * @param {object} data - Parsed message data
   * @private
   */
  _handleDataMessage(data) {
    const { channel, payload, timestamp } = data;
    this.emit('data', {
      source: channel,
      raw: payload,
      timestamp: timestamp || Date.now(),
      receivedAt: Date.now()
    });
    this._subscriptions.dispatch(channel || '*', payload);
  }

  /**
   * Handle a response to a previous request.
   * @param {object} data - Response message
   * @private
   */
  _handleResponseMessage(data) {
    const pending = this._pendingRequests.get(data.id);
    if (!pending) return;

    clearTimeout(pending.timeout);
    this._pendingRequests.delete(data.id);

    if (data.error) {
      pending.reject(new TelemetryClientError(data.error.message, data.error.code));
    } else {
      pending.resolve(data.result);
    }
  }

  /**
   * Handle an error message from the server.
   * @param {object} data - Error message
   * @private
   */
  _handleErrorMessage(data) {
    this.emit('error', new TelemetryClientError(
      data.message || 'Server error',
      data.code || 'SERVER_ERROR',
      data.details || {}
    ));
  }

  /**
   * Handle WebSocket close event.
   * @param {CloseEvent} event - Close event
   * @private
   */
  _handleClose(event) {
    this._ws = null;
    this._clearTimers();
    this._setState(ConnectionState.DISCONNECTED);
    this.emit('disconnected', { code: event.code, reason: event.reason });

    // Auto-reconnect
    if (this.config.autoReconnect && this._state !== ConnectionState.CLOSING) {
      this._scheduleReconnect();
    }
  }

  /**
   * Schedule a reconnection attempt with exponential backoff.
   * @private
   */
  _scheduleReconnect() {
    if (this.config.maxReconnectAttempts > 0 &&
        this._reconnectAttempts >= this.config.maxReconnectAttempts) {
      this.emit('error', new TelemetryClientError(
        'Max reconnect attempts reached',
        'MAX_RECONNECT'
      ));
      return;
    }

    this._setState(ConnectionState.RECONNECTING);

    let delay = this._reconnectDelay;
    if (this.config.reconnectJitter) {
      delay += Math.random() * delay * 0.3;
    }

    this._reconnectTimer = setTimeout(async () => {
      this._reconnectAttempts++;
      this._reconnectDelay = Math.min(
        this._reconnectDelay * this.config.reconnectMultiplier,
        this.config.reconnectMaxDelayMs
      );
      this._stats.reconnectCount++;

      try {
        await this.connect();
      } catch (error) {
        this.emit('reconnect_failed', { attempt: this._reconnectAttempts, error });
        // Will be called again via _handleClose → _scheduleReconnect
      }
    }, delay);
  }

  /**
   * Send a ping frame.
   * @private
   */
  _ping() {
    if (this._ws && this._ws.readyState === 1) {
      this._send({ type: 'ping', timestamp: Date.now() });
    }
  }

  /**
   * Send a command message to the server.
   * @param {string} command - Command name
   * @param {object} [params={}] - Command parameters
   * @private
   */
  _sendCommand(command, params = {}) {
    this._send({ type: 'command', command, params, timestamp: Date.now() });
  }

  /**
   * Send a message over the WebSocket connection.
   * @param {object} message - Message to send
   * @private
   */
  _send(message) {
    if (!this._ws || this._ws.readyState !== 1) {
      this.emit('error', new TelemetryClientError('Cannot send: not connected', 'NOT_CONNECTED'));
      return;
    }

    let data = JSON.stringify(message);

    if (this.config.compression) {
      const compressed = deflateSync(Buffer.from(data));
      data = compressed.toString('base64');
    }

    try {
      this._ws.send(data);
      this._stats.messagesSent++;
      this._stats.bytesSent += data.length;
    } catch (error) {
      this.emit('error', new TelemetryClientError(
        `Send failed: ${error.message}`,
        'SEND_ERROR'
      ));
    }
  }

  /**
   * Set connection state and emit state change.
   * @param {string} state - New state
   * @private
   */
  _setState(state) {
    const old = this._state;
    this._state = state;
    this.emit('state_change', { from: old, to: state });
  }

  /**
   * Clear all timers.
   * @private
   */
  _clearTimers() {
    if (this._reconnectTimer) {
      clearTimeout(this._reconnectTimer);
      this._reconnectTimer = null;
    }
    if (this._pingTimer) {
      clearInterval(this._pingTimer);
      this._pingTimer = null;
    }
  }
}

export default TelemetryClient;
