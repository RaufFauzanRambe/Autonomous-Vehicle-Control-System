/**
 * @module telemetry_server
 * @description Telemetry server for the Autonomous Vehicle Control System.
 * Provides Express + WebSocket server with broadcast channels, subscription-based
 * streaming, data buffering, rate limiting, client management, authentication,
 * and compression support.
 *
 * @author Autonomous Vehicle Control System
 * @version 2.0.0
 * @license MIT
 */

import { EventEmitter } from 'events';
import { deflateSync, inflateSync } from 'zlib';

// ─────────────────────────────────────────────────────────────────────────────
// Custom Error Classes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Error thrown by the telemetry server.
 * @extends Error
 */
export class TelemetryServerError extends Error {
  /**
   * @param {string} message - Error description
   * @param {string} [code='SERVER_ERROR'] - Machine-readable error code
   */
  constructor(message, code = 'SERVER_ERROR') {
    super(message);
    this.name = 'TelemetryServerError';
    this.code = code;
    this.timestamp = Date.now();
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Rate Limiter
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Token bucket rate limiter for per-client rate limiting.
 */
class RateLimiter {
  /**
   * @param {object} config - Rate limiter configuration
   * @param {number} [config.maxTokens=100] - Maximum tokens in bucket
   * @param {number} [config.refillRate=10] - Tokens added per second
   */
  constructor(config = {}) {
    /** @type {number} */
    this.maxTokens = config.maxTokens || 100;
    /** @type {number} */
    this.refillRate = config.refillRate || 10;
    /** @type {Map<string, {tokens:number, lastRefill:number}>} */
    this._clients = new Map();
  }

  /**
   * Check if a client has available rate limit tokens.
   * @param {string} clientId - Client identifier
   * @returns {boolean} True if request is allowed
   */
  allow(clientId) {
    let bucket = this._clients.get(clientId);
    if (!bucket) {
      bucket = { tokens: this.maxTokens, lastRefill: Date.now() };
      this._clients.set(clientId, bucket);
    }

    // Refill tokens based on elapsed time
    const now = Date.now();
    const elapsed = (now - bucket.lastRefill) / 1000;
    bucket.tokens = Math.min(this.maxTokens, bucket.tokens + elapsed * this.refillRate);
    bucket.lastRefill = now;

    if (bucket.tokens >= 1) {
      bucket.tokens -= 1;
      return true;
    }
    return false;
  }

  /**
   * Get remaining tokens for a client.
   * @param {string} clientId - Client identifier
   * @returns {number}
   */
  remaining(clientId) {
    const bucket = this._clients.get(clientId);
    return bucket ? Math.floor(bucket.tokens) : this.maxTokens;
  }

  /**
   * Remove a client from rate limiting.
   * @param {string} clientId - Client identifier
   */
  removeClient(clientId) {
    this._clients.delete(clientId);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Data Buffer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Circular buffer for storing recent telemetry data per channel.
 */
class DataBuffer {
  /**
   * @param {number} maxSize - Maximum items per channel
   * @param {number} maxAgeMs - Maximum age of items in ms
   */
  constructor(maxSize = 1000, maxAgeMs = 60000) {
    /** @type {number} */
    this.maxSize = maxSize;
    /** @type {number} */
    this.maxAgeMs = maxAgeMs;
    /** @type {Map<string, Array<{data:*, timestamp:number}>>} */
    this._buffers = new Map();
  }

  /**
   * Push data into a channel buffer.
   * @param {string} channel - Channel name
   * @param {*} data - Data payload
   */
  push(channel, data) {
    if (!this._buffers.has(channel)) {
      this._buffers.set(channel, []);
    }
    const buffer = this._buffers.get(channel);
    buffer.push({ data, timestamp: Date.now() });

    // Enforce size limit
    while (buffer.length > this.maxSize) {
      buffer.shift();
    }
  }

  /**
   * Get recent data from a channel.
   * @param {string} channel - Channel name
   * @param {number} [limit=100] - Maximum items to return
   * @returns {Array<*>}
   */
  getRecent(channel, limit = 100) {
    const buffer = this._buffers.get(channel);
    if (!buffer) return [];

    // Filter expired items
    const now = Date.now();
    const valid = buffer.filter(item => (now - item.timestamp) < this.maxAgeMs);
    this._buffers.set(channel, valid);

    return valid.slice(-limit).map(item => item.data);
  }

  /**
   * Clear all buffers.
   */
  clear() {
    this._buffers.clear();
  }

  /**
   * Get total buffered item count.
   * @returns {number}
   */
  get size() {
    let total = 0;
    for (const buffer of this._buffers.values()) {
      total += buffer.length;
    }
    return total;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Client Connection
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Represents a connected client with metadata and subscriptions.
 */
class ClientConnection {
  /**
   * @param {string} id - Unique client ID
   * @param {object} socket - WebSocket connection
   * @param {object} [meta={}] - Client metadata
   */
  constructor(id, socket, meta = {}) {
    /** @type {string} */
    this.id = id;
    /** @type {object} */
    this.socket = socket;
    /** @type {object} */
    this.meta = meta;
    /** @type {Set<string>} Subscribed channels */
    this.channels = new Set();
    /** @type {number} */
    this.connectedAt = Date.now();
    /** @type {boolean} */
    this.authenticated = false;
    /** @type {number} */
    this.messagesReceived = 0;
    /** @type {number} */
    this.messagesSent = 0;
    /** @type {number} */
    this.bytesSent = 0;
  }

  /**
   * Check if client is subscribed to a channel.
   * @param {string} channel - Channel name
   * @returns {boolean}
   */
  isSubscribed(channel) {
    return this.channels.has(channel) || this.channels.has('*');
  }

  /**
   * Subscribe to a channel.
   * @param {string} channel - Channel name
   */
  subscribe(channel) {
    this.channels.add(channel);
  }

  /**
   * Unsubscribe from a channel.
   * @param {string} channel - Channel name
   */
  unsubscribe(channel) {
    this.channels.delete(channel);
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Authentication Provider
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Simple token-based authentication provider.
 */
class AuthenticationProvider {
  /**
   * @param {object} config - Auth configuration
   * @param {string[]} [config.validTokens=[]] - List of valid auth tokens
   * @param {Function} [config.validateFn] - Custom validation function
   */
  constructor(config = {}) {
    /** @type {Set<string>} */
    this._validTokens = new Set(config.validTokens || []);
    /** @type {Function|null} */
    this._validateFn = config.validateFn || null;
  }

  /**
   * Validate an authentication token.
   * @param {string} token - Auth token to validate
   * @returns {Promise<{valid:boolean, meta?:object}>}
   */
  async validate(token) {
    if (!token) {
      return { valid: false, meta: { reason: 'No token provided' } };
    }

    if (this._validateFn) {
      try {
        const result = await this._validateFn(token);
        return { valid: !!result, meta: typeof result === 'object' ? result : {} };
      } catch (error) {
        return { valid: false, meta: { reason: error.message } };
      }
    }

    if (this._validTokens.has(token)) {
      return { valid: true, meta: {} };
    }

    return { valid: false, meta: { reason: 'Invalid token' } };
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// TelemetryServer
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Telemetry server providing WebSocket-based real-time data streaming
 * with subscription management, rate limiting, and authentication.
 *
 * @extends EventEmitter
 *
 * @example
 * const server = new TelemetryServer({ port: 8080, authTokens: ['secret'] });
 * server.on('client:connected', (client) => console.log(`Client ${client.id} connected`));
 * server.broadcast('gps', { lat: 37.7749, lng: -122.4194 });
 * await server.start();
 */
export class TelemetryServer extends EventEmitter {
  /**
   * @param {object} [config={}] - Server configuration
   * @param {number} [config.port=8080] - Server port
   * @param {string} [config.host='0.0.0.0'] - Server host
   * @param {string[]} [config.authTokens=[]] - Valid authentication tokens
   * @param {object} [config.rateLimit] - Rate limiting config
   * @param {number} [config.bufferSize=1000] - Per-channel buffer size
   * @param {number} [config.bufferMaxAgeMs=60000] - Buffer max age in ms
   * @param {boolean} [config.compression=false] - Enable compression
   * @param {number} [config.maxClients=100] - Maximum simultaneous clients
   * @param {number} [config.pingIntervalMs=30000] - Ping interval for keepalive
   */
  constructor(config = {}) {
    super();

    /** @type {object} */
    this.config = {
      port: config.port || 8080,
      host: config.host || '0.0.0.0',
      authTokens: config.authTokens || [],
      rateLimit: {
        maxTokens: config.rateLimit?.maxTokens || 100,
        refillRate: config.rateLimit?.refillRate || 10
      },
      bufferSize: config.bufferSize || 1000,
      bufferMaxAgeMs: config.bufferMaxAgeMs || 60000,
      compression: config.compression || false,
      maxClients: config.maxClients || 100,
      pingIntervalMs: config.pingIntervalMs || 30000
    };

    /** @type {Map<string, ClientConnection>} Connected clients */
    this._clients = new Map();

    /** @type {RateLimiter} */
    this._rateLimiter = new RateLimiter(this.config.rateLimit);

    /** @type {DataBuffer} */
    this._dataBuffer = new DataBuffer(this.config.bufferSize, this.config.bufferMaxAgeMs);

    /** @type {AuthenticationProvider} */
    this._authProvider = new AuthenticationProvider({
      validTokens: this.config.authTokens
    });

    /** @type {object|null} HTTP server instance */
    this._httpServer = null;

    /** @type {object|null} WebSocket server instance */
    this._wss = null;

    /** @type {boolean} */
    this._running = false;

    /** @type {NodeJS.Timer|null} */
    this._pingTimer = null;

    /** @type {number} */
    this._clientCounter = 0;

    /** @type {object} Server statistics */
    this._stats = {
      totalConnections: 0,
      totalMessagesReceived: 0,
      totalMessagesSent: 0,
      totalBytesSent: 0,
      authFailures: 0,
      rateLimitHits: 0,
      startTime: null
    };
  }

  // ── Server Lifecycle ────────────────────────────────────────────────────

  /**
   * Start the telemetry server.
   * @returns {Promise<void>}
   */
  async start() {
    if (this._running) {
      throw new TelemetryServerError('Server is already running', 'ALREADY_RUNNING');
    }

    // In a real implementation, this would create an Express + ws server:
    // const express = (await import('express')).default;
    // const http = require('http');
    // const WebSocket = (await import('ws')).WebSocketServer;
    //
    // const app = express();
    // app.use(express.json());
    // this._setupHTTPRoutes(app);
    // this._httpServer = http.createServer(app);
    // this._wss = new WebSocket({ server: this._httpServer });

    this._running = true;
    this._stats.startTime = Date.now();

    // Start ping interval
    if (this.config.pingIntervalMs > 0) {
      this._pingTimer = setInterval(() => this._pingClients(), this.config.pingIntervalMs);
    }

    this.emit('started', { port: this.config.port, host: this.config.host });
  }

  /**
   * Stop the telemetry server gracefully.
   * @param {number} [timeout=5000] - Graceful shutdown timeout in ms
   * @returns {Promise<void>}
   */
  async stop(timeout = 5000) {
    if (!this._running) return;

    this._running = false;

    if (this._pingTimer) {
      clearInterval(this._pingTimer);
      this._pingTimer = null;
    }

    // Close all client connections
    const closePromises = [];
    for (const [id, client] of this._clients) {
      closePromises.push(this._disconnectClient(id, 1001, 'Server shutting down'));
    }
    await Promise.all(closePromises);

    // Close server
    if (this._httpServer) {
      await new Promise(resolve => this._httpServer.close(resolve));
    }

    this._dataBuffer.clear();
    this.emit('stopped', { stats: this._stats });
  }

  // ── Data Broadcasting ───────────────────────────────────────────────────

  /**
   * Broadcast data to all clients subscribed to a channel.
   * @param {string} channel - Channel to broadcast on
   * @param {*} data - Data payload
   * @param {object} [options={}] - Broadcast options
   * @param {boolean} [options.buffer=true] - Store in data buffer
   * @param {boolean} [options.compress=false] - Compress payload
   * @returns {number} Number of clients that received the message
   */
  broadcast(channel, data, options = {}) {
    const { buffer = true, compress = false } = options;

    // Buffer the data
    if (buffer) {
      this._dataBuffer.push(channel, data);
    }

    const message = {
      type: 'data',
      channel,
      payload: data,
      timestamp: Date.now()
    };

    let recipientCount = 0;

    for (const [, client] of this._clients) {
      if (client.isSubscribed(channel) && client.authenticated) {
        if (this._sendToClient(client, message, compress)) {
          recipientCount++;
        }
      }
    }

    this.emit('broadcast', { channel, recipientCount });
    return recipientCount;
  }

  /**
   * Send data to a specific client.
   * @param {string} clientId - Target client ID
   * @param {string} channel - Channel name
   * @param {*} data - Data payload
   * @returns {boolean} True if message was sent
   */
  sendToClient(clientId, channel, data) {
    const client = this._clients.get(clientId);
    if (!client) return false;

    return this._sendToClient(client, {
      type: 'data',
      channel,
      payload: data,
      timestamp: Date.now()
    });
  }

  // ── Client Management ───────────────────────────────────────────────────

  /**
   * Get a list of connected client IDs.
   * @returns {string[]}
   */
  getClientIds() {
    return Array.from(this._clients.keys());
  }

  /**
   * Get information about a specific client.
   * @param {string} clientId - Client ID
   * @returns {object|null}
   */
  getClientInfo(clientId) {
    const client = this._clients.get(clientId);
    if (!client) return null;
    return {
      id: client.id,
      authenticated: client.authenticated,
      channels: Array.from(client.channels),
      connectedAt: client.connectedAt,
      messagesReceived: client.messagesReceived,
      messagesSent: client.messagesSent,
      bytesSent: client.bytesSent
    };
  }

  /**
   * Disconnect a specific client.
   * @param {string} clientId - Client ID to disconnect
   * @param {number} [code=1000] - Close code
   * @param {string} [reason=''] - Close reason
   * @returns {Promise<void>}
   */
  async _disconnectClient(clientId, code = 1000, reason = '') {
    const client = this._clients.get(clientId);
    if (!client) return;

    try {
      if (client.socket && typeof client.socket.close === 'function') {
        client.socket.close(code, reason);
      }
    } catch (_) { /* ignore */ }

    this._rateLimiter.removeClient(clientId);
    this._clients.delete(clientId);
    this.emit('client:disconnected', { clientId, code, reason });
  }

  // ── Query Endpoints ─────────────────────────────────────────────────────

  /**
   * Get buffered data for a channel.
   * @param {string} channel - Channel name
   * @param {number} [limit=100] - Maximum items
   * @returns {Array<*>}
   */
  getChannelData(channel, limit = 100) {
    return this._dataBuffer.getRecent(channel, limit);
  }

  /**
   * Get server statistics.
   * @returns {object}
   */
  getStats() {
    return {
      ...this._stats,
      connectedClients: this._clients.size,
      bufferSize: this._dataBuffer.size,
      uptimeMs: this._stats.startTime ? Date.now() - this._stats.startTime : 0
    };
  }

  /**
   * Get health status.
   * @returns {object}
   */
  getHealth() {
    return {
      status: this._running ? 'healthy' : 'stopped',
      connectedClients: this._clients.size,
      maxClients: this.config.maxClients,
      bufferSize: this._dataBuffer.size
    };
  }

  // ── Private Methods ─────────────────────────────────────────────────────

  /**
   * Handle a new WebSocket connection.
   * @param {object} socket - WebSocket connection
   * @param {object} request - HTTP upgrade request
   * @private
   */
  async _handleConnection(socket, request) {
    if (this._clients.size >= this.config.maxClients) {
      socket.close(1013, 'Maximum clients reached');
      return;
    }

    const clientId = `client_${++this._clientCounter}`;
    const client = new ClientConnection(clientId, socket, {
      remoteAddress: request?.socket?.remoteAddress,
      userAgent: request?.headers?.['user-agent']
    });

    this._clients.set(clientId, client);
    this._stats.totalConnections++;

    // Extract auth token from query params or headers
    const url = new URL(request?.url || '', `http://${request?.headers?.host || 'localhost'}`);
    const token = url.searchParams.get('token') || request?.headers?.['authorization']?.replace('Bearer ', '');

    if (token) {
      const authResult = await this._authProvider.validate(token);
      if (authResult.valid) {
        client.authenticated = true;
        client.meta = { ...client.meta, ...authResult.meta };
      } else {
        this._stats.authFailures++;
        this._sendToClient(client, {
          type: 'error',
          code: 'AUTH_FAILED',
          message: authResult.meta?.reason || 'Authentication failed'
        });
        await this._disconnectClient(clientId, 4001, 'Authentication failed');
        return;
      }
    } else if (this.config.authTokens.length > 0) {
      // Auth tokens configured but none provided
      this._stats.authFailures++;
      this._sendToClient(client, {
        type: 'error',
        code: 'AUTH_REQUIRED',
        message: 'Authentication token required'
      });
      await this._disconnectClient(clientId, 4001, 'Authentication required');
      return;
    } else {
      // No auth configured, auto-authenticate
      client.authenticated = true;
    }

    this.emit('client:connected', { clientId, meta: client.meta });

    // Send backlog for subscribed channels
    // (no subscriptions yet, but future subscribe will send backlog)

    // Set up message handler
    socket.onmessage = (event) => this._handleClientMessage(clientId, event);
    socket.onclose = (event) => this._handleClientClose(clientId, event);
    socket.onerror = (error) => this._handleClientError(clientId, error);

    // Welcome message
    this._sendToClient(client, {
      type: 'welcome',
      clientId,
      channels: this._getAvailableChannels(),
      timestamp: Date.now()
    });
  }

  /**
   * Handle a message from a client.
   * @param {string} clientId - Client ID
   * @param {MessageEvent} event - Message event
   * @private
   */
  async _handleClientMessage(clientId, event) {
    const client = this._clients.get(clientId);
    if (!client) return;

    // Rate limiting
    if (!this._rateLimiter.allow(clientId)) {
      this._stats.rateLimitHits++;
      this._sendToClient(client, {
        type: 'error',
        code: 'RATE_LIMITED',
        message: 'Rate limit exceeded'
      });
      return;
    }

    client.messagesReceived++;
    this._stats.totalMessagesReceived++;

    try {
      const raw = typeof event.data === 'string' ? event.data : event.data.toString();
      let message;

      try {
        message = JSON.parse(raw);
      } catch (_) {
        if (this.config.compression) {
          const decompressed = inflateSync(Buffer.from(raw, 'base64'));
          message = JSON.parse(decompressed.toString());
        } else {
          throw new Error('Invalid JSON');
        }
      }

      switch (message.type) {
        case 'subscribe':
          this._handleSubscribe(clientId, message);
          break;
        case 'unsubscribe':
          this._handleUnsubscribe(clientId, message);
          break;
        case 'request':
          await this._handleRequest(clientId, message);
          break;
        case 'command':
          this._handleCommand(clientId, message);
          break;
        case 'ping':
          this._sendToClient(client, { type: 'pong', timestamp: Date.now() });
          break;
        case 'data':
          this.emit('client:data', { clientId, channel: message.channel, data: message.payload });
          break;
        default:
          this._sendToClient(client, {
            type: 'error',
            code: 'UNKNOWN_MESSAGE_TYPE',
            message: `Unknown message type: ${message.type}`
          });
      }
    } catch (error) {
      this.emit('error', new TelemetryServerError(
        `Client message error: ${error.message}`,
        'MESSAGE_PARSE_ERROR'
      ));
    }
  }

  /**
   * Handle a subscribe message from a client.
   * @param {string} clientId - Client ID
   * @param {object} message - Subscribe message
   * @private
   */
  _handleSubscribe(clientId, message) {
    const client = this._clients.get(clientId);
    if (!client) return;

    const channels = message.channels || [];
    for (const ch of channels) {
      client.subscribe(ch);
    }

    // Send backlog for newly subscribed channels
    for (const ch of channels) {
      const backlog = this._dataBuffer.getRecent(ch, 50);
      if (backlog.length > 0) {
        for (const item of backlog) {
          this._sendToClient(client, {
            type: 'data',
            channel: ch,
            payload: item,
            timestamp: Date.now(),
            backlog: true
          });
        }
      }
    }

    this.emit('client:subscribed', { clientId, channels });
  }

  /**
   * Handle an unsubscribe message from a client.
   * @param {string} clientId - Client ID
   * @param {object} message - Unsubscribe message
   * @private
   */
  _handleUnsubscribe(clientId, message) {
    const client = this._clients.get(clientId);
    if (!client) return;

    const channels = message.channels || [];
    for (const ch of channels) {
      client.unsubscribe(ch);
    }

    this.emit('client:unsubscribed', { clientId, channels });
  }

  /**
   * Handle a request message from a client.
   * @param {string} clientId - Client ID
   * @param {object} message - Request message
   * @private
   */
  async _handleRequest(clientId, message) {
    const client = this._clients.get(clientId);
    if (!client) return;

    try {
      let result;

      switch (message.method) {
        case 'getChannelData':
          result = this._dataBuffer.getRecent(message.params?.channel, message.params?.limit || 100);
          break;
        case 'getChannels':
          result = this._getAvailableChannels();
          break;
        case 'getStats':
          result = this.getStats();
          break;
        case 'ping':
          result = { pong: true, serverTime: Date.now() };
          break;
        default:
          this._sendToClient(client, {
            type: 'response',
            id: message.id,
            error: { code: 'UNKNOWN_METHOD', message: `Unknown method: ${message.method}` }
          });
          return;
      }

      this._sendToClient(client, {
        type: 'response',
        id: message.id,
        result
      });
    } catch (error) {
      this._sendToClient(client, {
        type: 'response',
        id: message.id,
        error: { code: 'INTERNAL_ERROR', message: error.message }
      });
    }
  }

  /**
   * Handle a command message from a client.
   * @param {string} clientId - Client ID
   * @param {object} message - Command message
   * @private
   */
  _handleCommand(clientId, message) {
    this.emit('client:command', { clientId, command: message.command, params: message.params });
  }

  /**
   * Handle client WebSocket close event.
   * @param {string} clientId - Client ID
   * @param {CloseEvent} event - Close event
   * @private
   */
  _handleClientClose(clientId, event) {
    this._rateLimiter.removeClient(clientId);
    this._clients.delete(clientId);
    this.emit('client:disconnected', { clientId, code: event.code, reason: event.reason });
  }

  /**
   * Handle client WebSocket error.
   * @param {string} clientId - Client ID
   * @param {Error} error - Error
   * @private
   */
  _handleClientError(clientId, error) {
    this.emit('client:error', { clientId, error });
  }

  /**
   * Send a message to a specific client.
   * @param {ClientConnection} client - Client connection
   * @param {object} message - Message to send
   * @param {boolean} [compress=false] - Compress payload
   * @returns {boolean} True if sent successfully
   * @private
   */
  _sendToClient(client, message, compress = false) {
    if (!client.socket || client.socket.readyState !== 1) {
      return false;
    }

    try {
      let data = JSON.stringify(message);

      if (compress || this.config.compression) {
        const compressed = deflateSync(Buffer.from(data));
        data = compressed.toString('base64');
      }

      client.socket.send(data);
      client.messagesSent++;
      client.bytesSent += data.length;
      this._stats.totalMessagesSent++;
      this._stats.totalBytesSent += data.length;
      return true;
    } catch (error) {
      this.emit('error', new TelemetryServerError(
        `Send to client ${client.id} failed: ${error.message}`,
        'SEND_ERROR'
      ));
      return false;
    }
  }

  /**
   * Send ping to all connected clients.
   * @private
   */
  _pingClients() {
    for (const [, client] of this._clients) {
      if (client.socket && client.socket.readyState === 1) {
        this._sendToClient(client, { type: 'ping', timestamp: Date.now() });
      }
    }
  }

  /**
   * Get list of available channels from the data buffer.
   * @returns {string[]}
   * @private
   */
  _getAvailableChannels() {
    return Array.from(this._dataBuffer._buffers.keys());
  }
}

export default TelemetryServer;
