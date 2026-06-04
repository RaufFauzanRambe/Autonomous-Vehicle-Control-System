/**
 * @fileoverview WebSocketClient - Robust WebSocket client for the autonomous
 * vehicle monitoring system. Provides connection management with auto-reconnect
 * using exponential backoff (1s→30s), heartbeat/ping-pong mechanism, message
 * queuing during disconnection, subscription channel management, binary frame
 * support, and authentication handshake.
 *
 * @module realtime_monitoring/websocket_client
 * @version 2.1.0
 * @author Autonomous Vehicle Control System Team
 */

import { EventEmitter } from 'events';

/** @typedef {'disconnected'|'connecting'|'authenticating'|'connected'|'reconnecting'} ConnectionState */

/**
 * @typedef {Object} WSClientConfig
 * @property {string} url - WebSocket server URL
 * @property {Object} [headers={}] - Custom headers for handshake
 * @property {Object} [auth=null] - Authentication credentials
 * @property {string} [auth.token] - Auth token
 * @property {string} [auth.clientId] - Client identifier
 * @property {number} [reconnectBaseDelayMs=1000] - Initial reconnect delay
 * @property {number} [reconnectMaxDelayMs=30000] - Maximum reconnect delay
 * @property {number} [reconnectMaxAttempts=Infinity] - Max reconnect attempts
 * @property {number} [heartbeatIntervalMs=30000] - Heartbeat interval
 * @property {number} [heartbeatTimeoutMs=10000] - Heartbeat response timeout
 * @property {number} [maxQueueSize=1000] - Max queued messages during disconnect
 * @property {boolean} [binarySupported=true] - Enable binary frame support
 * @property {number} [connectTimeoutMs=10000] - Connection timeout
 */

/**
 * @typedef {Object} QueuedMessage
 * @property {string} channel - Target channel
 * @property {*} data - Message data
 * @property {number} timestamp - Queued timestamp
 * @property {number} [retryCount=0] - Number of send retries
 */

/**
 * @typedef {Object} Subscription
 * @property {string} channel - Channel name
 * @property {Function} handler - Message handler
 * @property {Date} subscribedAt - Subscription time
 * @property {number} messageCount - Received message count
 */

/**
 * WebSocketClient manages a persistent WebSocket connection with automatic
 * reconnection, heartbeat, message queuing, and channel subscriptions.
 *
 * @extends EventEmitter
 *
 * @example
 * const client = new WebSocketClient({
 *   url: 'wss://vehicle-cloud.example.com/monitoring',
 *   auth: { token: 'abc123', clientId: 'veh-001' },
 * });
 *
 * client.on('connected', () => console.log('Connected!'));
 * client.subscribe('telemetry', (data) => handleTelemetry(data));
 *
 * await client.connect();
 * client.send('telemetry', { speed: 60, heading: 270 });
 */
export class WebSocketClient extends EventEmitter {
  /** @type {WSClientConfig} */
  #config;

  /** @type {ConnectionState} */
  #state = 'disconnected';

  /** @type {WebSocket|null} */
  #ws = null;

  /** @type {number} */
  #reconnectAttempts = 0;

  /** @type {number|null} */
  #reconnectTimer = null;

  /** @type {number|null} */
  #heartbeatTimer = null;

  /** @type {number|null} */
  #heartbeatTimeoutTimer = null;

  /** @type {boolean} */
  #heartbeatAcked = false;

  /** @type {QueuedMessage[]} */
  #messageQueue = [];

  /** @type {Map<string, Subscription>} */
  #subscriptions = new Map();

  /** @type {number|null} */
  #connectTimeout = null;

  /** @type {number} */
  #lastMessageTime = 0;

  /** @type {number} */
  #messagesReceived = 0;

  /** @type {number} */
  #messagesSent = 0;

  /** @type {boolean} */
  #intentionalDisconnect = false;

  /**
   * Creates a new WebSocketClient.
   *
   * @param {WSClientConfig} config - Client configuration
   */
  constructor(config) {
    super();
    this.setMaxListeners(30);

    if (!config?.url) {
      throw new TypeError('WebSocket URL is required');
    }

    this.#config = {
      headers: {},
      auth: null,
      reconnectBaseDelayMs: 1000,
      reconnectMaxDelayMs: 30000,
      reconnectMaxAttempts: Infinity,
      heartbeatIntervalMs: 30000,
      heartbeatTimeoutMs: 10000,
      maxQueueSize: 1000,
      binarySupported: true,
      connectTimeoutMs: 10000,
      ...config,
    };
  }

  /**
   * Current connection state.
   * @type {ConnectionState}
   */
  get state() {
    return this.#state;
  }

  /**
   * Whether the client is currently connected.
   * @type {boolean}
   */
  get isConnected() {
    return this.#state === 'connected';
  }

  /**
   * Number of active channel subscriptions.
   * @type {number}
   */
  get subscriptionCount() {
    return this.#subscriptions.size;
  }

  /**
   * Number of messages in the offline queue.
   * @type {number}
   */
  get queuedMessageCount() {
    return this.#messageQueue.length;
  }

  /**
   * Current reconnect attempt number (0 if connected or first try).
   * @type {number}
   */
  get reconnectAttempts() {
    return this.#reconnectAttempts;
  }

  /**
   * Total messages received since last connect.
   * @type {number}
   */
  get messagesReceived() {
    return this.#messagesReceived;
  }

  /**
   * Total messages sent since last connect.
   * @type {number}
   */
  get messagesSent() {
    return this.#messagesSent;
  }

  /**
   * Connects to the WebSocket server.
   * Initiates the connection and performs authentication handshake if
   * credentials are configured.
   *
   * @returns {Promise<void>}
   * @throws {Error} If URL is invalid or connection fails
   *
   * @fires WebSocketClient#connecting
   * @fires WebSocketClient#connected
   */
  async connect() {
    if (this.#state === 'connected' || this.#state === 'connecting') {
      return;
    }

    this.#intentionalDisconnect = false;
    this.#setState('connecting');

    /**
     * @event WebSocketClient#connecting
     * @type {Object}
     * @property {string} url - Server URL
     */
    this.emit('connecting', { url: this.#config.url });

    return new Promise((resolve, reject) => {
      try {
        const protocols = this.#config.auth ? ['vehicle-monitor-v1'] : undefined;
        this.#ws = new WebSocket(this.#config.url, protocols);

        this.#connectTimeout = setTimeout(() => {
          this.#ws?.close();
          reject(new Error('Connection timeout'));
        }, this.#config.connectTimeoutMs);

        this.#ws.onopen = () => {
          clearTimeout(this.#connectTimeout);
          this.#connectTimeout = null;

          if (this.#config.auth) {
            this.#setState('authenticating');
            this.#performAuthHandshake()
              .then(() => {
                this.#onConnected();
                resolve();
              })
              .catch((err) => {
                this.#ws?.close();
                reject(err);
              });
          } else {
            this.#onConnected();
            resolve();
          }
        };

        this.#ws.onmessage = (event) => {
          this.#handleMessage(event);
        };

        this.#ws.onerror = (event) => {
          /**
           * @event WebSocketClient#error
           * @type {Object}
           * @property {string} message - Error description
           */
          this.emit('error', { message: 'WebSocket error', event });
        };

        this.#ws.onclose = (event) => {
          clearTimeout(this.#connectTimeout);
          this.#connectTimeout = null;
          this.#onClose(event.code, event.reason);
        };
      } catch (error) {
        this.#setState('disconnected');
        reject(new Error(`Failed to create WebSocket: ${error.message}`));
      }
    });
  }

  /**
   * Disconnects from the server.
   * Flushes the message queue and cleans up resources.
   *
   * @param {number} [code=1000] - Close code
   * @param {string} [reason='Client disconnect'] - Close reason
   * @returns {Promise<void>}
   *
   * @fires WebSocketClient#disconnected
   */
  async disconnect(code = 1000, reason = 'Client disconnect') {
    this.#intentionalDisconnect = true;
    this.#stopHeartbeat();
    this.#clearReconnectTimer();

    if (this.#ws && (this.#ws.readyState === WebSocket.OPEN || this.#ws.readyState === WebSocket.CONNECTING)) {
      this.#ws.close(code, reason);
    }

    this.#ws = null;
    this.#setState('disconnected');
    this.#messageQueue = [];
    this.#subscriptions.clear();

    /**
     * @event WebSocketClient#disconnected
     * @type {Object}
     * @property {number} code - Close code
     * @property {string} reason - Close reason
     */
    this.emit('disconnected', { code, reason });
  }

  /**
   * Sends data on a specific channel.
   * If disconnected, the message is queued for later delivery.
   *
   * @param {string} channel - Target channel
   * @param {*} data - Data to send (will be serialized)
   * @param {Object} [options={}] - Send options
   * @param {boolean} [options.binary=false] - Send as binary frame
   * @param {boolean} [options.highPriority=false] - Bypass queue position
   * @returns {boolean} Whether the message was sent or queued
   */
  send(channel, data, options = {}) {
    if (!channel || typeof channel !== 'string') {
      throw new TypeError('Channel must be a non-empty string');
    }

    const message = {
      type: 'publish',
      channel,
      data,
      timestamp: Date.now(),
      clientId: this.#config.auth?.clientId,
    };

    if (this.isConnected && this.#ws?.readyState === WebSocket.OPEN) {
      try {
        if (options.binary && this.#config.binarySupported && data instanceof ArrayBuffer) {
          this.#ws.send(data);
        } else {
          const serialized = JSON.stringify(message);
          this.#ws.send(serialized);
        }
        this.#messagesSent++;
        return true;
      } catch (error) {
        this.emit('error', { message: `Send failed: ${error.message}` });
        return this.#enqueueMessage(channel, data, options.highPriority);
      }
    }

    return this.#enqueueMessage(channel, data, options.highPriority);
  }

  /**
   * Sends binary data directly over the connection.
   *
   * @param {ArrayBuffer|Uint8Array} data - Binary data
   * @returns {boolean} Whether the data was sent
   */
  sendBinary(data) {
    if (!this.isConnected || this.#ws?.readyState !== WebSocket.OPEN) {
      return false;
    }

    if (!this.#config.binarySupported) {
      this.emit('error', { message: 'Binary frames not supported' });
      return false;
    }

    try {
      const buffer = data instanceof Uint8Array ? data.buffer : data;
      this.#ws.send(buffer);
      this.#messagesSent++;
      return true;
    } catch (error) {
      this.emit('error', { message: `Binary send failed: ${error.message}` });
      return false;
    }
  }

  /**
   * Subscribes to a channel for receiving messages.
   *
   * @param {string} channel - Channel name
   * @param {Function} handler - Message handler (data) => void
   * @returns {Function} Unsubscribe function
   *
   * @example
   * const unsub = client.subscribe('alerts', (alert) => {
   *   console.log('Alert:', alert);
   * });
   * // Later: unsub();
   */
  subscribe(channel, handler) {
    if (!channel || typeof channel !== 'string') {
      throw new TypeError('Channel must be a non-empty string');
    }
    if (typeof handler !== 'function') {
      throw new TypeError('Handler must be a function');
    }

    const existing = this.#subscriptions.get(channel);
    if (existing) {
      this.#subscriptions.set(channel, {
        ...existing,
        handler,
        messageCount: existing.messageCount,
      });
    } else {
      this.#subscriptions.set(channel, {
        channel,
        handler,
        subscribedAt: new Date(),
        messageCount: 0,
      });

      if (this.isConnected) {
        this.#sendSubscriptionMessage('subscribe', channel);
      }
    }

    return () => this.unsubscribe(channel);
  }

  /**
   * Unsubscribes from a channel.
   *
   * @param {string} channel - Channel name
   * @returns {void}
   */
  unsubscribe(channel) {
    if (this.#subscriptions.delete(channel) && this.isConnected) {
      this.#sendSubscriptionMessage('unsubscribe', channel);
    }
  }

  /**
   * Gets connection statistics.
   *
   * @returns {Object} Connection statistics
   */
  getStats() {
    return {
      state: this.#state,
      url: this.#config.url,
      reconnectAttempts: this.#reconnectAttempts,
      messagesReceived: this.#messagesReceived,
      messagesSent: this.#messagesSent,
      queuedMessages: this.#messageQueue.length,
      subscriptions: this.#subscriptions.size,
      lastMessageTime: this.#lastMessageTime,
      uptime: this.isConnected ? Date.now() - (this.#lastMessageTime || Date.now()) : 0,
    };
  }

  // ─── Private Methods ───────────────────────────────────────────────

  /**
   * Performs the authentication handshake with the server.
   * @private
   * @returns {Promise<void>}
   */
  #performAuthHandshake() {
    return new Promise((resolve, reject) => {
      const authMessage = JSON.stringify({
        type: 'auth',
        token: this.#config.auth.token,
        clientId: this.#config.auth.clientId,
        timestamp: Date.now(),
      });

      const authTimeout = setTimeout(() => {
        reject(new Error('Authentication timeout'));
      }, 5000);

      const authHandler = (event) => {
        try {
          const response = JSON.parse(event.data);
          if (response.type === 'auth:success') {
            clearTimeout(authTimeout);
            this.#ws?.removeEventListener('message', authHandler);
            resolve();
          } else if (response.type === 'auth:failure') {
            clearTimeout(authTimeout);
            this.#ws?.removeEventListener('message', authHandler);
            reject(new Error(`Authentication failed: ${response.reason || 'Unknown'}`));
          }
        } catch {
          // Ignore non-JSON messages during auth
        }
      };

      this.#ws?.addEventListener('message', authHandler);
      this.#ws?.send(authMessage);
    });
  }

  /**
   * Called when the connection is established.
   * @private
   */
  #onConnected() {
    this.#setState('connected');
    this.#reconnectAttempts = 0;
    this.#messagesReceived = 0;
    this.#messagesSent = 0;

    this.#startHeartbeat();
    this.#resubscribeAll();
    this.#flushMessageQueue();

    /**
     * @event WebSocketClient#connected
     * @type {Object}
     * @property {string} url - Server URL
     * @property {number} timestamp - Connection timestamp
     */
    this.emit('connected', {
      url: this.#config.url,
      timestamp: Date.now(),
    });
  }

  /**
   * Handles incoming WebSocket messages.
   * @private
   * @param {MessageEvent} event - WebSocket message event
   */
  #handleMessage(event) {
    this.#lastMessageTime = Date.now();
    this.#messagesReceived++;

    if (event.data instanceof ArrayBuffer || event.data instanceof Blob) {
      /**
       * @event WebSocketClient#binary
       * @type {ArrayBuffer}
       */
      this.emit('binary', event.data);
      return;
    }

    try {
      const message = JSON.parse(event.data);

      switch (message.type) {
        case 'pong':
          this.#handlePong();
          break;
        case 'publish':
          this.#dispatchToSubscription(message.channel, message.data);
          break;
        case 'subscribed':
          /**
           * @event WebSocketClient#subscribed
           * @type {string}
           */
          this.emit('subscribed', message.channel);
          break;
        case 'unsubscribed':
          this.emit('unsubscribed', message.channel);
          break;
        case 'error':
          this.emit('error', { message: message.error, code: message.code });
          break;
        default:
          /**
           * @event WebSocketClient#message
           * @type {Object}
           */
          this.emit('message', message);
      }
    } catch {
      this.emit('message', { raw: event.data });
    }
  }

  /**
   * Handles connection close events.
   * @private
   * @param {number} code - Close code
   * @param {string} reason - Close reason
   */
  #onClose(code, reason) {
    this.#stopHeartbeat();

    const wasConnected = this.#state === 'connected';
    this.#ws = null;

    /**
     * @event WebSocketClient#disconnected
     * @type {Object}
     */
    this.emit('disconnected', { code, reason, wasConnected });

    if (!this.#intentionalDisconnect) {
      this.#scheduleReconnect();
    } else {
      this.#setState('disconnected');
    }
  }

  /**
   * Schedules a reconnection attempt with exponential backoff.
   * @private
   */
  #scheduleReconnect() {
    if (this.#reconnectAttempts >= this.#config.reconnectMaxAttempts) {
      this.#setState('disconnected');
      this.emit('error', {
        message: `Max reconnect attempts reached (${this.#config.reconnectMaxAttempts})`,
      });
      return;
    }

    this.#setState('reconnecting');

    const delay = Math.min(
      this.#config.reconnectBaseDelayMs * Math.pow(2, this.#reconnectAttempts) +
        Math.random() * 1000,
      this.#config.reconnectMaxDelayMs
    );

    this.#reconnectAttempts++;

    /**
     * @event WebSocketClient#reconnecting
     * @type {Object}
     * @property {number} attempt - Current attempt number
     * @property {number} delayMs - Delay before attempt in ms
     */
    this.emit('reconnecting', {
      attempt: this.#reconnectAttempts,
      delayMs: delay,
    });

    this.#reconnectTimer = setTimeout(async () => {
      try {
        await this.connect();
      } catch {
        // connect() will trigger another reconnect on failure
      }
    }, delay);
  }

  /**
   * Clears the reconnect timer.
   * @private
   */
  #clearReconnectTimer() {
    if (this.#reconnectTimer !== null) {
      clearTimeout(this.#reconnectTimer);
      this.#reconnectTimer = null;
    }
  }

  /**
   * Starts the heartbeat mechanism.
   * @private
   */
  #startHeartbeat() {
    this.#stopHeartbeat();
    this.#heartbeatAcked = true;

    this.#heartbeatTimer = setInterval(() => {
      if (!this.#heartbeatAcked) {
        this.#ws?.close(4001, 'Heartbeat timeout');
        return;
      }

      this.#heartbeatAcked = false;

      if (this.#ws?.readyState === WebSocket.OPEN) {
        this.#ws.send(JSON.stringify({ type: 'ping', timestamp: Date.now() }));
      }

      this.#heartbeatTimeoutTimer = setTimeout(() => {
        if (!this.#heartbeatAcked) {
          this.#ws?.close(4001, 'Heartbeat timeout');
        }
      }, this.#config.heartbeatTimeoutMs);
    }, this.#config.heartbeatIntervalMs);
  }

  /**
   * Stops the heartbeat mechanism.
   * @private
   */
  #stopHeartbeat() {
    if (this.#heartbeatTimer !== null) {
      clearInterval(this.#heartbeatTimer);
      this.#heartbeatTimer = null;
    }
    if (this.#heartbeatTimeoutTimer !== null) {
      clearTimeout(this.#heartbeatTimeoutTimer);
      this.#heartbeatTimeoutTimer = null;
    }
  }

  /**
   * Handles a pong response from the server.
   * @private
   */
  #handlePong() {
    this.#heartbeatAcked = true;
    if (this.#heartbeatTimeoutTimer !== null) {
      clearTimeout(this.#heartbeatTimeoutTimer);
      this.#heartbeatTimeoutTimer = null;
    }
  }

  /**
   * Sends a subscription/unsubscription message to the server.
   * @private
   * @param {'subscribe'|'unsubscribe'} action - Action type
   * @param {string} channel - Channel name
   */
  #sendSubscriptionMessage(action, channel) {
    if (this.#ws?.readyState === WebSocket.OPEN) {
      this.#ws.send(JSON.stringify({
        type: action,
        channel,
        timestamp: Date.now(),
      }));
    }
  }

  /**
   * Resubscribes to all previously subscribed channels.
   * @private
   */
  #resubscribeAll() {
    for (const channel of this.#subscriptions.keys()) {
      this.#sendSubscriptionMessage('subscribe', channel);
    }
  }

  /**
   * Dispatches a message to the appropriate subscription handler.
   * @private
   * @param {string} channel - Target channel
   * @param {*} data - Message data
   */
  #dispatchToSubscription(channel, data) {
    const sub = this.#subscriptions.get(channel);
    if (sub) {
      sub.messageCount++;
      try {
        sub.handler(data);
      } catch (error) {
        this.emit('error', { message: `Subscription handler error on ${channel}: ${error.message}` });
      }
    }
  }

  /**
   * Enqueues a message for later delivery.
   * @private
   * @param {string} channel - Target channel
   * @param {*} data - Message data
   * @param {boolean} [highPriority=false] - Insert at front of queue
   * @returns {boolean} Whether the message was queued
   */
  #enqueueMessage(channel, data, highPriority = false) {
    if (this.#messageQueue.length >= this.#config.maxQueueSize) {
      if (highPriority) {
        this.#messageQueue.shift();
      } else {
        this.emit('error', { message: 'Message queue full, message dropped' });
        return false;
      }
    }

    const entry = { channel, data, timestamp: Date.now(), retryCount: 0 };

    if (highPriority) {
      this.#messageQueue.unshift(entry);
    } else {
      this.#messageQueue.push(entry);
    }

    return true;
  }

  /**
   * Flushes the message queue, sending all queued messages.
   * @private
   */
  #flushMessageQueue() {
    while (this.#messageQueue.length > 0 && this.isConnected) {
      const entry = this.#messageQueue.shift();
      this.send(entry.channel, entry.data);
    }
  }

  /**
   * Updates the connection state and emits state change.
   * @private
   * @param {ConnectionState} newState - New state
   */
  #setState(newState) {
    const oldState = this.#state;
    if (oldState === newState) return;
    this.#state = newState;

    /**
     * @event WebSocketClient#state:change
     * @type {Object}
     * @property {ConnectionState} oldState
     * @property {ConnectionState} newState
     */
    this.emit('state:change', { oldState, newState });
  }
}

export default WebSocketClient;
