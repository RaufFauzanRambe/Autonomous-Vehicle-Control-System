# Realtime Monitoring Module

> Autonomous Vehicle Control System — Realtime Monitoring Subsystem v2.1.0

Comprehensive real-time monitoring infrastructure for the autonomous vehicle
control system. Provides health monitoring, latency tracking, alert management,
event logging, performance tracking, system status aggregation, and WebSocket
communication for dashboard telemetry.

---

## Architecture Overview

```
┌───────────────────────────────────────────────────────────────┐
│                     RealtimeMonitor (Orchestrator)            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐    │
│  │HealthMonitor │  │LatencyMonitor│  │PerformanceTracker│    │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘    │
│         │                 │                    │              │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌────────┴─────────┐    │
│  │ AlertManager │  │  EventLogger │  │   SystemStatus   │    │
│  └──────┬───────┘  └──────┬───────┘  └────────┬─────────┘    │
│         │                 │                    │              │
│  ───────┴─────────────────┴────────────────────┴──────────    │
│                    Data Pipeline Bus                           │
│  ─────────────────────────────────────────────────────────    │
│  ┌──────────────────┐  ┌──────────────────┐                   │
│  │ WebSocketServer  │  │ WebSocketClient  │                   │
│  │  (Dashboard)     │  │  (Cloud Upload)  │                   │
│  └──────────────────┘  └──────────────────┘                   │
└───────────────────────────────────────────────────────────────┘
```

### Module Responsibilities

| Module | File | Purpose |
|--------|------|---------|
| RealtimeMonitor | `realtime_monitor.js` | Orchestrator; data pipeline, lifecycle, dashboard push |
| WebSocketClient | `websocket_client.js` | Outbound WS connection with auto-reconnect |
| WebSocketServer | `websocket_server.js` | Inbound WS server for dashboard clients |
| HealthMonitor | `health_monitor.js` | System health checks, heartbeat, auto-recovery |
| LatencyMonitor | `latency_monitor.js` | E2E latency, percentiles, budget tracking |
| EventLogger | `event_logger.js` | Structured logging, rotation, search, export |
| AlertManager | `alert_manager.js` | Alert lifecycle, rules engine, notifications |
| SystemStatus | `system_status.js` | Overall status, mode management, module tracking |
| PerformanceTracker | `performance_tracker.js` | FPS, memory, CPU, regression detection |

---

## Quick Start

```javascript
import { RealtimeMonitor } from './realtime_monitor.js';

const monitor = new RealtimeMonitor({
  tickIntervalMs: 100,
  wsServerPort: 8090,
  enableWebSocketServer: true,
});

// Listen for dashboard updates
monitor.on('dashboard:update', (snapshot) => {
  console.log(`System: ${snapshot.systemStatus.overallStatus}`);
  console.log(`Mode: ${snapshot.systemStatus.mode}`);
  console.log(`Health: ${snapshot.healthReport.score}`);
  console.log(`Active Alerts: ${snapshot.activeAlerts.length}`);
});

// Listen for critical alerts
monitor.on('alert:critical', (alert) => {
  emergencyHandler(alert);
});

// Start monitoring
await monitor.start();

// Ingest custom data
monitor.ingestData({
  source: 'perception',
  type: 'sensor',
  payload: { objectType: 'vehicle', distance: 45.2 },
});
```

---

## WebSocket Protocol

### Message Types

| Type | Direction | Description |
|------|-----------|-------------|
| `auth` | Client → Server | Authentication handshake |
| `auth:success` / `auth:failure` | Server → Client | Auth response |
| `subscribe` | Client → Server | Subscribe to a channel |
| `unsubscribe` | Client → Server | Unsubscribe from a channel |
| `publish` | Bidirectional | Publish data to a channel |
| `ping` / `pong` | Bidirectional | Heartbeat keep-alive |
| `welcome` | Server → Client | Initial connection info |
| `error` | Server → Client | Error notification |
| `dashboard` | Server → Client | Dashboard snapshot broadcast |

### Channel Architecture

```
dashboard    → Real-time system metrics and status updates
alerts       → Alert notifications (active, acknowledged, resolved)
telemetry    → Raw sensor and pipeline data
health       → Health check reports
performance  → Performance metrics and regressions
```

### Auto-Reconnect (Client)

The WebSocket client implements exponential backoff reconnection:

```
Delay = min(baseDelay × 2^attempt + jitter, maxDelay)
baseDelay = 1000ms
maxDelay  = 30000ms
jitter    = random(0, 1000ms)
```

Messages sent during disconnection are queued (up to `maxQueueSize`) and
flushed automatically upon reconnection.

---

## Alert Rules Reference

### Rule Types

| Type | Description | Condition Fields |
|------|-------------|------------------|
| `threshold` | Simple value comparison | `threshold`, `operator` (gt/lt/gte/lte/eq) |
| `anomaly` | Statistical outlier detection | `zScore` (default: 2.5), `windowMs` |
| `composite` | Multiple conditions combined | `conditions[]`, `logic` (and/or) |
| `manual` | Programmatically created | N/A |

### Example Rules

```javascript
// Threshold rule: CPU > 90% for 30 seconds
alertManager.addRule({
  name: 'High CPU Usage',
  type: 'threshold',
  severity: 'warning',
  source: 'system',
  metric: 'cpu_percent',
  condition: { threshold: 90, operator: 'gt', duration: 30000 },
  cooldownMs: 60000,
});

// Anomaly rule: Unusual latency spike
alertManager.addRule({
  name: 'Latency Anomaly',
  type: 'anomaly',
  severity: 'error',
  source: 'latencyMonitor',
  metric: 'p99_latency',
  condition: { zScore: 3.0, windowMs: 60000 },
  cooldownMs: 120000,
});
```

### Alert Lifecycle

```
active → acknowledged → resolved
  │                         ↑
  └──→ escalated ──→ acknowledged → resolved
  │
  └──→ expired (auto after 1hr)
```

---

## Performance Metrics Guide

### Key Metrics

| Metric | Source | Unit | Target | Warning | Critical |
|--------|--------|------|--------|---------|----------|
| FPS | PerformanceTracker | frames/s | 30 | <20 | <10 |
| Memory Usage | PerformanceTracker | % | <70% | >80% | >95% |
| CPU Usage | PerformanceTracker | % | <50% | >85% | >95% |
| Event Loop Lag | PerformanceTracker | ms | <10 | >50 | >100 |
| GC Pressure | PerformanceTracker | 0-1 | <0.1 | >0.3 | >0.5 |
| P50 Latency | LatencyMonitor | ms | varies | budget×0.8 | budget×0.95 |
| P99 Latency | LatencyMonitor | ms | varies | budget×0.8 | budget×0.95 |
| Health Score | HealthMonitor | 0-1 | >0.9 | <0.7 | <0.4 |

### Latency Budgets (Default)

| Pipeline Stage | Budget (ms) | Description |
|----------------|-------------|-------------|
| Perception | 50 | Sensor processing → object detection |
| Planning | 30 | Path planning → trajectory generation |
| Control | 20 | Control commands → actuation |
| Communication | 10 | V2X / cloud messaging |
| Full Pipeline | 150 | End-to-end sensor → actuator |

### Regression Detection

The PerformanceTracker uses baseline comparison to detect regressions:

- **Window size**: 30 samples (configurable)
- **Threshold**: 15% degradation (configurable)
- **Direction**: FPS/throughput (lower = worse), others (higher = worse)
- **Events**: `performance:regression` with metric details

---

## Event Reference

### RealtimeMonitor Events

| Event | Payload | Description |
|-------|---------|-------------|
| `started` | `{ timestamp }` | Monitor started |
| `stopped` | `{ timestamp }` | Monitor stopped |
| `state:change` | `{ oldState, newState }` | State transition |
| `dashboard:update` | `DashboardSnapshot` | Periodic dashboard push |
| `data:ingest` | `DataPoint` | Data point ingested |
| `alert:new` | `Alert` | New alert created |
| `alert:critical` | `Alert` | Critical-level alert |
| `emergency` | `Alert` | Emergency mode triggered |

### HealthMonitor Events

| Event | Payload | Description |
|-------|---------|-------------|
| `health:check` | `HealthReport` | Periodic health check |
| `health:degraded` | `HealthReport` | Health degraded |
| `health:recovered` | `HealthReport` | Health recovered |
| `health:critical` | `HealthReport` | Health critical |
| `recovery:attempted` | `{ serviceName, success }` | Auto-recovery result |

---

## Configuration Reference

```javascript
const monitor = new RealtimeMonitor({
  tickIntervalMs: 100,           // Main loop tick (ms)
  dashboardPushIntervalMs: 200,  // Dashboard push interval
  healthCheckIntervalMs: 1000,   // Health check interval
  latencySampleIntervalMs: 50,   // Latency sampling interval
  enableWebSocketServer: true,   // Enable WS server
  wsServerPort: 8090,            // WS server port
  enableCloudUpload: false,      // Upload to cloud
  cloudEndpoint: '',             // Cloud WS endpoint
  maxDataBufferSize: 10000,      // Max data points in buffer
  alertThrottleMs: 5000,         // Alert dedup throttle
  moduleConfig: {                // Per-module overrides
    healthMonitor: { degradedThreshold: 0.6 },
    alertManager: { throttleMs: 3000 },
  },
});
```

---

## License

Proprietary — Autonomous Vehicle Control System Project
