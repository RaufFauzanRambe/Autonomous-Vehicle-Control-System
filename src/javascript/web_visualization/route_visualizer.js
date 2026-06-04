/**
 * @fileoverview Route Visualizer Module - Planned route polyline rendering, completed route
 * segments, waypoint markers, turn-by-turn indicators, route color coding by speed,
 * alternative route display, and route statistics overlay.
 *
 * @module route_visualizer
 * @version 2.0.0
 * @author Autonomous Vehicle Control System Team
 */

// ============================================================
// Waypoint and Route Data Structures
// ============================================================

/**
 * Represents a single waypoint along a route.
 * @class Waypoint
 */
export class Waypoint {
  /**
   * @param {Object} config - Waypoint configuration
   * @param {number} config.lat - Latitude
   * @param {number} config.lng - Longitude
   * @param {number} [config.elevation=0] - Elevation in meters
   * @param {string} [config.name=''] - Waypoint name
   * @param {'start'|'end'|'waypoint'|'turn'} [config.type='waypoint'] - Waypoint type
   * @param {number} [config.speedLimit] - Speed limit at this waypoint (m/s)
   * @param {string} [config.turnDirection] - Turn direction ('left','right','straight','uturn')
   * @param {number} [config.turnAngle] - Turn angle in degrees
   * @param {number} [config.distanceFromStart=0] - Cumulative distance from route start
   * @param {number} [config.eta] - Estimated time of arrival (Unix timestamp)
   */
  constructor(config) {
    this.lat = config.lat;
    this.lng = config.lng;
    this.elevation = config.elevation ?? 0;
    this.name = config.name ?? '';
    this.type = config.type ?? 'waypoint';
    this.speedLimit = config.speedLimit ?? null;
    this.turnDirection = config.turnDirection ?? null;
    this.turnAngle = config.turnAngle ?? 0;
    this.distanceFromStart = config.distanceFromStart ?? 0;
    this.eta = config.eta ?? null;
    this.reached = false;
  }

  /**
   * Calculate distance to another waypoint using Haversine formula.
   * @param {Waypoint} other - Target waypoint
   * @returns {number} Distance in meters
   */
  distanceTo(other) {
    const R = 6371000;
    const dLat = ((other.lat - this.lat) * Math.PI) / 180;
    const dLon = ((other.lng - this.lng) * Math.PI) / 180;
    const a = Math.sin(dLat / 2) ** 2
      + Math.cos((this.lat * Math.PI) / 180) * Math.cos((other.lat * Math.PI) / 180)
      * Math.sin(dLon / 2) ** 2;
    return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  }

  /**
   * Calculate bearing to another waypoint.
   * @param {Waypoint} other - Target waypoint
   * @returns {number} Bearing in degrees (0=North, clockwise)
   */
  bearingTo(other) {
    const dLon = ((other.lng - this.lng) * Math.PI) / 180;
    const lat1 = (this.lat * Math.PI) / 180;
    const lat2 = (other.lat * Math.PI) / 180;
    const y = Math.sin(dLon) * Math.cos(lat2);
    const x = Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(dLon);
    return ((Math.atan2(y, x) * 180) / Math.PI + 360) % 360;
  }
}

/**
 * Represents a route segment between two waypoints with associated data.
 * @class RouteSegment
 */
export class RouteSegment {
  /**
   * @param {Waypoint} start - Start waypoint
   * @param {Waypoint} end - End waypoint
   */
  constructor(start, end) {
    this.start = start;
    this.end = end;
    /** @type {number} Distance in meters */
    this.distance = start.distanceTo(end);
    /** @type {number} Bearing in degrees */
    this.bearing = start.bearingTo(end);
    /** @type {number} Average speed for color coding (m/s) */
    this.averageSpeed = 0;
    /** @type {Array<{lat: number, lng: number}>} Intermediate shape points */
    this.shapePoints = [];
    /** @type {'planned'|'active'|'completed'|'alternative'} */
    this.status = 'planned';
  }

  /**
   * Set shape points for curved route rendering.
   * @param {Array<{lat: number, lng: number}>} points - Intermediate points
   */
  setShapePoints(points) {
    this.shapePoints = points;
  }

  /**
   * Get all points for rendering this segment.
   * @returns {Array<{lat: number, lng: number}>}
   */
  getAllPoints() {
    const points = [{ lat: this.start.lat, lng: this.start.lng }];
    for (const p of this.shapePoints) {
      points.push(p);
    }
    points.push({ lat: this.end.lat, lng: this.end.lng });
    return points;
  }
}

// ============================================================
// Speed Color Mapping
// ============================================================

/**
 * Map speed values to colors for route visualization.
 * @param {number} speed - Speed in m/s
 * @param {number} [maxSpeed=30] - Maximum speed for gradient scaling
 * @returns {string} CSS color string
 */
export function speedToColor(speed, maxSpeed = 30) {
  const ratio = Math.min(1, Math.max(0, speed / maxSpeed));
  // Green (slow) → Yellow (medium) → Red (fast)
  const r = ratio < 0.5 ? Math.round(ratio * 2 * 255) : 255;
  const g = ratio < 0.5 ? 255 : Math.round((1 - (ratio - 0.5) * 2) * 255);
  const b = 0;
  return `rgb(${r},${g},${b})`;
}

// ============================================================
// Route Statistics Overlay
// ============================================================

/**
 * Computes and renders route statistics.
 * @class RouteStatistics
 */
export class RouteStatistics {
  /**
   * @param {Array<RouteSegment>} segments - Route segments
   */
  constructor(segments) {
    this.segments = segments;
    this._stats = null;
  }

  /**
   * Compute route statistics.
   * @returns {{totalDistance: number, totalDuration: number, avgSpeed: number, maxSpeed: number,
   *           minSpeed: number, elevationGain: number, elevationLoss: number, turnCount: number}}
   */
  compute() {
    let totalDistance = 0;
    let totalDuration = 0;
    let maxSpeed = 0;
    let minSpeed = Infinity;
    let elevationGain = 0;
    let elevationLoss = 0;
    let turnCount = 0;

    for (const seg of this.segments) {
      totalDistance += seg.distance;
      if (seg.averageSpeed > 0) {
        totalDuration += seg.distance / seg.averageSpeed;
      }
      maxSpeed = Math.max(maxSpeed, seg.averageSpeed);
      if (seg.averageSpeed > 0) {
        minSpeed = Math.min(minSpeed, seg.averageSpeed);
      }
      const elevDiff = seg.end.elevation - seg.start.elevation;
      if (elevDiff > 0) elevationGain += elevDiff;
      else elevationLoss += Math.abs(elevDiff);
      if (seg.end.turnDirection && seg.end.turnDirection !== 'straight') {
        turnCount++;
      }
    }

    this._stats = {
      totalDistance,
      totalDuration,
      avgSpeed: totalDuration > 0 ? totalDistance / totalDuration : 0,
      maxSpeed,
      minSpeed: minSpeed === Infinity ? 0 : minSpeed,
      elevationGain,
      elevationLoss,
      turnCount,
    };
    return this._stats;
  }

  /**
   * Render statistics overlay to a canvas context.
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {number} x - X position
   * @param {number} y - Y position
   * @param {number} width - Panel width
   */
  render(ctx, x, y, width) {
    if (!this._stats) this.compute();
    const s = this._stats;
    const padding = 12;
    const lineHeight = 20;
    const lines = [
      `Distance: ${(s.totalDistance / 1000).toFixed(1)} km`,
      `Duration: ${Math.floor(s.totalDuration / 60)} min`,
      `Avg Speed: ${(s.avgSpeed * 3.6).toFixed(1)} km/h`,
      `Max Speed: ${(s.maxSpeed * 3.6).toFixed(1)} km/h`,
      `Elevation +${s.elevationGain.toFixed(0)}m / -${s.elevationLoss.toFixed(0)}m`,
      `Turns: ${s.turnCount}`,
    ];

    const height = lines.length * lineHeight + padding * 2;
    ctx.fillStyle = 'rgba(15, 23, 42, 0.85)';
    ctx.beginPath();
    ctx.roundRect(x, y, width, height, 8);
    ctx.fill();

    ctx.font = '13px Inter, monospace';
    ctx.fillStyle = '#e2e8f0';
    ctx.textAlign = 'left';
    for (let i = 0; i < lines.length; i++) {
      ctx.fillText(lines[i], x + padding, y + padding + (i + 1) * lineHeight - 4);
    }
  }
}

// ============================================================
// Turn Indicator Renderer
// ============================================================

/**
 * Renders turn-by-turn indicators on the map.
 * @class TurnIndicatorRenderer
 */
export class TurnIndicatorRenderer {
  /**
   * @param {Object} [options={}] - Rendering options
   * @param {number} [options.size=28] - Indicator size in pixels
   * @param {string} [options.bgColor='#1e293b'] - Background color
   * @param {string} [options.arrowColor='#f8fafc'] - Arrow color
   */
  constructor(options = {}) {
    this.size = options.size ?? 28;
    this.bgColor = options.bgColor ?? '#1e293b';
    this.arrowColor = options.arrowColor ?? '#f8fafc';
  }

  /**
   * Render a turn indicator at a given position.
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {number} cx - Center X pixel
   * @param {number} cy - Center Y pixel
   * @param {string} direction - Turn direction
   * @param {number} [angle=0] - Turn angle in degrees
   */
  render(ctx, cx, cy, direction, angle = 0) {
    const r = this.size / 2;
    ctx.save();
    ctx.translate(cx, cy);

    // Background circle
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fillStyle = this.bgColor;
    ctx.fill();
    ctx.strokeStyle = '#475569';
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Arrow
    ctx.strokeStyle = this.arrowColor;
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const arrowLen = r * 0.65;
    switch (direction) {
      case 'left':
        ctx.beginPath();
        ctx.moveTo(arrowLen * 0.3, -arrowLen * 0.3);
        ctx.lineTo(-arrowLen * 0.4, 0);
        ctx.lineTo(arrowLen * 0.3, arrowLen * 0.3);
        ctx.stroke();
        break;
      case 'right':
        ctx.beginPath();
        ctx.moveTo(-arrowLen * 0.3, -arrowLen * 0.3);
        ctx.lineTo(arrowLen * 0.4, 0);
        ctx.lineTo(-arrowLen * 0.3, arrowLen * 0.3);
        ctx.stroke();
        break;
      case 'uturn':
        ctx.beginPath();
        ctx.arc(0, -arrowLen * 0.1, arrowLen * 0.35, -Math.PI * 0.8, Math.PI * 0.3);
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(arrowLen * 0.25, arrowLen * 0.1);
        ctx.lineTo(arrowLen * 0.1, -arrowLen * 0.2);
        ctx.stroke();
        break;
      case 'straight':
      default:
        ctx.beginPath();
        ctx.moveTo(0, arrowLen * 0.4);
        ctx.lineTo(0, -arrowLen * 0.4);
        ctx.moveTo(-arrowLen * 0.2, -arrowLen * 0.15);
        ctx.lineTo(0, -arrowLen * 0.4);
        ctx.lineTo(arrowLen * 0.2, -arrowLen * 0.15);
        ctx.stroke();
        break;
    }

    ctx.restore();
  }
}

// ============================================================
// Main Route Visualizer
// ============================================================

/**
 * Renders planned/completed routes on a canvas with waypoint markers,
 * turn-by-turn indicators, speed-based coloring, and alternative routes.
 * @class RouteVisualizer
 */
export class RouteVisualizer {
  /**
   * @param {HTMLCanvasElement} canvas - Target canvas element
   * @param {Object} [options={}] - Visualization options
   * @param {Function} [options.projectFn] - Lat/lng to pixel projection function
   * @param {boolean} [options.showSpeedColors=true] - Color route by speed
   * @param {boolean} [options.showTurnIndicators=true] - Show turn indicators
   * @param {boolean} [options.showStats=true] - Show statistics overlay
   * @param {number} [options.maxSpeed=30] - Maximum speed for color scaling
   * @param {number} [options.lineWidth=4] - Route line width
   */
  constructor(canvas, options = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError('RouteVisualizer requires an HTMLCanvasElement');
    }
    this._canvas = canvas;
    this._ctx = canvas.getContext('2d');
    this._projectFn = options.projectFn ?? ((lat, lng) => ({ x: lng, y: -lat }));
    this._showSpeedColors = options.showSpeedColors ?? true;
    this._showTurnIndicators = options.showTurnIndicators ?? true;
    this._showStats = options.showStats ?? true;
    this._maxSpeed = options.maxSpeed ?? 30;
    this._lineWidth = options.lineWidth ?? 4;

    /** @type {Array<RouteSegment>} */
    this.primaryRoute = [];
    /** @type {Array<Array<RouteSegment>>} */
    this.alternativeRoutes = [];
    /** @type {RouteStatistics|null} */
    this._statistics = null;
    this._turnRenderer = new TurnIndicatorRenderer();
    /** @type {number} - Progress index for completed segments */
    this._completedUpTo = -1;
  }

  /**
   * Set the primary route from an array of waypoints.
   * @param {Array<Waypoint>} waypoints - Ordered waypoints
   * @param {Array<number>} [speeds] - Speed at each segment
   */
  setRoute(waypoints, speeds = []) {
    if (waypoints.length < 2) {
      throw new Error('Route requires at least 2 waypoints');
    }
    this.primaryRoute = [];
    let cumDistance = 0;
    for (let i = 0; i < waypoints.length - 1; i++) {
      const seg = new RouteSegment(waypoints[i], waypoints[i + 1]);
      seg.averageSpeed = speeds[i] ?? 0;
      cumDistance += seg.distance;
      seg.end.distanceFromStart = cumDistance;
      this.primaryRoute.push(seg);
    }
    this._statistics = new RouteStatistics(this.primaryRoute);
    this._completedUpTo = -1;
  }

  /**
   * Add an alternative route.
   * @param {Array<Waypoint>} waypoints - Alternative route waypoints
   * @param {Array<number>} [speeds] - Speed at each segment
   */
  addAlternativeRoute(waypoints, speeds = []) {
    if (waypoints.length < 2) return;
    const altRoute = [];
    for (let i = 0; i < waypoints.length - 1; i++) {
      const seg = new RouteSegment(waypoints[i], waypoints[i + 1]);
      seg.averageSpeed = speeds[i] ?? 0;
      seg.status = 'alternative';
      altRoute.push(seg);
    }
    this.alternativeRoutes.push(altRoute);
  }

  /**
   * Mark segments up to a given index as completed.
   * @param {number} segmentIndex - Last completed segment index
   */
  markCompleted(segmentIndex) {
    for (let i = 0; i < this.primaryRoute.length; i++) {
      this.primaryRoute[i].status = i <= segmentIndex ? 'completed' : 'planned';
    }
    this._completedUpTo = segmentIndex;
    for (let i = 0; i <= segmentIndex && i < this.primaryRoute.length; i++) {
      this.primaryRoute[i].start.reached = true;
      this.primaryRoute[i].end.reached = i === segmentIndex ? false : true;
    }
  }

  /**
   * Set the projection function for converting lat/lng to pixel coordinates.
   * @param {Function} fn - Projection function (lat, lng) => {x, y}
   */
  setProjection(fn) {
    this._projectFn = fn;
  }

  /**
   * Render all route elements to the canvas.
   */
  render() {
    const ctx = this._ctx;
    const w = this._canvas.width / (window.devicePixelRatio || 1);
    const h = this._canvas.height / (window.devicePixelRatio || 1);
    ctx.clearRect(0, 0, w, h);

    // Alternative routes (rendered first, behind primary)
    for (const altRoute of this.alternativeRoutes) {
      this._renderRouteSegments(ctx, altRoute, {
        baseColor: '#94a3b8',
        lineWidth: this._lineWidth - 1,
        dashPattern: [8, 6],
        opacity: 0.6,
      });
    }

    // Primary route
    this._renderRouteSegments(ctx, this.primaryRoute, {
      baseColor: '#3b82f6',
      lineWidth: this._lineWidth,
      dashPattern: null,
      opacity: 1.0,
    });

    // Waypoint markers
    this._renderWaypointMarkers(ctx);

    // Turn indicators
    if (this._showTurnIndicators) {
      this._renderTurnIndicators(ctx);
    }

    // Statistics overlay
    if (this._showStats && this._statistics) {
      this._statistics.render(ctx, 10, 10, 220);
    }
  }

  /**
   * @private Render route segments with styling
   * @param {CanvasRenderingContext2D} ctx
   * @param {Array<RouteSegment>} segments
   * @param {Object} style
   */
  _renderRouteSegments(ctx, segments, style) {
    for (const seg of segments) {
      const points = seg.getAllPoints();
      if (points.length < 2) continue;

      ctx.save();
      ctx.globalAlpha = style.opacity;

      if (style.dashPattern) {
        ctx.setLineDash(style.dashPattern);
      }

      ctx.lineWidth = style.lineWidth;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';

      if (this._showSpeedColors && seg.averageSpeed > 0) {
        ctx.strokeStyle = speedToColor(seg.averageSpeed, this._maxSpeed);
      } else if (seg.status === 'completed') {
        ctx.strokeStyle = '#22c55e';
      } else {
        ctx.strokeStyle = style.baseColor;
      }

      // Draw completed segment with dashed overlay
      ctx.beginPath();
      const p0 = this._projectFn(points[0].lat, points[0].lng);
      ctx.moveTo(p0.x, p0.y);
      for (let i = 1; i < points.length; i++) {
        const p = this._projectFn(points[i].lat, points[i].lng);
        ctx.lineTo(p.x, p.y);
      }
      ctx.stroke();

      // Completed segment overlay
      if (seg.status === 'completed') {
        ctx.strokeStyle = 'rgba(34, 197, 94, 0.4)';
        ctx.lineWidth = style.lineWidth + 2;
        ctx.setLineDash([]);
        ctx.beginPath();
        ctx.moveTo(p0.x, p0.y);
        for (let i = 1; i < points.length; i++) {
          const p = this._projectFn(points[i].lat, points[i].lng);
          ctx.lineTo(p.x, p.y);
        }
        ctx.stroke();
      }

      ctx.restore();
    }
  }

  /**
   * @private Render waypoint markers
   * @param {CanvasRenderingContext2D} ctx
   */
  _renderWaypointMarkers(ctx) {
    const renderedWaypoints = new Set();

    for (const seg of this.primaryRoute) {
      for (const wp of [seg.start, seg.end]) {
        const key = `${wp.lat.toFixed(6)},${wp.lng.toFixed(6)}`;
        if (renderedWaypoints.has(key)) continue;
        renderedWaypoints.add(key);

        const pos = this._projectFn(wp.lat, wp.lng);
        const r = wp.type === 'start' || wp.type === 'end' ? 8 : 5;

        ctx.save();
        // Outer ring
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r + 2, 0, Math.PI * 2);
        ctx.fillStyle = '#ffffff';
        ctx.fill();

        // Inner circle
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
        const colorMap = { start: '#22c55e', end: '#ef4444', turn: '#f59e0b', waypoint: '#3b82f6' };
        ctx.fillStyle = colorMap[wp.type] ?? '#3b82f6';
        ctx.fill();

        // Reached checkmark
        if (wp.reached) {
          ctx.beginPath();
          ctx.arc(pos.x, pos.y, r + 5, 0, Math.PI * 2);
          ctx.strokeStyle = '#22c55e';
          ctx.lineWidth = 2;
          ctx.stroke();
        }

        // Label
        if (wp.name) {
          ctx.font = '11px Inter, sans-serif';
          ctx.fillStyle = '#1e293b';
          ctx.textAlign = 'center';
          ctx.fillText(wp.name, pos.x, pos.y - r - 6);
        }

        ctx.restore();
      }
    }
  }

  /**
   * @private Render turn-by-turn indicators
   * @param {CanvasRenderingContext2D} ctx
   */
  _renderTurnIndicators(ctx) {
    for (const seg of this.primaryRoute) {
      const wp = seg.end;
      if (!wp.turnDirection || wp.turnDirection === 'straight') continue;
      const pos = this._projectFn(wp.lat, wp.lng);
      this._turnRenderer.render(ctx, pos.x, pos.y - 24, wp.turnDirection, wp.turnAngle);
    }
  }

  /**
   * Get current route statistics.
   * @returns {Object|null} Statistics object
   */
  getStatistics() {
    return this._statistics ? this._statistics.compute() : null;
  }

  /**
   * Find the nearest segment to a given coordinate.
   * @param {number} lat - Latitude
   * @param {number} lng - Longitude
   * @returns {{segment: RouteSegment, index: number, distance: number}|null}
   */
  findNearestSegment(lat, lng) {
    let best = null;
    let bestDist = Infinity;
    for (let i = 0; i < this.primaryRoute.length; i++) {
      const seg = this.primaryRoute[i];
      const dStart = Math.sqrt((seg.start.lat - lat) ** 2 + (seg.start.lng - lng) ** 2);
      const dEnd = Math.sqrt((seg.end.lat - lat) ** 2 + (seg.end.lng - lng) ** 2);
      const d = Math.min(dStart, dEnd);
      if (d < bestDist) {
        bestDist = d;
        best = { segment: seg, index: i, distance: d };
      }
    }
    return best;
  }

  /**
   * Clear all routes and reset state.
   */
  clear() {
    this.primaryRoute = [];
    this.alternativeRoutes = [];
    this._statistics = null;
    this._completedUpTo = -1;
  }
}

export default RouteVisualizer;
