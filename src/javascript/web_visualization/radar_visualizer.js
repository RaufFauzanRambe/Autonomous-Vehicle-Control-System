/**
 * @fileoverview Radar Visualizer Module - Polar plot rendering with target blips and trails,
 * range rings, bearing lines, Doppler velocity coloring, target tracking lines,
 * radar sweep animation, and configurable range/azimuth.
 *
 * @module radar_visualizer
 * @version 2.0.0
 * @author Autonomous Vehicle Control System Team
 */

// ============================================================
// Radar Target Data
// ============================================================

/**
 * Represents a tracked radar target with position, velocity, and history.
 * @class RadarTarget
 */
export class RadarTarget {
  /**
   * @param {Object} config
   * @param {string} config.id - Unique target identifier
   * @param {number} config.range - Range from radar in meters
   * @param {number} config.azimuth - Azimuth angle in degrees (0=North, clockwise)
   * @param {number} [config.elevation=0] - Elevation angle in degrees
   * @param {number} [config.radialVelocity=0] - Radial/Doppler velocity in m/s
   * @param {number} [config.rcs=1.0] - Radar cross-section estimate
   * @param {'vehicle'|'pedestrian'|'cyclist'|'unknown'} [config.type='unknown'] - Classification
   * @param {number} [config.confidence=0.5] - Detection confidence [0, 1]
   */
  constructor(config) {
    this.id = config.id;
    this.range = config.range;
    this.azimuth = config.azimuth;
    this.elevation = config.elevation ?? 0;
    this.radialVelocity = config.radialVelocity ?? 0;
    this.rcs = config.rcs ?? 1.0;
    this.type = config.type ?? 'unknown';
    this.confidence = config.confidence ?? 0.5;
    this.timestamp = Date.now();
    /** @type {Array<{range: number, azimuth: number, time: number}>} */
    this.trail = [];
    this.maxTrailLength = 20;
    this.active = true;
  }

  /**
   * Update target position and record trail.
   * @param {number} range - New range in meters
   * @param {number} azimuth - New azimuth in degrees
   */
  update(range, azimuth) {
    this.trail.push({ range: this.range, azimuth: this.azimuth, time: this.timestamp });
    if (this.trail.length > this.maxTrailLength) {
      this.trail.shift();
    }
    this.range = range;
    this.azimuth = azimuth;
    this.timestamp = Date.now();
  }

  /**
   * Get Cartesian position relative to radar.
   * @returns {{x: number, y: number}} Position in meters
   */
  toCartesian() {
    const rad = (this.azimuth * Math.PI) / 180;
    return {
      x: this.range * Math.sin(rad),
      y: this.range * Math.cos(rad),
    };
  }
}

// ============================================================
// Doppler Velocity Color Mapping
// ============================================================

/**
 * Map Doppler radial velocity to a color for visualization.
 * Positive = approaching (warm colors), Negative = receding (cool colors).
 * @param {number} velocity - Radial velocity in m/s
 * @param {number} [maxVelocity=30] - Maximum velocity for scaling
 * @returns {string} CSS color string
 */
export function dopplerColor(velocity, maxVelocity = 30) {
  const normalized = Math.max(-1, Math.min(1, velocity / maxVelocity));
  if (normalized > 0) {
    // Approaching: green → yellow → red
    const r = Math.round(255 * Math.min(1, normalized * 2));
    const g = Math.round(255 * Math.max(0, 1 - normalized));
    return `rgb(${r},${g},0)`;
  } else {
    // Receding: cyan → blue → purple
    const t = Math.abs(normalized);
    const b = Math.round(255 * Math.min(1, t * 1.5 + 0.3));
    const g = Math.round(255 * Math.max(0, 0.8 - t * 1.5));
    return `rgb(80,${g},${b})`;
  }
}

// ============================================================
// Radar Sweep Animation
// ============================================================

/**
 * Manages the radar sweep line animation state.
 * @class RadarSweep
 */
export class RadarSweep {
  /**
   * @param {Object} [options={}] - Sweep options
   * @param {number} [options.rpm=12] - Rotations per minute
   * @param {string} [options.color='#22c55e'] - Sweep line color
   * @param {number} [options.trailLength=30] - Sweep trail width in degrees
   */
  constructor(options = {}) {
    this.rpm = options.rpm ?? 12;
    this.color = options.color ?? '#22c55e';
    this.trailLength = options.trailLength ?? 30;
    /** @type {number} Current sweep angle in degrees */
    this.angle = 0;
    this._lastTime = performance.now();
  }

  /**
   * Update sweep angle based on elapsed time.
   * @param {number} [deltaMs] - Elapsed time in milliseconds
   */
  update(deltaMs) {
    const dt = deltaMs ?? (performance.now() - this._lastTime);
    this._lastTime = performance.now();
    const degreesPerMs = (this.rpm * 360) / 60000;
    this.angle = (this.angle + degreesPerMs * dt) % 360;
  }
}

// ============================================================
// Radar Visualizer
// ============================================================

/**
 * Renders a radar display with polar plot, targets, sweep animation,
 * range rings, and bearing lines on a Canvas2D element.
 * @class RadarVisualizer
 */
export class RadarVisualizer {
  /**
   * @param {HTMLCanvasElement} canvas - Target canvas element
   * @param {Object} [options={}] - Visualization options
   * @param {number} [options.maxRange=200] - Maximum radar range in meters
   * @param {number} [options.rangeStep=50] - Range ring interval in meters
   * @param {number} [options.bearingStep=30] - Bearing line interval in degrees
   * @param {boolean} [options.showSweep=true] - Show sweep animation
   * @param {boolean} [options.showTrails=true] - Show target trails
   * @param {boolean} [options.showTrackingLines=true] - Show tracking prediction lines
   * @param {'doppler'|'type'|'confidence'} [options.colorMode='doppler'] - Target color mode
   */
  constructor(canvas, options = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError('RadarVisualizer requires an HTMLCanvasElement');
    }

    this._canvas = canvas;
    this._ctx = canvas.getContext('2d');
    this._maxRange = options.maxRange ?? 200;
    this._rangeStep = options.rangeStep ?? 50;
    this._bearingStep = options.bearingStep ?? 30;
    this._showSweep = options.showSweep ?? true;
    this._showTrails = options.showTrails ?? true;
    this._showTrackingLines = options.showTrackingLines ?? true;
    this._colorMode = options.colorMode ?? 'doppler';

    /** @type {Map<string, RadarTarget>} */
    this._targets = new Map();
    this._sweep = new RadarSweep();

    this._centerX = 0;
    this._centerY = 0;
    this._radius = 0;

    this._resize();
    this._animFrameId = null;
    this._renderLoop();

    window.addEventListener('resize', () => this._resize());
  }

  /** @private Recalculate center and radius */
  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this._canvas.getBoundingClientRect();
    this._canvas.width = rect.width * dpr;
    this._canvas.height = rect.height * dpr;
    this._ctx.scale(dpr, dpr);
    this._canvasW = rect.width;
    this._canvasH = rect.height;
    this._centerX = rect.width / 2;
    this._centerY = rect.height / 2;
    this._radius = Math.min(this._centerX, this._centerY) - 20;
  }

  /**
   * Convert polar coordinates to canvas pixel coordinates.
   * @param {number} range - Range in meters
   * @param {number} azimuth - Azimuth in degrees (0=North, clockwise)
   * @returns {{x: number, y: number}}
   */
  polarToPixel(range, azimuth) {
    const normalizedRange = range / this._maxRange;
    const rad = (azimuth - 90) * Math.PI / 180; // -90 to align North upward
    return {
      x: this._centerX + normalizedRange * this._radius * Math.cos(rad),
      y: this._centerY + normalizedRange * this._radius * Math.sin(rad),
    };
  }

  /**
   * Add or update a radar target.
   * @param {RadarTarget} target - Target to add/update
   */
  addTarget(target) {
    if (!(target instanceof RadarTarget)) {
      throw new TypeError('Expected a RadarTarget instance');
    }
    const existing = this._targets.get(target.id);
    if (existing) {
      existing.update(target.range, target.azimuth);
      existing.radialVelocity = target.radialVelocity;
      existing.rcs = target.rcs;
      existing.type = target.type;
      existing.confidence = target.confidence;
      existing.elevation = target.elevation;
    } else {
      this._targets.set(target.id, target);
    }
  }

  /**
   * Remove a target by ID.
   * @param {string} targetId - Target ID to remove
   */
  removeTarget(targetId) {
    this._targets.delete(targetId);
  }

  /**
   * Clear all targets.
   */
  clearTargets() {
    this._targets.clear();
  }

  /**
   * Set the maximum radar range.
   * @param {number} range - Maximum range in meters
   */
  setMaxRange(range) {
    this._maxRange = Math.max(10, range);
  }

  /**
   * Set the sweep rotation speed.
   * @param {number} rpm - Rotations per minute
   */
  setSweepSpeed(rpm) {
    this._sweep.rpm = Math.max(1, Math.min(60, rpm));
  }

  /** @private Main render loop */
  _renderLoop() {
    this._render();
    this._animFrameId = requestAnimationFrame(() => this._renderLoop());
  }

  /** @private Render all radar elements */
  _render() {
    const ctx = this._ctx;
    const w = this._canvasW;
    const h = this._canvasH;
    const cx = this._centerX;
    const cy = this._centerY;
    const r = this._radius;

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#0a0f1a';
    ctx.fillRect(0, 0, w, h);

    // Range rings
    this._drawRangeRings(ctx, cx, cy, r);

    // Bearing lines
    this._drawBearingLines(ctx, cx, cy, r);

    // Sweep
    if (this._showSweep) {
      this._sweep.update(16.67);
      this._drawSweep(ctx, cx, cy, r);
    }

    // Target trails
    if (this._showTrails) {
      this._drawTargetTrails(ctx);
    }

    // Tracking lines
    if (this._showTrackingLines) {
      this._drawTrackingLines(ctx);
    }

    // Target blips
    this._drawTargets(ctx);

    // Center marker
    ctx.beginPath();
    ctx.arc(cx, cy, 3, 0, Math.PI * 2);
    ctx.fillStyle = '#22c55e';
    ctx.fill();

    // Range scale label
    ctx.font = '10px Inter, monospace';
    ctx.fillStyle = '#64748b';
    ctx.textAlign = 'center';
    ctx.fillText(`${this._maxRange}m`, cx, cy + r + 14);
  }

  /** @private Draw concentric range rings */
  _drawRangeRings(ctx, cx, cy, r) {
    const steps = Math.floor(this._maxRange / this._rangeStep);
    for (let i = 1; i <= steps; i++) {
      const ringR = (i / steps) * r;
      ctx.beginPath();
      ctx.arc(cx, cy, ringR, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.15)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Range label
      ctx.font = '9px Inter, monospace';
      ctx.fillStyle = 'rgba(100, 116, 139, 0.7)';
      ctx.textAlign = 'left';
      ctx.fillText(`${i * this._rangeStep}m`, cx + 4, cy - ringR + 10);
    }
  }

  /** @private Draw bearing reference lines */
  _drawBearingLines(ctx, cx, cy, r) {
    for (let deg = 0; deg < 360; deg += this._bearingStep) {
      const rad = (deg - 90) * Math.PI / 180;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + r * Math.cos(rad), cy + r * Math.sin(rad));
      ctx.strokeStyle = 'rgba(34, 197, 94, 0.1)';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Bearing label
      const labelR = r + 12;
      ctx.font = '10px Inter, monospace';
      ctx.fillStyle = '#64748b';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(`${deg}°`, cx + labelR * Math.cos(rad), cy + labelR * Math.sin(rad));
    }

    // Cardinal labels
    const cardinals = [
      { label: 'N', deg: 0 }, { label: 'E', deg: 90 },
      { label: 'S', deg: 180 }, { label: 'W', deg: 270 },
    ];
    ctx.font = 'bold 12px Inter, sans-serif';
    ctx.fillStyle = '#94a3b8';
    for (const c of cardinals) {
      const rad = (c.deg - 90) * Math.PI / 180;
      const labelR = r + 24;
      ctx.fillText(c.label, cx + labelR * Math.cos(rad), cy + labelR * Math.sin(rad));
    }
  }

  /** @private Draw radar sweep line with trail */
  _drawSweep(ctx, cx, cy, r) {
    const sweepAngle = this._sweep.angle;

    // Sweep trail (fading arc)
    const trailStart = sweepAngle - this._sweep.trailLength;
    const steps = 20;
    for (let i = 0; i < steps; i++) {
      const t = i / steps;
      const startDeg = trailStart + t * this._sweep.trailLength;
      const endDeg = trailStart + (t + 1 / steps) * this._sweep.trailLength;
      const startRad = (startDeg - 90) * Math.PI / 180;
      const endRad = (endDeg - 90) * Math.PI / 180;

      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.arc(cx, cy, r, startRad, endRad);
      ctx.closePath();
      ctx.fillStyle = `rgba(34, 197, 94, ${t * 0.08})`;
      ctx.fill();
    }

    // Main sweep line
    const sweepRad = (sweepAngle - 90) * Math.PI / 180;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(cx + r * Math.cos(sweepRad), cy + r * Math.sin(sweepRad));
    ctx.strokeStyle = 'rgba(34, 197, 94, 0.7)';
    ctx.lineWidth = 2;
    ctx.stroke();
  }

  /** @private Draw target trails */
  _drawTargetTrails(ctx) {
    for (const target of this._targets.values()) {
      if (!target.active || target.trail.length < 2) continue;
      for (let i = 1; i < target.trail.length; i++) {
        const prev = this.polarToPixel(target.trail[i - 1].range, target.trail[i - 1].azimuth);
        const curr = this.polarToPixel(target.trail[i].range, target.trail[i].azimuth);
        const opacity = (i / target.trail.length) * 0.5;
        ctx.beginPath();
        ctx.moveTo(prev.x, prev.y);
        ctx.lineTo(curr.x, curr.y);
        ctx.strokeStyle = `rgba(148, 163, 184, ${opacity})`;
        ctx.lineWidth = 1;
        ctx.stroke();
      }
      // Line from last trail point to current position
      if (target.trail.length > 0) {
        const last = this.polarToPixel(target.trail[target.trail.length - 1].range, target.trail[target.trail.length - 1].azimuth);
        const curr = this.polarToPixel(target.range, target.azimuth);
        ctx.beginPath();
        ctx.moveTo(last.x, last.y);
        ctx.lineTo(curr.x, curr.y);
        ctx.strokeStyle = 'rgba(148, 163, 184, 0.6)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  /** @private Draw tracking prediction lines */
  _drawTrackingLines(ctx) {
    for (const target of this._targets.values()) {
      if (!target.active || target.trail.length < 3) continue;

      // Simple linear extrapolation
      const recent = target.trail.slice(-3);
      const avgVelocity = {
        range: (target.range - recent[0].range) / (recent.length),
        azimuth: (target.azimuth - recent[0].azimuth) / (recent.length),
      };

      const predictedRange = target.range + avgVelocity.range * 5;
      const predictedAzimuth = target.azimuth + avgVelocity.azimuth * 5;

      if (predictedRange > 0 && predictedRange <= this._maxRange) {
        const curr = this.polarToPixel(target.range, target.azimuth);
        const pred = this.polarToPixel(predictedRange, predictedAzimuth);
        ctx.beginPath();
        ctx.moveTo(curr.x, curr.y);
        ctx.lineTo(pred.x, pred.y);
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)';
        ctx.lineWidth = 1;
        ctx.setLineDash([4, 4]);
        ctx.stroke();
        ctx.setLineDash([]);

        // Predicted position marker
        ctx.beginPath();
        ctx.arc(pred.x, pred.y, 3, 0, Math.PI * 2);
        ctx.strokeStyle = 'rgba(251, 191, 36, 0.4)';
        ctx.lineWidth = 1;
        ctx.stroke();
      }
    }
  }

  /** @private Draw target blips */
  _drawTargets(ctx) {
    for (const target of this._targets.values()) {
      if (!target.active) continue;
      const pos = this.polarToPixel(target.range, target.azimuth);

      // Determine color
      let color;
      switch (this._colorMode) {
        case 'doppler':
          color = dopplerColor(target.radialVelocity);
          break;
        case 'type': {
          const typeColors = { vehicle: '#ef4444', pedestrian: '#f59e0b', cyclist: '#3b82f6', unknown: '#94a3b8' };
          color = typeColors[target.type] ?? '#94a3b8';
          break;
        }
        case 'confidence': {
          const g = Math.round(target.confidence * 255);
          color = `rgb(${255 - g},${g},80)`;
          break;
        }
        default:
          color = '#22c55e';
      }

      // Blip size based on RCS
      const blipSize = 3 + Math.min(5, target.rcs * 2);

      // Glow effect
      const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, blipSize * 3);
      gradient.addColorStop(0, color);
      gradient.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, blipSize * 3, 0, Math.PI * 2);
      ctx.fillStyle = gradient;
      ctx.fill();

      // Core blip
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, blipSize, 0, Math.PI * 2);
      ctx.fillStyle = color;
      ctx.fill();

      // ID label
      ctx.font = '9px Inter, monospace';
      ctx.fillStyle = '#e2e8f0';
      ctx.textAlign = 'left';
      ctx.fillText(target.id, pos.x + blipSize + 3, pos.y - 3);

      // Velocity label
      if (Math.abs(target.radialVelocity) > 0.1) {
        const velKmh = Math.abs(target.radialVelocity * 3.6).toFixed(0);
        const dir = target.radialVelocity > 0 ? '↑' : '↓';
        ctx.fillStyle = '#94a3b8';
        ctx.fillText(`${velKmh}${dir}`, pos.x + blipSize + 3, pos.y + 8);
      }
    }
  }

  /**
   * Get all current targets.
   * @returns {Array<RadarTarget>}
   */
  getTargets() {
    return Array.from(this._targets.values());
  }

  /**
   * Destroy the visualizer.
   */
  destroy() {
    if (this._animFrameId) {
      cancelAnimationFrame(this._animFrameId);
    }
    this._targets.clear();
  }
}

export default RadarVisualizer;
