/**
 * @fileoverview Graph Renderer Module - Canvas-based line/bar/gauge charts with real-time data feed,
 * auto-scaling axes, grid lines, legend, tooltip on hover, multiple series,
 * animation on data change, and responsive resize.
 *
 * @module graph_renderer
 * @version 2.0.0
 * @author Autonomous Vehicle Control System Team
 */

// ============================================================
// Chart Series
// ============================================================

/**
 * Represents a single data series for chart rendering.
 * @class ChartSeries
 */
export class ChartSeries {
  /**
   * @param {Object} config
   * @param {string} config.name - Series display name
   * @param {string} [config.color='#3b82f6'] - Series color
   * @param {'line'|'bar'|'area'} [config.type='line'] - Chart type
   * @param {number} [config.lineWidth=2] - Line width for line/area charts
   * @param {boolean} [config.fill=false] - Fill area under line
   * @param {number} [config.maxPoints=200] - Maximum data points to keep
   * @param {number} [config.minY] - Minimum Y value override
   * @param {number} [config.maxY] - Maximum Y value override
   */
  constructor(config) {
    this.name = config.name;
    this.color = config.color ?? '#3b82f6';
    this.type = config.type ?? 'line';
    this.lineWidth = config.lineWidth ?? 2;
    this.fill = config.fill ?? false;
    this.maxPoints = config.maxPoints ?? 200;
    this.minY = config.minY ?? null;
    this.maxY = config.maxY ?? null;

    /** @type {Array<{x: number, y: number}>} */
    this.data = [];
    /** @private @type {Array<{x: number, y: number}>} - Previous data for animation */
    this._prevData = [];
    /** @private @type {number} - Animation progress [0, 1] */
    this._animProgress = 1;
  }

  /**
   * Add a data point.
   * @param {number} x - X value (typically timestamp)
   * @param {number} y - Y value
   */
  addPoint(x, y) {
    this.data.push({ x, y });
    if (this.data.length > this.maxPoints) {
      this.data.shift();
    }
    this._animProgress = 0;
  }

  /**
   * Set all data points at once.
   * @param {Array<{x: number, y: number}>} points
   */
  setData(points) {
    this._prevData = [...this.data];
    this.data = points.slice(-this.maxPoints);
    this._animProgress = 0;
  }

  /**
   * Get the interpolated data for animation.
   * @param {number} t - Animation progress [0, 1]
   * @returns {Array<{x: number, y: number}>}
   */
  getAnimatedData(t) {
    if (this._prevData.length === 0 || t >= 1) {
      return this.data;
    }
    const maxLen = Math.max(this._prevData.length, this.data.length);
    const result = [];
    for (let i = 0; i < this.data.length; i++) {
      const prev = this._prevData[i] ?? this._prevData[this._prevData.length - 1] ?? this.data[i];
      result.push({
        x: this.data[i].x,
        y: prev.y + (this.data[i].y - prev.y) * t,
      });
    }
    return result;
  }

  /**
   * Clear all data.
   */
  clear() {
    this._prevData = [...this.data];
    this.data = [];
    this._animProgress = 0;
  }
}

// ============================================================
// Axis Configuration
// ============================================================

/**
 * Auto-scaling axis with grid line calculation.
 * @class Axis
 */
export class Axis {
  /**
   * @param {Object} [options={}] - Axis options
   * @param {'linear'|'time'} [options.type='linear'] - Axis scale type
   * @param {number} [options.min] - Minimum value (auto if undefined)
   * @param {number} [options.max] - Maximum value (auto if undefined)
   * @param {number} [options.gridLines=5] - Target number of grid lines
   * @param {string} [options.label=''] - Axis label
   * @param {string} [options.format] - Value format function
   */
  constructor(options = {}) {
    this.type = options.type ?? 'linear';
    this._minOverride = options.min;
    this._maxOverride = options.max;
    this.gridLines = options.gridLines ?? 5;
    this.label = options.label ?? '';
    this.format = options.format ?? ((v) => v.toFixed(1));
    /** @type {number} */
    this.min = 0;
    /** @type {number} */
    this.max = 100;
    /** @type {number} */
    this.step = 20;
  }

  /**
   * Compute nice axis bounds from data range.
   * @param {number} dataMin - Data minimum
   * @param {number} dataMax - Data maximum
   */
  computeScale(dataMin, dataMax) {
    const min = this._minOverride ?? dataMin;
    const max = this._maxOverride ?? dataMax;
    const range = max - min || 1;

    // Calculate nice step size
    const roughStep = range / this.gridLines;
    const magnitude = Math.pow(10, Math.floor(Math.log10(roughStep)));
    const residual = roughStep / magnitude;
    let niceStep;
    if (residual <= 1.5) niceStep = magnitude;
    else if (residual <= 3) niceStep = 2 * magnitude;
    else if (residual <= 7) niceStep = 5 * magnitude;
    else niceStep = 10 * magnitude;

    this.min = Math.floor(min / niceStep) * niceStep;
    this.max = Math.ceil(max / niceStep) * niceStep;
    this.step = niceStep;
  }

  /**
   * Get grid line values.
   * @returns {Array<number>}
   */
  getGridValues() {
    const values = [];
    for (let v = this.min; v <= this.max + this.step * 0.5; v += this.step) {
      values.push(Math.round(v * 1e10) / 1e10); // Avoid floating point drift
    }
    return values;
  }

  /**
   * Normalize a value to [0, 1] range.
   * @param {number} value - Input value
   * @returns {number} Normalized value
   */
  normalize(value) {
    const range = this.max - this.min;
    return range > 0 ? (value - this.min) / range : 0;
  }
}

// ============================================================
// Tooltip
// ============================================================

/**
 * Interactive tooltip that follows the mouse and displays data values.
 * @class Tooltip
 */
export class Tooltip {
  /**
   * @param {Object} [options={}] - Tooltip options
   * @param {string} [options.bgColor='rgba(15,23,42,0.9)'] - Background color
   * @param {string} [options.textColor='#e2e8f0'] - Text color
   * @param {string} [options.font='12px Inter, monospace'] - Font
   * @param {number} [options.padding=8] - Padding in pixels
   * @param {number} [options.offset=12] - Offset from cursor in pixels
   */
  constructor(options = {}) {
    this.bgColor = options.bgColor ?? 'rgba(15,23,42,0.9)';
    this.textColor = options.textColor ?? '#e2e8f0';
    this.font = options.font ?? '12px Inter, monospace';
    this.padding = options.padding ?? 8;
    this.offset = options.offset ?? 12;
    /** @type {{x: number, y: number, lines: Array<{label: string, value: string, color: string}>}|null} */
    this.data = null;
  }

  /**
   * Set tooltip data.
   * @param {number} x - Cursor X
   * @param {number} y - Cursor Y
   * @param {Array<{label: string, value: string, color: string}>} lines
   */
  setData(x, y, lines) {
    this.data = { x, y, lines };
  }

  /**
   * Clear tooltip.
   */
  clear() {
    this.data = null;
  }

  /**
   * Render the tooltip.
   * @param {CanvasRenderingContext2D} ctx
   */
  render(ctx) {
    if (!this.data || this.data.lines.length === 0) return;

    ctx.save();
    ctx.font = this.font;
    const { x, y, lines } = this.data;

    // Measure text
    const lineHeight = 18;
    let maxTextWidth = 0;
    for (const line of lines) {
      const text = `${line.label}: ${line.value}`;
      maxTextWidth = Math.max(maxTextWidth, ctx.measureText(text).width);
    }

    const boxW = maxTextWidth + this.padding * 2 + 12; // +12 for color dot
    const boxH = lines.length * lineHeight + this.padding * 2;

    // Position tooltip (keep on screen)
    let tx = x + this.offset;
    let ty = y - boxH / 2;
    const canvasW = ctx.canvas.width / (window.devicePixelRatio || 1);
    const canvasH = ctx.canvas.height / (window.devicePixelRatio || 1);
    if (tx + boxW > canvasW) tx = x - boxW - this.offset;
    if (ty < 0) ty = 4;
    if (ty + boxH > canvasH) ty = canvasH - boxH - 4;

    // Background
    ctx.fillStyle = this.bgColor;
    ctx.beginPath();
    ctx.roundRect(tx, ty, boxW, boxH, 6);
    ctx.fill();
    ctx.strokeStyle = 'rgba(148,163,184,0.3)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Text
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < lines.length; i++) {
      const line = lines[i];
      const ly = ty + this.padding + i * lineHeight + lineHeight / 2;

      // Color dot
      ctx.beginPath();
      ctx.arc(tx + this.padding + 4, ly, 4, 0, Math.PI * 2);
      ctx.fillStyle = line.color;
      ctx.fill();

      // Text
      ctx.fillStyle = this.textColor;
      ctx.fillText(`${line.label}: ${line.value}`, tx + this.padding + 14, ly);
    }

    ctx.restore();
  }
}

// ============================================================
// Gauge Renderer
// ============================================================

/**
 * Renders a gauge chart (speedometer style).
 * @class GaugeRenderer
 */
export class GaugeRenderer {
  /**
   * @param {Object} [options={}] - Gauge options
   * @param {number} [options.minAngle=225] - Start angle in degrees
   * @param {number} [options.maxAngle=-45] - End angle in degrees
   * @param {number} [options.minValue=0] - Minimum value
   * @param {number} [options.maxValue=100] - Maximum value
   * @param {string} [options.label=''] - Gauge label
   * @param {string} [options.unit=''] - Value unit
   * @param {Array<{value: number, color: string}>} [options.thresholds] - Color thresholds
   */
  constructor(options = {}) {
    this.minAngle = options.minAngle ?? 225;
    this.maxAngle = options.maxAngle ?? -45;
    this.minValue = options.minValue ?? 0;
    this.maxValue = options.maxValue ?? 100;
    this.label = options.label ?? '';
    this.unit = options.unit ?? '';
    this.thresholds = options.thresholds ?? [
      { value: 0.6, color: '#22c55e' },
      { value: 0.8, color: '#f59e0b' },
      { value: 1.0, color: '#ef4444' },
    ];
    /** @type {number} Current value */
    this.value = 0;
    /** @private @type {number} Animated display value */
    this._displayValue = 0;
  }

  /**
   * Set the current gauge value.
   * @param {number} value - New value
   */
  setValue(value) {
    this.value = Math.max(this.minValue, Math.min(this.maxValue, value));
  }

  /**
   * Render the gauge.
   * @param {CanvasRenderingContext2D} ctx
   * @param {number} cx - Center X
   * @param {number} cy - Center Y
   * @param {number} r - Gauge radius
   */
  render(ctx, cx, cy, r) {
    // Smooth animation
    this._displayValue += (this.value - this._displayValue) * 0.1;

    const startRad = (this.minAngle * Math.PI) / 180;
    const endRad = (this.maxAngle * Math.PI) / 180;
    const totalAngle = (this.minAngle - this.maxAngle) * Math.PI / 180;
    const normalized = (this._displayValue - this.minValue) / (this.maxValue - this.minValue);
    const valueAngle = startRad - normalized * totalAngle;

    // Background arc
    ctx.beginPath();
    ctx.arc(cx, cy, r, -endRad, -startRad);
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = r * 0.15;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Colored arc segments
    for (const threshold of this.thresholds) {
      const tStart = startRad - threshold.value * totalAngle;
      ctx.beginPath();
      ctx.arc(cx, cy, r, -startRad, -tStart);
      ctx.strokeStyle = threshold.color;
      ctx.lineWidth = r * 0.15;
      ctx.lineCap = 'round';
      ctx.stroke();
    }

    // Needle
    const needleLen = r * 0.8;
    ctx.beginPath();
    ctx.moveTo(cx, cy);
    ctx.lineTo(
      cx + needleLen * Math.cos(-valueAngle),
      cy + needleLen * Math.sin(-valueAngle),
    );
    ctx.strokeStyle = '#f8fafc';
    ctx.lineWidth = 2.5;
    ctx.lineCap = 'round';
    ctx.stroke();

    // Center circle
    ctx.beginPath();
    ctx.arc(cx, cy, r * 0.08, 0, Math.PI * 2);
    ctx.fillStyle = '#f8fafc';
    ctx.fill();

    // Value text
    ctx.font = `bold ${r * 0.3}px Inter, sans-serif`;
    ctx.fillStyle = '#f8fafc';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(this._displayValue.toFixed(0), cx, cy + r * 0.35);

    // Unit
    if (this.unit) {
      ctx.font = `${r * 0.15}px Inter, sans-serif`;
      ctx.fillStyle = '#94a3b8';
      ctx.fillText(this.unit, cx, cy + r * 0.5);
    }

    // Label
    if (this.label) {
      ctx.font = `${r * 0.12}px Inter, sans-serif`;
      ctx.fillStyle = '#64748b';
      ctx.fillText(this.label, cx, cy + r * 0.65);
    }
  }
}

// ============================================================
// Graph Renderer
// ============================================================

/**
 * Canvas-based chart renderer supporting line, bar, area, and gauge charts
 * with real-time data, auto-scaling, tooltips, and animations.
 * @class GraphRenderer
 */
export class GraphRenderer {
  /**
   * @param {HTMLCanvasElement} canvas - Target canvas
   * @param {Object} [options={}] - Renderer options
   * @param {{top: number, right: number, bottom: number, left: number}} [options.padding={top:20,right:20,bottom:40,left:60}]
   * @param {string} [options.bgColor='#0f172a'] - Chart background color
   * @param {string} [options.gridColor='rgba(148,163,184,0.1)'] - Grid line color
   * @param {string} [options.axisColor='#64748b'] - Axis text color
   * @param {boolean} [options.animate=true] - Animate data changes
   * @param {number} [options.animDuration=300] - Animation duration in ms
   */
  constructor(canvas, options = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError('GraphRenderer requires an HTMLCanvasElement');
    }

    this._canvas = canvas;
    this._ctx = canvas.getContext('2d');
    this._padding = options.padding ?? { top: 20, right: 20, bottom: 40, left: 60 };
    this._bgColor = options.bgColor ?? '#0f172a';
    this._gridColor = options.gridColor ?? 'rgba(148,163,184,0.1)';
    this._axisColor = options.axisColor ?? '#64748b';
    this._animate = options.animate ?? true;
    this._animDuration = options.animDuration ?? 300;

    /** @type {Array<ChartSeries>} */
    this._series = [];
    this._xAxis = new Axis({ type: 'linear' });
    this._yAxis = new Axis({ type: 'linear' });
    this._tooltip = new Tooltip();
    this._gauges = [];

    this._animStartTime = 0;
    this._isAnimating = false;
    this._animFrameId = null;

    this._resize();
    this._bindEvents();
  }

  /** @private Resize canvas */
  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this._canvas.getBoundingClientRect();
    this._canvas.width = rect.width * dpr;
    this._canvas.height = rect.height * dpr;
    this._ctx.scale(dpr, dpr);
    this._canvasW = rect.width;
    this._canvasH = rect.height;
    window.addEventListener('resize', () => {
      const r2 = this._canvas.getBoundingClientRect();
      this._canvas.width = r2.width * dpr;
      this._canvas.height = r2.height * dpr;
      this._ctx.scale(dpr, dpr);
      this._canvasW = r2.width;
      this._canvasH = r2.height;
    });
  }

  /** @private Bind mouse events for tooltip */
  _bindEvents() {
    this._canvas.addEventListener('mousemove', (e) => {
      const rect = this._canvas.getBoundingClientRect();
      const mx = e.clientX - rect.left;
      const my = e.clientY - rect.top;
      this._handleHover(mx, my);
    });
    this._canvas.addEventListener('mouseleave', () => {
      this._tooltip.clear();
    });
  }

  /** @private Handle hover for tooltip */
  _handleHover(mx, my) {
    const plotArea = this._getPlotArea();
    if (mx < plotArea.x || mx > plotArea.x + plotArea.w || my < plotArea.y || my > plotArea.y + plotArea.h) {
      this._tooltip.clear();
      return;
    }

    const normX = (mx - plotArea.x) / plotArea.w;
    const dataX = this._xAxis.min + normX * (this._xAxis.max - this._xAxis.min);

    const lines = [];
    for (const s of this._series) {
      if (s.data.length === 0) continue;
      // Find nearest data point
      let nearest = s.data[0];
      let nearestDist = Math.abs(s.data[0].x - dataX);
      for (const p of s.data) {
        const d = Math.abs(p.x - dataX);
        if (d < nearestDist) {
          nearestDist = d;
          nearest = p;
        }
      }
      lines.push({
        label: s.name,
        value: this._yAxis.format(nearest.y),
        color: s.color,
      });
    }

    if (lines.length > 0) {
      this._tooltip.setData(mx, my, lines);
    }
  }

  /** @private Get plot area dimensions */
  _getPlotArea() {
    const p = this._padding;
    return {
      x: p.left,
      y: p.top,
      w: this._canvasW - p.left - p.right,
      h: this._canvasH - p.top - p.bottom,
    };
  }

  /**
   * Add a data series.
   * @param {ChartSeries} series - Series to add
   */
  addSeries(series) {
    if (!(series instanceof ChartSeries)) {
      throw new TypeError('Expected a ChartSeries instance');
    }
    this._series.push(series);
  }

  /**
   * Add a gauge.
   * @param {GaugeRenderer} gauge - Gauge to add
   */
  addGauge(gauge) {
    this._gauges.push(gauge);
  }

  /**
   * Push real-time data to a series.
   * @param {number} seriesIndex - Series index
   * @param {number} x - X value
   * @param {number} y - Y value
   */
  pushData(seriesIndex, x, y) {
    if (seriesIndex >= 0 && seriesIndex < this._series.length) {
      this._series[seriesIndex].addPoint(x, y);
      if (!this._isAnimating) {
        this._startAnimation();
      }
    }
  }

  /** @private Start animation cycle */
  _startAnimation() {
    this._isAnimating = true;
    this._animStartTime = performance.now();
    this._tick();
  }

  /** @private Animation tick */
  _tick() {
    const elapsed = performance.now() - this._animStartTime;
    const t = Math.min(1, elapsed / this._animDuration);
    const easedT = t < 0.5 ? 2 * t * t : 1 - Math.pow(-2 * t + 2, 2) / 2;

    // Update series animation progress
    for (const s of this._series) {
      s._animProgress = easedT;
    }

    this.render();

    if (t < 1) {
      this._animFrameId = requestAnimationFrame(() => this._tick());
    } else {
      this._isAnimating = false;
      // Finalize
      for (const s of this._series) {
        s._prevData = [...s.data];
      }
    }
  }

  /**
   * Compute axis scales from all series data.
   * @private
   */
  _computeScales() {
    let xMin = Infinity, xMax = -Infinity, yMin = Infinity, yMax = -Infinity;

    for (const s of this._series) {
      if (s.data.length === 0) continue;
      for (const p of s.data) {
        if (p.x < xMin) xMin = p.x;
        if (p.x > xMax) xMax = p.x;
        if (p.y < yMin) yMin = p.y;
        if (p.y > yMax) yMax = p.y;
      }
      if (s.minY !== null) yMin = Math.min(yMin, s.minY);
      if (s.maxY !== null) yMax = Math.max(yMax, s.maxY);
    }

    if (xMin === Infinity) { xMin = 0; xMax = 100; }
    if (yMin === Infinity) { yMin = 0; yMax = 100; }

    this._xAxis.computeScale(xMin, xMax);
    this._yAxis.computeScale(yMin, yMax);
  }

  /**
   * Render the complete chart.
   */
  render() {
    const ctx = this._ctx;
    const w = this._canvasW;
    const h = this._canvasH;
    const plot = this._getPlotArea();

    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = this._bgColor;
    ctx.fillRect(0, 0, w, h);

    // Gauges (render in dedicated area if present)
    if (this._gauges.length > 0 && this._series.length === 0) {
      this._renderGauges(ctx, plot);
      return;
    }

    this._computeScales();

    // Grid lines
    this._drawGrid(ctx, plot);

    // Data series
    for (const series of this._series) {
      this._drawSeries(ctx, series, plot);
    }

    // Axis labels
    this._drawAxisLabels(ctx, plot);

    // Legend
    this._drawLegend(ctx);

    // Tooltip
    this._tooltip.render(ctx);
  }

  /** @private Draw grid lines */
  _drawGrid(ctx, plot) {
    ctx.strokeStyle = this._gridColor;
    ctx.lineWidth = 1;

    // Y grid
    const yValues = this._yAxis.getGridValues();
    for (const v of yValues) {
      const y = plot.y + plot.h - this._yAxis.normalize(v) * plot.h;
      ctx.beginPath();
      ctx.moveTo(plot.x, y);
      ctx.lineTo(plot.x + plot.w, y);
      ctx.stroke();
    }

    // X grid
    const xValues = this._xAxis.getGridValues();
    for (const v of xValues) {
      const x = plot.x + this._xAxis.normalize(v) * plot.w;
      ctx.beginPath();
      ctx.moveTo(x, plot.y);
      ctx.lineTo(x, plot.y + plot.h);
      ctx.stroke();
    }
  }

  /** @private Draw axis labels */
  _drawAxisLabels(ctx, plot) {
    ctx.font = '10px Inter, monospace';
    ctx.fillStyle = this._axisColor;

    // Y axis labels
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    const yValues = this._yAxis.getGridValues();
    for (const v of yValues) {
      const y = plot.y + plot.h - this._yAxis.normalize(v) * plot.h;
      ctx.fillText(this._yAxis.format(v), plot.x - 6, y);
    }

    // X axis labels
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    const xValues = this._xAxis.getGridValues();
    for (const v of xValues) {
      const x = plot.x + this._xAxis.normalize(v) * plot.w;
      ctx.fillText(this._xAxis.format(v), x, plot.y + plot.h + 6);
    }

    // Axis titles
    if (this._yAxis.label) {
      ctx.save();
      ctx.translate(14, plot.y + plot.h / 2);
      ctx.rotate(-Math.PI / 2);
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(this._yAxis.label, 0, 0);
      ctx.restore();
    }
    if (this._xAxis.label) {
      ctx.font = '11px Inter, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText(this._xAxis.label, plot.x + plot.w / 2, plot.y + plot.h + 24);
    }
  }

  /** @private Draw a data series */
  _drawSeries(ctx, series, plot) {
    const data = this._animate ? series.getAnimatedData(series._animProgress) : series.data;
    if (data.length < 1) return;

    const toPixelX = (v) => plot.x + this._xAxis.normalize(v) * plot.w;
    const toPixelY = (v) => plot.y + plot.h - this._yAxis.normalize(v) * plot.h;

    if (series.type === 'bar') {
      this._drawBarSeries(ctx, data, series, plot, toPixelX, toPixelY);
    } else {
      this._drawLineSeries(ctx, data, series, plot, toPixelX, toPixelY);
    }
  }

  /** @private Draw line/area series */
  _drawLineSeries(ctx, data, series, plot, toPixelX, toPixelY) {
    if (data.length < 2) return;
    ctx.save();

    ctx.beginPath();
    ctx.moveTo(toPixelX(data[0].x), toPixelY(data[0].y));
    for (let i = 1; i < data.length; i++) {
      ctx.lineTo(toPixelX(data[i].x), toPixelY(data[i].y));
    }

    if (series.fill || series.type === 'area') {
      ctx.lineTo(toPixelX(data[data.length - 1].x), plot.y + plot.h);
      ctx.lineTo(toPixelX(data[0].x), plot.y + plot.h);
      ctx.closePath();

      const gradient = ctx.createLinearGradient(0, plot.y, 0, plot.y + plot.h);
      const baseColor = series.color;
      gradient.addColorStop(0, baseColor + '60');
      gradient.addColorStop(1, baseColor + '05');
      ctx.fillStyle = gradient;
      ctx.fill();
    }

    // Line stroke
    ctx.beginPath();
    ctx.moveTo(toPixelX(data[0].x), toPixelY(data[0].y));
    for (let i = 1; i < data.length; i++) {
      ctx.lineTo(toPixelX(data[i].x), toPixelY(data[i].y));
    }
    ctx.strokeStyle = series.color;
    ctx.lineWidth = series.lineWidth;
    ctx.lineJoin = 'round';
    ctx.stroke();

    ctx.restore();
  }

  /** @private Draw bar series */
  _drawBarSeries(ctx, data, series, plot, toPixelX, toPixelY) {
    if (data.length === 0) return;
    const barWidth = Math.max(2, (plot.w / data.length) * 0.7);

    for (const point of data) {
      const x = toPixelX(point.x) - barWidth / 2;
      const y = toPixelY(point.y);
      const h = plot.y + plot.h - y;

      ctx.fillStyle = series.color;
      ctx.beginPath();
      ctx.roundRect(x, y, barWidth, h, [2, 2, 0, 0]);
      ctx.fill();
    }
  }

  /** @private Draw legend */
  _drawLegend(ctx) {
    if (this._series.length < 2) return;
    const legendX = this._canvasW - this._padding.right - 120;
    const legendY = this._padding.top + 4;

    ctx.fillStyle = 'rgba(15,23,42,0.7)';
    ctx.beginPath();
    ctx.roundRect(legendX, legendY, 116, this._series.length * 20 + 8, 4);
    ctx.fill();

    ctx.font = '11px Inter, sans-serif';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';

    for (let i = 0; i < this._series.length; i++) {
      const s = this._series[i];
      const y = legendY + 14 + i * 20;

      ctx.beginPath();
      ctx.arc(legendX + 12, y, 4, 0, Math.PI * 2);
      ctx.fillStyle = s.color;
      ctx.fill();

      ctx.fillStyle = '#e2e8f0';
      ctx.fillText(s.name, legendX + 22, y);
    }
  }

  /** @private Render gauges */
  _renderGauges(ctx, plot) {
    const n = this._gauges.length;
    if (n === 0) return;
    const cols = Math.ceil(Math.sqrt(n));
    const rows = Math.ceil(n / cols);
    const cellW = plot.w / cols;
    const cellH = plot.h / rows;

    for (let i = 0; i < n; i++) {
      const col = i % cols;
      const row = Math.floor(i / cols);
      const cx = plot.x + col * cellW + cellW / 2;
      const cy = plot.y + row * cellH + cellH / 2;
      const r = Math.min(cellW, cellH) * 0.35;
      this._gauges[i].render(ctx, cx, cy, r);
    }
  }

  /**
   * Destroy the renderer.
   */
  destroy() {
    if (this._animFrameId) {
      cancelAnimationFrame(this._animFrameId);
    }
    this._series = [];
    this._gauges = [];
  }
}

export default GraphRenderer;
