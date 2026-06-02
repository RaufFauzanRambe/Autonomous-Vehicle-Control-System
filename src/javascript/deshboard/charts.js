/**
 * @fileoverview Chart Rendering System
 * @description ChartWrapper class, LineChart (speed over time), BarChart (object counts),
 *   PieChart (sensor distribution), GaugeChart (speedometer), ChartTheme,
 *   real-time data updates, and animation for the Autonomous Vehicle Dashboard.
 * @module charts
 */

import { EventEmitter } from 'events';

// ─── Chart Theme ─────────────────────────────────────────────────────────────

/**
 * Theme configuration for chart rendering
 */
export class ChartTheme {
  /**
   * @param {Object} [options={}]
   * @param {string[]} [options.colors] - Color palette
   * @param {string} [options.background] - Background color
   * @param {string} [options.gridColor] - Grid line color
   * @param {string} [options.textColor] - Text/label color
   * @param {string} [options.fontFamily] - Font family
   * @param {number} [options.fontSize] - Font size
   * @param {number} [options.lineWidth] - Default line width
   * @param {boolean} [options.animate] - Enable animations
   * @param {number} [options.animationDuration] - Animation duration in ms
   */
  constructor(options = {}) {
    /** @type {string[]} */ this.colors = options.colors || ['#00d4aa', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#ec4899', '#84cc16'];
    /** @type {string} */ this.background = options.background || '#1a1d27';
    /** @type {string} */ this.gridColor = options.gridColor || '#2d3348';
    /** @type {string} */ this.textColor = options.textColor || '#9ca3af';
    /** @type {string} */ this.fontFamily = options.fontFamily || "'JetBrains Mono', 'Fira Code', monospace";
    /** @type {number} */ this.fontSize = options.fontSize || 12;
    /** @type {number} */ this.lineWidth = options.lineWidth || 2;
    /** @type {boolean} */ this.animate = options.animate !== false;
    /** @type {number} */ this.animationDuration = options.animationDuration ?? 300;
  }

  /**
   * Create a dark theme preset
   * @returns {ChartTheme}
   */
  static dark() {
    return new ChartTheme({
      colors: ['#00d4aa', '#3b82f6', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4'],
      background: '#1a1d27',
      gridColor: '#2d3348',
      textColor: '#9ca3af',
    });
  }

  /**
   * Create a light theme preset
   * @returns {ChartTheme}
   */
  static light() {
    return new ChartTheme({
      colors: ['#059669', '#2563eb', '#d97706', '#dc2626', '#7c3aed', '#0891b2'],
      background: '#ffffff',
      gridColor: '#e2e8f0',
      textColor: '#64748b',
    });
  }

  /**
   * Get a color from the palette by index
   * @param {number} index - Color index
   * @returns {string}
   */
  getColor(index) {
    return this.colors[index % this.colors.length];
  }
}

// ─── Chart Wrapper Base ──────────────────────────────────────────────────────

/**
 * Base chart wrapper providing canvas management and common rendering
 * @extends EventEmitter
 */
export class ChartWrapper extends EventEmitter {
  /**
   * @param {HTMLCanvasElement} canvas - Target canvas element
   * @param {Object} [options={}]
   * @param {ChartTheme} [options.theme] - Chart theme
   * @param {Object} [options.padding] - Chart padding
   * @param {boolean} [options.showGrid=true] - Show grid lines
   * @param {boolean} [options.showLegend=true] - Show legend
   * @param {boolean} [options.responsive=true] - Auto-resize with container
   */
  constructor(canvas, options = {}) {
    super();
    /** @protected */ this.canvas = canvas;
    /** @protected */ this.ctx = canvas.getContext('2d');
    /** @protected */ this.theme = options.theme || ChartTheme.dark();
    /** @protected */ this.padding = options.padding || { top: 30, right: 20, bottom: 40, left: 50 };
    /** @protected */ this.showGrid = options.showGrid !== false;
    /** @protected */ this.showLegend = options.showLegend !== false;
    /** @protected */ this.responsive = options.responsive !== false;
    /** @protected @type {Map<string, number[]>} */ this.datasets = new Map();
    /** @protected @type {string[]} */ this.labels = [];
    /** @protected @type {number|null} */ this._animFrame = null;
    /** @protected */ this._animationProgress = 1;
    /** @protected */ this._resizeObserver = null;

    if (this.responsive) this._setupResize();
  }

  /**
   * Set up resize observer for responsive behavior
   * @private
   */
  _setupResize() {
    if (typeof ResizeObserver !== 'undefined') {
      this._resizeObserver = new ResizeObserver(() => {
        this._handleResize();
      });
      this._resizeObserver.observe(this.canvas.parentElement);
    }
  }

  /**
   * Handle canvas resize
   * @private
   */
  _handleResize() {
    const parent = this.canvas.parentElement;
    if (parent) {
      const dpr = window.devicePixelRatio || 1;
      const rect = parent.getBoundingClientRect();
      this.canvas.width = rect.width * dpr;
      this.canvas.height = rect.height * dpr;
      this.canvas.style.width = `${rect.width}px`;
      this.canvas.style.height = `${rect.height}px`;
      this.ctx.scale(dpr, dpr);
      this.render();
    }
  }

  /**
   * Get the drawable area dimensions
   * @protected
   * @returns {{ x: number, y: number, width: number, height: number }}
   */
  _getPlotArea() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.width / dpr;
    const h = this.canvas.height / dpr;
    return {
      x: this.padding.left,
      y: this.padding.top,
      width: w - this.padding.left - this.padding.right,
      height: h - this.padding.top - this.padding.bottom,
    };
  }

  /**
   * Clear the canvas and draw background
   * @protected
   */
  _clearCanvas() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.width / dpr;
    const h = this.canvas.height / dpr;
    this.ctx.clearRect(0, 0, w, h);
    this.ctx.fillStyle = this.theme.background;
    this.ctx.fillRect(0, 0, w, h);
  }

  /**
   * Draw grid lines on the plot area
   * @protected
   * @param {number} [gridLines=5] - Number of horizontal grid lines
   */
  _drawGrid(gridLines = 5) {
    if (!this.showGrid) return;
    const area = this._getPlotArea();
    this.ctx.strokeStyle = this.theme.gridColor;
    this.ctx.lineWidth = 0.5;
    this.ctx.font = `${this.theme.fontSize}px ${this.theme.fontFamily}`;
    this.ctx.fillStyle = this.theme.textColor;
    this.ctx.textAlign = 'right';

    for (let i = 0; i <= gridLines; i++) {
      const y = area.y + (area.height / gridLines) * i;
      this.ctx.beginPath();
      this.ctx.moveTo(area.x, y);
      this.ctx.lineTo(area.x + area.width, y);
      this.ctx.stroke();
    }
  }

  /**
   * Draw the chart title
   * @protected
   * @param {string} title - Chart title
   */
  _drawTitle(title) {
    this.ctx.font = `bold ${this.theme.fontSize + 2}px ${this.theme.fontFamily}`;
    this.ctx.fillStyle = this.theme.textColor;
    this.ctx.textAlign = 'left';
    this.ctx.fillText(title, this.padding.left, 18);
  }

  /**
   * Render the chart — override in subclasses
   */
  render() {
    this._clearCanvas();
  }

  /**
   * Add a dataset to the chart
   * @param {string} name - Dataset name
   * @param {number[]} values - Data values
   * @param {string[]} [lbls] - Labels
   */
  setData(name, values, lbls) {
    this.datasets.set(name, values);
    if (lbls) this.labels = lbls;
    this.render();
  }

  /**
   * Append a data point to a dataset
   * @param {string} name - Dataset name
   * @param {number} value - New value
   * @param {string} [label] - New label
   * @param {number} [maxPoints=100] - Maximum number of points
   */
  appendData(name, value, label, maxPoints = 100) {
    if (!this.datasets.has(name)) this.datasets.set(name, []);
    const data = this.datasets.get(name);
    data.push(value);
    if (data.length > maxPoints) data.shift();
    if (label) {
      this.labels.push(label);
      if (this.labels.length > maxPoints) this.labels.shift();
    }
    this.render();
  }

  /**
   * Destroy the chart and clean up
   */
  destroy() {
    if (this._animFrame) cancelAnimationFrame(this._animFrame);
    if (this._resizeObserver) this._resizeObserver.disconnect();
    this.datasets.clear();
    this.labels = [];
    this.removeAllListeners();
  }
}

// ─── LineChart ───────────────────────────────────────────────────────────────

/**
 * Line chart for time-series data (e.g., speed over time)
 * @extends ChartWrapper
 */
export class LineChart extends ChartWrapper {
  /**
   * @param {HTMLCanvasElement} canvas - Canvas element
   * @param {Object} [options={}]
   * @param {boolean} [options.fill=false] - Fill area under the line
   * @param {boolean} [options.smooth=true] - Smooth line with bezier curves
   * @param {number} [options.dotRadius=0] - Data point dot radius
   */
  constructor(canvas, options = {}) {
    super(canvas, options);
    /** @private */ this._fill = options.fill ?? false;
    /** @private */ this._smooth = options.smooth ?? true;
    /** @private */ this._dotRadius = options.dotRadius ?? 0;
    /** @private */ this._yMin = options.yMin ?? null;
    /** @private */ this._yMax = options.yMax ?? null;
  }

  /** @override */ render() {
    super.render();
    if (this.datasets.size === 0) return;

    const area = this._getPlotArea();
    this._drawGrid();
    this._drawTitle('Speed Over Time');

    // Compute Y range
    let allValues = [];
    for (const values of this.datasets.values()) allValues.push(...values);
    const yMin = this._yMin ?? Math.min(...allValues, 0);
    const yMax = this._yMax ?? Math.max(...allValues) * 1.1 || 1;
    const yRange = yMax - yMin || 1;

    // Draw Y axis labels
    this.ctx.font = `${this.theme.fontSize}px ${this.theme.fontFamily}`;
    this.ctx.fillStyle = this.theme.textColor;
    this.ctx.textAlign = 'right';
    for (let i = 0; i <= 5; i++) {
      const val = yMin + (yRange / 5) * (5 - i);
      const y = area.y + (area.height / 5) * i;
      this.ctx.fillText(val.toFixed(0), area.x - 8, y + 4);
    }

    // Draw datasets
    let colorIdx = 0;
    for (const [name, values] of this.datasets.entries()) {
      const color = this.theme.getColor(colorIdx++);
      if (values.length < 2) continue;

      const xStep = area.width / Math.max(values.length - 1, 1);

      // Fill area
      if (this._fill) {
        this.ctx.beginPath();
        this.ctx.moveTo(area.x, area.y + area.height);
        for (let i = 0; i < values.length; i++) {
          const x = area.x + i * xStep;
          const y = area.y + area.height - ((values[i] - yMin) / yRange) * area.height;
          if (i === 0) this.ctx.lineTo(x, y); else if (this._smooth) this._bezierTo(this.ctx, values, i, xStep, area, yMin, yRange); else this.ctx.lineTo(x, y);
        }
        this.ctx.lineTo(area.x + (values.length - 1) * xStep, area.y + area.height);
        this.ctx.closePath();
        const grad = this.ctx.createLinearGradient(0, area.y, 0, area.y + area.height);
        grad.addColorStop(0, color + '40');
        grad.addColorStop(1, color + '05');
        this.ctx.fillStyle = grad;
        this.ctx.fill();
      }

      // Draw line
      this.ctx.beginPath();
      this.ctx.strokeStyle = color;
      this.ctx.lineWidth = this.theme.lineWidth;
      this.ctx.lineJoin = 'round';
      for (let i = 0; i < values.length; i++) {
        const x = area.x + i * xStep;
        const y = area.y + area.height - ((values[i] - yMin) / yRange) * area.height;
        if (i === 0) this.ctx.moveTo(x, y); else if (this._smooth) this._bezierTo(this.ctx, values, i, xStep, area, yMin, yRange); else this.ctx.lineTo(x, y);
      }
      this.ctx.stroke();

      // Draw dots
      if (this._dotRadius > 0) {
        for (let i = 0; i < values.length; i++) {
          const x = area.x + i * xStep;
          const y = area.y + area.height - ((values[i] - yMin) / yRange) * area.height;
          this.ctx.beginPath();
          this.ctx.arc(x, y, this._dotRadius, 0, Math.PI * 2);
          this.ctx.fillStyle = color;
          this.ctx.fill();
        }
      }

      // Legend
      if (this.showLegend) {
        const legendX = area.x + area.width - 100;
        const legendY = area.y + 10 + (colorIdx - 1) * 18;
        this.ctx.fillStyle = color;
        this.ctx.fillRect(legendX, legendY, 12, 12);
        this.ctx.fillStyle = this.theme.textColor;
        this.ctx.textAlign = 'left';
        this.ctx.fillText(name, legendX + 16, legendY + 10);
      }
    }

    this.emit('rendered');
  }

  /**
   * Draw a smooth bezier curve to a data point
   * @private
   */
  _bezierTo(ctx, values, i, xStep, area, yMin, yRange) {
    const prev = values[i - 1];
    const curr = values[i];
    const x0 = area.x + (i - 1) * xStep;
    const x1 = area.x + i * xStep;
    const y0 = area.y + area.height - ((prev - yMin) / yRange) * area.height;
    const y1 = area.y + area.height - ((curr - yMin) / yRange) * area.height;
    const cpx = (x0 + x1) / 2;
    ctx.bezierCurveTo(cpx, y0, cpx, y1, x1, y1);
  }
}

// ─── BarChart ────────────────────────────────────────────────────────────────

/**
 * Bar chart for categorical data (e.g., detected object counts by type)
 * @extends ChartWrapper
 */
export class BarChart extends ChartWrapper {
  /**
   * @param {HTMLCanvasElement} canvas - Canvas element
   * @param {Object} [options={}]
   * @param {number} [options.barPadding=4] - Padding between bars
   * @param {boolean} [options.horizontal=false] - Horizontal bars
   */
  constructor(canvas, options = {}) {
    super(canvas, options);
    /** @private */ this._barPadding = options.barPadding ?? 4;
    /** @private */ this._horizontal = options.horizontal ?? false;
  }

  /** @override */ render() {
    super.render();
    if (this.datasets.size === 0) return;

    const area = this._getPlotArea();
    this._drawGrid();
    this._drawTitle('Detected Objects by Type');

    const labels = this.labels;
    const [name, values] = this.datasets.entries().next().value;
    const maxVal = Math.max(...values, 1);

    if (!this._horizontal) {
      const barWidth = (area.width - this._barPadding * (labels.length + 1)) / labels.length;

      // Y-axis labels
      this.ctx.font = `${this.theme.fontSize}px ${this.theme.fontFamily}`;
      this.ctx.fillStyle = this.theme.textColor;
      this.ctx.textAlign = 'right';
      for (let i = 0; i <= 5; i++) {
        const val = (maxVal / 5) * (5 - i);
        const y = area.y + (area.height / 5) * i;
        this.ctx.fillText(val.toFixed(0), area.x - 8, y + 4);
      }

      // Draw bars
      for (let i = 0; i < values.length; i++) {
        const x = area.x + this._barPadding + i * (barWidth + this._barPadding);
        const barHeight = (values[i] / maxVal) * area.height;
        const y = area.y + area.height - barHeight;

        const color = this.theme.getColor(i);
        this.ctx.fillStyle = color;
        this.ctx.fillRect(x, y, barWidth, barHeight);

        // Value on top
        this.ctx.fillStyle = this.theme.textColor;
        this.ctx.textAlign = 'center';
        this.ctx.fillText(values[i].toString(), x + barWidth / 2, y - 4);

        // Label below
        this.ctx.fillText(labels[i] || `#${i}`, x + barWidth / 2, area.y + area.height + 16);
      }
    }

    this.emit('rendered');
  }
}

// ─── PieChart ────────────────────────────────────────────────────────────────

/**
 * Pie/Donut chart for distribution data (e.g., sensor type distribution)
 * @extends ChartWrapper
 */
export class PieChart extends ChartWrapper {
  /**
   * @param {HTMLCanvasElement} canvas - Canvas element
   * @param {Object} [options={}]
   * @param {boolean} [options.donut=true] - Donut style
   * @param {number} [options.innerRadiusRatio=0.6] - Inner radius ratio for donut
   * @param {boolean} [options.showLabels=true] - Show slice labels
   * @param {number} [options.startAngle=-Math.PI/2] - Starting angle
   */
  constructor(canvas, options = {}) {
    super(canvas, options);
    /** @private */ this._donut = options.donut ?? true;
    /** @private */ this._innerRatio = options.innerRadiusRatio ?? 0.6;
    /** @private */ this._showLabels = options.showLabels ?? true;
    /** @private */ this._startAngle = options.startAngle ?? -Math.PI / 2;
  }

  /** @override */ render() {
    super.render();
    if (this.datasets.size === 0) return;

    const area = this._getPlotArea();
    this._drawTitle('Sensor Distribution');

    const labels = this.labels;
    const [_, values] = this.datasets.entries().next().value;
    const total = values.reduce((a, b) => a + b, 0) || 1;

    const centerX = area.x + area.width / 2;
    const centerY = area.y + area.height / 2;
    const radius = Math.min(area.width, area.height) / 2 - 10;
    const innerRadius = this._donut ? radius * this._innerRatio : 0;

    let currentAngle = this._startAngle;

    for (let i = 0; i < values.length; i++) {
      const sliceAngle = (values[i] / total) * Math.PI * 2;
      const color = this.theme.getColor(i);

      // Draw slice
      this.ctx.beginPath();
      this.ctx.moveTo(
        centerX + Math.cos(currentAngle) * innerRadius,
        centerY + Math.sin(currentAngle) * innerRadius
      );
      this.ctx.arc(centerX, centerY, radius, currentAngle, currentAngle + sliceAngle);
      this.ctx.arc(centerX, centerY, innerRadius, currentAngle + sliceAngle, currentAngle, true);
      this.ctx.closePath();
      this.ctx.fillStyle = color;
      this.ctx.fill();

      // Slice border
      this.ctx.strokeStyle = this.theme.background;
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      // Label
      if (this._showLabels && sliceAngle > 0.15) {
        const labelAngle = currentAngle + sliceAngle / 2;
        const labelRadius = (radius + innerRadius) / 2;
        const lx = centerX + Math.cos(labelAngle) * labelRadius;
        const ly = centerY + Math.sin(labelAngle) * labelRadius;
        this.ctx.fillStyle = '#ffffff';
        this.ctx.font = `bold ${this.theme.fontSize}px ${this.theme.fontFamily}`;
        this.ctx.textAlign = 'center';
        this.ctx.textBaseline = 'middle';
        this.ctx.fillText(`${((values[i] / total) * 100).toFixed(0)}%`, lx, ly);
      }

      currentAngle += sliceAngle;
    }

    // Center text for donut
    if (this._donut) {
      this.ctx.fillStyle = this.theme.textColor;
      this.ctx.font = `bold ${this.theme.fontSize + 6}px ${this.theme.fontFamily}`;
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(total.toString(), centerX, centerY - 6);
      this.ctx.font = `${this.theme.fontSize - 2}px ${this.theme.fontFamily}`;
      this.ctx.fillText('total', centerX, centerY + 12);
    }

    // Legend
    if (this.showLegend) {
      const legendX = area.x + area.width + 10;
      let legendY = area.y;
      for (let i = 0; i < labels.length; i++) {
        this.ctx.fillStyle = this.theme.getColor(i);
        this.ctx.fillRect(legendX, legendY, 10, 10);
        this.ctx.fillStyle = this.theme.textColor;
        this.ctx.textAlign = 'left';
        this.ctx.font = `${this.theme.fontSize}px ${this.theme.fontFamily}`;
        this.ctx.fillText(`${labels[i]} (${values[i]})`, legendX + 14, legendY + 9);
        legendY += 18;
      }
    }

    this.emit('rendered');
  }
}

// ─── GaugeChart ──────────────────────────────────────────────────────────────

/**
 * Gauge/speedometer chart for single-value display
 * @extends ChartWrapper
 */
export class GaugeChart extends ChartWrapper {
  /**
   * @param {HTMLCanvasElement} canvas - Canvas element
   * @param {Object} [options={}]
   * @param {number} [options.min=0] - Minimum value
   * @param {number} [options.max=220] - Maximum value
   * @param {number} [options.warningThreshold=0.6] - Warning zone start (ratio)
   * @param {number} [options.criticalThreshold=0.8] - Critical zone start (ratio)
   * @param {string} [options.unit='km/h'] - Value unit
   */
  constructor(canvas, options = {}) {
    super(canvas, options);
    /** @private */ this._min = options.min ?? 0;
    /** @private */ this._max = options.max ?? 220;
    /** @private */ this._warningThreshold = options.warningThreshold ?? 0.6;
    /** @private */ this._criticalThreshold = options.criticalThreshold ?? 0.8;
    /** @private */ this._unit = options.unit || 'km/h';
    /** @private */ this._currentValue = this._min;
    /** @private */ this._targetValue = this._min;
  }

  /**
   * Set the gauge value with optional animation
   * @param {number} value - New value
   * @param {boolean} [animate=true] - Animate the transition
   */
  setValue(value, animate = true) {
    this._targetValue = Math.max(this._min, Math.min(value, this._max));
    if (animate && this.theme.animate) {
      this._animateTo();
    } else {
      this._currentValue = this._targetValue;
      this.render();
    }
  }

  /**
   * Animate to target value
   * @private
   */
  _animateTo() {
    const diff = this._targetValue - this._currentValue;
    if (Math.abs(diff) < 0.5) {
      this._currentValue = this._targetValue;
      this.render();
      return;
    }
    this._currentValue += diff * 0.15;
    this.render();
    this._animFrame = requestAnimationFrame(() => this._animateTo());
  }

  /** @override */ render() {
    super.render();
    const area = this._getPlotArea();
    const centerX = area.x + area.width / 2;
    const centerY = area.y + area.height * 0.65;
    const radius = Math.min(area.width, area.height) * 0.45;

    const startAngle = Math.PI * 0.8;
    const endAngle = Math.PI * 0.2;
    const totalAngle = Math.PI * 1.4;

    // Draw background arc
    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY, radius, startAngle, Math.PI * 2 + endAngle);
    this.ctx.strokeStyle = this.theme.gridColor;
    this.ctx.lineWidth = 20;
    this.ctx.lineCap = 'round';
    this.ctx.stroke();

    // Draw warning zone
    const warningAngle = startAngle + totalAngle * this._warningThreshold;
    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY, radius, warningAngle, Math.PI * 2 + endAngle);
    this.ctx.strokeStyle = '#f59e0b40';
    this.ctx.lineWidth = 20;
    this.ctx.stroke();

    // Draw critical zone
    const criticalAngle = startAngle + totalAngle * this._criticalThreshold;
    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY, radius, criticalAngle, Math.PI * 2 + endAngle);
    this.ctx.strokeStyle = '#ef444440';
    this.ctx.lineWidth = 20;
    this.ctx.stroke();

    // Draw value arc
    const valueRatio = (this._currentValue - this._min) / (this._max - this._min);
    const valueAngle = startAngle + totalAngle * valueRatio;
    const arcColor = valueRatio > this._criticalThreshold ? '#ef4444' : valueRatio > this._warningThreshold ? '#f59e0b' : '#00d4aa';

    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY, radius, startAngle, valueAngle);
    this.ctx.strokeStyle = arcColor;
    this.ctx.lineWidth = 20;
    this.ctx.lineCap = 'round';
    this.ctx.stroke();

    // Draw tick marks
    for (let i = 0; i <= 10; i++) {
      const angle = startAngle + (totalAngle / 10) * i;
      const x1 = centerX + Math.cos(angle) * (radius - 15);
      const y1 = centerY + Math.sin(angle) * (radius - 15);
      const x2 = centerX + Math.cos(angle) * (radius + 5);
      const y2 = centerY + Math.sin(angle) * (radius + 5);
      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.strokeStyle = this.theme.gridColor;
      this.ctx.lineWidth = 2;
      this.ctx.stroke();

      // Tick labels
      const labelVal = this._min + ((this._max - this._min) / 10) * i;
      const lx = centerX + Math.cos(angle) * (radius - 28);
      const ly = centerY + Math.sin(angle) * (radius - 28);
      this.ctx.fillStyle = this.theme.textColor;
      this.ctx.font = `${this.theme.fontSize - 2}px ${this.theme.fontFamily}`;
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(labelVal.toFixed(0), lx, ly);
    }

    // Draw needle
    const needleAngle = startAngle + totalAngle * valueRatio;
    const needleLen = radius - 40;
    const nx = centerX + Math.cos(needleAngle) * needleLen;
    const ny = centerY + Math.sin(needleAngle) * needleLen;
    this.ctx.beginPath();
    this.ctx.moveTo(centerX, centerY);
    this.ctx.lineTo(nx, ny);
    this.ctx.strokeStyle = arcColor;
    this.ctx.lineWidth = 3;
    this.ctx.lineCap = 'round';
    this.ctx.stroke();

    // Center dot
    this.ctx.beginPath();
    this.ctx.arc(centerX, centerY, 6, 0, Math.PI * 2);
    this.ctx.fillStyle = arcColor;
    this.ctx.fill();

    // Value text
    this.ctx.fillStyle = this.theme.textColor;
    this.ctx.font = `bold ${this.theme.fontSize + 16}px ${this.theme.fontFamily}`;
    this.ctx.textAlign = 'center';
    this.ctx.fillText(this._currentValue.toFixed(0), centerX, centerY + 40);
    this.ctx.font = `${this.theme.fontSize}px ${this.theme.fontFamily}`;
    this.ctx.fillText(this._unit, centerX, centerY + 58);

    this.emit('rendered', { value: this._currentValue });
  }
}

/**
 * Factory function to create chart instances
 * @param {string} type - Chart type (line, bar, pie, gauge)
 * @param {HTMLCanvasElement} canvas - Canvas element
 * @param {Object} [options={}] - Chart options
 * @returns {ChartWrapper} Chart instance
 */
export function createChart(type, canvas, options = {}) {
  const chartMap = { line: LineChart, bar: BarChart, pie: PieChart, gauge: GaugeChart };
  const ChartClass = chartMap[type];
  if (!ChartClass) throw new Error(`Unknown chart type: ${type}. Available: ${Object.keys(chartMap).join(', ')}`);
  return new ChartClass(canvas, options);
}
