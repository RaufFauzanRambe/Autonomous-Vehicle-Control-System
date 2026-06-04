/**
 * @fileoverview Heatmap Generator Module - 2D density heatmap from coordinate data with
 * color gradient mapping, Gaussian kernel density estimation, tile-based rendering for
 * large datasets, export as image, and configurable radius/intensity.
 *
 * @module heatmap_generator
 * @version 2.0.0
 * @author Autonomous Vehicle Control System Team
 */

// ============================================================
// Color Gradient Presets
// ============================================================

/**
 * Predefined color gradient palettes for heatmap rendering.
 * Each gradient is an array of {stop, color} objects.
 * @constant {Object}
 */
export const ColorGradients = Object.freeze({
  /** Classic thermal: blue → cyan → green → yellow → red */
  THERMAL: [
    { stop: 0.0, color: { r: 0, g: 0, b: 139 } },
    { stop: 0.25, color: { r: 0, g: 200, b: 200 } },
    { stop: 0.5, color: { r: 0, g: 255, b: 0 } },
    { stop: 0.75, color: { r: 255, g: 255, b: 0 } },
    { stop: 1.0, color: { r: 255, g: 0, b: 0 } },
  ],
  /** Monochrome: transparent → white */
  MONO: [
    { stop: 0.0, color: { r: 255, g: 255, b: 255 } },
    { stop: 0.5, color: { r: 255, g: 255, b: 255 } },
    { stop: 1.0, color: { r: 255, g: 255, b: 255 } },
  ],
  /** Plasma: dark purple → magenta → orange → yellow */
  PLASMA: [
    { stop: 0.0, color: { r: 13, g: 8, b: 135 } },
    { stop: 0.25, color: { r: 126, g: 3, b: 168 } },
    { stop: 0.5, color: { r: 204, g: 71, b: 120 } },
    { stop: 0.75, color: { r: 248, g: 149, b: 64 } },
    { stop: 1.0, color: { r: 240, g: 249, b: 33 } },
  ],
  /** Viridis: dark blue → teal → green → yellow */
  VIRIDIS: [
    { stop: 0.0, color: { r: 68, g: 1, b: 84 } },
    { stop: 0.25, color: { r: 59, g: 82, b: 139 } },
    { stop: 0.5, color: { r: 33, g: 145, b: 140 } },
    { stop: 0.75, color: { r: 94, g: 201, b: 98 } },
    { stop: 1.0, color: { r: 253, g: 231, b: 37 } },
  ],
});

/**
 * Interpolate a color from a gradient at a given position.
 * @param {number} t - Position along the gradient [0, 1]
 * @param {Array<{stop: number, color: {r: number, g: number, b: number}}>} gradient
 * @returns {{r: number, g: number, b: number}} Interpolated color
 */
export function interpolateGradient(t, gradient) {
  const clamped = Math.max(0, Math.min(1, t));
  if (clamped <= gradient[0].stop) return { ...gradient[0].color };
  if (clamped >= gradient[gradient.length - 1].stop) return { ...gradient[gradient.length - 1].color };

  for (let i = 0; i < gradient.length - 1; i++) {
    if (clamped >= gradient[i].stop && clamped <= gradient[i + 1].stop) {
      const range = gradient[i + 1].stop - gradient[i].stop;
      const localT = range > 0 ? (clamped - gradient[i].stop) / range : 0;
      const c0 = gradient[i].color;
      const c1 = gradient[i + 1].color;
      return {
        r: Math.round(c0.r + (c1.r - c0.r) * localT),
        g: Math.round(c0.g + (c1.g - c0.g) * localT),
        b: Math.round(c0.b + (c1.b - c0.b) * localT),
      };
    }
  }
  return { ...gradient[gradient.length - 1].color };
}

// ============================================================
// Gaussian Kernel
// ============================================================

/**
 * Compute the Gaussian kernel value.
 * @param {number} x - Input value
 * @param {number} sigma - Standard deviation
 * @returns {number} Kernel density value
 */
export function gaussianKernel(x, sigma) {
  const s2 = 2 * sigma * sigma;
  return Math.exp(-x * x / s2) / (Math.sqrt(2 * Math.PI) * sigma);
}

/**
 * Pre-compute a discrete Gaussian kernel for convolution.
 * @param {number} radius - Kernel radius in pixels
 * @param {number} [sigma] - Standard deviation (defaults to radius/3)
 * @returns {Float32Array} Normalized kernel weights
 */
export function createGaussianKernel(radius, sigma) {
  const s = sigma ?? radius / 3;
  const size = radius * 2 + 1;
  const kernel = new Float32Array(size);
  let sum = 0;
  for (let i = 0; i < size; i++) {
    const x = i - radius;
    kernel[i] = gaussianKernel(x, s);
    sum += kernel[i];
  }
  // Normalize
  for (let i = 0; i < size; i++) {
    kernel[i] /= sum;
  }
  return kernel;
}

// ============================================================
// Heatmap Data Point
// ============================================================

/**
 * Represents a data point for heatmap generation.
 * @class HeatmapPoint
 */
export class HeatmapPoint {
  /**
   * @param {number} x - X coordinate (pixel or normalized)
   * @param {number} y - Y coordinate (pixel or normalized)
   * @param {number} [value=1] - Point intensity/value
   */
  constructor(x, y, value = 1) {
    this.x = x;
    this.y = y;
    this.value = value;
  }
}

// ============================================================
// Heatmap Generator
// ============================================================

/**
 * Generates a 2D density heatmap from coordinate data using Gaussian kernel density
 * estimation with tile-based rendering for large datasets and configurable visual parameters.
 * @class HeatmapGenerator
 */
export class HeatmapGenerator {
  /**
   * @param {HTMLCanvasElement} canvas - Target canvas element
   * @param {Object} [options={}] - Generator options
   * @param {number} [options.radius=25] - Kernel radius in pixels
   * @param {number} [options.intensity=1.0] - Global intensity multiplier
   * @param {number} [options.opacity=0.8] - Heatmap opacity
   * @param {number} [options.minOpacity=0.05] - Minimum opacity for rendered cells
   * @param {Array} [options.gradient=ColorGradients.THERMAL] - Color gradient
   * @param {number} [options.tileSize=64] - Tile size for chunked rendering
   * @param {number} [options.maxPoints=50000] - Maximum data points
   * @param {boolean} [options.useWebGL=false] - Use WebGL acceleration (experimental)
   */
  constructor(canvas, options = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError('HeatmapGenerator requires an HTMLCanvasElement');
    }

    this._canvas = canvas;
    this._ctx = canvas.getContext('2d');
    this._radius = options.radius ?? 25;
    this._intensity = options.intensity ?? 1.0;
    this._opacity = options.opacity ?? 0.8;
    this._minOpacity = options.minOpacity ?? 0.05;
    this._gradient = options.gradient ?? ColorGradients.THERMAL;
    this._tileSize = options.tileSize ?? 64;
    this._maxPoints = options.maxPoints ?? 50000;

    /** @type {Array<HeatmapPoint>} */
    this._points = [];
    /** @type {Float32Array} - Density field buffer */
    this._densityField = null;
    /** @type {number} - Field width */
    this._fieldW = 0;
    /** @type {number} - Field height */
    this._fieldH = 0;
    /** @type {number} - Maximum density value for normalization */
    this._maxDensity = 0;

    // Pre-computed gradient lookup table (256 entries)
    this._gradientLUT = this._buildGradientLUT();

    // Shadow canvas for alpha-based rendering
    this._shadowCanvas = document.createElement('canvas');
    this._shadowCtx = this._shadowCanvas.getContext('2d');

    this._resize();
    window.addEventListener('resize', () => this._resize());
  }

  /** @private Build 256-entry gradient lookup table */
  _buildGradientLUT() {
    const lut = new Uint8Array(256 * 4);
    for (let i = 0; i < 256; i++) {
      const t = i / 255;
      const color = interpolateGradient(t, this._gradient);
      lut[i * 4] = color.r;
      lut[i * 4 + 1] = color.g;
      lut[i * 4 + 2] = color.b;
      lut[i * 4 + 3] = 255;
    }
    return lut;
  }

  /** @private Resize canvases to match container */
  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this._canvas.getBoundingClientRect();
    this._canvas.width = rect.width * dpr;
    this._canvas.height = rect.height * dpr;
    this._ctx.scale(dpr, dpr);
    this._canvasW = rect.width;
    this._canvasH = rect.height;
    this._shadowCanvas.width = rect.width;
    this._shadowCanvas.height = rect.height;
    this._fieldW = Math.ceil(rect.width);
    this._fieldH = Math.ceil(rect.height);
  }

  /**
   * Add a data point to the heatmap.
   * @param {number} x - X coordinate
   * @param {number} y - Y coordinate
   * @param {number} [value=1] - Point intensity
   */
  addPoint(x, y, value = 1) {
    if (this._points.length >= this._maxPoints) {
      this._points.shift();
    }
    this._points.push(new HeatmapPoint(x, y, value));
  }

  /**
   * Set all data points at once.
   * @param {Array<{x: number, y: number, value?: number}>} points - Data points
   */
  setData(points) {
    this._points = points.slice(0, this._maxPoints).map(
      (p) => new HeatmapPoint(p.x, p.y, p.value ?? 1),
    );
  }

  /**
   * Clear all data points.
   */
  clear() {
    this._points = [];
    this._densityField = null;
    this._maxDensity = 0;
  }

  /**
   * Compute the density field using Gaussian KDE.
   * Uses a two-pass separable convolution for performance.
   * @returns {Float32Array} Density field
   */
  computeDensityField() {
    const w = this._fieldW;
    const h = this._fieldH;
    const field = new Float32Array(w * h);

    // Stamp each point's contribution onto the field
    for (const point of this._points) {
      const px = Math.round(point.x);
      const py = Math.round(point.y);
      const r = this._radius;
      const value = point.value * this._intensity;

      // Only compute within the kernel radius
      const startX = Math.max(0, px - r);
      const endX = Math.min(w - 1, px + r);
      const startY = Math.max(0, py - r);
      const endY = Math.min(h - 1, py + r);

      for (let y = startY; y <= endY; y++) {
        for (let x = startX; x <= endX; x++) {
          const dx = x - px;
          const dy = y - py;
          const dist2 = dx * dx + dy * dy;
          const r2 = r * r;
          if (dist2 <= r2) {
            // Gaussian contribution
            const contribution = value * Math.exp(-dist2 / (2 * (r / 3) * (r / 3)));
            field[y * w + x] += contribution;
          }
        }
      }
    }

    // Find maximum density for normalization
    this._maxDensity = 0;
    for (let i = 0; i < field.length; i++) {
      if (field[i] > this._maxDensity) {
        this._maxDensity = field[i];
      }
    }

    this._densityField = field;
    return field;
  }

  /**
   * Render the heatmap to the canvas.
   * Uses the shadow canvas technique: first render alpha circles,
   * then colorize using the gradient LUT.
   */
  render() {
    const ctx = this._ctx;
    const w = this._canvasW;
    const h = this._canvasH;

    ctx.clearRect(0, 0, w, h);

    if (this._points.length === 0) return;

    // Ensure density field is computed
    if (!this._densityField) {
      this.computeDensityField();
    }

    // Use shadow canvas for alpha-based rendering
    const sCtx = this._shadowCtx;
    sCtx.clearRect(0, 0, w, h);

    // Draw radial gradient circles for each point (alpha accumulation)
    for (const point of this._points) {
      const r = this._radius;
      const gradient = sCtx.createRadialGradient(
        point.x, point.y, 0,
        point.x, point.y, r,
      );
      const alpha = Math.min(1, (point.value * this._intensity) / Math.max(0.001, this._maxDensity) * 0.5);
      gradient.addColorStop(0, `rgba(0,0,0,${alpha})`);
      gradient.addColorStop(1, 'rgba(0,0,0,0)');

      sCtx.fillStyle = gradient;
      sCtx.fillRect(point.x - r, point.y - r, r * 2, r * 2);
    }

    // Read alpha channel and colorize
    const imageData = sCtx.getImageData(0, 0, Math.round(w), Math.round(h));
    const pixels = imageData.data;

    // Apply color gradient based on alpha values
    for (let i = 0; i < pixels.length; i += 4) {
      const alpha = pixels[i + 3]; // Use accumulated alpha as intensity
      if (alpha < 1) continue; // Skip transparent pixels

      const normalizedAlpha = Math.min(255, alpha);
      const lutIndex = normalizedAlpha * 4;
      pixels[i] = this._gradientLUT[lutIndex];
      pixels[i + 1] = this._gradientLUT[lutIndex + 1];
      pixels[i + 2] = this._gradientLUT[lutIndex + 2];
      pixels[i + 3] = Math.round(normalizedAlpha * this._opacity);
    }

    // Draw colorized heatmap onto main canvas
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = Math.round(w);
    tempCanvas.height = Math.round(h);
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.putImageData(imageData, 0, 0);

    ctx.drawImage(tempCanvas, 0, 0);
  }

  /**
   * Tile-based rendering for large datasets.
   * Renders the heatmap in tiles to avoid blocking the main thread.
   * @param {Function} [onTileComplete] - Called after each tile with (tileX, tileY)
   * @param {Function} [onComplete] - Called when all tiles are rendered
   * @returns {Promise<void>}
   */
  async renderTiled(onTileComplete, onComplete) {
    if (this._points.length === 0) return;

    if (!this._densityField) {
      this.computeDensityField();
    }

    const ctx = this._ctx;
    const w = this._canvasW;
    const h = this._canvasH;
    ctx.clearRect(0, 0, w, h);

    const tileSize = this._tileSize;
    const tilesX = Math.ceil(w / tileSize);
    const tilesY = Math.ceil(h / tileSize);

    for (let ty = 0; ty < tilesY; ty++) {
      for (let tx = 0; tx < tilesX; tx++) {
        const startX = tx * tileSize;
        const startY = ty * tileSize;
        const endX = Math.min(startX + tileSize, w);
        const endY = Math.min(startY + tileSize, h);

        // Render this tile from the density field
        const tileW = endX - startX;
        const tileH = endY - startY;
        const imageData = ctx.createImageData(tileW, tileH);
        const pixels = imageData.data;

        for (let y = 0; y < tileH; y++) {
          for (let x = 0; x < tileW; x++) {
            const fieldX = startX + x;
            const fieldY = startY + y;
            const density = this._densityField[fieldY * this._fieldW + fieldX] ?? 0;
            const normalized = this._maxDensity > 0 ? density / this._maxDensity : 0;

            if (normalized < this._minOpacity) continue;

            const lutIndex = Math.round(normalized * 255) * 4;
            const pi = (y * tileW + x) * 4;
            pixels[pi] = this._gradientLUT[lutIndex];
            pixels[pi + 1] = this._gradientLUT[lutIndex + 1];
            pixels[pi + 2] = this._gradientLUT[lutIndex + 2];
            pixels[pi + 3] = Math.round(normalized * 255 * this._opacity);
          }
        }

        ctx.putImageData(imageData, startX, startY);

        if (onTileComplete) onTileComplete(tx, ty);

        // Yield to main thread between tiles
        await new Promise((resolve) => requestAnimationFrame(resolve));
      }
    }

    if (onComplete) onComplete();
  }

  /**
   * Export the current heatmap as a PNG image.
   * @param {string} [filename='heatmap.png'] - Output filename
   * @returns {string} Data URL of the exported image
   */
  exportAsImage(filename = 'heatmap.png') {
    const dataURL = this._canvas.toDataURL('image/png');
    const link = document.createElement('a');
    link.download = filename;
    link.href = dataURL;
    link.click();
    return dataURL;
  }

  /**
   * Set the color gradient.
   * @param {Array<{stop: number, color: {r: number, g: number, b: number}}>} gradient
   */
  setGradient(gradient) {
    this._gradient = gradient;
    this._gradientLUT = this._buildGradientLUT();
  }

  /**
   * Set the kernel radius.
   * @param {number} radius - Radius in pixels
   */
  setRadius(radius) {
    this._radius = Math.max(1, radius);
    this._densityField = null; // Invalidate cached field
  }

  /**
   * Set the global intensity multiplier.
   * @param {number} intensity - Intensity value
   */
  setIntensity(intensity) {
    this._intensity = Math.max(0.01, intensity);
    this._densityField = null;
  }

  /**
   * Get the current density at a specific pixel.
   * @param {number} x - X pixel
   * @param {number} y - Y pixel
   * @returns {number} Normalized density [0, 1]
   */
  getDensityAt(x, y) {
    if (!this._densityField || this._maxDensity === 0) return 0;
    const idx = Math.round(y) * this._fieldW + Math.round(x);
    return (this._densityField[idx] ?? 0) / this._maxDensity;
  }

  /**
   * Get heatmap statistics.
   * @returns {{pointCount: number, maxDensity: number, fieldSize: number}}
   */
  getStats() {
    return {
      pointCount: this._points.length,
      maxDensity: this._maxDensity,
      fieldSize: this._fieldW * this._fieldH,
    };
  }

  /**
   * Destroy the generator and free resources.
   */
  destroy() {
    this._points = [];
    this._densityField = null;
    this._gradientLUT = null;
  }
}

export default HeatmapGenerator;
