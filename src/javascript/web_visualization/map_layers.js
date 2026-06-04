/**
 * @fileoverview Map Layers Module - Tile layer management with satellite/terrain/roadmap styles,
 * custom overlay layers (sensors, zones, detections), layer visibility toggle, layer ordering,
 * legend generation, and style configuration.
 *
 * @module map_layers
 * @version 2.0.0
 * @author Autonomous Vehicle Control System Team
 */

// ============================================================
// Tile Source Configuration
// ============================================================

/**
 * Predefined tile source configurations for common map providers.
 * @constant {Object}
 */
export const TileSources = Object.freeze({
  /** OpenStreetMap standard */
  OSM_STANDARD: {
    id: 'osm-standard',
    name: 'OpenStreetMap',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    subdomains: ['a', 'b', 'c'],
    attribution: '&copy; OpenStreetMap contributors',
    minZoom: 1,
    maxZoom: 19,
    tileSize: 256,
    category: 'roadmap',
  },
  /** OpenStreetMap HD (for retina displays) */
  OSM_HD: {
    id: 'osm-hd',
    name: 'OpenStreetMap HD',
    url: 'https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
    subdomains: ['a', 'b', 'c'],
    attribution: '&copy; OpenStreetMap contributors',
    minZoom: 1,
    maxZoom: 19,
    tileSize: 512,
    category: 'roadmap',
  },
  /** ArcGIS World Imagery (satellite) */
  ARCGIS_SATELLITE: {
    id: 'arcgis-satellite',
    name: 'ArcGIS Satellite',
    url: 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
    subdomains: [],
    attribution: '&copy; Esri',
    minZoom: 1,
    maxZoom: 19,
    tileSize: 256,
    category: 'satellite',
  },
  /** OpenTopoMap (terrain) */
  OPENTOPO: {
    id: 'opentopo',
    name: 'OpenTopoMap',
    url: 'https://{s}.tile.opentopomap.org/{z}/{x}/{y}.png',
    subdomains: ['a', 'b', 'c'],
    attribution: '&copy; OpenTopoMap',
    minZoom: 1,
    maxZoom: 17,
    tileSize: 256,
    category: 'terrain',
  },
  /** CartoDB Dark Matter (dark themed) */
  CARTO_DARK: {
    id: 'carto-dark',
    name: 'CartoDB Dark',
    url: 'https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png',
    subdomains: ['a', 'b', 'c', 'd'],
    attribution: '&copy; CartoDB',
    minZoom: 1,
    maxZoom: 19,
    tileSize: 256,
    category: 'dark',
  },
  /** CartoDB Positron (light themed) */
  CARTO_LIGHT: {
    id: 'carto-light',
    name: 'CartoDB Light',
    url: 'https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png',
    subdomains: ['a', 'b', 'c', 'd'],
    attribution: '&copy; CartoDB',
    minZoom: 1,
    maxZoom: 19,
    tileSize: 256,
    category: 'roadmap',
  },
});

// ============================================================
// Tile Layer
// ============================================================

/**
 * Represents a single tile layer on the map.
 * @class TileLayer
 */
export class TileLayer {
  /**
   * @param {Object} source - Tile source configuration
   * @param {Object} [options={}] - Layer options
   * @param {number} [options.opacity=1.0] - Layer opacity
   * @param {boolean} [options.visible=true] - Initial visibility
   * @param {number} [options.zIndex=0] - Z-index for ordering
   */
  constructor(source, options = {}) {
    this.id = source.id;
    this.name = source.name;
    this.url = source.url;
    this.subdomains = source.subdomains ?? [];
    this.attribution = source.attribution ?? '';
    this.minZoom = source.minZoom ?? 1;
    this.maxZoom = source.maxZoom ?? 19;
    this.tileSize = source.tileSize ?? 256;
    this.category = source.category ?? 'roadmap';

    this.opacity = options.opacity ?? 1.0;
    this.visible = options.visible ?? true;
    this.zIndex = options.zIndex ?? 0;
    this.loaded = false;

    /** @private @type {Map<string, HTMLImageElement>} */
    this._tileCache = new Map();
    /** @private @type {number} */
    this._cacheLimit = 512;
  }

  /**
   * Build the URL for a specific tile.
   * @param {number} x - Tile X coordinate
   * @param {number} y - Tile Y coordinate
   * @param {number} z - Zoom level
   * @returns {string} Tile URL
   */
  getTileURL(x, y, z) {
    let url = this.url
      .replace('{x}', String(x))
      .replace('{y}', String(y))
      .replace('{z}', String(z))
      .replace('{r}', '@2x');

    if (this.subdomains.length > 0) {
      const server = this.subdomains[Math.abs(x + y) % this.subdomains.length];
      url = url.replace('{s}', server);
    }

    return url;
  }

  /**
   * Load a tile image.
   * @param {number} x - Tile X
   * @param {number} y - Tile Y
   * @param {number} z - Zoom level
   * @returns {Promise<HTMLImageElement>}
   */
  async loadTile(x, y, z) {
    const key = `${z}/${x}/${y}`;
    if (this._tileCache.has(key)) {
      return this._tileCache.get(key);
    }

    const url = this.getTileURL(x, y, z);

    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        this._evictCache();
        this._tileCache.set(key, img);
        this.loaded = true;
        resolve(img);
      };
      img.onerror = () => reject(new Error(`Tile load failed: ${key}`));
      img.src = url;
    });
  }

  /** @private */
  _evictCache() {
    if (this._tileCache.size >= this._cacheLimit) {
      const firstKey = this._tileCache.keys().next().value;
      this._tileCache.delete(firstKey);
    }
  }

  /**
   * Clear the tile cache.
   */
  clearCache() {
    this._tileCache.clear();
    this.loaded = false;
  }

  /**
   * Set layer opacity.
   * @param {number} opacity - Opacity [0, 1]
   */
  setOpacity(opacity) {
    this.opacity = Math.max(0, Math.min(1, opacity));
  }

  /**
   * Toggle layer visibility.
   * @param {boolean} [visible] - Force visibility, or toggle if omitted
   */
  toggleVisibility(visible) {
    this.visible = visible ?? !this.visible;
  }
}

// ============================================================
// Overlay Layer
// ============================================================

/**
 * Custom overlay layer for rendering sensor data, zones, detections, etc.
 * @class OverlayLayer
 */
export class OverlayLayer {
  /**
   * @param {Object} config
   * @param {string} config.id - Layer identifier
   * @param {string} config.name - Display name
   * @param {'sensors'|'zones'|'detections'|'route'|'heatmap'|'custom'} config.type - Overlay type
   * @param {string} [config.color='#3b82f6'] - Default color
   * @param {boolean} [config.visible=true] - Initial visibility
   * @param {number} [config.zIndex=10] - Z-index
   * @param {Function} [config.renderFn] - Custom render function(ctx, projectFn)
   * @param {Object} [config.style={}] - Style configuration
   */
  constructor(config) {
    this.id = config.id;
    this.name = config.name;
    this.type = config.type;
    this.color = config.color ?? '#3b82f6';
    this.visible = config.visible ?? true;
    this.zIndex = config.zIndex ?? 10;
    this.renderFn = config.renderFn ?? null;
    this.style = config.style ?? {};

    /** @type {Array<Object>} */
    this.features = [];
    /** @type {Object} */
    this.metadata = {};
  }

  /**
   * Add a feature to the overlay.
   * @param {Object} feature - Feature data (geometry, properties)
   */
  addFeature(feature) {
    this.features.push(feature);
  }

  /**
   * Set all features for the overlay.
   * @param {Array<Object>} features
   */
  setFeatures(features) {
    this.features = features;
  }

  /**
   * Remove a feature by index.
   * @param {number} index
   */
  removeFeature(index) {
    if (index >= 0 && index < this.features.length) {
      this.features.splice(index, 1);
    }
  }

  /**
   * Render this overlay layer.
   * @param {CanvasRenderingContext2D} ctx - Canvas context
   * @param {Function} projectFn - Lat/lng to pixel projection
   */
  render(ctx, projectFn) {
    if (!this.visible) return;

    if (this.renderFn) {
      this.renderFn(ctx, projectFn, this.features, this.style);
      return;
    }

    // Default rendering based on type
    switch (this.type) {
      case 'sensors':
        this._renderSensors(ctx, projectFn);
        break;
      case 'zones':
        this._renderZones(ctx, projectFn);
        break;
      case 'detections':
        this._renderDetections(ctx, projectFn);
        break;
      case 'route':
        this._renderRoute(ctx, projectFn);
        break;
      case 'heatmap':
        this._renderHeatmap(ctx, projectFn);
        break;
      default:
        break;
    }
  }

  /** @private Render sensor positions */
  _renderSensors(ctx, projectFn) {
    for (const feature of this.features) {
      const pos = projectFn(feature.lat, feature.lng);
      if (!pos) continue;

      // Sensor icon
      const r = feature.size ?? 8;
      ctx.beginPath();
      ctx.arc(pos.x, pos.y, r, 0, Math.PI * 2);
      ctx.fillStyle = feature.color ?? this.color;
      ctx.fill();
      ctx.strokeStyle = '#1e293b';
      ctx.lineWidth = 1.5;
      ctx.stroke();

      // FOV arc
      if (feature.fov && feature.range) {
        const rangePixels = feature.range * (this.style.pixelsPerMeter ?? 3);
        const startAngle = ((feature.heading - feature.fov / 2) - 90) * Math.PI / 180;
        const endAngle = ((feature.heading + feature.fov / 2) - 90) * Math.PI / 180;
        ctx.beginPath();
        ctx.moveTo(pos.x, pos.y);
        ctx.arc(pos.x, pos.y, rangePixels, startAngle, endAngle);
        ctx.closePath();
        ctx.fillStyle = (feature.color ?? this.color) + '20';
        ctx.fill();
        ctx.strokeStyle = (feature.color ?? this.color) + '60';
        ctx.lineWidth = 1;
        ctx.stroke();
      }

      // Label
      if (feature.label) {
        ctx.font = '10px Inter, sans-serif';
        ctx.fillStyle = '#e2e8f0';
        ctx.textAlign = 'center';
        ctx.fillText(feature.label, pos.x, pos.y - r - 4);
      }
    }
  }

  /** @private Render zone polygons */
  _renderZones(ctx, projectFn) {
    for (const feature of this.features) {
      if (!feature.vertices || feature.vertices.length < 3) continue;
      const pixels = feature.vertices.map((v) => projectFn(v.lat, v.lng)).filter(Boolean);
      if (pixels.length < 3) continue;

      const fillColor = feature.fillColor ?? this.color + '30';
      const strokeColor = feature.strokeColor ?? this.color;

      ctx.beginPath();
      ctx.moveTo(pixels[0].x, pixels[0].y);
      for (let i = 1; i < pixels.length; i++) {
        ctx.lineTo(pixels[i].x, pixels[i].y);
      }
      ctx.closePath();
      ctx.fillStyle = fillColor;
      ctx.fill();
      ctx.strokeStyle = strokeColor;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Zone label
      if (feature.label) {
        const cx = pixels.reduce((s, p) => s + p.x, 0) / pixels.length;
        const cy = pixels.reduce((s, p) => s + p.y, 0) / pixels.length;
        ctx.font = '11px Inter, sans-serif';
        ctx.fillStyle = strokeColor;
        ctx.textAlign = 'center';
        ctx.fillText(feature.label, cx, cy);
      }
    }
  }

  /** @private Render detection markers */
  _renderDetections(ctx, projectFn) {
    for (const feature of this.features) {
      const pos = projectFn(feature.lat, feature.lng);
      if (!pos) continue;

      const color = feature.color ?? this.color;
      const size = feature.size ?? 6;

      // Diamond marker
      ctx.beginPath();
      ctx.moveTo(pos.x, pos.y - size);
      ctx.lineTo(pos.x + size, pos.y);
      ctx.lineTo(pos.x, pos.y + size);
      ctx.lineTo(pos.x - size, pos.y);
      ctx.closePath();
      ctx.fillStyle = color;
      ctx.fill();
      ctx.strokeStyle = '#0f172a';
      ctx.lineWidth = 1;
      ctx.stroke();

      // Confidence ring
      if (feature.confidence !== undefined) {
        ctx.beginPath();
        ctx.arc(pos.x, pos.y, size + 3, 0, Math.PI * 2 * feature.confidence);
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.stroke();
      }

      // Type label
      if (feature.type) {
        ctx.font = '9px Inter, sans-serif';
        ctx.fillStyle = '#94a3b8';
        ctx.textAlign = 'left';
        ctx.fillText(feature.type, pos.x + size + 4, pos.y + 3);
      }
    }
  }

  /** @private Render route polyline */
  _renderRoute(ctx, projectFn) {
    if (this.features.length === 0) return;
    for (const route of this.features) {
      if (!route.points || route.points.length < 2) continue;
      const pixels = route.points.map((p) => projectFn(p.lat, p.lng)).filter(Boolean);
      if (pixels.length < 2) continue;

      ctx.beginPath();
      ctx.moveTo(pixels[0].x, pixels[0].y);
      for (let i = 1; i < pixels.length; i++) {
        ctx.lineTo(pixels[i].x, pixels[i].y);
      }
      ctx.strokeStyle = route.color ?? this.color;
      ctx.lineWidth = route.width ?? 3;
      ctx.lineCap = 'round';
      ctx.lineJoin = 'round';
      if (route.dashArray) {
        ctx.setLineDash(route.dashArray);
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  /** @private Render heatmap overlay */
  _renderHeatmap(ctx, projectFn) {
    for (const feature of this.features) {
      const pos = projectFn(feature.lat, feature.lng);
      if (!pos) continue;

      const r = feature.radius ?? 20;
      const intensity = feature.intensity ?? 0.5;
      const gradient = ctx.createRadialGradient(pos.x, pos.y, 0, pos.x, pos.y, r);
      gradient.addColorStop(0, `rgba(239, 68, 68, ${intensity})`);
      gradient.addColorStop(0.5, `rgba(245, 158, 11, ${intensity * 0.5})`);
      gradient.addColorStop(1, 'rgba(59, 130, 246, 0)');
      ctx.fillStyle = gradient;
      ctx.fillRect(pos.x - r, pos.y - r, r * 2, r * 2);
    }
  }

  /**
   * Clear all features.
   */
  clear() {
    this.features = [];
  }
}

// ============================================================
// Layer Manager
// ============================================================

/**
 * Manages tile layers and overlay layers with visibility, ordering, and legend generation.
 * @class LayerManager
 */
export class LayerManager {
  /**
   * @param {Object} [options={}] - Manager options
   * @param {number} [options.maxConcurrentLoads=6] - Max parallel tile loads
   * @param {boolean} [options.enableCaching=true] - Enable tile caching
   */
  constructor(options = {}) {
    /** @type {Map<string, TileLayer>} */
    this._tileLayers = new Map();
    /** @type {Map<string, OverlayLayer>} */
    this._overlayLayers = new Map();
    /** @type {string|null} - Currently active base tile layer */
    this._activeBaseLayer = null;
    this._maxConcurrentLoads = options.maxConcurrentLoads ?? 6;
    this._enableCaching = options.enableCaching ?? true;
    /** @type {Function|null} */
    this._onChange = null;
  }

  /**
   * Add a tile layer.
   * @param {Object} source - Tile source configuration
   * @param {Object} [options] - Layer options
   * @returns {TileLayer} The created tile layer
   */
  addTileLayer(source, options = {}) {
    const layer = new TileLayer(source, options);
    this._tileLayers.set(layer.id, layer);
    if (!this._activeBaseLayer) {
      this._activeBaseLayer = layer.id;
    }
    this._emitChange();
    return layer;
  }

  /**
   * Add an overlay layer.
   * @param {Object} config - Overlay configuration
   * @returns {OverlayLayer} The created overlay layer
   */
  addOverlayLayer(config) {
    const layer = new OverlayLayer(config);
    this._overlayLayers.set(layer.id, layer);
    this._emitChange();
    return layer;
  }

  /**
   * Remove a tile layer by ID.
   * @param {string} layerId
   */
  removeTileLayer(layerId) {
    const layer = this._tileLayers.get(layerId);
    if (layer) {
      layer.clearCache();
      this._tileLayers.delete(layerId);
      if (this._activeBaseLayer === layerId) {
        this._activeBaseLayer = this._tileLayers.keys().next().value ?? null;
      }
      this._emitChange();
    }
  }

  /**
   * Remove an overlay layer by ID.
   * @param {string} layerId
   */
  removeOverlayLayer(layerId) {
    this._overlayLayers.delete(layerId);
    this._emitChange();
  }

  /**
   * Set the active base tile layer.
   * @param {string} layerId - Tile layer ID to activate
   */
  setActiveBaseLayer(layerId) {
    if (this._tileLayers.has(layerId)) {
      this._activeBaseLayer = layerId;
      this._emitChange();
    }
  }

  /**
   * Get the active base tile layer.
   * @returns {TileLayer|null}
   */
  getActiveBaseLayer() {
    return this._activeBaseLayer ? this._tileLayers.get(this._activeBaseLayer) : null;
  }

  /**
   * Toggle visibility of a layer.
   * @param {string} layerId - Layer ID
   * @param {boolean} [visible] - Force visibility
   */
  toggleLayer(layerId, visible) {
    const tileLayer = this._tileLayers.get(layerId);
    if (tileLayer) {
      tileLayer.toggleVisibility(visible);
      this._emitChange();
      return;
    }
    const overlayLayer = this._overlayLayers.get(layerId);
    if (overlayLayer) {
      overlayLayer.visible = visible ?? !overlayLayer.visible;
      this._emitChange();
    }
  }

  /**
   * Reorder overlay layers by z-index.
   * @param {string} layerId - Layer to reorder
   * @param {number} zIndex - New z-index
   */
  reorderLayer(layerId, zIndex) {
    const layer = this._overlayLayers.get(layerId);
    if (layer) {
      layer.zIndex = zIndex;
      this._emitChange();
    }
  }

  /**
   * Get all layers sorted by z-index for rendering.
   * @returns {{tileLayers: Array<TileLayer>, overlayLayers: Array<OverlayLayer>}}
   */
  getLayersForRendering() {
    const tileLayers = Array.from(this._tileLayers.values()).filter((l) => l.visible);
    const overlayLayers = Array.from(this._overlayLayers.values())
      .filter((l) => l.visible)
      .sort((a, b) => a.zIndex - b.zIndex);
    return { tileLayers, overlayLayers };
  }

  /**
   * Generate a legend for all visible layers.
   * @returns {Array<{id: string, name: string, type: string, color: string, visible: boolean, features: number}>}
   */
  generateLegend() {
    const legend = [];

    for (const layer of this._tileLayers.values()) {
      legend.push({
        id: layer.id,
        name: layer.name,
        type: 'tile',
        color: '#94a3b8',
        visible: layer.visible,
        features: 0,
        category: layer.category,
      });
    }

    for (const layer of this._overlayLayers.values()) {
      legend.push({
        id: layer.id,
        name: layer.name,
        type: layer.type,
        color: layer.color,
        visible: layer.visible,
        features: layer.features.length,
      });
    }

    return legend;
  }

  /**
   * Render the legend to a DOM container.
   * @param {HTMLElement} container - Legend container
   */
  renderLegendUI(container) {
    if (!(container instanceof HTMLElement)) return;
    container.innerHTML = '';

    const legend = this.generateLegend();
    const list = document.createElement('div');
    list.style.cssText = 'display:flex;flex-direction:column;gap:4px;font-size:12px;';

    // Tile layer section
    const tileHeader = document.createElement('div');
    tileHeader.textContent = 'Base Maps';
    tileHeader.style.cssText = 'font-weight:bold;color:#94a3b8;margin-top:8px;margin-bottom:4px;';
    list.appendChild(tileHeader);

    for (const item of legend.filter((l) => l.type === 'tile')) {
      const row = this._createLegendRow(item, item.id === this._activeBaseLayer);
      list.appendChild(row);
    }

    // Overlay section
    const overlayHeader = document.createElement('div');
    overlayHeader.textContent = 'Overlays';
    overlayHeader.style.cssText = 'font-weight:bold;color:#94a3b8;margin-top:12px;margin-bottom:4px;';
    list.appendChild(overlayHeader);

    for (const item of legend.filter((l) => l.type !== 'tile')) {
      const row = this._createLegendRow(item);
      list.appendChild(row);
    }

    container.appendChild(list);
  }

  /** @private Create a legend row element */
  _createLegendRow(item, isBase = false) {
    const row = document.createElement('div');
    row.style.cssText = `display:flex;align-items:center;gap:6px;padding:3px 6px;border-radius:4px;cursor:pointer;transition:background 0.15s;${isBase ? 'background:rgba(59,130,246,0.15);' : ''}`;

    row.addEventListener('mouseenter', () => { row.style.background = 'rgba(148,163,184,0.1)'; });
    row.addEventListener('mouseleave', () => { row.style.background = isBase ? 'rgba(59,130,246,0.15)' : 'transparent'; });

    // Color dot
    const dot = document.createElement('span');
    dot.style.cssText = `width:8px;height:8px;border-radius:50%;background:${item.color};flex-shrink:0;`;
    row.appendChild(dot);

    // Name
    const name = document.createElement('span');
    name.textContent = item.name;
    name.style.cssText = `color:${item.visible ? '#e2e8f0' : '#64748b'};flex:1;`;
    row.appendChild(name);

    // Feature count
    if (item.features > 0) {
      const count = document.createElement('span');
      count.textContent = `(${item.features})`;
      count.style.cssText = 'color:#64748b;font-size:10px;';
      row.appendChild(count);
    }

    // Visibility toggle
    const toggle = document.createElement('input');
    toggle.type = 'checkbox';
    toggle.checked = item.visible;
    toggle.style.cssText = 'margin:0;cursor:pointer;';
    toggle.addEventListener('change', () => {
      this.toggleLayer(item.id, toggle.checked);
    });
    row.appendChild(toggle);

    // Base layer selection
    if (item.type === 'tile') {
      row.addEventListener('click', (e) => {
        if (e.target !== toggle) {
          this.setActiveBaseLayer(item.id);
          this.renderLegendUI(row.parentElement.parentElement);
        }
      });
    }

    return row;
  }

  /**
   * Apply a style configuration to all layers.
   * @param {Object} styleConfig - Style configuration
   * @param {string} [styleConfig.theme] - Theme preset ('dark'|'light'|'satellite')
   * @param {number} [styleConfig.opacity] - Global tile opacity
   * @param {Object} [styleConfig.overlays] - Per-overlay style overrides
   */
  applyStyle(styleConfig) {
    if (styleConfig.theme) {
      const themeMap = {
        dark: 'carto-dark',
        light: 'carto-light',
        satellite: 'arcgis-satellite',
      };
      const layerId = themeMap[styleConfig.theme];
      if (layerId && this._tileLayers.has(layerId)) {
        this.setActiveBaseLayer(layerId);
      }
    }

    if (styleConfig.opacity !== undefined) {
      for (const layer of this._tileLayers.values()) {
        layer.setOpacity(styleConfig.opacity);
      }
    }

    if (styleConfig.overlays) {
      for (const [id, style] of Object.entries(styleConfig.overlays)) {
        const layer = this._overlayLayers.get(id);
        if (layer) {
          Object.assign(layer.style, style);
        }
      }
    }

    this._emitChange();
  }

  /**
   * Register a change callback.
   * @param {Function} callback
   */
  onChange(callback) {
    this._onChange = callback;
  }

  /** @private */
  _emitChange() {
    if (this._onChange) {
      this._onChange({
        tileLayers: this._tileLayers.size,
        overlayLayers: this._overlayLayers.size,
        activeBase: this._activeBaseLayer,
      });
    }
  }

  /**
   * Clear all layers and caches.
   */
  clearAll() {
    for (const layer of this._tileLayers.values()) {
      layer.clearCache();
    }
    this._tileLayers.clear();
    this._overlayLayers.clear();
    this._activeBaseLayer = null;
  }
}

export default LayerManager;
