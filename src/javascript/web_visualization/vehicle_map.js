/**
 * @fileoverview Vehicle Map Module - Interactive vehicle map with Leaflet/MapLibre integration,
 * vehicle position marker with heading, trajectory trail, geofencing zones, coordinate conversion,
 * auto-follow mode, and zoom controls.
 *
 * @module vehicle_map
 * @version 2.0.0
 * @author Autonomous Vehicle Control System Team
 */

// ============================================================
// Coordinate Conversion Utilities (WGS84 ↔ UTM)
// ============================================================

/**
 * WGS84 ellipsoid parameters for geodetic calculations.
 * @constant {Object}
 */
const WGS84 = Object.freeze({
  /** Semi-major axis in meters */
  A: 6378137.0,
  /** Semi-minor axis in meters */
  B: 6356752.314245,
  /** Flattening */
  F: 1 / 298.257223563,
  /** First eccentricity squared */
  E2: 0.00669437999014,
  /** Second eccentricity squared */
  EP2: 0.00673949674228,
});

/**
 * UTM zone lookup from longitude.
 * @param {number} longitude - Longitude in degrees (-180 to 180)
 * @returns {number} UTM zone number (1-60)
 */
export function longitudeToUTMZone(longitude) {
  if (longitude < -180 || longitude > 180) {
    throw new RangeError(`Longitude must be between -180 and 180, got ${longitude}`);
  }
  return Math.floor((longitude + 180) / 6) + 1;
}

/**
 * Convert WGS84 latitude/longitude to UTM coordinates.
 * @param {number} lat - Latitude in degrees
 * @param {number} lon - Longitude in degrees
 * @returns {{easting: number, northing: number, zone: number, letter: string}}
 */
export function wgs84ToUTM(lat, lon) {
  if (lat < -80 || lat > 84) {
    throw new RangeError(`UTM conversion only valid for latitudes -80 to 84, got ${lat}`);
  }

  const zone = longitudeToUTMZone(lon);
  const letter = lat >= 0
    ? String.fromCharCode(78 + Math.floor(lat / 8))
    : String.fromCharCode(67 + Math.floor((-lat) / 8));

  const latRad = (lat * Math.PI) / 180;
  const lonRad = (lon * Math.PI) / 180;
  const lonOrigin = ((zone - 1) * 6 - 180 + 3) * Math.PI / 180;

  const N = WGS84.A / Math.sqrt(1 - WGS84.E2 * Math.sin(latRad) ** 2);
  const T = Math.tan(latRad) ** 2;
  const C = WGS84.EP2 * Math.cos(latRad) ** 2;
  const A_val = Math.cos(latRad) * (lonRad - lonOrigin);

  const M = WGS84.A * (
    (1 - WGS84.E2 / 4 - 3 * WGS84.E2 ** 2 / 64 - 5 * WGS84.E2 ** 3 / 256) * latRad
    - (3 * WGS84.E2 / 8 + 3 * WGS84.E2 ** 2 / 32 + 45 * WGS84.E2 ** 3 / 1024) * Math.sin(2 * latRad)
    + (15 * WGS84.E2 ** 2 / 256 + 45 * WGS84.E2 ** 3 / 1024) * Math.sin(4 * latRad)
    - (35 * WGS84.E2 ** 3 / 3072) * Math.sin(6 * latRad)
  );

  const k0 = 0.9996;
  const easting = k0 * N * (
    A_val
    + (1 - T + C) * A_val ** 3 / 6
    + (5 - 18 * T + T ** 2 + 72 * C - 58 * WGS84.EP2) * A_val ** 5 / 120
  ) + 500000.0;

  const northing = k0 * (
    M + N * Math.tan(latRad) * (
      A_val ** 2 / 2
      + (5 - T + 9 * C + 4 * C ** 2) * A_val ** 4 / 24
      + (61 - 58 * T + T ** 2 + 600 * C - 330 * WGS84.EP2) * A_val ** 6 / 720
    )
  );

  return {
    easting: Math.round(easting * 100) / 100,
    northing: Math.round((lat < 0 ? northing + 10000000.0 : northing) * 100) / 100,
    zone,
    letter,
  };
}

/**
 * Convert UTM coordinates back to WGS84 latitude/longitude.
 * @param {number} easting - UTM easting in meters
 * @param {number} northing - UTM northing in meters
 * @param {number} zone - UTM zone number
 * @param {boolean} [northern=true] - Northern hemisphere
 * @returns {{latitude: number, longitude: number}}
 */
export function utmToWGS84(easting, northing, zone, northern = true) {
  const k0 = 0.9996;
  const e1 = (1 - Math.sqrt(1 - WGS84.E2)) / (1 + Math.sqrt(1 - WGS84.E2));
  const x = easting - 500000.0;
  const y = northern ? northing : northing - 10000000.0;
  const lonOrigin = (zone - 1) * 6 - 180 + 3;

  const M = y / k0;
  const mu = M / (WGS84.A * (1 - WGS84.E2 / 4 - 3 * WGS84.E2 ** 2 / 64 - 5 * WGS84.E2 ** 3 / 256));

  const phi1 = mu
    + (3 * e1 / 2 - 27 * e1 ** 3 / 32) * Math.sin(2 * mu)
    + (21 * e1 ** 2 / 16 - 55 * e1 ** 4 / 32) * Math.sin(4 * mu)
    + (151 * e1 ** 3 / 96) * Math.sin(6 * mu);

  const N1 = WGS84.A / Math.sqrt(1 - WGS84.E2 * Math.sin(phi1) ** 2);
  const T1 = Math.tan(phi1) ** 2;
  const C1 = WGS84.EP2 * Math.cos(phi1) ** 2;
  const R1 = WGS84.A * (1 - WGS84.E2) / (1 - WGS84.E2 * Math.sin(phi1) ** 2) ** 1.5;
  const D = x / (N1 * k0);

  const lat = phi1 - (N1 * Math.tan(phi1) / R1) * (
    D ** 2 / 2
    - (5 + 3 * T1 + 10 * C1 - 4 * C1 ** 2 - 9 * WGS84.EP2) * D ** 4 / 24
    + (61 + 90 * T1 + 298 * C1 + 45 * T1 ** 2 - 252 * WGS84.EP2 - 3 * C1 ** 2) * D ** 6 / 720
  );

  const lon = (D
    - (1 + 2 * T1 + C1) * D ** 3 / 6
    + (5 - 2 * C1 + 28 * T1 - 3 * C1 ** 2 + 8 * WGS84.EP2 + 24 * T1 ** 2) * D ** 5 / 120
  ) / Math.cos(phi1);

  return {
    latitude: (lat * 180) / Math.PI,
    longitude: lonOrigin + (lon * 180) / Math.PI,
  };
}

// ============================================================
// Map Tile Manager
// ============================================================

/**
 * Manages map tile sources and caching for efficient rendering.
 * @class TileManager
 */
export class TileManager {
  /**
   * @param {Object} [options={}] - Configuration options
   * @param {number} [options.cacheSize=256] - Maximum cached tiles
   * @param {number} [options.tileSize=256] - Tile size in pixels
   * @param {number} [options.maxZoom=19] - Maximum zoom level
   * @param {number} [options.minZoom=1] - Minimum zoom level
   */
  constructor(options = {}) {
    /** @private @type {Map<string, HTMLImageElement>} */
    this._cache = new Map();
    this._cacheSize = options.cacheSize ?? 256;
    this._tileSize = options.tileSize ?? 256;
    this._maxZoom = options.maxZoom ?? 19;
    this._minZoom = options.minZoom ?? 1;
    /** @type {boolean} */
    this.loading = false;
    /** @type {Set<string>} */
    this._pendingRequests = new Set();
  }

  /**
   * Build tile URL from coordinates.
   * @param {number} x - Tile X coordinate
   * @param {number} y - Tile Y coordinate
   * @param {number} z - Zoom level
   * @param {string} [style='streets'] - Map style
   * @returns {string} Tile URL
   */
  buildTileURL(x, y, z, style = 'streets') {
    const servers = ['a', 'b', 'c'];
    const server = servers[Math.abs(x + y) % 3];
    const styleMap = {
      streets: `https://${server}.tile.openstreetmap.org/${z}/${x}/${y}.png`,
      satellite: `https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/${z}/${y}/${x}`,
      terrain: `https://${server}.tile.opentopomap.org/${z}/${x}/${y}.png`,
    };
    return styleMap[style] ?? styleMap.streets;
  }

  /**
   * Load a tile image with caching.
   * @param {number} x - Tile X
   * @param {number} y - Tile Y
   * @param {number} z - Zoom level
   * @param {string} [style] - Map style
   * @returns {Promise<HTMLImageElement>} Loaded tile image
   */
  async loadTile(x, y, z, style = 'streets') {
    const key = `${style}/${z}/${x}/${y}`;
    if (this._cache.has(key)) {
      return this._cache.get(key);
    }
    if (this._pendingRequests.has(key)) {
      return new Promise((resolve) => {
        const check = setInterval(() => {
          if (this._cache.has(key)) {
            clearInterval(check);
            resolve(this._cache.get(key));
          }
        }, 50);
      });
    }

    this._pendingRequests.add(key);
    const url = this.buildTileURL(x, y, z, style);

    return new Promise((resolve, reject) => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        this._evictIfNeeded();
        this._cache.set(key, img);
        this._pendingRequests.delete(key);
        resolve(img);
      };
      img.onerror = () => {
        this._pendingRequests.delete(key);
        reject(new Error(`Failed to load tile: ${key}`));
      };
      img.src = url;
    });
  }

  /** @private Evict oldest cache entries if size exceeded */
  _evictIfNeeded() {
    if (this._cache.size >= this._cacheSize) {
      const firstKey = this._cache.keys().next().value;
      this._cache.delete(firstKey);
    }
  }

  /**
   * Clear the tile cache.
   */
  clearCache() {
    this._cache.clear();
    this._pendingRequests.clear();
  }
}

// ============================================================
// Geofencing Zone
// ============================================================

/**
 * Represents a geofencing zone on the map.
 * @class GeofenceZone
 */
export class GeofenceZone {
  /**
   * @param {Object} config - Zone configuration
   * @param {string} config.id - Unique zone identifier
   * @param {string} config.name - Zone display name
   * @param {Array<{lat: number, lng: number}>} config.vertices - Polygon vertices
   * @param {'restricted'|'warning'|'safe'} config.type - Zone type
   * @param {number} [config.maxSpeed] - Speed limit in m/s for restricted zones
   */
  constructor({ id, name, vertices, type, maxSpeed }) {
    this.id = id;
    this.name = name;
    this.vertices = vertices;
    this.type = type;
    this.maxSpeed = maxSpeed ?? null;
    this.active = true;
    this._colorMap = {
      restricted: { fill: '#ef4444', stroke: '#dc2626', opacity: 0.25 },
      warning: { fill: '#f59e0b', stroke: '#d97706', opacity: 0.2 },
      safe: { fill: '#22c55e', stroke: '#16a34a', opacity: 0.15 },
    };
  }

  /**
   * Get styling for this zone.
   * @returns {{fill: string, stroke: string, opacity: number}}
   */
  getStyle() {
    return this._colorMap[this.type] ?? this._colorMap.safe;
  }

  /**
   * Check if a point is inside this geofence using ray-casting algorithm.
   * @param {number} lat - Point latitude
   * @param {number} lng - Point longitude
   * @returns {boolean} True if point is inside the zone
   */
  containsPoint(lat, lng) {
    const n = this.vertices.length;
    if (n < 3) return false;
    let inside = false;
    for (let i = 0, j = n - 1; i < n; j = i++) {
      const vi = this.vertices[i];
      const vj = this.vertices[j];
      if (((vi.lat > lng) !== (vj.lat > lng))
        && (lat < (vj.lng - vi.lng) * (lng - vi.lat) / (vj.lat - vi.lat) + vi.lng)) {
        inside = !inside;
      }
    }
    return inside;
  }
}

// ============================================================
// Vehicle Marker
// ============================================================

/**
 * Custom vehicle marker with heading indicator rendered on canvas.
 * @class VehicleMarker
 */
export class VehicleMarker {
  /**
   * @param {Object} [options={}] - Marker options
   * @param {number} [options.size=40] - Marker size in pixels
   * @param {string} [options.color='#3b82f6'] - Marker fill color
   * @param {string} [options.headingColor='#ffffff'] - Heading arrow color
   * @param {boolean} [options.showHeading=true] - Show heading indicator
   */
  constructor(options = {}) {
    this.size = options.size ?? 40;
    this.color = options.color ?? '#3b82f6';
    this.headingColor = options.headingColor ?? '#ffffff';
    this.showHeading = options.showHeading ?? true;
    /** @type {{lat: number, lng: number, heading: number}} */
    this.position = { lat: 0, lng: 0, heading: 0 };
    /** @private @type {HTMLCanvasElement|null} */
    this._canvas = null;
  }

  /**
   * Update the vehicle's position and heading.
   * @param {number} lat - Latitude
   * @param {number} lng - Longitude
   * @param {number} heading - Heading in degrees (0=North, clockwise)
   */
  updatePosition(lat, lng, heading) {
    this.position = { lat, lng, heading: heading % 360 };
  }

  /**
   * Render the vehicle marker to an offscreen canvas.
   * @returns {HTMLCanvasElement} Rendered marker canvas
   */
  render() {
    const s = this.size;
    if (!this._canvas) {
      this._canvas = document.createElement('canvas');
    }
    this._canvas.width = s * 2;
    this._canvas.height = s * 2;

    const ctx = this._canvas.getContext('2d');
    ctx.clearRect(0, 0, s * 2, s * 2);
    ctx.save();
    ctx.translate(s, s);

    // Vehicle body circle
    ctx.beginPath();
    ctx.arc(0, 0, s * 0.4, 0, Math.PI * 2);
    ctx.fillStyle = this.color;
    ctx.fill();
    ctx.strokeStyle = '#1e293b';
    ctx.lineWidth = 2;
    ctx.stroke();

    // Inner circle
    ctx.beginPath();
    ctx.arc(0, 0, s * 0.2, 0, Math.PI * 2);
    ctx.fillStyle = 'rgba(255,255,255,0.3)';
    ctx.fill();

    // Heading arrow
    if (this.showHeading) {
      const headingRad = (this.position.heading - 90) * Math.PI / 180;
      ctx.rotate(headingRad);
      ctx.beginPath();
      ctx.moveTo(s * 0.55, 0);
      ctx.lineTo(s * 0.25, -s * 0.12);
      ctx.lineTo(s * 0.25, s * 0.12);
      ctx.closePath();
      ctx.fillStyle = this.headingColor;
      ctx.fill();
    }

    ctx.restore();
    return this._canvas;
  }
}

// ============================================================
// Trajectory Trail
// ============================================================

/**
 * Manages the vehicle's trajectory trail with configurable length and styling.
 * @class TrajectoryTrail
 */
export class TrajectoryTrail {
  /**
   * @param {Object} [options={}] - Trail options
   * @param {number} [options.maxPoints=500] - Maximum trail points
   * @param {string} [options.color='#3b82f6'] - Trail color
   * @param {number} [options.width=3] - Trail width in pixels
   * @param {boolean} [options.fadeOut=true] - Fade older points
   * @param {number} [options.fadeSteps=10] - Number of gradient steps for fading
   */
  constructor(options = {}) {
    this.maxPoints = options.maxPoints ?? 500;
    this.color = options.color ?? '#3b82f6';
    this.width = options.width ?? 3;
    this.fadeOut = options.fadeOut ?? true;
    this.fadeSteps = options.fadeSteps ?? 10;
    /** @type {Array<{lat: number, lng: number, timestamp: number}>} */
    this.points = [];
  }

  /**
   * Add a point to the trail.
   * @param {number} lat - Latitude
   * @param {number} lng - Longitude
   * @param {number} [timestamp=Date.now()] - Timestamp
   */
  addPoint(lat, lng, timestamp = Date.now()) {
    this.points.push({ lat, lng, timestamp });
    if (this.points.length > this.maxPoints) {
      this.points.shift();
    }
  }

  /**
   * Get trail segments grouped for gradient rendering.
   * @returns {Array<{points: Array, opacity: number}>}
   */
  getSegments() {
    if (!this.fadeOut || this.points.length < 2) {
      return [{ points: this.points, opacity: 1.0 }];
    }
    const total = this.points.length;
    const segSize = Math.max(1, Math.ceil(total / this.fadeSteps));
    const segments = [];
    for (let i = 0; i < total; i += segSize) {
      const chunk = this.points.slice(i, i + segSize + 1);
      if (chunk.length < 2) continue;
      const opacity = ((i + segSize / 2) / total);
      segments.push({ points: chunk, opacity: Math.min(1, opacity) });
    }
    return segments;
  }

  /**
   * Clear the trail.
   */
  clear() {
    this.points = [];
  }
}

// ============================================================
// Main Vehicle Map
// ============================================================

/**
 * Interactive vehicle map integrating map tiles, vehicle markers, trajectory trails,
 * geofencing zones, auto-follow mode, and zoom controls.
 * @class VehicleMap
 */
export class VehicleMap {
  /**
   * @param {HTMLElement} container - DOM container for the map
   * @param {Object} [options={}] - Map configuration
   * @param {{lat: number, lng: number}} [options.center={lat: 39.9042, lng: 116.4074}] - Initial center
   * @param {number} [options.zoom=15] - Initial zoom level
   * @param {string} [options.style='streets'] - Default map style
   * @param {boolean} [options.autoFollow=true] - Enable auto-follow on vehicle position
   * @param {number} [options.trailLength=500] - Trajectory trail max points
   */
  constructor(container, options = {}) {
    if (!(container instanceof HTMLElement)) {
      throw new TypeError('VehicleMap requires a valid HTMLElement container');
    }

    this._container = container;
    this._center = options.center ?? { lat: 39.9042, lng: 116.4074 };
    this._zoom = options.zoom ?? 15;
    this._style = options.style ?? 'streets';
    this._autoFollow = options.autoFollow ?? true;
    this._minZoom = 1;
    this._maxZoom = 19;

    // Sub-systems
    this.tileManager = new TileManager({ maxZoom: this._maxZoom, minZoom: this._minZoom });
    this.marker = new VehicleMarker();
    this.trail = new TrajectoryTrail({ maxPoints: options.trailLength ?? 500 });
    /** @type {Map<string, GeofenceZone>} */
    this.geofences = new Map();

    // State
    this._offset = { x: 0, y: 0 };
    this._isDragging = false;
    this._dragStart = { x: 0, y: 0 };
    this._animFrameId = null;

    /** @type {Map<string, Function>} */
    this._listeners = new Map();

    this._initCanvas();
    this._bindEvents();
    this._renderLoop();
  }

  /** @private Initialize canvas element */
  _initCanvas() {
    this._canvas = document.createElement('canvas');
    this._canvas.style.width = '100%';
    this._canvas.style.height = '100%';
    this._canvas.style.display = 'block';
    this._container.appendChild(this._canvas);

    const rect = this._container.getBoundingClientRect();
    this._canvas.width = rect.width * window.devicePixelRatio;
    this._canvas.height = rect.height * window.devicePixelRatio;
    this._ctx = this._canvas.getContext('2d');
    this._ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    this._canvasWidth = rect.width;
    this._canvasHeight = rect.height;
  }

  /** @private Bind mouse/touch events */
  _bindEvents() {
    this._canvas.addEventListener('mousedown', (e) => {
      this._isDragging = true;
      this._dragStart = { x: e.clientX - this._offset.x, y: e.clientY - this._offset.y };
    });
    window.addEventListener('mousemove', (e) => {
      if (!this._isDragging) return;
      this._offset.x = e.clientX - this._dragStart.x;
      this._offset.y = e.clientY - this._dragStart.y;
    });
    window.addEventListener('mouseup', () => { this._isDragging = false; });
    this._canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      const delta = e.deltaY > 0 ? -1 : 1;
      this.setZoom(this._zoom + delta);
    }, { passive: false });

    // Touch events
    this._canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this._isDragging = true;
        this._dragStart = {
          x: e.touches[0].clientX - this._offset.x,
          y: e.touches[0].clientY - this._offset.y,
        };
      }
    }, { passive: true });
    this._canvas.addEventListener('touchmove', (e) => {
      if (!this._isDragging || e.touches.length !== 1) return;
      this._offset.x = e.touches[0].clientX - this._dragStart.x;
      this._offset.y = e.touches[0].clientY - this._dragStart.y;
    }, { passive: true });
    this._canvas.addEventListener('touchend', () => { this._isDragging = false; });

    window.addEventListener('resize', () => this._handleResize());
  }

  /** @private Handle window resize */
  _handleResize() {
    const rect = this._container.getBoundingClientRect();
    this._canvas.width = rect.width * window.devicePixelRatio;
    this._canvas.height = rect.height * window.devicePixelRatio;
    this._ctx = this._canvas.getContext('2d');
    this._ctx.scale(window.devicePixelRatio, window.devicePixelRatio);
    this._canvasWidth = rect.width;
    this._canvasHeight = rect.height;
  }

  /**
   * Convert lat/lng to pixel coordinates on the canvas.
   * @param {number} lat - Latitude
   * @param {number} lng - Longitude
   * @returns {{x: number, y: number}} Pixel coordinates
   */
  latLngToPixel(lat, lng) {
    const n = Math.pow(2, this._zoom);
    const x = ((lng + 180) / 360) * 256 * n + this._offset.x + this._canvasWidth / 2;
    const latRad = (lat * Math.PI) / 180;
    const y = (1 - Math.log(Math.tan(latRad) + 1 / Math.cos(latRad)) / Math.PI) / 2 * 256 * n
      + this._offset.y + this._canvasHeight / 2;
    return { x, y };
  }

  /**
   * Convert pixel coordinates to lat/lng.
   * @param {number} px - Pixel X
   * @param {number} py - Pixel Y
   * @returns {{lat: number, lng: number}}
   */
  pixelToLatLng(px, py) {
    const n = Math.pow(2, this._zoom);
    const x = px - this._offset.x - this._canvasWidth / 2;
    const y = py - this._offset.y - this._canvasHeight / 2;
    const lng = (x / (256 * n)) * 360 - 180;
    const latRad = Math.atan(Math.sinh(Math.PI * (1 - 2 * y / (256 * n))));
    return { lat: (latRad * 180) / Math.PI, lng };
  }

  /**
   * Set the map zoom level.
   * @param {number} zoom - New zoom level (clamped to min/max)
   */
  setZoom(zoom) {
    const old = this._zoom;
    this._zoom = Math.max(this._minZoom, Math.min(this._maxZoom, Math.round(zoom)));
    if (this._zoom !== old) {
      const factor = Math.pow(2, this._zoom - old);
      this._offset.x *= factor;
      this._offset.y *= factor;
      this._emit('zoomchange', { zoom: this._zoom });
    }
  }

  /**
   * Center the map on a coordinate.
   * @param {number} lat - Latitude
   * @param {number} lng - Longitude
   */
  setCenter(lat, lng) {
    const target = this.latLngToPixel(lat, lng);
    const center = { x: this._canvasWidth / 2, y: this._canvasHeight / 2 };
    this._offset.x += center.x - target.x;
    this._offset.y += center.y - target.y;
    this._center = { lat, lng };
  }

  /**
   * Update the vehicle position on the map.
   * @param {number} lat - Latitude
   * @param {number} lng - Longitude
   * @param {number} heading - Heading in degrees
   */
  updateVehicle(lat, lng, heading) {
    this.marker.updatePosition(lat, lng, heading);
    this.trail.addPoint(lat, lng);
    if (this._autoFollow) {
      this.setCenter(lat, lng);
    }
    this._emit('vehicleupdate', { lat, lng, heading });
  }

  /**
   * Add a geofencing zone to the map.
   * @param {GeofenceZone} zone - The geofence zone to add
   */
  addGeofence(zone) {
    if (!(zone instanceof GeofenceZone)) {
      throw new TypeError('Expected a GeofenceZone instance');
    }
    this.geofences.set(zone.id, zone);
    this._emit('geofenceadd', { zone });
  }

  /**
   * Remove a geofencing zone by ID.
   * @param {string} zoneId - Zone ID to remove
   */
  removeGeofence(zoneId) {
    this.geofences.delete(zoneId);
    this._emit('geofenceremove', { zoneId });
  }

  /**
   * Toggle auto-follow mode.
   * @param {boolean} [enabled] - Force enable/disable, or toggle if omitted
   */
  toggleAutoFollow(enabled) {
    this._autoFollow = enabled ?? !this._autoFollow;
    this._emit('autofollowchange', { enabled: this._autoFollow });
  }

  /**
   * Set the map tile style.
   * @param {'streets'|'satellite'|'terrain'} style - Map style
   */
  setStyle(style) {
    this._style = style;
    this.tileManager.clearCache();
    this._emit('stylechange', { style });
  }

  /** @private Main render loop */
  _renderLoop() {
    this._render();
    this._animFrameId = requestAnimationFrame(() => this._renderLoop());
  }

  /** @private Render all map layers */
  _render() {
    const ctx = this._ctx;
    const w = this._canvasWidth;
    const h = this._canvasHeight;
    ctx.clearRect(0, 0, w, h);

    // Background
    ctx.fillStyle = '#e2e8f0';
    ctx.fillRect(0, 0, w, h);

    // Tile grid
    this._renderTiles(ctx, w, h);

    // Geofence zones
    this._renderGeofences(ctx);

    // Trajectory trail
    this._renderTrail(ctx);

    // Vehicle marker
    this._renderMarker(ctx);
  }

  /** @private Render map tiles */
  _renderTiles(ctx, w, h) {
    const n = Math.pow(2, this._zoom);
    const tileSize = 256;
    const topLeft = this.pixelToLatLng(0, 0);
    const bottomRight = this.pixelToLatLng(w, h);

    const tileMinX = Math.max(0, Math.floor(((topLeft.lng + 180) / 360) * n));
    const tileMaxX = Math.min(n - 1, Math.ceil(((bottomRight.lng + 180) / 360) * n));
    const latRadTL = (topLeft.lat * Math.PI) / 180;
    const latRadBR = (bottomRight.lat * Math.PI) / 180;
    const tileMinY = Math.max(0, Math.floor((1 - Math.log(Math.tan(latRadTL) + 1 / Math.cos(latRadTL)) / Math.PI) / 2 * n));
    const tileMaxY = Math.min(n - 1, Math.ceil((1 - Math.log(Math.tan(latRadBR) + 1 / Math.cos(latRadBR)) / Math.PI) / 2 * n));

    for (let tx = tileMinX; tx <= tileMaxX; tx++) {
      for (let ty = tileMinY; ty <= tileMaxY; ty++) {
        const key = `${this._style}/${this._zoom}/${tx}/${ty}`;
        const cached = this.tileManager._cache.get(key);
        if (cached) {
          const px = this.latLngToPixel(0, (tx / n) * 360 - 180);
          const py = this.latLngToPixel((Math.atan(Math.sinh(Math.PI * (1 - 2 * ty / n))) * 180) / Math.PI, 0);
          const pos = this.latLngToPixel(
            (Math.atan(Math.sinh(Math.PI * (1 - 2 * ty / n))) * 180) / Math.PI,
            (tx / n) * 360 - 180,
          );
          ctx.drawImage(cached, pos.x - tileSize / 2, pos.y - tileSize / 2, tileSize, tileSize);
        }
      }
    }
  }

  /** @private Render geofence zones */
  _renderGeofences(ctx) {
    for (const zone of this.geofences.values()) {
      if (!zone.active) continue;
      const style = zone.getStyle();
      const vertices = zone.vertices.map((v) => this.latLngToPixel(v.lat, v.lng));
      if (vertices.length < 3) continue;

      ctx.beginPath();
      ctx.moveTo(vertices[0].x, vertices[0].y);
      for (let i = 1; i < vertices.length; i++) {
        ctx.lineTo(vertices[i].x, vertices[i].y);
      }
      ctx.closePath();
      ctx.fillStyle = style.fill;
      ctx.globalAlpha = style.opacity;
      ctx.fill();
      ctx.globalAlpha = 1;
      ctx.strokeStyle = style.stroke;
      ctx.lineWidth = 2;
      ctx.stroke();

      // Zone label
      const cx = vertices.reduce((s, v) => s + v.x, 0) / vertices.length;
      const cy = vertices.reduce((s, v) => s + v.y, 0) / vertices.length;
      ctx.font = '12px Inter, sans-serif';
      ctx.fillStyle = style.stroke;
      ctx.textAlign = 'center';
      ctx.fillText(zone.name, cx, cy);
    }
  }

  /** @private Render trajectory trail */
  _renderTrail(ctx) {
    const segments = this.trail.getSegments();
    for (const seg of segments) {
      if (seg.points.length < 2) continue;
      ctx.beginPath();
      const first = this.latLngToPixel(seg.points[0].lat, seg.points[0].lng);
      ctx.moveTo(first.x, first.y);
      for (let i = 1; i < seg.points.length; i++) {
        const p = this.latLngToPixel(seg.points[i].lat, seg.points[i].lng);
        ctx.lineTo(p.x, p.y);
      }
      ctx.strokeStyle = this.trail.color;
      ctx.lineWidth = this.trail.width;
      ctx.globalAlpha = seg.opacity;
      ctx.stroke();
      ctx.globalAlpha = 1;
    }
  }

  /** @private Render vehicle marker */
  _renderMarker(ctx) {
    const { lat, lng } = this.marker.position;
    if (lat === 0 && lng === 0) return;
    const pos = this.latLngToPixel(lat, lng);
    const markerCanvas = this.marker.render();
    ctx.drawImage(markerCanvas, pos.x - this.marker.size, pos.y - this.marker.size);
  }

  /**
   * Register an event listener.
   * @param {string} event - Event name
   * @param {Function} callback - Event handler
   */
  on(event, callback) {
    this._listeners.set(event, callback);
  }

  /** @private Emit an event */
  _emit(event, data) {
    const cb = this._listeners.get(event);
    if (cb) cb(data);
  }

  /**
   * Destroy the map and clean up resources.
   */
  destroy() {
    if (this._animFrameId) {
      cancelAnimationFrame(this._animFrameId);
    }
    this.tileManager.clearCache();
    this.trail.clear();
    this.geofences.clear();
    this._listeners.clear();
    this._canvas.remove();
  }
}

export default VehicleMap;
