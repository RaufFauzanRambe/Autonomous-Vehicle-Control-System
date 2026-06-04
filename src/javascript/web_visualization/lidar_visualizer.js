/**
 * @fileoverview LiDAR Point Cloud Visualizer - WebGL/Canvas rendering of point cloud data
 * with coloring by intensity/height/distance, rotation/zoom/pan, ground plane filtering,
 * bounding box overlay, point size adjustment, sector view, and downsampling for performance.
 *
 * @module lidar_visualizer
 * @version 2.0.0
 * @author Autonomous Vehicle Control System Team
 */

// ============================================================
// Matrix and Vector Math Utilities
// ============================================================

/**
 * 4x4 matrix operations for 3D transformations.
 */
export class Mat4 {
  /**
   * Create an identity matrix.
   * @returns {Float32Array} 4x4 identity matrix
   */
  static identity() {
    const m = new Float32Array(16);
    m[0] = m[5] = m[10] = m[15] = 1;
    return m;
  }

  /**
   * Create a perspective projection matrix.
   * @param {number} fov - Field of view in radians
   * @param {number} aspect - Aspect ratio (width/height)
   * @param {number} near - Near clipping plane
   * @param {number} far - Far clipping plane
   * @returns {Float32Array} Perspective matrix
   */
  static perspective(fov, aspect, near, far) {
    const f = 1.0 / Math.tan(fov / 2);
    const nf = 1 / (near - far);
    const m = new Float32Array(16);
    m[0] = f / aspect;
    m[5] = f;
    m[10] = (far + near) * nf;
    m[11] = -1;
    m[14] = 2 * far * near * nf;
    return m;
  }

  /**
   * Create a lookAt view matrix.
   * @param {number[]} eye - Camera position [x,y,z]
   * @param {number[]} center - Look-at target [x,y,z]
   * @param {number[]} up - Up vector [x,y,z]
   * @returns {Float32Array} View matrix
   */
  static lookAt(eye, center, up) {
    const zx = eye[0] - center[0], zy = eye[1] - center[1], zz = eye[2] - center[2];
    let len = Math.max(1e-6, Math.sqrt(zx * zx + zy * zy + zz * zz));
    const fz = [zx / len, zy / len, zz / len];

    const sx = up[1] * fz[2] - up[2] * fz[1];
    const sy = up[2] * fz[0] - up[0] * fz[2];
    const sz = up[0] * fz[1] - up[1] * fz[0];
    len = Math.max(1e-6, Math.sqrt(sx * sx + sy * sy + sz * sz));
    const s = [sx / len, sy / len, sz / len];

    const u = [s[1] * fz[2] - s[2] * fz[1], s[2] * fz[0] - s[0] * fz[2], s[0] * fz[1] - s[1] * fz[0]];

    const m = new Float32Array(16);
    m[0] = s[0]; m[4] = s[1]; m[8] = s[2];  m[12] = -(s[0] * eye[0] + s[1] * eye[1] + s[2] * eye[2]);
    m[1] = u[0]; m[5] = u[1]; m[9] = u[2];  m[13] = -(u[0] * eye[0] + u[1] * eye[1] + u[2] * eye[2]);
    m[2] = fz[0]; m[6] = fz[1]; m[10] = fz[2]; m[14] = -(fz[0] * eye[0] + fz[1] * eye[1] + fz[2] * eye[2]);
    m[3] = 0; m[7] = 0; m[11] = 0; m[15] = 1;
    return m;
  }

  /**
   * Multiply two 4x4 matrices.
   * @param {Float32Array} a - First matrix
   * @param {Float32Array} b - Second matrix
   * @returns {Float32Array} Product matrix
   */
  static multiply(a, b) {
    const out = new Float32Array(16);
    for (let i = 0; i < 4; i++) {
      for (let j = 0; j < 4; j++) {
        out[j * 4 + i] = a[i] * b[j * 4] + a[4 + i] * b[j * 4 + 1] + a[8 + i] * b[j * 4 + 2] + a[12 + i] * b[j * 4 + 3];
      }
    }
    return out;
  }

  /**
   * Create a rotation matrix around an arbitrary axis.
   * @param {number} angle - Angle in radians
   * @param {number} x - Axis X component
   * @param {number} y - Axis Y component
   * @param {number} z - Axis Z component
   * @returns {Float32Array} Rotation matrix
   */
  static fromAxisRotation(angle, x, y, z) {
    let len = Math.sqrt(x * x + y * y + z * z);
    if (len < 1e-6) return Mat4.identity();
    len = 1 / len;
    const ax = x * len, ay = y * len, az = z * len;
    const s = Math.sin(angle), c = Math.cos(angle), t = 1 - c;
    const m = new Float32Array(16);
    m[0] = t * ax * ax + c;     m[1] = t * ax * ay + s * az; m[2] = t * ax * az - s * ay;
    m[4] = t * ax * ay - s * az; m[5] = t * ay * ay + c;     m[6] = t * ay * az + s * ax;
    m[8] = t * ax * az + s * ay; m[9] = t * ay * az - s * ax; m[10] = t * az * az + c;
    m[15] = 1;
    return m;
  }
}

// ============================================================
// Point Cloud Data Structure
// ============================================================

/**
 * Represents a LiDAR point cloud dataset with optimized storage.
 * @class PointCloud
 */
export class PointCloud {
  /**
   * @param {number} capacity - Pre-allocated point capacity
   */
  constructor(capacity = 100000) {
    /** @type {Float32Array} X, Y, Z positions */
    this.positions = new Float32Array(capacity * 3);
    /** @type {Float32Array} Intensity values [0, 1] */
    this.intensities = new Float32Array(capacity);
    /** @type {Float32Array} Ring/channel indices */
    this.rings = new Float32Array(capacity);
    /** @type {number} Actual number of points */
    this.count = 0;
    /** @type {number} Maximum capacity */
    this.capacity = capacity;
    /** @type {{minX: number, maxX: number, minY: number, maxY: number, minZ: number, maxZ: number}} */
    this.bounds = { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity, minZ: Infinity, maxZ: -Infinity };
  }

  /**
   * Add a point to the cloud.
   * @param {number} x - X coordinate
   * @param {number} y - Y coordinate
   * @param {number} z - Z coordinate
   * @param {number} [intensity=1.0] - Intensity value [0, 1]
   * @param {number} [ring=0] - Ring index
   */
  addPoint(x, y, z, intensity = 1.0, ring = 0) {
    if (this.count >= this.capacity) return;
    const i = this.count;
    const i3 = i * 3;
    this.positions[i3] = x;
    this.positions[i3 + 1] = y;
    this.positions[i3 + 2] = z;
    this.intensities[i] = intensity;
    this.rings[i] = ring;
    this.count++;

    this.bounds.minX = Math.min(this.bounds.minX, x);
    this.bounds.maxX = Math.max(this.bounds.maxX, x);
    this.bounds.minY = Math.min(this.bounds.minY, y);
    this.bounds.maxY = Math.max(this.bounds.maxY, y);
    this.bounds.minZ = Math.min(this.bounds.minZ, z);
    this.bounds.maxZ = Math.max(this.bounds.maxZ, z);
  }

  /**
   * Get the center of the bounding box.
   * @returns {{x: number, y: number, z: number}}
   */
  getCenter() {
    return {
      x: (this.bounds.minX + this.bounds.maxX) / 2,
      y: (this.bounds.minY + this.bounds.maxY) / 2,
      z: (this.bounds.minZ + this.bounds.maxZ) / 2,
    };
  }

  /**
   * Downsample the point cloud using voxel grid filtering.
   * @param {number} voxelSize - Voxel grid cell size
   * @returns {PointCloud} Downsampled point cloud
   */
  downsample(voxelSize) {
    const voxelMap = new Map();
    for (let i = 0; i < this.count; i++) {
      const i3 = i * 3;
      const vx = Math.floor(this.positions[i3] / voxelSize);
      const vy = Math.floor(this.positions[i3 + 1] / voxelSize);
      const vz = Math.floor(this.positions[i3 + 2] / voxelSize);
      const key = `${vx},${vy},${vz}`;
      if (!voxelMap.has(key)) {
        voxelMap.set(key, { x: 0, y: 0, z: 0, intensity: 0, ring: 0, count: 0 });
      }
      const v = voxelMap.get(key);
      v.x += this.positions[i3];
      v.y += this.positions[i3 + 1];
      v.z += this.positions[i3 + 2];
      v.intensity += this.intensities[i];
      v.ring = this.rings[i];
      v.count++;
    }

    const result = new PointCloud(voxelMap.size);
    for (const v of voxelMap.values()) {
      result.addPoint(v.x / v.count, v.y / v.count, v.z / v.count, v.intensity / v.count, v.ring);
    }
    return result;
  }

  /**
   * Filter out ground plane points using height threshold.
   * @param {number} heightThreshold - Points below this Z value are ground
   * @returns {PointCloud} Filtered point cloud without ground points
   */
  filterGroundPlane(heightThreshold) {
    const result = new PointCloud(this.count);
    for (let i = 0; i < this.count; i++) {
      const i3 = i * 3;
      if (this.positions[i3 + 2] > heightThreshold) {
        result.addPoint(
          this.positions[i3], this.positions[i3 + 1], this.positions[i3 + 2],
          this.intensities[i], this.rings[i],
        );
      }
    }
    return result;
  }

  /**
   * Extract a sector (angular wedge) of the point cloud.
   * @param {number} startAngle - Start angle in degrees
   * @param {number} endAngle - End angle in degrees
   * @returns {PointCloud} Sector point cloud
   */
  extractSector(startAngle, endAngle) {
    const result = new PointCloud(this.count);
    const sRad = (startAngle * Math.PI) / 180;
    const eRad = (endAngle * Math.PI) / 180;
    for (let i = 0; i < this.count; i++) {
      const i3 = i * 3;
      const angle = Math.atan2(this.positions[i3 + 1], this.positions[i3]);
      const normalizedAngle = ((angle + 2 * Math.PI) % (2 * Math.PI));
      if (normalizedAngle >= sRad && normalizedAngle <= eRad) {
        result.addPoint(
          this.positions[i3], this.positions[i3 + 1], this.positions[i3 + 2],
          this.intensities[i], this.rings[i],
        );
      }
    }
    return result;
  }
}

// ============================================================
// Color Mapping
// ============================================================

/**
 * Color mapping modes for point cloud rendering.
 * @enum {string}
 */
export const ColorMode = {
  INTENSITY: 'intensity',
  HEIGHT: 'height',
  DISTANCE: 'distance',
  RING: 'ring',
  FLAT: 'flat',
};

/**
 * Map a value to a color using the specified color mode.
 * @param {number} value - Normalized value [0, 1]
 * @param {ColorMode} mode - Color mode
 * @returns {{r: number, g: number, b: number}} Color components [0, 1]
 */
export function mapColor(value, mode) {
  const t = Math.max(0, Math.min(1, value));
  switch (mode) {
    case ColorMode.INTENSITY: {
      return { r: t, g: t, b: t };
    }
    case ColorMode.HEIGHT: {
      // Blue → Cyan → Green → Yellow → Red
      if (t < 0.25) return { r: 0, g: t * 4, b: 1 };
      if (t < 0.5) return { r: 0, g: 1, b: 1 - (t - 0.25) * 4 };
      if (t < 0.75) return { r: (t - 0.5) * 4, g: 1, b: 0 };
      return { r: 1, g: 1 - (t - 0.75) * 4, b: 0 };
    }
    case ColorMode.DISTANCE: {
      // Near = warm, Far = cool
      return { r: 1 - t * 0.7, g: 0.3 + t * 0.3, b: 0.2 + t * 0.8 };
    }
    case ColorMode.RING: {
      // Distinct colors per ring
      const hue = (t * 360) % 360;
      const s = 0.8, l = 0.55;
      const c = (1 - Math.abs(2 * l - 1)) * s;
      const x = c * (1 - Math.abs(((hue / 60) % 2) - 1));
      const m = l - c / 2;
      let r, g, b;
      if (hue < 60) { r = c; g = x; b = 0; }
      else if (hue < 120) { r = x; g = c; b = 0; }
      else if (hue < 180) { r = 0; g = c; b = x; }
      else if (hue < 240) { r = 0; g = x; b = c; }
      else if (hue < 300) { r = x; g = 0; b = c; }
      else { r = c; g = 0; b = x; }
      return { r: r + m, g: g + m, b: b + m };
    }
    case ColorMode.FLAT:
    default:
      return { r: 0.4, g: 0.7, b: 1.0 };
  }
}

// ============================================================
// WebGL LiDAR Renderer
// ============================================================

/** @private Vertex shader source */
const VERT_SRC = `
  attribute vec3 aPosition;
  attribute vec3 aColor;
  uniform mat4 uMVP;
  uniform float uPointSize;
  varying vec3 vColor;
  void main() {
    gl_Position = uMVP * vec4(aPosition, 1.0);
    gl_PointSize = uPointSize;
    vColor = aColor;
  }
`;

/** @private Fragment shader source */
const FRAG_SRC = `
  precision mediump float;
  varying vec3 vColor;
  void main() {
    vec2 coord = gl_PointCoord - vec2(0.5);
    float dist = length(coord);
    if (dist > 0.5) discard;
    float alpha = 1.0 - smoothstep(0.35, 0.5, dist);
    gl_FragColor = vec4(vColor, alpha);
  }
`;

/**
 * WebGL-based LiDAR point cloud renderer with interactive camera controls.
 * @class LidarVisualizer
 */
export class LidarVisualizer {
  /**
   * @param {HTMLCanvasElement} canvas - Target WebGL canvas
   * @param {Object} [options={}] - Renderer options
   * @param {ColorMode} [options.colorMode=ColorMode.HEIGHT] - Default color mode
   * @param {number} [options.pointSize=2.0] - Default point size
   * @param {number} [options.fov=60] - Camera field of view in degrees
   * @param {number} [options.near=0.1] - Near clipping plane
   * @param {number} [options.far=1000] - Far clipping plane
   * @param {number} [options.groundThreshold=-1.5] - Ground plane filter height
   * @param {boolean} [options.showBoundingBox=false] - Show bounding box overlay
   */
  constructor(canvas, options = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError('LidarVisualizer requires an HTMLCanvasElement');
    }

    this._canvas = canvas;
    this._colorMode = options.colorMode ?? ColorMode.HEIGHT;
    this._pointSize = options.pointSize ?? 2.0;
    this._fov = (options.fov ?? 60) * Math.PI / 180;
    this._near = options.near ?? 0.1;
    this._far = options.far ?? 1000;
    this._groundThreshold = options.groundThreshold ?? -1.5;
    this._showBoundingBox = options.showBoundingBox ?? false;

    // Camera state (orbit camera)
    this._cameraAzimuth = 45;
    this._cameraElevation = 30;
    this._cameraDistance = 50;
    this._cameraTarget = [0, 0, 0];

    // Interaction state
    this._isDragging = false;
    this._lastMouse = { x: 0, y: 0 };

    // Point cloud data
    /** @type {PointCloud|null} */
    this._pointCloud = null;

    // WebGL setup
    this._gl = canvas.getContext('webgl', { antialias: true, alpha: true });
    if (!this._gl) {
      throw new Error('WebGL not supported');
    }

    this._initGL();
    this._bindEvents();
    this._renderLoop();
  }

  /** @private Initialize WebGL context and shaders */
  _initGL() {
    const gl = this._gl;
    gl.enable(gl.DEPTH_TEST);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
    gl.clearColor(0.06, 0.09, 0.13, 1.0);

    // Compile shaders
    const vs = this._compileShader(gl.VERTEX_SHADER, VERT_SRC);
    const fs = this._compileShader(gl.FRAGMENT_SHADER, FRAG_SRC);

    this._program = gl.createProgram();
    gl.attachShader(this._program, vs);
    gl.attachShader(this._program, fs);
    gl.linkProgram(this._program);

    if (!gl.getProgramParameter(this._program, gl.LINK_STATUS)) {
      throw new Error(`Shader link error: ${gl.getProgramInfoLog(this._program)}`);
    }

    gl.useProgram(this._program);

    // Attribute and uniform locations
    this._aPosition = gl.getAttribLocation(this._program, 'aPosition');
    this._aColor = gl.getAttribLocation(this._program, 'aColor');
    this._uMVP = gl.getUniformLocation(this._program, 'uMVP');
    this._uPointSize = gl.getUniformLocation(this._program, 'uPointSize');

    // Buffers
    this._positionBuffer = gl.createBuffer();
    this._colorBuffer = gl.createBuffer();
    this._pointCount = 0;

    // Bounding box buffer
    this._bboxBuffer = gl.createBuffer();
    this._bboxColorBuffer = gl.createBuffer();
  }

  /** @private Compile a WebGL shader */
  _compileShader(type, source) {
    const gl = this._gl;
    const shader = gl.createShader(type);
    gl.shaderSource(shader, source);
    gl.compileShader(shader);
    if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
      const info = gl.getShaderInfoLog(shader);
      gl.deleteShader(shader);
      throw new Error(`Shader compile error: ${info}`);
    }
    return shader;
  }

  /** @private Bind mouse/touch events for orbit camera */
  _bindEvents() {
    this._canvas.addEventListener('mousedown', (e) => {
      this._isDragging = true;
      this._lastMouse = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('mousemove', (e) => {
      if (!this._isDragging) return;
      const dx = e.clientX - this._lastMouse.x;
      const dy = e.clientY - this._lastMouse.y;
      this._cameraAzimuth += dx * 0.5;
      this._cameraElevation = Math.max(-89, Math.min(89, this._cameraElevation + dy * 0.5));
      this._lastMouse = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('mouseup', () => { this._isDragging = false; });

    this._canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this._cameraDistance *= e.deltaY > 0 ? 1.1 : 0.9;
      this._cameraDistance = Math.max(1, Math.min(500, this._cameraDistance));
    }, { passive: false });

    // Touch support
    let lastTouchDist = 0;
    this._canvas.addEventListener('touchstart', (e) => {
      if (e.touches.length === 1) {
        this._isDragging = true;
        this._lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        lastTouchDist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY,
        );
      }
    }, { passive: true });
    this._canvas.addEventListener('touchmove', (e) => {
      if (e.touches.length === 1 && this._isDragging) {
        const dx = e.touches[0].clientX - this._lastMouse.x;
        const dy = e.touches[0].clientY - this._lastMouse.y;
        this._cameraAzimuth += dx * 0.5;
        this._cameraElevation = Math.max(-89, Math.min(89, this._cameraElevation + dy * 0.5));
        this._lastMouse = { x: e.touches[0].clientX, y: e.touches[0].clientY };
      } else if (e.touches.length === 2) {
        const dist = Math.hypot(
          e.touches[0].clientX - e.touches[1].clientX,
          e.touches[0].clientY - e.touches[1].clientY,
        );
        if (lastTouchDist > 0) {
          this._cameraDistance *= lastTouchDist / dist;
          this._cameraDistance = Math.max(1, Math.min(500, this._cameraDistance));
        }
        lastTouchDist = dist;
      }
    }, { passive: true });
    this._canvas.addEventListener('touchend', () => { this._isDragging = false; lastTouchDist = 0; });
  }

  /**
   * Set the point cloud data to render.
   * @param {PointCloud} cloud - Point cloud data
   */
  setPointCloud(cloud) {
    if (!(cloud instanceof PointCloud)) {
      throw new TypeError('Expected a PointCloud instance');
    }
    this._pointCloud = cloud;
    this._uploadPointCloud(cloud);
  }

  /**
   * @private Upload point cloud data to WebGL buffers.
   * @param {PointCloud} cloud
   */
  _uploadPointCloud(cloud) {
    const gl = this._gl;
    const n = cloud.count;
    this._pointCount = n;

    // Upload positions
    gl.bindBuffer(gl.ARRAY_BUFFER, this._positionBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, cloud.positions.subarray(0, n * 3), gl.STATIC_DRAW);

    // Compute and upload colors
    const colors = new Float32Array(n * 3);
    const b = cloud.bounds;
    const heightRange = Math.max(1e-6, b.maxZ - b.minZ);
    const maxDist = Math.sqrt(Math.max(b.maxX, -b.minX) ** 2 + Math.max(b.maxY, -b.minY) ** 2);
    const distRange = Math.max(1e-6, maxDist);
    const maxRing = Math.max(1, ...Array.from(cloud.rings.subarray(0, n)));

    for (let i = 0; i < n; i++) {
      const i3 = i * 3;
      let normalizedValue;
      switch (this._colorMode) {
        case ColorMode.INTENSITY:
          normalizedValue = cloud.intensities[i];
          break;
        case ColorMode.HEIGHT:
          normalizedValue = (cloud.positions[i3 + 2] - b.minZ) / heightRange;
          break;
        case ColorMode.DISTANCE:
          normalizedValue = Math.sqrt(cloud.positions[i3] ** 2 + cloud.positions[i3 + 1] ** 2) / distRange;
          break;
        case ColorMode.RING:
          normalizedValue = cloud.rings[i] / maxRing;
          break;
        default:
          normalizedValue = 0.5;
      }
      const c = mapColor(normalizedValue, this._colorMode);
      colors[i3] = c.r;
      colors[i3 + 1] = c.g;
      colors[i3 + 2] = c.b;
    }

    gl.bindBuffer(gl.ARRAY_BUFFER, this._colorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, colors, gl.STATIC_DRAW);

    // Bounding box lines
    if (this._showBoundingBox) {
      this._uploadBoundingBox(cloud);
    }

    // Update camera target to cloud center
    const center = cloud.getCenter();
    this._cameraTarget = [center.x, center.y, center.z];
  }

  /** @private Upload bounding box wireframe */
  _uploadBoundingBox(cloud) {
    const gl = this._gl;
    const b = cloud.bounds;
    const v = [
      b.minX, b.minY, b.minZ, b.maxX, b.minY, b.minZ,
      b.maxX, b.minY, b.minZ, b.maxX, b.maxY, b.minZ,
      b.maxX, b.maxY, b.minZ, b.minX, b.maxY, b.minZ,
      b.minX, b.maxY, b.minZ, b.minX, b.minY, b.minZ,
      b.minX, b.minY, b.maxZ, b.maxX, b.minY, b.maxZ,
      b.maxX, b.minY, b.maxZ, b.maxX, b.maxY, b.maxZ,
      b.maxX, b.maxY, b.maxZ, b.minX, b.maxY, b.maxZ,
      b.minX, b.maxY, b.maxZ, b.minX, b.minY, b.maxZ,
      b.minX, b.minY, b.minZ, b.minX, b.minY, b.maxZ,
      b.maxX, b.minY, b.minZ, b.maxX, b.minY, b.maxZ,
      b.maxX, b.maxY, b.minZ, b.maxX, b.maxY, b.maxZ,
      b.minX, b.maxY, b.minZ, b.minX, b.maxY, b.maxZ,
    ];
    gl.bindBuffer(gl.ARRAY_BUFFER, this._bboxBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(v), gl.STATIC_DRAW);

    const bboxColors = new Float32Array(24 * 3).fill(0);
    for (let i = 0; i < 24; i++) {
      bboxColors[i * 3] = 0.4;
      bboxColors[i * 3 + 1] = 0.9;
      bboxColors[i * 3 + 2] = 0.4;
    }
    gl.bindBuffer(gl.ARRAY_BUFFER, this._bboxColorBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, bboxColors, gl.STATIC_DRAW);
  }

  /** @private Compute camera position from orbit parameters */
  _getCameraPosition() {
    const azRad = (this._cameraAzimuth * Math.PI) / 180;
    const elRad = (this._cameraElevation * Math.PI) / 180;
    const t = this._cameraTarget;
    return [
      t[0] + this._cameraDistance * Math.cos(elRad) * Math.sin(azRad),
      t[1] + this._cameraDistance * Math.sin(elRad),
      t[2] + this._cameraDistance * Math.cos(elRad) * Math.cos(azRad),
    ];
  }

  /** @private Compute model-view-projection matrix */
  _computeMVP() {
    const aspect = this._canvas.width / this._canvas.height;
    const proj = Mat4.perspective(this._fov, aspect, this._near, this._far);
    const eye = this._getCameraPosition();
    const view = Mat4.lookAt(eye, this._cameraTarget, [0, 1, 0]);
    return Mat4.multiply(proj, view);
  }

  /** @private Main render loop */
  _renderLoop() {
    this._render();
    this._animFrameId = requestAnimationFrame(() => this._renderLoop());
  }

  /** @private Render frame */
  _render() {
    const gl = this._gl;
    gl.viewport(0, 0, this._canvas.width, this._canvas.height);
    gl.clear(gl.COLOR_BUFFER_BIT | gl.DEPTH_BUFFER_BIT);

    gl.useProgram(this._program);

    const mvp = this._computeMVP();
    gl.uniformMatrix4fv(this._uMVP, false, mvp);
    gl.uniform1f(this._uPointSize, this._pointSize);

    // Draw point cloud
    if (this._pointCount > 0) {
      gl.bindBuffer(gl.ARRAY_BUFFER, this._positionBuffer);
      gl.enableVertexAttribArray(this._aPosition);
      gl.vertexAttribPointer(this._aPosition, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this._colorBuffer);
      gl.enableVertexAttribArray(this._aColor);
      gl.vertexAttribPointer(this._aColor, 3, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.POINTS, 0, this._pointCount);
    }

    // Draw bounding box
    if (this._showBoundingBox && this._pointCloud) {
      gl.uniform1f(this._uPointSize, 1.0);
      gl.bindBuffer(gl.ARRAY_BUFFER, this._bboxBuffer);
      gl.enableVertexAttribArray(this._aPosition);
      gl.vertexAttribPointer(this._aPosition, 3, gl.FLOAT, false, 0, 0);

      gl.bindBuffer(gl.ARRAY_BUFFER, this._bboxColorBuffer);
      gl.enableVertexAttribArray(this._aColor);
      gl.vertexAttribPointer(this._aColor, 3, gl.FLOAT, false, 0, 0);

      gl.drawArrays(gl.LINES, 0, 24);
    }
  }

  /**
   * Set the color mode for point rendering.
   * @param {ColorMode} mode - Color mode
   */
  setColorMode(mode) {
    this._colorMode = mode;
    if (this._pointCloud) {
      this._uploadPointCloud(this._pointCloud);
    }
  }

  /**
   * Set the point size.
   * @param {number} size - Point size in pixels
   */
  setPointSize(size) {
    this._pointSize = Math.max(0.5, Math.min(20, size));
  }

  /**
   * Toggle bounding box visibility.
   * @param {boolean} visible - Show bounding box
   */
  showBoundingBox(visible) {
    this._showBoundingBox = visible;
    if (visible && this._pointCloud) {
      this._uploadBoundingBox(this._pointCloud);
    }
  }

  /**
   * Reset the camera to default view.
   */
  resetCamera() {
    this._cameraAzimuth = 45;
    this._cameraElevation = 30;
    this._cameraDistance = 50;
    if (this._pointCloud) {
      const c = this._pointCloud.getCenter();
      this._cameraTarget = [c.x, c.y, c.z];
    }
  }

  /**
   * Destroy the visualizer and release WebGL resources.
   */
  destroy() {
    if (this._animFrameId) {
      cancelAnimationFrame(this._animFrameId);
    }
    const gl = this._gl;
    gl.deleteBuffer(this._positionBuffer);
    gl.deleteBuffer(this._colorBuffer);
    gl.deleteBuffer(this._bboxBuffer);
    gl.deleteBuffer(this._bboxColorBuffer);
    gl.deleteProgram(this._program);
  }
}

export default LidarVisualizer;
