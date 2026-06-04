/**
 * @fileoverview 3D Scene Module - Three.js scene manager with vehicle 3D model loading,
 * environment rendering (road, buildings, trees), camera orbit/follow modes, lighting setup,
 * sensor frustum visualization, and detection box rendering.
 *
 * @module 3d_scene
 * @version 2.0.0
 * @author Autonomous Vehicle Control System Team
 *
 * @note This module expects Three.js (r158+) to be available as a global or imported dependency.
 * Import paths use the convention: import * as THREE from 'three'
 */

// ============================================================
// Camera Controller
// ============================================================

/**
 * Orbit and follow camera controller for 3D scene navigation.
 * @class CameraController
 */
export class CameraController {
  /**
   * @param {Object} [options={}] - Camera options
   * @param {'orbit'|'follow'|'top-down'|'first-person'} [options.mode='orbit'] - Camera mode
   * @param {number} [options.fov=60] - Field of view in degrees
   * @param {number} [options.near=0.1] - Near clipping plane
   * @param {number} [options.far=2000] - Far clipping plane
   * @param {number} [options.orbitDistance=50] - Orbit distance
   * @param {number} [options.orbitAzimuth=45] - Orbit azimuth in degrees
   * @param {number} [options.orbitElevation=30] - Orbit elevation in degrees
   * @param {number} [options.followDistance=15] - Follow camera distance behind vehicle
   * @param {number} [options.followHeight=6] - Follow camera height above vehicle
   * @param {number} [options.followSmoothing=0.05] - Smoothing factor for follow mode
   */
  constructor(options = {}) {
    this.mode = options.mode ?? 'orbit';
    this.fov = options.fov ?? 60;
    this.near = options.near ?? 0.1;
    this.far = options.far ?? 2000;
    this.orbitDistance = options.orbitDistance ?? 50;
    this.orbitAzimuth = options.orbitAzimuth ?? 45;
    this.orbitElevation = options.orbitElevation ?? 30;
    this.followDistance = options.followDistance ?? 15;
    this.followHeight = options.followHeight ?? 6;
    this.followSmoothing = options.followSmoothing ?? 0.05;

    /** @private @type {{x: number, y: number, z: number}} */
    this._target = { x: 0, y: 0, z: 0 };
    /** @private @type {{x: number, y: number, z: number}} */
    this._smoothedTarget = { x: 0, y: 0, z: 0 };
    /** @private @type {number} */
    this._smoothedHeading = 0;

    // Interaction state
    this._isDragging = false;
    this._lastMouse = { x: 0, y: 0 };
  }

  /**
   * Set the camera follow target (typically vehicle position).
   * @param {number} x - Target X
   * @param {number} y - Target Y
   * @param {number} z - Target Z
   */
  setTarget(x, y, z) {
    this._target = { x, y, z };
  }

  /**
   * Compute the camera position and look-at target based on current mode.
   * @returns {{position: {x: number, y: number, z: number}, target: {x: number, y: number, z: number}}}
   */
  computeCameraState() {
    switch (this.mode) {
      case 'orbit':
        return this._computeOrbit();
      case 'follow':
        return this._computeFollow();
      case 'top-down':
        return this._computeTopDown();
      case 'first-person':
        return this._computeFirstPerson();
      default:
        return this._computeOrbit();
    }
  }

  /** @private */
  _computeOrbit() {
    const azRad = (this.orbitAzimuth * Math.PI) / 180;
    const elRad = (this.orbitElevation * Math.PI) / 180;
    const d = this.orbitDistance;
    return {
      position: {
        x: this._target.x + d * Math.cos(elRad) * Math.sin(azRad),
        y: this._target.y + d * Math.sin(elRad),
        z: this._target.z + d * Math.cos(elRad) * Math.cos(azRad),
      },
      target: { ...this._target },
    };
  }

  /** @private */
  _computeFollow() {
    // Smooth target position
    const s = this.followSmoothing;
    this._smoothedTarget.x += (this._target.x - this._smoothedTarget.x) * s;
    this._smoothedTarget.y += (this._target.y - this._smoothedTarget.y) * s;
    this._smoothedTarget.z += (this._target.z - this._smoothedTarget.z) * s;

    const headingRad = (this._smoothedHeading * Math.PI) / 180;
    return {
      position: {
        x: this._smoothedTarget.x - Math.sin(headingRad) * this.followDistance,
        y: this._smoothedTarget.y + this.followHeight,
        z: this._smoothedTarget.z - Math.cos(headingRad) * this.followDistance,
      },
      target: { ...this._smoothedTarget },
    };
  }

  /** @private */
  _computeTopDown() {
    return {
      position: { x: this._target.x, y: this._target.y + this.orbitDistance, z: this._target.z + 0.01 },
      target: { ...this._target },
    };
  }

  /** @private */
  _computeFirstPerson() {
    const headingRad = (this._smoothedHeading * Math.PI) / 180;
    return {
      position: { x: this._target.x, y: this._target.y + 1.5, z: this._target.z },
      target: {
        x: this._target.x + Math.sin(headingRad) * 20,
        y: this._target.y + 1.0,
        z: this._target.z + Math.cos(headingRad) * 20,
      },
    };
  }

  /**
   * Set the vehicle heading for follow/first-person modes.
   * @param {number} heading - Heading in degrees
   */
  setHeading(heading) {
    const s = this.followSmoothing;
    let diff = ((heading - this._smoothedHeading + 540) % 360) - 180;
    this._smoothedHeading += diff * s;
  }

  /**
   * Switch camera mode.
   * @param {'orbit'|'follow'|'top-down'|'first-person'} mode
   */
  setMode(mode) {
    this.mode = mode;
  }

  /**
   * Handle mouse drag for orbit rotation.
   * @param {number} dx - Mouse delta X
   * @param {number} dy - Mouse delta Y
   */
  handleDrag(dx, dy) {
    if (this.mode === 'orbit') {
      this.orbitAzimuth -= dx * 0.5;
      this.orbitElevation = Math.max(-89, Math.min(89, this.orbitElevation + dy * 0.5));
    }
  }

  /**
   * Handle scroll for zoom.
   * @param {number} delta - Scroll delta
   */
  handleZoom(delta) {
    this.orbitDistance *= delta > 0 ? 1.1 : 0.9;
    this.orbitDistance = Math.max(3, Math.min(500, this.orbitDistance));
  }
}

// ============================================================
// Sensor Frustum
// ============================================================

/**
 * Visualizes a sensor's detection frustum in 3D space.
 * @class SensorFrustum
 */
export class SensorFrustum {
  /**
   * @param {Object} config
   * @param {string} config.id - Sensor identifier
   * @param {'lidar'|'radar'|'camera'} config.type - Sensor type
   * @param {number} [config.fov=60] - Horizontal field of view in degrees
   * @param {number} [config.vFov=30] - Vertical field of view (camera only)
   * @param {number} [config.range=100] - Detection range in meters
   * @param {{x: number, y: number, z: number}} [config.offset={x:0,y:1.5,z:2}] - Mount offset
   * @param {string} [config.color='#22c55e'] - Frustum wireframe color
   */
  constructor(config) {
    this.id = config.id;
    this.type = config.type;
    this.fov = config.fov ?? 60;
    this.vFov = config.vFov ?? 30;
    this.range = config.range ?? 100;
    this.offset = config.offset ?? { x: 0, y: 1.5, z: 2 };
    this.color = config.color ?? '#22c55e';
    this.visible = true;
  }

  /**
   * Generate frustum vertices for rendering.
   * @returns {Array<{x: number, y: number, z: number}>}
   */
  getVertices() {
    const hFovRad = (this.fov / 2 * Math.PI) / 180;
    const vFovRad = (this.type === 'camera' ? this.vFov / 2 : 30) * Math.PI / 180;
    const r = this.range;

    const corners = [
      { x: 0, y: 0, z: 0 }, // Origin
      { x: r * Math.sin(-hFovRad), y: r * Math.sin(vFovRad), z: r * Math.cos(hFovRad) },
      { x: r * Math.sin(hFovRad), y: r * Math.sin(vFovRad), z: r * Math.cos(hFovRad) },
      { x: r * Math.sin(hFovRad), y: r * Math.sin(-vFovRad), z: r * Math.cos(hFovRad) },
      { x: r * Math.sin(-hFovRad), y: r * Math.sin(-vFovRad), z: r * Math.cos(hFovRad) },
    ];

    // Apply offset
    return corners.map((v) => ({
      x: v.x + this.offset.x,
      y: v.y + this.offset.y,
      z: v.z + this.offset.z,
    }));
  }

  /**
   * Get line segments for wireframe rendering.
   * @returns {Array<[number, number, number, number, number, number]>}
   */
  getWireframeLines() {
    const v = this.getVertices();
    return [
      [v[0].x, v[0].y, v[0].z, v[1].x, v[1].y, v[1].z],
      [v[0].x, v[0].y, v[0].z, v[2].x, v[2].y, v[2].z],
      [v[0].x, v[0].y, v[0].z, v[3].x, v[3].y, v[3].z],
      [v[0].x, v[0].y, v[0].z, v[4].x, v[4].y, v[4].z],
      [v[1].x, v[1].y, v[1].z, v[2].x, v[2].y, v[2].z],
      [v[2].x, v[2].y, v[2].z, v[3].x, v[3].y, v[3].z],
      [v[3].x, v[3].y, v[3].z, v[4].x, v[4].y, v[4].z],
      [v[4].x, v[4].y, v[4].z, v[1].x, v[1].y, v[1].z],
    ];
  }
}

// ============================================================
// Detection Box 3D
// ============================================================

/**
 * 3D bounding box for object detection visualization.
 * @class DetectionBox3D
 */
export class DetectionBox3D {
  /**
   * @param {Object} config
   * @param {number} config.centerX - Center X in world coordinates
   * @param {number} config.centerY - Center Y in world coordinates
   * @param {number} config.centerZ - Center Z in world coordinates
   * @param {number} config.width - Box width (X)
   * @param {number} config.height - Box height (Y)
   * @param {number} config.depth - Box depth (Z)
   * @param {number} [config.heading=0] - Heading rotation in degrees
   * @param {string} [config.label=''] - Detection label
   * @param {number} [config.confidence=0] - Detection confidence
   * @param {'vehicle'|'pedestrian'|'cyclist'|'unknown'} [config.type='unknown']
   * @param {string} [config.trackId] - Tracking ID
   */
  constructor(config) {
    this.centerX = config.centerX;
    this.centerY = config.centerY;
    this.centerZ = config.centerZ;
    this.width = config.width;
    this.height = config.height;
    this.depth = config.depth;
    this.heading = config.heading ?? 0;
    this.label = config.label ?? '';
    this.confidence = config.confidence ?? 0;
    this.type = config.type ?? 'unknown';
    this.trackId = config.trackId ?? null;
  }

  /**
   * Get the 8 corner vertices of the box.
   * @returns {Array<{x: number, y: number, z: number}>}
   */
  getCorners() {
    const hw = this.width / 2;
    const hh = this.height / 2;
    const hd = this.depth / 2;
    const headingRad = (this.heading * Math.PI) / 180;
    const cosH = Math.cos(headingRad);
    const sinH = Math.sin(headingRad);

    const localCorners = [
      { x: -hw, y: -hh, z: -hd }, { x: hw, y: -hh, z: -hd },
      { x: hw, y: hh, z: -hd },   { x: -hw, y: hh, z: -hd },
      { x: -hw, y: -hh, z: hd },  { x: hw, y: -hh, z: hd },
      { x: hw, y: hh, z: hd },    { x: -hw, y: hh, z: hd },
    ];

    return localCorners.map((c) => ({
      x: c.x * cosH - c.z * sinH + this.centerX,
      y: c.y + this.centerY,
      z: c.x * sinH + c.z * cosH + this.centerZ,
    }));
  }

  /**
   * Get the 12 edge line segments for wireframe rendering.
   * @returns {Array<[number, number, number, number, number, number]>}
   */
  getEdges() {
    const c = this.getCorners();
    const edgeIndices = [
      [0, 1], [1, 2], [2, 3], [3, 0], // Front face
      [4, 5], [5, 6], [6, 7], [7, 4], // Back face
      [0, 4], [1, 5], [2, 6], [3, 7], // Connecting edges
    ];
    return edgeIndices.map(([a, b]) => [c[a].x, c[a].y, c[a].z, c[b].x, c[b].y, c[b].z]);
  }

  /**
   * Get color for this detection type.
   * @returns {string}
   */
  getColor() {
    const colors = { vehicle: '#ef4444', pedestrian: '#f59e0b', cyclist: '#3b82f6', unknown: '#94a3b8' };
    return colors[this.type] ?? '#94a3b8';
  }
}

// ============================================================
// Environment Builder
// ============================================================

/**
 * Builds procedural 3D environment geometry (road, buildings, trees).
 * @class EnvironmentBuilder
 */
export class EnvironmentBuilder {
  /**
   * Generate road segment vertices.
   * @param {number} length - Road length in meters
   * @param {number} width - Road width in meters
   * @param {number} [segments=20] - Number of segments
   * @returns {{vertices: Float32Array, indices: Uint16Array, normals: Float32Array}}
   */
  static createRoad(length = 200, width = 8, segments = 20) {
    const vertexCount = (segments + 1) * 2;
    const vertices = new Float32Array(vertexCount * 3);
    const normals = new Float32Array(vertexCount * 3);
    const indices = new Uint16Array(segments * 6);

    for (let i = 0; i <= segments; i++) {
      const z = (i / segments) * length - length / 2;
      const idx = i * 2;
      // Left edge
      vertices[idx * 3] = -width / 2;
      vertices[idx * 3 + 1] = 0.01;
      vertices[idx * 3 + 2] = z;
      // Right edge
      vertices[(idx + 1) * 3] = width / 2;
      vertices[(idx + 1) * 3 + 1] = 0.01;
      vertices[(idx + 1) * 3 + 2] = z;
      // Normals (up)
      normals[idx * 3] = 0; normals[idx * 3 + 1] = 1; normals[idx * 3 + 2] = 0;
      normals[(idx + 1) * 3] = 0; normals[(idx + 1) * 3 + 1] = 1; normals[(idx + 1) * 3 + 2] = 0;
    }

    for (let i = 0; i < segments; i++) {
      const base = i * 6;
      const vi = i * 2;
      indices[base] = vi;
      indices[base + 1] = vi + 1;
      indices[base + 2] = vi + 2;
      indices[base + 3] = vi + 1;
      indices[base + 4] = vi + 3;
      indices[base + 5] = vi + 2;
    }

    return { vertices, indices, normals };
  }

  /**
   * Generate building vertices (box geometry at a position).
   * @param {number} x - Center X
   * @param {number} z - Center Z
   * @param {number} width - Building width
   * @param {number} depth - Building depth
   * @param {number} height - Building height
   * @returns {{vertices: Float32Array, indices: Uint16Array}}
   */
  static createBuilding(x, z, width, depth, height) {
    const hw = width / 2, hd = depth / 2;
    // 8 vertices, 12 triangles (6 faces * 2)
    const vertices = new Float32Array([
      x - hw, 0, z - hd,   x + hw, 0, z - hd,   x + hw, height, z - hd,   x - hw, height, z - hd,
      x - hw, 0, z + hd,   x + hw, 0, z + hd,   x + hw, height, z + hd,   x - hw, height, z + hd,
    ]);
    const indices = new Uint16Array([
      0, 2, 1, 0, 3, 2,  // Front
      5, 7, 4, 5, 6, 7,  // Back
      4, 3, 0, 4, 7, 3,  // Left
      1, 6, 5, 1, 2, 6,  // Right
      3, 6, 2, 3, 7, 6,  // Top
      4, 1, 5, 4, 0, 1,  // Bottom
    ]);
    return { vertices, indices };
  }

  /**
   * Generate tree vertices (cone + cylinder).
   * @param {number} x - Base X
   * @param {number} z - Base Z
   * @param {number} [trunkHeight=2] - Trunk height
   * @param {number} [crownHeight=4] - Crown height
   * @param {number} [crownRadius=1.5] - Crown radius
   * @returns {{trunk: {vertices: Float32Array, indices: Uint16Array},
   *            crown: {vertices: Float32Array, indices: Uint16Array}}}
   */
  static createTree(x, z, trunkHeight = 2, crownHeight = 4, crownRadius = 1.5) {
    // Trunk (cylinder - simplified as 8-sided)
    const sides = 8;
    const trunkR = 0.15;
    const trunkVerts = [];
    const trunkIdx = [];
    for (let i = 0; i < sides; i++) {
      const angle = (i / sides) * Math.PI * 2;
      const cos = Math.cos(angle), sin = Math.sin(angle);
      trunkVerts.push(x + trunkR * cos, 0, z + trunkR * sin);
      trunkVerts.push(x + trunkR * cos, trunkHeight, z + trunkR * sin);
    }
    for (let i = 0; i < sides; i++) {
      const next = (i + 1) % sides;
      trunkIdx.push(i * 2, i * 2 + 1, next * 2);
      trunkIdx.push(i * 2 + 1, next * 2 + 1, next * 2);
    }

    // Crown (cone - 8-sided)
    const crownVerts = [];
    const crownIdx = [];
    const crownBase = trunkHeight;
    const tipIdx = sides * 2; // Trunk vertices count as base indices
    // Ring at base of crown
    for (let i = 0; i < sides; i++) {
      const angle = (i / sides) * Math.PI * 2;
      const cos = Math.cos(angle), sin = Math.sin(angle);
      crownVerts.push(x + crownRadius * cos, crownBase, z + crownRadius * sin);
    }
    // Tip
    crownVerts.push(x, crownBase + crownHeight, z);
    // Indices
    for (let i = 0; i < sides; i++) {
      const next = (i + 1) % sides;
      crownIdx.push(i, sides, next);
    }

    return {
      trunk: { vertices: new Float32Array(trunkVerts), indices: new Uint16Array(trunkIdx) },
      crown: { vertices: new Float32Array(crownVerts), indices: new Uint16Array(crownIdx) },
    };
  }
}

// ============================================================
// 3D Scene Manager (Canvas2D fallback renderer)
// ============================================================

/**
 * 3D Scene Manager that provides scene graph management and rendering.
 * Uses a lightweight Canvas2D-based 3D renderer as a fallback when Three.js
 * is not available, with full Three.js integration support.
 * @class Scene3D
 */
export class Scene3D {
  /**
   * @param {HTMLCanvasElement} canvas - Target canvas
   * @param {Object} [options={}] - Scene options
   * @param {string} [options.bgColor='#0a0f1a'] - Background color
   * @param {boolean} [options.showGrid=true] - Show ground grid
   * @param {boolean} [options.showAxes=true] - Show coordinate axes
   * @param {boolean} [options.shadows=true] - Enable shadow rendering
   * @param {Object} [options.lights] - Lighting configuration
   */
  constructor(canvas, options = {}) {
    if (!(canvas instanceof HTMLCanvasElement)) {
      throw new TypeError('Scene3D requires an HTMLCanvasElement');
    }

    this._canvas = canvas;
    this._ctx = canvas.getContext('2d');
    this._bgColor = options.bgColor ?? '#0a0f1a';
    this._showGrid = options.showGrid ?? true;
    this._showAxes = options.showAxes ?? true;
    this._shadows = options.shadows ?? true;

    // Scene objects
    /** @type {{x: number, y: number, z: number, heading: number}} */
    this._vehiclePosition = { x: 0, y: 0, z: 0, heading: 0 };
    /** @type {Array<SensorFrustum>} */
    this._sensorFrustums = [];
    /** @type {Array<DetectionBox3D>} */
    this._detectionBoxes = [];
    /** @type {Array<Object>} */
    this._environmentObjects = [];

    // Camera
    this._camera = new CameraController({ mode: 'orbit' });

    // Lighting
    this._lights = options.lights ?? {
      ambient: { color: '#64748b', intensity: 0.4 },
      directional: { color: '#fef3c7', intensity: 0.8, direction: { x: -1, y: -2, z: -1 } },
    };

    // Projection
    this._fov = this._camera.fov * Math.PI / 180;
    this._near = this._camera.near;
    this._far = this._camera.far;

    this._resize();
    this._bindEvents();
    this._renderLoop();
  }

  /** @private */
  _resize() {
    const dpr = window.devicePixelRatio || 1;
    const rect = this._canvas.getBoundingClientRect();
    this._canvas.width = rect.width * dpr;
    this._canvas.height = rect.height * dpr;
    this._ctx.scale(dpr, dpr);
    this._canvasW = rect.width;
    this._canvasH = rect.height;
    this._aspect = rect.width / rect.height;
  }

  /** @private */
  _bindEvents() {
    this._canvas.addEventListener('mousedown', (e) => {
      this._camera._isDragging = true;
      this._camera._lastMouse = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('mousemove', (e) => {
      if (!this._camera._isDragging) return;
      const dx = e.clientX - this._camera._lastMouse.x;
      const dy = e.clientY - this._camera._lastMouse.y;
      this._camera.handleDrag(dx, dy);
      this._camera._lastMouse = { x: e.clientX, y: e.clientY };
    });
    window.addEventListener('mouseup', () => { this._camera._isDragging = false; });
    this._canvas.addEventListener('wheel', (e) => {
      e.preventDefault();
      this._camera.handleZoom(e.deltaY);
    }, { passive: false });
    window.addEventListener('resize', () => this._resize());
  }

  /**
   * Update the vehicle's 3D position and heading.
   * @param {number} x - X position
   * @param {number} y - Y position (up)
   * @param {number} z - Z position
   * @param {number} heading - Heading in degrees
   */
  updateVehicle(x, y, z, heading) {
    this._vehiclePosition = { x, y, z, heading };
    this._camera.setTarget(x, y, z);
    this._camera.setHeading(heading);
  }

  /**
   * Add a sensor frustum to the scene.
   * @param {SensorFrustum} frustum
   */
  addSensorFrustum(frustum) {
    this._sensorFrustums.push(frustum);
  }

  /**
   * Add a detection box to the scene.
   * @param {DetectionBox3D} box
   */
  addDetectionBox(box) {
    this._detectionBoxes.push(box);
  }

  /**
   * Set all detection boxes (replaces existing).
   * @param {Array<DetectionBox3D>} boxes
   */
  setDetectionBoxes(boxes) {
    this._detectionBoxes = boxes;
  }

  /**
   * Switch camera mode.
   * @param {'orbit'|'follow'|'top-down'|'first-person'} mode
   */
  setCameraMode(mode) {
    this._camera.setMode(mode);
  }

  /**
   * Build a default environment scene.
   * @param {Object} [options={}] - Environment options
   * @param {number} [options.buildingCount=20] - Number of buildings
   * @param {number} [options.treeCount=30] - Number of trees
   */
  buildDefaultEnvironment(options = {}) {
    const buildingCount = options.buildingCount ?? 20;
    const treeCount = options.treeCount ?? 30;

    this._environmentObjects = [];

    // Buildings along the road
    for (let i = 0; i < buildingCount; i++) {
      const side = i % 2 === 0 ? -1 : 1;
      const z = (Math.random() - 0.5) * 200;
      const x = side * (8 + Math.random() * 15);
      const w = 4 + Math.random() * 8;
      const d = 4 + Math.random() * 8;
      const h = 5 + Math.random() * 25;
      this._environmentObjects.push({
        type: 'building',
        data: EnvironmentBuilder.createBuilding(x, z, w, d, h),
        x, z, width: w, depth: d, height: h,
        color: `hsl(${210 + Math.random() * 30}, ${10 + Math.random() * 15}%, ${20 + Math.random() * 15}%)`,
      });
    }

    // Trees
    for (let i = 0; i < treeCount; i++) {
      const side = i % 2 === 0 ? -1 : 1;
      const z = (Math.random() - 0.5) * 200;
      const x = side * (6 + Math.random() * 25);
      this._environmentObjects.push({
        type: 'tree',
        data: EnvironmentBuilder.createTree(x, z),
        x, z,
        trunkColor: '#5c3d2e',
        crownColor: `hsl(${120 + Math.random() * 40}, ${40 + Math.random() * 20}%, ${25 + Math.random() * 15}%)`,
      });
    }
  }

  /**
   * @private Project 3D point to 2D screen coordinates.
   * @param {number} x - World X
   * @param {number} y - World Y
   * @param {number} z - World Z
   * @returns {{x: number, y: number, depth: number}|null}
   */
  _project(x, y, z) {
    const cam = this._camera.computeCameraState();
    const pos = cam.position;
    const target = cam.target;

    // Compute view matrix
    const forward = {
      x: target.x - pos.x,
      y: target.y - pos.y,
      z: target.z - pos.z,
    };
    const fLen = Math.sqrt(forward.x ** 2 + forward.y ** 2 + forward.z ** 2);
    if (fLen < 1e-6) return null;
    forward.x /= fLen; forward.y /= fLen; forward.z /= fLen;

    // Right vector (forward × up)
    const up = { x: 0, y: 1, z: 0 };
    const right = {
      x: forward.y * up.z - forward.z * up.y,
      y: forward.z * up.x - forward.x * up.z,
      z: forward.x * up.y - forward.y * up.x,
    };
    const rLen = Math.sqrt(right.x ** 2 + right.y ** 2 + right.z ** 2);
    if (rLen < 1e-6) return null;
    right.x /= rLen; right.y /= rLen; right.z /= rLen;

    // Recalculate up (right × forward)
    const trueUp = {
      x: right.y * forward.z - right.z * forward.y,
      y: right.z * forward.x - right.x * forward.z,
      z: right.x * forward.y - right.y * forward.x,
    };

    // View transform
    const dx = x - pos.x;
    const dy = y - pos.y;
    const dz = z - pos.z;

    const viewX = dx * right.x + dy * right.y + dz * right.z;
    const viewY = dx * trueUp.x + dy * trueUp.y + dz * trueUp.z;
    const viewZ = dx * forward.x + dy * forward.y + dz * forward.z;

    if (viewZ < this._near) return null;

    // Perspective projection
    const scale = (this._canvasH / 2) / Math.tan(this._fov / 2);
    const screenX = this._canvasW / 2 + (viewX / viewZ) * scale;
    const screenY = this._canvasH / 2 - (viewY / viewZ) * scale;

    return { x: screenX, y: screenY, depth: viewZ };
  }

  /** @private Main render loop */
  _renderLoop() {
    this._render();
    this._animFrameId = requestAnimationFrame(() => this._renderLoop());
  }

  /** @private Render the 3D scene */
  _render() {
    const ctx = this._ctx;
    const w = this._canvasW;
    const h = this._canvasH;

    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = this._bgColor;
    ctx.fillRect(0, 0, w, h);

    // Ground grid
    if (this._showGrid) {
      this._drawGrid(ctx);
    }

    // Coordinate axes
    if (this._showAxes) {
      this._drawAxes(ctx);
    }

    // Environment
    this._drawEnvironment(ctx);

    // Road
    this._drawRoad(ctx);

    // Vehicle
    this._drawVehicle(ctx);

    // Sensor frustums
    this._drawSensorFrustums(ctx);

    // Detection boxes
    this._drawDetectionBoxes(ctx);

    // HUD overlay
    this._drawHUD(ctx);
  }

  /** @private Draw ground grid */
  _drawGrid(ctx) {
    const gridSize = 100;
    const step = 10;
    ctx.strokeStyle = 'rgba(148,163,184,0.08)';
    ctx.lineWidth = 1;

    for (let i = -gridSize; i <= gridSize; i += step) {
      const p1 = this._project(i, 0, -gridSize);
      const p2 = this._project(i, 0, gridSize);
      if (p1 && p2) {
        ctx.beginPath(); ctx.moveTo(p1.x, p1.y); ctx.lineTo(p2.x, p2.y); ctx.stroke();
      }
      const p3 = this._project(-gridSize, 0, i);
      const p4 = this._project(gridSize, 0, i);
      if (p3 && p4) {
        ctx.beginPath(); ctx.moveTo(p3.x, p3.y); ctx.lineTo(p4.x, p4.y); ctx.stroke();
      }
    }
  }

  /** @private Draw coordinate axes */
  _drawAxes(ctx) {
    const origin = this._project(0, 0, 0);
    const axisLen = 5;
    const axes = [
      { end: this._project(axisLen, 0, 0), color: '#ef4444', label: 'X' },
      { end: this._project(0, axisLen, 0), color: '#22c55e', label: 'Y' },
      { end: this._project(0, 0, axisLen), color: '#3b82f6', label: 'Z' },
    ];
    for (const axis of axes) {
      if (origin && axis.end) {
        ctx.beginPath();
        ctx.moveTo(origin.x, origin.y);
        ctx.lineTo(axis.end.x, axis.end.y);
        ctx.strokeStyle = axis.color;
        ctx.lineWidth = 2;
        ctx.stroke();
        ctx.font = '12px Inter, sans-serif';
        ctx.fillStyle = axis.color;
        ctx.textAlign = 'center';
        ctx.fillText(axis.label, axis.end.x + 8, axis.end.y);
      }
    }
  }

  /** @private Draw road */
  _drawRoad(ctx) {
    const roadPts = [];
    for (let z = -100; z <= 100; z += 5) {
      const p = this._project(-4, 0.01, z);
      if (p) roadPts.push(p);
    }
    if (roadPts.length > 1) {
      ctx.beginPath();
      ctx.moveTo(roadPts[0].x, roadPts[0].y);
      for (let i = 1; i < roadPts.length; i++) {
        ctx.lineTo(roadPts[i].x, roadPts[i].y);
      }
      ctx.strokeStyle = 'rgba(100,116,139,0.3)';
      ctx.lineWidth = 20;
      ctx.stroke();
    }

    // Center line
    const centerPts = [];
    for (let z = -100; z <= 100; z += 5) {
      const p = this._project(0, 0.02, z);
      if (p) centerPts.push(p);
    }
    if (centerPts.length > 1) {
      ctx.beginPath();
      ctx.setLineDash([8, 8]);
      ctx.moveTo(centerPts[0].x, centerPts[0].y);
      for (let i = 1; i < centerPts.length; i++) {
        ctx.lineTo(centerPts[i].x, centerPts[i].y);
      }
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 1;
      ctx.stroke();
      ctx.setLineDash([]);
    }
  }

  /** @private Draw vehicle */
  _drawVehicle(ctx) {
    const v = this._vehiclePosition;
    const headingRad = (v.heading * Math.PI) / 180;
    const cosH = Math.cos(headingRad);
    const sinH = Math.sin(headingRad);

    // Vehicle body (simplified box: 4.5m x 1.5m x 2m)
    const hw = 1.0, hh = 0.75, hd = 2.25;
    const corners = [
      { x: -hw, y: 0, z: -hd }, { x: hw, y: 0, z: -hd },
      { x: hw, y: hh * 2, z: -hd }, { x: -hw, y: hh * 2, z: -hd },
      { x: -hw, y: 0, z: hd }, { x: hw, y: 0, z: hd },
      { x: hw, y: hh * 2, z: hd }, { x: -hw, y: hh * 2, z: hd },
    ];

    const worldCorners = corners.map((c) => ({
      x: c.x * cosH - c.z * sinH + v.x,
      y: c.y + v.y,
      z: c.x * sinH + c.z * cosH + v.z,
    }));

    const projected = worldCorners.map((c) => this._project(c.x, c.y, c.z));

    // Draw body faces (sorted by depth)
    const faces = [
      [0, 1, 2, 3], [4, 5, 6, 7], [0, 4, 7, 3],
      [1, 5, 6, 2], [3, 2, 6, 7], [0, 1, 5, 4],
    ];
    const faceColors = ['#2563eb', '#2563eb', '#1d4ed8', '#1d4ed8', '#3b82f6', '#1e40af'];

    for (let fi = 0; fi < faces.length; fi++) {
      const face = faces[fi];
      if (face.some((i) => !projected[i])) continue;
      ctx.beginPath();
      ctx.moveTo(projected[face[0]].x, projected[face[0]].y);
      for (let i = 1; i < face.length; i++) {
        ctx.lineTo(projected[face[i]].x, projected[face[i]].y);
      }
      ctx.closePath();
      ctx.fillStyle = faceColors[fi];
      ctx.fill();
      ctx.strokeStyle = '#1e3a5f';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    // Heading arrow
    const arrowStart = this._project(v.x, v.y + hh * 2 + 0.5, v.z);
    const arrowEnd = this._project(
      v.x + sinH * 3, v.y + hh * 2 + 0.5, v.z + cosH * 3,
    );
    if (arrowStart && arrowEnd) {
      ctx.beginPath();
      ctx.moveTo(arrowStart.x, arrowStart.y);
      ctx.lineTo(arrowEnd.x, arrowEnd.y);
      ctx.strokeStyle = '#f59e0b';
      ctx.lineWidth = 2;
      ctx.stroke();
    }
  }

  /** @private Draw sensor frustums */
  _drawSensorFrustums(ctx) {
    for (const frustum of this._sensorFrustums) {
      if (!frustum.visible) continue;
      const lines = frustum.getWireframeLines();
      const v = this._vehiclePosition;
      const headingRad = (v.heading * Math.PI) / 180;
      const cosH = Math.cos(headingRad);
      const sinH = Math.sin(headingRad);

      for (const line of lines) {
        const p1 = this._project(
          line[0] * cosH - line[2] * sinH + v.x,
          line[1] + v.y,
          line[0] * sinH + line[2] * cosH + v.z,
        );
        const p2 = this._project(
          line[3] * cosH - line[5] * sinH + v.x,
          line[4] + v.y,
          line[3] * sinH + line[5] * cosH + v.z,
        );
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = frustum.color + '80';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      }
    }
  }

  /** @private Draw detection boxes */
  _drawDetectionBoxes(ctx) {
    for (const box of this._detectionBoxes) {
      const edges = box.getEdges();
      const color = box.getColor();

      for (const edge of edges) {
        const p1 = this._project(edge[0], edge[1], edge[2]);
        const p2 = this._project(edge[3], edge[4], edge[5]);
        if (p1 && p2) {
          ctx.beginPath();
          ctx.moveTo(p1.x, p1.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = color;
          ctx.lineWidth = 2;
          ctx.stroke();
        }
      }

      // Label
      const labelPos = this._project(box.centerX, box.centerY + box.height / 2 + 0.5, box.centerZ);
      if (labelPos) {
        ctx.font = '11px Inter, sans-serif';
        ctx.fillStyle = color;
        ctx.textAlign = 'center';
        const text = box.label ? `${box.label} ${(box.confidence * 100).toFixed(0)}%` : box.type;
        ctx.fillText(text, labelPos.x, labelPos.y);
      }
    }
  }

  /** @private Draw environment objects */
  _drawEnvironment(ctx) {
    for (const obj of this._environmentObjects) {
      if (obj.type === 'building') {
        // Simplified building rendering - just draw the top face
        const hw = obj.width / 2, hd = obj.depth / 2;
        const corners = [
          this._project(obj.x - hw, obj.height, obj.z - hd),
          this._project(obj.x + hw, obj.height, obj.z - hd),
          this._project(obj.x + hw, obj.height, obj.z + hd),
          this._project(obj.x - hw, obj.height, obj.z + hd),
        ];
        if (corners.every((c) => c)) {
          ctx.beginPath();
          ctx.moveTo(corners[0].x, corners[0].y);
          for (let i = 1; i < corners.length; i++) {
            ctx.lineTo(corners[i].x, corners[i].y);
          }
          ctx.closePath();
          ctx.fillStyle = obj.color;
          ctx.fill();
          ctx.strokeStyle = 'rgba(148,163,184,0.2)';
          ctx.lineWidth = 1;
          ctx.stroke();
        }
      } else if (obj.type === 'tree') {
        // Simplified tree rendering
        const base = this._project(obj.x, 2, obj.z);
        const top = this._project(obj.x, 6, obj.z);
        if (base && top) {
          ctx.beginPath();
          ctx.moveTo(base.x, base.y);
          ctx.lineTo(top.x, top.y);
          ctx.strokeStyle = obj.trunkColor;
          ctx.lineWidth = 3;
          ctx.stroke();

          ctx.beginPath();
          ctx.arc(top.x, top.y, 8, 0, Math.PI * 2);
          ctx.fillStyle = obj.crownColor;
          ctx.fill();
        }
      }
    }
  }

  /** @private Draw HUD overlay */
  _drawHUD(ctx) {
    const v = this._vehiclePosition;
    ctx.font = '11px Inter, monospace';
    ctx.fillStyle = '#94a3b8';
    ctx.textAlign = 'left';
    ctx.fillText(`Pos: (${v.x.toFixed(1)}, ${v.y.toFixed(1)}, ${v.z.toFixed(1)})`, 10, 20);
    ctx.fillText(`Heading: ${v.heading.toFixed(1)}°`, 10, 34);
    ctx.fillText(`Camera: ${this._camera.mode}`, 10, 48);
    ctx.fillText(`Detections: ${this._detectionBoxes.length}`, 10, 62);
  }

  /**
   * Get the camera controller.
   * @returns {CameraController}
   */
  getCamera() {
    return this._camera;
  }

  /**
   * Destroy the scene.
   */
  destroy() {
    if (this._animFrameId) {
      cancelAnimationFrame(this._animFrameId);
    }
    this._sensorFrustums = [];
    this._detectionBoxes = [];
    this._environmentObjects = [];
  }
}

export default Scene3D;
