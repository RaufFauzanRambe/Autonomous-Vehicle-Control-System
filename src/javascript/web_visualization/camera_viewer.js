/**
 * @fileoverview Camera Viewer Module - MJPEG/WebSocket video stream viewer with multi-camera
 * grid layout, camera switching, snapshot capture, recording toggle, overlay (detection boxes,
 * timestamps), fullscreen mode, and PTZ controls stub.
 *
 * @module camera_viewer
 * @version 2.0.0
 * @author Autonomous Vehicle Control System Team
 */

// ============================================================
// Detection Box Overlay
// ============================================================

/**
 * Represents a detection bounding box with label and confidence.
 * @class DetectionBox
 */
export class DetectionBox {
  /**
   * @param {Object} config
   * @param {number} config.x - Top-left X (normalized 0-1)
   * @param {number} config.y - Top-left Y (normalized 0-1)
   * @param {number} config.width - Box width (normalized 0-1)
   * @param {number} config.height - Box height (normalized 0-1)
   * @param {string} [config.label=''] - Detection label
   * @param {number} [config.confidence=0] - Detection confidence [0, 1]
   * @param {'vehicle'|'pedestrian'|'cyclist'|'traffic_light'|'sign'|'unknown'} [config.type='unknown']
   * @param {string} [config.trackId] - Tracking ID for the detection
   */
  constructor(config) {
    this.x = config.x;
    this.y = config.y;
    this.width = config.width;
    this.height = config.height;
    this.label = config.label ?? '';
    this.confidence = config.confidence ?? 0;
    this.type = config.type ?? 'unknown';
    this.trackId = config.trackId ?? null;
  }

  /**
   * Get the color associated with this detection type.
   * @returns {string} CSS color
   */
  getColor() {
    const colors = {
      vehicle: '#ef4444',
      pedestrian: '#f59e0b',
      cyclist: '#3b82f6',
      traffic_light: '#22c55e',
      sign: '#a855f7',
      unknown: '#94a3b8',
    };
    return colors[this.type] ?? '#94a3b8';
  }

  /**
   * Check if a point is inside this detection box.
   * @param {number} px - Point X (normalized)
   * @param {number} py - Point Y (normalized)
   * @returns {boolean}
   */
  containsPoint(px, py) {
    return px >= this.x && px <= this.x + this.width && py >= this.y && py <= this.y + this.height;
  }
}

// ============================================================
// PTZ Controller Stub
// ============================================================

/**
 * Pan-Tilt-Zoom controller interface (stub for future integration).
 * @class PTZController
 */
export class PTZController {
  /**
   * @param {Object} [options={}] - PTZ options
   * @param {number} [options.panSpeed=1.0] - Pan speed multiplier
   * @param {number} [options.tiltSpeed=1.0] - Tilt speed multiplier
   * @param {number} [options.zoomSpeed=1.0] - Zoom speed multiplier
   */
  constructor(options = {}) {
    this.panSpeed = options.panSpeed ?? 1.0;
    this.tiltSpeed = options.tiltSpeed ?? 1.0;
    this.zoomSpeed = options.zoomSpeed ?? 1.0;
    /** @type {{pan: number, tilt: number, zoom: number}} */
    this.position = { pan: 0, tilt: 0, zoom: 1 };
    /** @type {Function|null} */
    this._onCommand = null;
  }

  /**
   * Pan the camera.
   * @param {number} delta - Pan delta (-1 to 1, negative=left, positive=right)
   */
  pan(delta) {
    this.position.pan += delta * this.panSpeed;
    this._sendCommand('pan', this.position.pan);
  }

  /**
   * Tilt the camera.
   * @param {number} delta - Tilt delta (-1 to 1, negative=down, positive=up)
   */
  tilt(delta) {
    this.position.tilt += delta * this.tiltSpeed;
    this._sendCommand('tilt', this.position.tilt);
  }

  /**
   * Zoom the camera.
   * @param {number} delta - Zoom delta (positive=in, negative=out)
   */
  zoom(delta) {
    this.position.zoom = Math.max(1, Math.min(20, this.position.zoom + delta * this.zoomSpeed));
    this._sendCommand('zoom', this.position.zoom);
  }

  /**
   * Reset PTZ to home position.
   */
  reset() {
    this.position = { pan: 0, tilt: 0, zoom: 1 };
    this._sendCommand('reset', null);
  }

  /**
   * Set the command callback for sending PTZ commands.
   * @param {Function} callback - Called with (command, value)
   */
  onCommand(callback) {
    this._onCommand = callback;
  }

  /** @private */
  _sendCommand(command, value) {
    if (this._onCommand) {
      this._onCommand(command, value);
    }
  }
}

// ============================================================
// Camera Stream
// ============================================================

/**
 * Manages a single camera video stream (MJPEG or WebSocket-based).
 * @class CameraStream
 */
export class CameraStream {
  /**
   * @param {Object} config
   * @param {string} config.id - Camera identifier
   * @param {string} config.name - Display name
   * @param {string} [config.url=''] - MJPEG stream URL
   * @param {'front'|'rear'|'left'|'right'|'top'} [config.position='front'] - Camera position
   * @param {number} [config.width=640] - Stream width
   * @param {number} [config.height=480] - Stream height
   * @param {number} [config.fps=30] - Expected frames per second
   */
  constructor(config) {
    this.id = config.id;
    this.name = config.name;
    this.url = config.url ?? '';
    this.position = config.position ?? 'front';
    this.width = config.width ?? 640;
    this.height = config.height ?? 480;
    this.fps = config.fps ?? 30;

    /** @type {HTMLImageElement|null} */
    this._img = null;
    /** @type {WebSocket|null} */
    this._ws = null;
    /** @type {'disconnected'|'connecting'|'connected'|'error'} */
    this.state = 'disconnected';
    /** @type {Array<DetectionBox>} */
    this.detections = [];
    /** @type {boolean} */
    this.recording = false;
    /** @type {MediaRecorder|null} */
    this._recorder = null;
    /** @type {Array<Blob>} */
    this._recordedChunks = [];
    this._frameCount = 0;
    this._lastFpsTime = performance.now();
    this._currentFps = 0;
    this._canvas = document.createElement('canvas');
    this._canvas.width = this.width;
    this._canvas.height = this.height;
    this._ctx = this._canvas.getContext('2d');
    this._offscreenCanvas = document.createElement('canvas');
    this._offscreenCanvas.width = this.width;
    this._offscreenCanvas.height = this.height;
    this._offscreenCtx = this._offscreenCanvas.getContext('2d');
  }

  /**
   * Connect to the camera stream.
   * @param {'mjpeg'|'websocket'} [protocol='mjpeg'] - Streaming protocol
   */
  connect(protocol = 'mjpeg') {
    if (this.state === 'connected' || this.state === 'connecting') return;

    this.state = 'connecting';
    try {
      if (protocol === 'mjpeg') {
        this._connectMJPEG();
      } else {
        this._connectWebSocket();
      }
    } catch (err) {
      this.state = 'error';
      console.error(`Camera ${this.id} connection error:`, err);
    }
  }

  /** @private Connect via MJPEG */
  _connectMJPEG() {
    this._img = new Image();
    this._img.crossOrigin = 'anonymous';
    this._img.onload = () => {
      this.state = 'connected';
      this._frameCount++;
      this._drawFrame();
    };
    this._img.onerror = () => {
      this.state = 'error';
    };
    this._img.src = this.url;
    // MJPEG browsers will keep loading frames
  }

  /** @private Connect via WebSocket */
  _connectWebSocket() {
    if (!this.url) {
      this.state = 'error';
      return;
    }
    try {
      this._ws = new WebSocket(this.url);
      this._ws.binaryType = 'arrayblob';
      this._ws.onopen = () => { this.state = 'connected'; };
      this._ws.onmessage = (event) => {
        if (event.data instanceof Blob) {
          const img = new Image();
          img.onload = () => {
            this._offscreenCtx.drawImage(img, 0, 0, this.width, this.height);
            this._frameCount++;
            this._drawFrame();
            URL.revokeObjectURL(img.src);
          };
          img.src = URL.createObjectURL(event.data);
        }
      };
      this._ws.onerror = () => { this.state = 'error'; };
      this._ws.onclose = () => { this.state = 'disconnected'; };
    } catch (err) {
      this.state = 'error';
    }
  }

  /** @private Draw a frame with overlays */
  _drawFrame() {
    const ctx = this._ctx;
    const w = this.width;
    const h = this.height;

    // Draw base image
    ctx.clearRect(0, 0, w, h);
    if (this._img && this._img.complete) {
      ctx.drawImage(this._img, 0, 0, w, h);
    } else {
      ctx.fillStyle = '#1e293b';
      ctx.fillRect(0, 0, w, h);
    }

    // Draw detection boxes
    for (const det of this.detections) {
      this._drawDetectionBox(ctx, det, w, h);
    }

    // Draw timestamp
    this._drawTimestamp(ctx, w);

    // Draw recording indicator
    if (this.recording) {
      this._drawRecordingIndicator(ctx, w, h);
    }

    // FPS calculation
    const now = performance.now();
    if (now - this._lastFpsTime >= 1000) {
      this._currentFps = this._frameCount;
      this._frameCount = 0;
      this._lastFpsTime = now;
    }
  }

  /** @private Draw a single detection box */
  _drawDetectionBox(ctx, det, w, h) {
    const x = det.x * w;
    const y = det.y * h;
    const bw = det.width * w;
    const bh = det.height * h;
    const color = det.getColor();

    // Box
    ctx.strokeStyle = color;
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, bw, bh);

    // Fill with transparency
    ctx.fillStyle = color.replace(')', ',0.1)').replace('rgb', 'rgba');
    ctx.fillRect(x, y, bw, bh);

    // Label background
    const labelText = det.label ? `${det.label} ${(det.confidence * 100).toFixed(0)}%` : `${(det.confidence * 100).toFixed(0)}%`;
    ctx.font = '11px Inter, sans-serif';
    const textMetrics = ctx.measureText(labelText);
    const labelH = 16;
    const labelW = textMetrics.width + 8;

    ctx.fillStyle = color;
    ctx.fillRect(x, y - labelH, labelW, labelH);
    ctx.fillStyle = '#ffffff';
    ctx.textAlign = 'left';
    ctx.textBaseline = 'middle';
    ctx.fillText(labelText, x + 4, y - labelH / 2);

    // Track ID
    if (det.trackId) {
      ctx.font = '9px Inter, monospace';
      ctx.fillStyle = '#94a3b8';
      ctx.textAlign = 'right';
      ctx.fillText(`ID:${det.trackId}`, x + bw - 2, y + bh + 12);
    }
  }

  /** @private Draw timestamp overlay */
  _drawTimestamp(ctx, w) {
    const now = new Date();
    const timeStr = now.toISOString().replace('T', ' ').substring(0, 23);
    ctx.font = '12px monospace';
    ctx.fillStyle = 'rgba(0,0,0,0.5)';
    const metrics = ctx.measureText(timeStr);
    ctx.fillRect(w - metrics.width - 10, h - 22, metrics.width + 8, 18);
    ctx.fillStyle = '#e2e8f0';
    ctx.textAlign = 'right';
    ctx.textBaseline = 'middle';
    ctx.fillText(timeStr, w - 6, h - 13);
  }

  /** @private Draw recording indicator */
  _drawRecordingIndicator(ctx, w, h) {
    const blinkOn = Math.floor(Date.now() / 500) % 2 === 0;
    if (blinkOn) {
      ctx.beginPath();
      ctx.arc(w - 20, 20, 6, 0, Math.PI * 2);
      ctx.fillStyle = '#ef4444';
      ctx.fill();
    }
    ctx.font = 'bold 11px Inter, sans-serif';
    ctx.fillStyle = '#ef4444';
    ctx.textAlign = 'left';
    ctx.fillText('REC', w - 52, 24);
  }

  /**
   * Set detection boxes to overlay on the stream.
   * @param {Array<DetectionBox>} detections - Detection boxes
   */
  setDetections(detections) {
    this.detections = Array.isArray(detections) ? detections : [];
  }

  /**
   * Capture a snapshot of the current frame.
   * @returns {string} Data URL of the snapshot
   */
  captureSnapshot() {
    return this._canvas.toDataURL('image/png');
  }

  /**
   * Toggle recording on/off.
   * @returns {boolean} New recording state
   */
  toggleRecording() {
    if (this.recording) {
      this.recording = false;
      if (this._recorder && this._recorder.state === 'recording') {
        this._recorder.stop();
      }
    } else {
      this.recording = true;
      this._recordedChunks = [];
      try {
        const stream = this._canvas.captureStream(this.fps);
        this._recorder = new MediaRecorder(stream, { mimeType: 'video/webm;codecs=vp9' });
        this._recorder.ondataavailable = (e) => {
          if (e.data.size > 0) this._recordedChunks.push(e.data);
        };
        this._recorder.start(100);
      } catch (err) {
        this.recording = false;
        console.error('Recording not supported:', err);
      }
    }
    return this.recording;
  }

  /**
   * Get the recorded video as a Blob.
   * @returns {Blob|null} Recorded video blob
   */
  getRecording() {
    if (this._recordedChunks.length === 0) return null;
    return new Blob(this._recordedChunks, { type: 'video/webm' });
  }

  /**
   * Disconnect the camera stream.
   */
  disconnect() {
    this.state = 'disconnected';
    if (this._img) {
      this._img.src = '';
      this._img = null;
    }
    if (this._ws) {
      this._ws.close();
      this._ws = null;
    }
    if (this.recording) {
      this.toggleRecording();
    }
  }
}

// ============================================================
// Multi-Camera Grid Viewer
// ============================================================

/**
 * Multi-camera grid viewer with camera switching, fullscreen, and PTZ controls.
 * @class CameraViewer
 */
export class CameraViewer {
  /**
   * @param {HTMLElement} container - DOM container
   * @param {Object} [options={}] - Viewer options
   * @param {1|2|4|6} [options.gridLayout=4] - Grid layout (1=fullscreen, 2=split, 4=quad, 6=3x2)
   * @param {boolean} [options.showControls=true] - Show camera control bar
   * @param {boolean} [options.showLabels=true] - Show camera name labels
   */
  constructor(container, options = {}) {
    if (!(container instanceof HTMLElement)) {
      throw new TypeError('CameraViewer requires an HTMLElement container');
    }

    this._container = container;
    this._gridLayout = options.gridLayout ?? 4;
    this._showControls = options.showControls ?? true;
    this._showLabels = options.showLabels ?? true;

    /** @type {Map<string, CameraStream>} */
    this._streams = new Map();
    /** @type {string|null} */
    this._activeStreamId = null;
    /** @type {Map<string, PTZController>} */
    this._ptzControllers = new Map();
    this._isFullscreen = false;

    this._initUI();
  }

  /** @private Build the viewer UI */
  _initUI() {
    this._wrapper = document.createElement('div');
    this._wrapper.style.cssText = 'width:100%;height:100%;display:flex;flex-direction:column;background:#0f172a;';

    // Grid container
    this._gridContainer = document.createElement('div');
    this._gridContainer.style.cssText = 'flex:1;display:grid;gap:2px;padding:2px;overflow:hidden;';

    this._wrapper.appendChild(this._gridContainer);

    // Control bar
    if (this._showControls) {
      this._controlBar = document.createElement('div');
      this._controlBar.style.cssText = 'height:40px;display:flex;align-items:center;gap:8px;padding:0 12px;background:#1e293b;';
      this._buildControlBar();
      this._wrapper.appendChild(this._controlBar);
    }

    this._container.appendChild(this._wrapper);
  }

  /** @private Build control bar buttons */
  _buildControlBar() {
    const btnStyle = 'padding:4px 10px;font-size:12px;border:1px solid #475569;border-radius:4px;background:#334155;color:#e2e8f0;cursor:pointer;';

    // Grid layout buttons
    const layouts = [1, 2, 4, 6];
    for (const layout of layouts) {
      const btn = document.createElement('button');
      btn.textContent = `${layout}cam`;
      btn.style.cssText = btnStyle;
      btn.addEventListener('click', () => this.setLayout(layout));
      this._controlBar.appendChild(btn);
    }

    // Separator
    const sep = document.createElement('span');
    sep.style.cssText = 'width:1px;height:24px;background:#475569;margin:0 4px;';
    this._controlBar.appendChild(sep);

    // Fullscreen toggle
    const fsBtn = document.createElement('button');
    fsBtn.textContent = '⛶';
    fsBtn.style.cssText = btnStyle;
    fsBtn.addEventListener('click', () => this.toggleFullscreen());
    this._controlBar.appendChild(fsBtn);

    // Snapshot
    const snapBtn = document.createElement('button');
    snapBtn.textContent = '📷';
    snapBtn.style.cssText = btnStyle;
    snapBtn.addEventListener('click', () => this.captureActiveSnapshot());
    this._controlBar.appendChild(snapBtn);

    // Record toggle
    const recBtn = document.createElement('button');
    recBtn.textContent = '⏺';
    recBtn.style.cssText = btnStyle;
    recBtn.id = 'rec-btn';
    recBtn.addEventListener('click', () => this.toggleActiveRecording());
    this._controlBar.appendChild(recBtn);
  }

  /**
   * Add a camera stream to the viewer.
   * @param {CameraStream} stream - Camera stream to add
   */
  addStream(stream) {
    if (!(stream instanceof CameraStream)) {
      throw new TypeError('Expected a CameraStream instance');
    }
    this._streams.set(stream.id, stream);
    this._ptzControllers.set(stream.id, new PTZController());
    this._refreshGrid();
    if (!this._activeStreamId) {
      this._activeStreamId = stream.id;
    }
  }

  /**
   * Remove a camera stream.
   * @param {string} streamId - Stream ID to remove
   */
  removeStream(streamId) {
    const stream = this._streams.get(streamId);
    if (stream) {
      stream.disconnect();
      this._streams.delete(streamId);
      this._ptzControllers.delete(streamId);
      if (this._activeStreamId === streamId) {
        this._activeStreamId = this._streams.keys().next().value ?? null;
      }
      this._refreshGrid();
    }
  }

  /**
   * Switch the active camera for fullscreen/controls.
   * @param {string} streamId - Stream ID to make active
   */
  switchTo(streamId) {
    if (!this._streams.has(streamId)) return;
    this._activeStreamId = streamId;
    // Highlight active camera
    const panels = this._gridContainer.children;
    for (const panel of panels) {
      panel.style.border = panel.dataset.streamId === streamId ? '2px solid #3b82f6' : '2px solid transparent';
    }
  }

  /**
   * Set the grid layout.
   * @param {1|2|4|6} layout - Number of camera panels
   */
  setLayout(layout) {
    this._gridLayout = layout;
    this._refreshGrid();
  }

  /** @private Refresh the grid layout */
  _refreshGrid() {
    this._gridContainer.innerHTML = '';
    let cols, rows;
    switch (this._gridLayout) {
      case 1: cols = 1; rows = 1; break;
      case 2: cols = 2; rows = 1; break;
      case 4: cols = 2; rows = 2; break;
      case 6: cols = 3; rows = 2; break;
      default: cols = 2; rows = 2;
    }
    this._gridContainer.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    this._gridContainer.style.gridTemplateRows = `repeat(${rows}, 1fr)`;

    let index = 0;
    for (const [id, stream] of this._streams) {
      if (index >= this._gridLayout) break;
      const panel = document.createElement('div');
      panel.dataset.streamId = id;
      panel.style.cssText = 'position:relative;overflow:hidden;background:#1e293b;border:2px solid transparent;cursor:pointer;';
      panel.addEventListener('click', () => this.switchTo(id));
      panel.addEventListener('dblclick', () => {
        this._activeStreamId = id;
        this.setLayout(1);
      });

      // Canvas for rendering
      panel.appendChild(stream._canvas);
      stream._canvas.style.cssText = 'width:100%;height:100%;object-fit:contain;';

      // Camera label
      if (this._showLabels) {
        const label = document.createElement('div');
        label.textContent = stream.name;
        label.style.cssText = 'position:absolute;top:4px;left:6px;font-size:11px;color:#e2e8f0;background:rgba(0,0,0,0.5);padding:2px 6px;border-radius:3px;';
        panel.appendChild(label);
      }

      // Status indicator
      const status = document.createElement('div');
      status.className = `status-${stream.state}`;
      status.style.cssText = 'position:absolute;top:4px;right:6px;width:8px;height:8px;border-radius:50%;';
      this._updateStatusDot(status, stream.state);
      panel.appendChild(status);

      if (id === this._activeStreamId) {
        panel.style.border = '2px solid #3b82f6';
      }

      this._gridContainer.appendChild(panel);
      index++;
    }

    // Fill empty panels
    while (index < this._gridLayout) {
      const empty = document.createElement('div');
      empty.style.cssText = 'display:flex;align-items:center;justify-content:center;background:#1e293b;color:#475569;font-size:13px;';
      empty.textContent = 'No Camera';
      this._gridContainer.appendChild(empty);
      index++;
    }
  }

  /** @private Update status dot color */
  _updateStatusDot(dot, state) {
    const colors = { connected: '#22c55e', connecting: '#f59e0b', disconnected: '#64748b', error: '#ef4444' };
    dot.style.backgroundColor = colors[state] ?? '#64748b';
  }

  /**
   * Toggle fullscreen mode for the active camera.
   */
  toggleFullscreen() {
    if (!document.fullscreenElement) {
      this._wrapper.requestFullscreen?.();
      this._isFullscreen = true;
    } else {
      document.exitFullscreen?.();
      this._isFullscreen = false;
    }
  }

  /**
   * Capture a snapshot from the active camera.
   * @returns {string|null} Data URL of the snapshot
   */
  captureActiveSnapshot() {
    const stream = this._streams.get(this._activeStreamId);
    return stream ? stream.captureSnapshot() : null;
  }

  /**
   * Toggle recording on the active camera.
   * @returns {boolean} New recording state
   */
  toggleActiveRecording() {
    const stream = this._streams.get(this._activeStreamId);
    if (!stream) return false;
    const recording = stream.toggleRecording();
    const recBtn = document.getElementById('rec-btn');
    if (recBtn) {
      recBtn.style.background = recording ? '#ef4444' : '#334155';
    }
    return recording;
  }

  /**
   * Get the PTZ controller for a specific camera.
   * @param {string} streamId - Camera stream ID
   * @returns {PTZController|null}
   */
  getPTZController(streamId) {
    return this._ptzControllers.get(streamId) ?? null;
  }

  /**
   * Destroy the viewer and disconnect all streams.
   */
  destroy() {
    for (const stream of this._streams.values()) {
      stream.disconnect();
    }
    this._streams.clear();
    this._ptzControllers.clear();
    this._wrapper.remove();
  }
}

export default CameraViewer;
