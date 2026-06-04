/**
 * @module telemetry_parser
 * @description Data parser for the Autonomous Vehicle Control System.
 * Parses CAN bus messages, GPS NMEA sentences, IMU data packets, LiDAR point
 * cloud headers, radar tracks, and vehicle state messages. Includes binary
 * protocol decoding and checksum validation.
 *
 * @author Autonomous Vehicle Control System
 * @version 2.0.0
 * @license MIT
 */

import { EventEmitter } from 'events';

// ─────────────────────────────────────────────────────────────────────────────
// Custom Error Classes
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Error thrown when parsing fails.
 * @extends Error
 */
export class ParseError extends Error {
  /**
   * @param {string} message - Error description
   * @param {string} protocol - Protocol that failed
   * @param {Buffer|string} [rawData] - Original raw data
   */
  constructor(message, protocol, rawData = null) {
    super(message);
    this.name = 'ParseError';
    this.protocol = protocol;
    this.rawData = rawData;
    this.timestamp = Date.now();
  }
}

/**
 * Error thrown when checksum validation fails.
 * @extends Error
 */
export class ChecksumError extends ParseError {
  /**
   * @param {string} protocol - Protocol name
   * @param {number} expected - Expected checksum value
   * @param {number} actual - Actual checksum value
   */
  constructor(protocol, expected, actual) {
    super(`Checksum mismatch in ${protocol}: expected 0x${expected.toString(16).toUpperCase()}, got 0x${actual.toString(16).toUpperCase()}`, protocol);
    this.name = 'ChecksumError';
    this.expected = expected;
    this.actual = actual;
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// Constants & Protocol Definitions
// ─────────────────────────────────────────────────────────────────────────────

/** @enum {number} CAN bus message types */
export const CANMessageType = {
  STANDARD: 0x00,
  EXTENDED: 0x01,
  REMOTE_FRAME: 0x02,
  ERROR_FRAME: 0x03
};

/** @enum {string} Sensor data type identifiers */
export const DataType = {
  CAN_BUS: 'can_bus',
  GPS_NMEA: 'gps_nmea',
  IMU: 'imu',
  LIDAR: 'lidar',
  RADAR: 'radar',
  VEHICLE_STATE: 'vehicle_state',
  UNKNOWN: 'unknown'
};

/** @type {Map<number, string>} CAN bus PGN (Parameter Group Number) mapping */
const CAN_PGN_MAP = new Map([
  [0x0CF00400, 'EC1_EngineController'],
  [0x0CF00300, 'EC2_EngineController'],
  [0x18FEF100, 'CC1_CruiseControl'],
  [0x18FEDF00, 'HRVD_HighResolutionVehicleDistance'],
  [0x18FEF200, 'VD_VehicleDistance'],
  [0x18FEBF00, 'ERC1_ElectronicRetarderController'],
  [0x18FEE500, 'VEP_VehicleElectricalPower'],
  [0x18FEE000, 'VTG_VehicleDirectionSpeed'],
  [0x0CF00400, 'TC1_TransmissionControl'],
  [0x18FF5000, 'ABS_AntilockBrakeSystem']
]);

/** @type {object} Default parser configuration */
const DEFAULT_CONFIG = {
  /** Validate checksums */
  validateChecksums: true,
  /** Throw on parse errors (false = return error objects) */
  strictMode: true,
  /** Maximum CAN bus data payload length */
  maxCANPayloadLength: 8,
  /** Maximum NMEA sentence length */
  maxNMEALength: 256,
  /** Maximum LiDAR packet size */
  maxLidarPacketSize: 131072,
  /** Enable binary protocol auto-detection */
  autoDetectProtocol: true,
  /** Byte order for binary decoding */
  byteOrder: 'little-endian'
};

// ─────────────────────────────────────────────────────────────────────────────
// Utility Functions
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Calculate CRC-16/CCITT checksum.
 * @param {Buffer} data - Data to checksum
 * @param {number} [offset=0] - Start offset
 * @param {number} [length] - Number of bytes
 * @returns {number} 16-bit CRC value
 */
function crc16CCITT(data, offset = 0, length) {
  let crc = 0xFFFF;
  const end = length ? offset + length : data.length;
  for (let i = offset; i < end; i++) {
    crc ^= data[i] << 8;
    for (let j = 0; j < 8; j++) {
      if (crc & 0x8000) {
        crc = (crc << 1) ^ 0x1021;
      } else {
        crc = crc << 1;
      }
    }
    crc &= 0xFFFF;
  }
  return crc;
}

/**
 * Calculate XOR checksum (used in NMEA).
 * @param {string} sentence - NMEA sentence (between $ and *)
 * @returns {string} Two-character hex checksum
 */
function nmeaChecksum(sentence) {
  let checksum = 0;
  for (let i = 0; i < sentence.length; i++) {
    checksum ^= sentence.charCodeAt(i);
  }
  return checksum.toString(16).toUpperCase().padStart(2, '0');
}

/**
 * Read a 16-bit unsigned integer from a buffer.
 * @param {Buffer} buf - Buffer
 * @param {number} offset - Byte offset
 * @param {boolean} [le=true] - Little-endian
 * @returns {number}
 */
function readUInt16(buf, offset, le = true) {
  return le ? buf.readUInt16LE(offset) : buf.readUInt16BE(offset);
}

/**
 * Read a 32-bit unsigned integer from a buffer.
 * @param {Buffer} buf - Buffer
 * @param {number} offset - Byte offset
 * @param {boolean} [le=true] - Little-endian
 * @returns {number}
 */
function readUInt32(buf, offset, le = true) {
  return le ? buf.readUInt32LE(offset) : buf.readUInt32BE(offset);
}

/**
 * Read a 32-bit float from a buffer.
 * @param {Buffer} buf - Buffer
 * @param {number} offset - Byte offset
 * @param {boolean} [le=true] - Little-endian
 * @returns {number}
 */
function readFloat32(buf, offset, le = true) {
  return le ? buf.readFloatLE(offset) : buf.readFloatBE(offset);
}

/**
 * Read a 64-bit double from a buffer.
 * @param {Buffer} buf - Buffer
 * @param {number} offset - Byte offset
 * @param {boolean} [le=true] - Little-endian
 * @returns {number}
 */
function readFloat64(buf, offset, le = true) {
  return le ? buf.readDoubleLE(offset) : buf.readDoubleBE(offset);
}

/**
 * Convert degrees-minutes to decimal degrees.
 * @param {number} dm - Degrees-minutes value (e.g., 3749.2872 = 37° 49.2872')
 * @param {string} direction - N, S, E, or W
 * @returns {number} Decimal degrees
 */
function dmToDecimal(dm, direction) {
  const degrees = Math.floor(dm / 100);
  const minutes = dm - (degrees * 100);
  let decimal = degrees + (minutes / 60);
  if (direction === 'S' || direction === 'W') {
    decimal = -decimal;
  }
  return decimal;
}

// ─────────────────────────────────────────────────────────────────────────────
// TelemetryParser
// ─────────────────────────────────────────────────────────────────────────────

/**
 * Telemetry data parser supporting multiple vehicle protocols.
 * Parses CAN bus, GPS NMEA, IMU, LiDAR, radar, and vehicle state messages.
 *
 * @extends EventEmitter
 *
 * @example
 * const parser = new TelemetryParser({ validateChecksums: true });
 * parser.on('parsed', (result) => console.log(result.type, result.data));
 * const result = parser.parse(canBuffer);
 */
export class TelemetryParser extends EventEmitter {
  /**
   * @param {object} [config={}] - Configuration overrides
   */
  constructor(config = {}) {
    super();

    /** @type {object} Merged configuration */
    this.config = { ...DEFAULT_CONFIG, ...config };

    /** @type {boolean} Whether byte order is little-endian */
    this._le = this.config.byteOrder !== 'big-endian';

    /** @type {object} Parser statistics */
    this._stats = {
      totalParsed: 0,
      parseErrors: 0,
      checksumErrors: 0,
      byType: {}
    };

    // Initialize per-type counters
    for (const type of Object.values(DataType)) {
      this._stats.byType[type] = 0;
    }
  }

  // ── Main Parse Method ───────────────────────────────────────────────────

  /**
   * Parse raw telemetry data, auto-detecting the protocol.
   * @param {Buffer|string|object} rawData - Raw data to parse
   * @param {string} [hint] - Optional protocol hint (e.g. 'can_bus', 'gps_nmea')
   * @returns {object} Parsed result with type, data, and metadata
   * @throws {ParseError} If parsing fails in strict mode
   */
  parse(rawData, hint) {
    try {
      // Already parsed object
      if (typeof rawData === 'object' && !Buffer.isBuffer(rawData) && typeof rawData !== 'string') {
        return this._wrapResult(DataType.VEHICLE_STATE, rawData);
      }

      // Determine protocol
      let dataType = hint || this._detectProtocol(rawData);

      let result;
      switch (dataType) {
        case DataType.CAN_BUS:
          result = this.parseCAN(rawData);
          break;
        case DataType.GPS_NMEA:
          result = this.parseNMEA(rawData);
          break;
        case DataType.IMU:
          result = this.parseIMU(rawData);
          break;
        case DataType.LIDAR:
          result = this.parseLidar(rawData);
          break;
        case DataType.RADAR:
          result = this.parseRadar(rawData);
          break;
        case DataType.VEHICLE_STATE:
          result = this.parseVehicleState(rawData);
          break;
        default:
          result = this._wrapResult(DataType.UNKNOWN, rawData);
      }

      this._stats.totalParsed++;
      this._stats.byType[result.type] = (this._stats.byType[result.type] || 0) + 1;
      this.emit('parsed', result);
      return result;
    } catch (error) {
      this._stats.parseErrors++;
      if (error instanceof ParseError) {
        this._stats.checksumErrors += (error instanceof ChecksumError) ? 1 : 0;
        if (this.config.strictMode) throw error;
        return { type: DataType.UNKNOWN, data: null, error: error.message, timestamp: Date.now() };
      }
      const parseError = new ParseError(error.message, hint || 'unknown', rawData);
      if (this.config.strictMode) throw parseError;
      return { type: DataType.UNKNOWN, data: null, error: parseError.message, timestamp: Date.now() };
    }
  }

  // ── CAN Bus Parser ──────────────────────────────────────────────────────

  /**
   * Parse a CAN bus message.
   * Binary format: [type:1][id:4][dlc:1][data:0-8][checksum:2]
   * @param {Buffer|string} rawData - CAN bus message
   * @returns {object} Parsed CAN message
   * @throws {ParseError}
   */
  parseCAN(rawData) {
    const buf = Buffer.isBuffer(rawData) ? rawData : Buffer.from(rawData, 'hex');

    if (buf.length < 6) {
      throw new ParseError('CAN message too short (min 6 bytes)', DataType.CAN_BUS, rawData);
    }

    const msgType = buf.readUInt8(0);
    const arbId = readUInt32(buf, 1, this._le);
    const dlc = buf.readUInt8(5);

    if (dlc > this.config.maxCANPayloadLength) {
      throw new ParseError(`Invalid DLC: ${dlc} (max ${this.config.maxCANPayloadLength})`, DataType.CAN_BUS, rawData);
    }

    const dataStart = 6;
    const dataEnd = dataStart + dlc;
    const payload = buf.slice(dataStart, dataEnd);

    // Validate checksum if present
    if (this.config.validateChecksums && buf.length >= dataEnd + 2) {
      const expectedCrc = readUInt16(buf, dataEnd, this._le);
      const actualCrc = crc16CCITT(buf, 0, dataEnd);
      if (expectedCrc !== actualCrc) {
        throw new ChecksumError(DataType.CAN_BUS, expectedCrc, actualCrc);
      }
    }

    // Decode PGN from arbitration ID (J1939 format)
    const pgn = (arbId >> 8) & 0x3FFFF;
    const pgnName = CAN_PGN_MAP.get(arbId & 0xFF0000FF) || `PGN_${pgn}`;

    const result = {
      type: DataType.CAN_BUS,
      data: {
        messageType: Object.entries(CANMessageType).find(([, v]) => v === msgType)?.[0] || 'UNKNOWN',
        arbitrationId: arbId,
        pgn,
        pgnName,
        dlc,
        payload: payload.toString('hex'),
        payloadBytes: Array.from(payload),
        timestamp: Date.now()
      },
      metadata: { rawLength: buf.length, protocol: 'CAN/J1939' }
    };

    this.emit('can:parsed', result);
    return result;
  }

  // ── GPS NMEA Parser ─────────────────────────────────────────────────────

  /**
   * Parse a GPS NMEA sentence (GGA, RMC, GSA, GSV, VTG).
   * @param {string} rawData - NMEA sentence string
   * @returns {object} Parsed GPS data
   * @throws {ParseError}
   */
  parseNMEA(rawData) {
    const sentence = typeof rawData === 'string' ? rawData.trim() : rawData.toString().trim();

    if (sentence.length > this.config.maxNMEALength) {
      throw new ParseError(`NMEA sentence too long (${sentence.length} chars)`, DataType.GPS_NMEA, rawData);
    }

    if (!sentence.startsWith('$')) {
      throw new ParseError('NMEA sentence must start with $', DataType.GPS_NMEA, rawData);
    }

    // Validate checksum
    if (this.config.validateChecksums && sentence.includes('*')) {
      const starIdx = sentence.indexOf('*');
      const content = sentence.substring(1, starIdx);
      const providedChecksum = sentence.substring(starIdx + 1, starIdx + 3);
      const computedChecksum = nmeaChecksum(content);
      if (providedChecksum.toUpperCase() !== computedChecksum) {
        throw new ChecksumError(DataType.GPS_NMEA, parseInt(providedChecksum, 16), parseInt(computedChecksum, 16));
      }
    }

    // Parse fields
    const starIdx = sentence.indexOf('*');
    const contentPart = starIdx > 0 ? sentence.substring(1, starIdx) : sentence.substring(1);
    const fields = contentPart.split(',');

    const talkerId = fields[0].substring(0, 2);
    const sentenceType = fields[0].substring(2);

    let data;
    switch (sentenceType) {
      case 'GGA':
        data = this._parseGGA(fields);
        break;
      case 'RMC':
        data = this._parseRMC(fields);
        break;
      case 'GSA':
        data = this._parseGSA(fields);
        break;
      case 'GSV':
        data = this._parseGSV(fields);
        break;
      case 'VTG':
        data = this._parseVTG(fields);
        break;
      default:
        data = { talkerId, sentenceType, raw: fields };
    }

    const result = {
      type: DataType.GPS_NMEA,
      data: {
        talkerId,
        sentenceType,
        ...data,
        timestamp: Date.now()
      },
      metadata: { rawLength: sentence.length, protocol: 'NMEA-0183' }
    };

    this.emit('nmea:parsed', result);
    return result;
  }

  // ── IMU Parser ──────────────────────────────────────────────────────────

  /**
   * Parse an IMU data packet.
   * Binary format: [header:4][seq:2][timestamp:8][accel_x:4][accel_y:4][accel_z:4]
   *                [gyro_x:4][gyro_y:4][gyro_z:4][mag_x:4][mag_y:4][mag_z:4]
   *                [temperature:4][checksum:2]
   * @param {Buffer|string} rawData - IMU binary data
   * @returns {object} Parsed IMU data
   * @throws {ParseError}
   */
  parseIMU(rawData) {
    const buf = Buffer.isBuffer(rawData) ? rawData : Buffer.from(rawData, 'hex');

    if (buf.length < 56) {
      throw new ParseError('IMU packet too short (min 56 bytes)', DataType.IMU, rawData);
    }

    // Validate header magic bytes
    const header = readUInt32(buf, 0, this._le);
    if (header !== 0x494D5531) { // "IMU1"
      throw new ParseError(`Invalid IMU header: 0x${header.toString(16)}`, DataType.IMU, rawData);
    }

    // Validate checksum
    if (this.config.validateChecksums && buf.length >= 58) {
      const expectedCrc = readUInt16(buf, buf.length - 2, this._le);
      const actualCrc = crc16CCITT(buf, 0, buf.length - 2);
      if (expectedCrc !== actualCrc) {
        throw new ChecksumError(DataType.IMU, expectedCrc, actualCrc);
      }
    }

    const result = {
      type: DataType.IMU,
      data: {
        sequenceId: readUInt16(buf, 4, this._le),
        sensorTimestamp: readFloat64(buf, 6, this._le),
        accelerometer: {
          x: readFloat32(buf, 14, this._le), // m/s²
          y: readFloat32(buf, 18, this._le),
          z: readFloat32(buf, 22, this._le)
        },
        gyroscope: {
          x: readFloat32(buf, 26, this._le), // rad/s
          y: readFloat32(buf, 30, this._le),
          z: readFloat32(buf, 34, this._le)
        },
        magnetometer: {
          x: readFloat32(buf, 38, this._le), // µT
          y: readFloat32(buf, 42, this._le),
          z: readFloat32(buf, 46, this._le)
        },
        temperature: readFloat32(buf, 50, this._le), // °C
        timestamp: Date.now()
      },
      metadata: { rawLength: buf.length, protocol: 'IMU-Binary-v1' }
    };

    this.emit('imu:parsed', result);
    return result;
  }

  // ── LiDAR Parser ────────────────────────────────────────────────────────

  /**
   * Parse a LiDAR point cloud packet header.
   * Binary format: [magic:4][version:2][seq:4][timestamp:8][azimuth:2]
   *                [blockCount:1][pointsPerBlock:1][returnMode:1]
   *                [horizontalResolution:2][verticalRange:4][checksum:2]
   * @param {Buffer|string} rawData - LiDAR binary data
   * @returns {object} Parsed LiDAR header + point data
   * @throws {ParseError}
   */
  parseLidar(rawData) {
    const buf = Buffer.isBuffer(rawData) ? rawData : Buffer.from(rawData, 'hex');

    if (buf.length < 32) {
      throw new ParseError('LiDAR packet too short (min 32 bytes)', DataType.LIDAR, rawData);
    }

    const magic = buf.toString('ascii', 0, 4);
    if (magic !== 'LIDR') {
      throw new ParseError(`Invalid LiDAR magic: '${magic}'`, DataType.LIDAR, rawData);
    }

    const version = readUInt16(buf, 4, this._le);
    const seq = readUInt32(buf, 6, this._le);
    const sensorTimestamp = readFloat64(buf, 10, this._le);
    const azimuth = readUInt16(buf, 18, this._le) * 0.01; // 0.01 degree resolution
    const blockCount = buf.readUInt8(20);
    const pointsPerBlock = buf.readUInt8(21);
    const returnMode = buf.readUInt8(22);
    const hResolution = readUInt16(buf, 23, this._le) * 0.001; // degrees
    const vRangeMin = readFloat32(buf, 25, this._le);
    const vRangeMax = readFloat32(buf, 29, this._le);

    // Parse point blocks if present
    const points = [];
    const pointDataStart = 33;
    const pointSize = 6; // 2 bytes distance + 2 bytes intensity + 2 bytes vertical angle index

    for (let block = 0; block < blockCount && pointDataStart + (block + 1) * pointsPerBlock * pointSize <= buf.length; block++) {
      for (let p = 0; p < pointsPerBlock; p++) {
        const offset = pointDataStart + (block * pointsPerBlock + p) * pointSize;
        const distance = readUInt16(buf, offset, this._le) * 0.002; // 2mm resolution
        const intensity = readUInt16(buf, offset + 2, this._le);
        const verticalIdx = readUInt16(buf, offset + 4, this._le);

        if (distance > 0) {
          points.push({ distance, intensity, verticalIndex: verticalIdx, azimuth });
        }
      }
    }

    const result = {
      type: DataType.LIDAR,
      data: {
        version,
        sequenceId: seq,
        sensorTimestamp,
        azimuth,
        blockCount,
        pointsPerBlock,
        returnMode: returnMode === 0 ? 'single' : returnMode === 1 ? 'dual' : 'unknown',
        horizontalResolution: hResolution,
        verticalRange: { min: vRangeMin, max: vRangeMax },
        pointCount: points.length,
        points: points.slice(0, 100), // Limit stored points
        timestamp: Date.now()
      },
      metadata: { rawLength: buf.length, protocol: `LiDAR-Binary-v${version}` }
    };

    this.emit('lidar:parsed', result);
    return result;
  }

  // ── Radar Parser ────────────────────────────────────────────────────────

  /**
   * Parse a radar track message.
   * Binary format: [header:4][timestamp:8][trackCount:1][tracks...][checksum:2]
   * Track format: [id:2][status:1][distance:4][azimuth:4][elevation:4]
   *               [velocity:4][acceleration:4][rcs:4]
   * @param {Buffer|string} rawData - Radar binary data
   * @returns {object} Parsed radar tracks
   * @throws {ParseError}
   */
  parseRadar(rawData) {
    const buf = Buffer.isBuffer(rawData) ? rawData : Buffer.from(rawData, 'hex');

    if (buf.length < 14) {
      throw new ParseError('Radar packet too short', DataType.RADAR, rawData);
    }

    const header = readUInt32(buf, 0, this._le);
    if (header !== 0x52415231) { // "RAR1"
      throw new ParseError(`Invalid radar header: 0x${header.toString(16)}`, DataType.RADAR, rawData);
    }

    const sensorTimestamp = readFloat64(buf, 4, this._le);
    const trackCount = buf.readUInt8(12);
    const trackSize = 31; // bytes per track
    const tracksStart = 13;

    const tracks = [];
    for (let i = 0; i < trackCount && tracksStart + (i + 1) * trackSize <= buf.length - 2; i++) {
      const offset = tracksStart + i * trackSize;
      tracks.push({
        id: readUInt16(buf, offset, this._le),
        status: buf.readUInt8(offset + 2) === 0 ? 'lost' : buf.readUInt8(offset + 2) === 1 ? 'new' : 'tracked',
        distance: readFloat32(buf, offset + 3, this._le),       // meters
        azimuth: readFloat32(buf, offset + 7, this._le),        // radians
        elevation: readFloat32(buf, offset + 11, this._le),     // radians
        velocity: readFloat32(buf, offset + 15, this._le),      // m/s
        acceleration: readFloat32(buf, offset + 19, this._le),  // m/s²
        rcs: readFloat32(buf, offset + 23, this._le),           // dBm²
        confidence: readFloat32(buf, offset + 27, this._le)
      });
    }

    const result = {
      type: DataType.RADAR,
      data: {
        sensorTimestamp,
        trackCount,
        tracks,
        timestamp: Date.now()
      },
      metadata: { rawLength: buf.length, protocol: 'Radar-Binary-v1' }
    };

    this.emit('radar:parsed', result);
    return result;
  }

  // ── Vehicle State Parser ────────────────────────────────────────────────

  /**
   * Parse a vehicle state message (JSON format).
   * @param {Buffer|string} rawData - Vehicle state data (JSON)
   * @returns {object} Parsed vehicle state
   * @throws {ParseError}
   */
  parseVehicleState(rawData) {
    let data;
    try {
      const str = Buffer.isBuffer(rawData) ? rawData.toString('utf-8') : String(rawData);
      data = JSON.parse(str);
    } catch (error) {
      throw new ParseError(`Vehicle state JSON parse error: ${error.message}`, DataType.VEHICLE_STATE, rawData);
    }

    // Normalize vehicle state structure
    const normalized = {
      speed: data.speed ?? data.velocity ?? 0,                 // m/s
      heading: data.heading ?? data.yaw ?? 0,                  // radians
      location: data.location ?? data.position ?? null,
      acceleration: data.acceleration ?? null,
      steeringAngle: data.steeringAngle ?? data.steer_angle ?? 0, // radians
      throttle: data.throttle ?? 0,                            // 0-1
      brake: data.brake ?? 0,                                  // 0-1
      gear: data.gear ?? data.gearPosition ?? 'P',
      turnSignal: data.turnSignal ?? data.turn_signal ?? 'none',
      timestamp: data.timestamp ?? Date.now()
    };

    const result = {
      type: DataType.VEHICLE_STATE,
      data: normalized,
      metadata: { protocol: 'VehicleState-JSON' }
    };

    this.emit('vehicle_state:parsed', result);
    return result;
  }

  // ── Statistics ──────────────────────────────────────────────────────────

  /**
   * Get parser statistics.
   * @returns {object}
   */
  getStats() {
    return { ...this._stats };
  }

  /**
   * Get health status.
   * @returns {object}
   */
  getHealth() {
    const errorRate = this._stats.totalParsed > 0
      ? this._stats.parseErrors / this._stats.totalParsed
      : 0;
    return {
      status: errorRate < 0.05 ? 'healthy' : errorRate < 0.2 ? 'degraded' : 'unhealthy',
      errorRate,
      ...this._stats
    };
  }

  // ── Private Methods ─────────────────────────────────────────────────────

  /**
   * Auto-detect the data protocol from raw data.
   * @param {Buffer|string} rawData - Raw data
   * @returns {string} Detected DataType
   * @private
   */
  _detectProtocol(rawData) {
    // NMEA: starts with '$'
    if (typeof rawData === 'string' && rawData.startsWith('$')) {
      return DataType.GPS_NMEA;
    }
    if (Buffer.isBuffer(rawData) && rawData.length > 0 && rawData[0] === 0x24) {
      return DataType.GPS_NMEA;
    }

    // Check binary magic bytes
    if (Buffer.isBuffer(rawData) && rawData.length >= 4) {
      const magic = rawData.toString('ascii', 0, 4);
      if (magic === 'IMU1') return DataType.IMU;
      if (magic === 'LIDR') return DataType.LIDAR;
      if (magic === 'RAR1') return DataType.RADAR;
    }

    // Try JSON parse for vehicle state
    if (typeof rawData === 'string') {
      try {
        const parsed = JSON.parse(rawData);
        if (parsed.speed !== undefined || parsed.heading !== undefined || parsed.throttle !== undefined) {
          return DataType.VEHICLE_STATE;
        }
      } catch (_) { /* not JSON */ }
    }

    // Try CAN bus detection (binary with valid-ish structure)
    if (Buffer.isBuffer(rawData) && rawData.length >= 6) {
      const msgType = rawData.readUInt8(0);
      if (msgType <= 0x03) {
        return DataType.CAN_BUS;
      }
    }

    return DataType.UNKNOWN;
  }

  /**
   * Wrap parsed data into a standard result format.
   * @param {string} type - DataType
   * @param {*} data - Parsed data
   * @returns {object}
   * @private
   */
  _wrapResult(type, data) {
    return {
      type,
      data: { ...data, timestamp: data.timestamp || Date.now() },
      metadata: { protocol: 'pass-through' }
    };
  }

  /**
   * Parse GGA (Global Positioning System Fix Data) sentence fields.
   * @param {string[]} fields - NMEA fields
   * @returns {object}
   * @private
   */
  _parseGGA(fields) {
    return {
      utcTime: fields[1] || '',
      latitude: fields[2] ? dmToDecimal(parseFloat(fields[2]), fields[3]) : null,
      longitude: fields[4] ? dmToDecimal(parseFloat(fields[4]), fields[5]) : null,
      fixQuality: parseInt(fields[6]) || 0,
      satellitesInView: parseInt(fields[7]) || 0,
      hdop: parseFloat(fields[8]) || 0,
      altitude: { value: parseFloat(fields[9]) || 0, unit: fields[10] || 'M' },
      geoidalSeparation: { value: parseFloat(fields[11]) || 0, unit: fields[12] || 'M' },
      dgpsAge: fields[13] || '',
      dgpsStation: fields[14] || ''
    };
  }

  /**
   * Parse RMC (Recommended Minimum) sentence fields.
   * @param {string[]} fields - NMEA fields
   * @returns {object}
   * @private
   */
  _parseRMC(fields) {
    return {
      utcTime: fields[1] || '',
      status: fields[2] === 'A' ? 'active' : 'void',
      latitude: fields[3] ? dmToDecimal(parseFloat(fields[3]), fields[4]) : null,
      longitude: fields[5] ? dmToDecimal(parseFloat(fields[5]), fields[6]) : null,
      speedKnots: parseFloat(fields[7]) || 0,
      course: parseFloat(fields[8]) || 0,
      date: fields[9] || '',
      magneticVariation: fields[10] ? parseFloat(fields[10]) : null,
      variationDirection: fields[11] || ''
    };
  }

  /**
   * Parse GSA (DOP and Active Satellites) sentence fields.
   * @param {string[]} fields - NMEA fields
   * @returns {object}
   * @private
   */
  _parseGSA(fields) {
    const satellites = [];
    for (let i = 3; i <= 14; i++) {
      if (fields[i]) satellites.push(parseInt(fields[i]));
    }
    return {
      mode: fields[1] === 'A' ? 'automatic' : 'manual',
      fixType: parseInt(fields[2]) || 1,
      satellites,
      pdop: parseFloat(fields[15]) || 0,
      hdop: parseFloat(fields[16]) || 0,
      vdop: parseFloat(fields[17]) || 0
    };
  }

  /**
   * Parse GSV (Satellites in View) sentence fields.
   * @param {string[]} fields - NMEA fields
   * @returns {object}
   * @private
   */
  _parseGSV(fields) {
    const satellites = [];
    for (let i = 4; i < fields.length - 1; i += 4) {
      satellites.push({
        prn: parseInt(fields[i]) || 0,
        elevation: parseInt(fields[i + 1]) || 0,
        azimuth: parseInt(fields[i + 2]) || 0,
        snr: parseInt(fields[i + 3]) || 0
      });
    }
    return {
      totalMessages: parseInt(fields[1]) || 1,
      messageNumber: parseInt(fields[2]) || 1,
      satellitesInView: parseInt(fields[3]) || 0,
      satellites
    };
  }

  /**
   * Parse VTG (Track Made Good and Ground Speed) sentence fields.
   * @param {string[]} fields - NMEA fields
   * @returns {object}
   * @private
   */
  _parseVTG(fields) {
    return {
      courseTrue: parseFloat(fields[1]) || 0,
      courseMagnetic: parseFloat(fields[3]) || 0,
      speedKnots: parseFloat(fields[5]) || 0,
      speedKph: parseFloat(fields[7]) || 0,
      mode: fields[9] || 'N'
    };
  }
}

export default TelemetryParser;
