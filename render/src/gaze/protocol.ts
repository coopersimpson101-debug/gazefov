/**
 * GZFV gaze packet v1 — 92 bytes, little-endian. Byte-for-byte compatible with
 * gazefov/protocol.py (cross-checked by a shared golden vector in tests/fixtures).
 * See docs/protocol.md for the table.
 */
export const MAGIC = 'GZFV';
export const VERSION = 1;
export const PACKET_SIZE = 92;

export const FLAG_VALID = 1 << 0;
export const FLAG_LEFT = 1 << 1;
export const FLAG_RIGHT = 1 << 2;
export const FLAG_CALIBRATED = 1 << 3;
export const FLAG_SACCADE = 1 << 4;
export const FLAG_OPENXR = 1 << 5;

export interface EyeSample {
  pupilMm: number;
  confidence: number;
  yawDeg: number;
  pitchDeg: number;
}

export interface GazePacket {
  flags: number;
  seq: number;
  tNs: bigint;
  origin: [number, number, number];
  direction: [number, number, number];
  ndc: [number, number];
  left: EyeSample;
  right: EyeSample;
  errorDeg: number;
}

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

export function crc32(buf: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

export function encodeGaze(p: GazePacket): Uint8Array {
  const out = new Uint8Array(PACKET_SIZE);
  const dv = new DataView(out.buffer);
  for (let i = 0; i < 4; i++) out[i] = MAGIC.charCodeAt(i);
  dv.setUint16(4, VERSION, true);
  dv.setUint16(6, p.flags & 0xffff, true);
  dv.setUint32(8, p.seq >>> 0, true);
  dv.setBigUint64(12, p.tNs, true);
  const f = [...p.origin, ...p.direction, ...p.ndc,
    p.left.pupilMm, p.left.confidence, p.left.yawDeg, p.left.pitchDeg,
    p.right.pupilMm, p.right.confidence, p.right.yawDeg, p.right.pitchDeg, p.errorDeg];
  f.forEach((v, i) => dv.setFloat32(20 + 4 * i, v, true));
  dv.setUint32(88, crc32(out.subarray(0, 88)), true);
  return out;
}

export function decodeGaze(buf: ArrayBuffer | Uint8Array): GazePacket {
  const u8 = buf instanceof Uint8Array ? buf : new Uint8Array(buf);
  if (u8.length !== PACKET_SIZE) throw new Error(`bad packet size ${u8.length}`);
  const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
  if (crc32(u8.subarray(0, 88)) !== dv.getUint32(88, true)) throw new Error('CRC mismatch');
  const magic = String.fromCharCode(u8[0], u8[1], u8[2], u8[3]);
  if (magic !== MAGIC || dv.getUint16(4, true) !== VERSION) throw new Error('bad magic/version');
  const f = (i: number) => dv.getFloat32(20 + 4 * i, true);
  return {
    flags: dv.getUint16(6, true),
    seq: dv.getUint32(8, true),
    tNs: dv.getBigUint64(12, true),
    origin: [f(0), f(1), f(2)],
    direction: [f(3), f(4), f(5)],
    ndc: [f(6), f(7)],
    left: { pupilMm: f(8), confidence: f(9), yawDeg: f(10), pitchDeg: f(11) },
    right: { pupilMm: f(12), confidence: f(13), yawDeg: f(14), pitchDeg: f(15) },
    errorDeg: f(16),
  };
}
