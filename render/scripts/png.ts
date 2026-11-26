/**
 * Minimal PNG writer (RGBA8, filter 0, zlib via node:zlib) and helpers for drawing overlays.
 * Adapted from splatweb v0.1.0 scripts/headless.ts (MIT).
 */
import { deflateSync } from 'node:zlib';

const CRC_TABLE = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) {
    let c = n;
    for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1;
    t[n] = c >>> 0;
  }
  return t;
})();

function crc32(buf: Uint8Array): number {
  let c = 0xffffffff;
  for (let i = 0; i < buf.length; i++) c = CRC_TABLE[(c ^ buf[i]) & 0xff] ^ (c >>> 8);
  return (c ^ 0xffffffff) >>> 0;
}

function chunk(type: string, data: Uint8Array): Uint8Array {
  const out = new Uint8Array(12 + data.length);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, data.length);
  for (let i = 0; i < 4; i++) out[4 + i] = type.charCodeAt(i);
  out.set(data, 8);
  dv.setUint32(8 + data.length, crc32(out.subarray(4, 8 + data.length)));
  return out;
}

export function encodePng(w: number, h: number, rgba: Uint8Array): Uint8Array {
  const ihdr = new Uint8Array(13);
  const dv = new DataView(ihdr.buffer);
  dv.setUint32(0, w);
  dv.setUint32(4, h);
  ihdr.set([8, 6, 0, 0, 0], 8);
  const raw = new Uint8Array(h * (w * 4 + 1));
  for (let y = 0; y < h; y++) raw.set(rgba.subarray(y * w * 4, (y + 1) * w * 4), y * (w * 4 + 1) + 1);
  const parts = [new Uint8Array([137, 80, 78, 71, 13, 10, 26, 10]), chunk('IHDR', ihdr), chunk('IDAT', deflateSync(raw, { level: 9 })), chunk('IEND', new Uint8Array(0))];
  const out = new Uint8Array(parts.reduce((s, p) => s + p.length, 0));
  let o = 0;
  for (const p of parts) {
    out.set(p, o);
    o += p.length;
  }
  return out;
}

/** Composite premultiplied RGBA over an opaque background colour (in place copy). */
export function overBackground(src: Uint8Array, bg: [number, number, number]): Uint8Array {
  const out = new Uint8Array(src.length);
  for (let i = 0; i < src.length; i += 4) {
    const a = src[i + 3] / 255;
    for (let c = 0; c < 3; c++) out[i + c] = Math.min(255, Math.round(src[i + c] + (1 - a) * bg[c]));
    out[i + 3] = 255;
  }
  return out;
}

function blendPx(img: Uint8Array, w: number, h: number, x: number, y: number, col: [number, number, number], a: number) {
  const xi = Math.round(x);
  const yi = Math.round(y);
  if (xi < 0 || yi < 0 || xi >= w || yi >= h) return;
  const i = (yi * w + xi) * 4;
  for (let c = 0; c < 3; c++) img[i + c] = Math.round(img[i + c] * (1 - a) + col[c] * a);
}

export function drawCircle(img: Uint8Array, w: number, h: number, cx: number, cy: number, r: number, col: [number, number, number], a = 1, dash = 0, thick = 1) {
  const n = Math.max(32, Math.ceil(2 * Math.PI * r * 1.5));
  for (let k = 0; k < n; k++) {
    if (dash > 0 && Math.floor((k / n) * 2 * Math.PI * r / dash) % 2 === 1) continue;
    const t = (2 * Math.PI * k) / n;
    for (let d = 0; d < thick; d++) blendPx(img, w, h, cx + (r + d) * Math.cos(t), cy + (r + d) * Math.sin(t), col, a);
  }
}

export function drawCross(img: Uint8Array, w: number, h: number, cx: number, cy: number, size: number, gap: number, col: [number, number, number], thick = 2) {
  for (let s = gap; s <= size; s++)
    for (let d = -Math.floor(thick / 2); d < Math.ceil(thick / 2); d++) {
      blendPx(img, w, h, cx + s, cy + d, col, 1);
      blendPx(img, w, h, cx - s, cy + d, col, 1);
      blendPx(img, w, h, cx + d, cy + s, col, 1);
      blendPx(img, w, h, cx + d, cy - s, col, 1);
    }
}

export function drawRect(img: Uint8Array, w: number, h: number, x0: number, y0: number, rw: number, rh: number, col: [number, number, number], a = 1) {
  for (let x = x0; x < x0 + rw; x++) {
    blendPx(img, w, h, x, y0, col, a);
    blendPx(img, w, h, x, y0 + rh - 1, col, a);
  }
  for (let y = y0; y < y0 + rh; y++) {
    blendPx(img, w, h, x0, y, col, a);
    blendPx(img, w, h, x0 + rw - 1, y, col, a);
  }
}
