import { readFileSync } from 'node:fs';
import { describe, expect, it } from 'vitest';
import { MAR0_DEG, centralPpd, fovealRadiusDeg, marDeg, minEccentricityForPitch } from '../src/foveation/acuity';
import { computeLayout, radiusPx, shadedPixelFraction } from '../src/foveation/layout';
import { luma, regionalPsnr, regionalSsim } from '../src/foveation/metrics';
import { GazeFilter, OneEuro } from '../src/gaze/filter';
import { PACKET_SIZE, crc32, decodeGaze, encodeGaze } from '../src/gaze/protocol';
import { acceptKey, encodeFrame } from '../scripts/wsframe';

describe('acuity model', () => {
  it('MAR grows linearly from 1/48 deg', () => {
    expect(marDeg(0)).toBeCloseTo(MAR0_DEG, 12);
    expect(marDeg(10) - marDeg(0)).toBeCloseTo(0.22, 12);
  });
  it('a coarser layer needs a larger eccentricity', () => {
    const ppd = centralPpd(2160, Math.PI / 2);
    expect(ppd).toBeCloseTo((1080 * Math.PI) / 180, 9);
    const e2 = minEccentricityForPitch(1 / (0.5 * ppd));
    const e3 = minEccentricityForPitch(1 / ((1 / 3) * ppd));
    expect(e3).toBeGreaterThan(e2);
    expect(marDeg(e3)).toBeCloseTo(1 / ((1 / 3) * ppd), 9);
    expect(fovealRadiusDeg(1 / 3, ppd, 1, 2)).toBeCloseTo(e3 + 3, 9);
  });
});

describe('layout', () => {
  it('radius in px matches f tan(r) at the centre', () => {
    expect(radiusPx(1000, 0, 10)).toBeCloseTo(1000 * Math.tan((10 * Math.PI) / 180), 9);
    expect(radiusPx(1000, 0.5, 10)).toBeGreaterThan(radiusPx(1000, 0, 10));
  });
  it('centred gaze gives a centred, clamped inset and consistent y-up offset', () => {
    const l = computeLayout(1000, 800, Math.PI / 3, { ndc: [0, 0] }, { peripheryScale: 0.25, innerDeg: 5, blendDeg: 3 });
    expect(l.gazePx).toEqual([500, 400]);
    const [x, y, w, h] = l.insetRect;
    expect(x + w / 2).toBeCloseTo(500, 0);
    expect(y + h / 2).toBeCloseTo(400, 0);
    expect(l.insetOffsetYUp[1]).toBe(800 - y - h);
    expect(l.periphery).toEqual([250, 200]);
    expect(w).toBeGreaterThanOrEqual(2 * l.rOuterPx);
    expect(shadedPixelFraction(l)).toBeLessThan(0.25);
  });
  it('inset is clamped at the screen edge', () => {
    const l = computeLayout(1000, 800, Math.PI / 3, { ndc: [1, -1] }, { peripheryScale: 0.5, innerDeg: 10, blendDeg: 5 });
    const [x, y, w, h] = l.insetRect;
    expect(x + w).toBeLessThanOrEqual(1000);
    expect(y + h).toBeLessThanOrEqual(800);
    expect(l.insetOffsetYUp[1]).toBe(0);
  });
});

describe('GZFV protocol', () => {
  const fix = JSON.parse(readFileSync(new URL('../../tests/fixtures/gaze_packet_v1.json', import.meta.url), 'utf8'));
  const bytes = Uint8Array.from(fix.hex.match(/../g).map((h: string) => parseInt(h, 16)));
  it('decodes the Python golden vector', () => {
    const p = decodeGaze(bytes);
    expect(p.seq).toBe(fix.fields.seq);
    expect(p.tNs).toBe(BigInt(fix.fields.t_ns));
    expect(p.flags).toBe(fix.fields.flags);
    expect(p.ndc[0]).toBeCloseTo(fix.fields.ndc[0], 6);
    expect(p.left.pupilMm).toBeCloseTo(3.5, 6);
    expect(p.errorDeg).toBeCloseTo(0.75, 6);
  });
  it('re-encodes byte-identically', () => {
    expect(Buffer.from(encodeGaze(decodeGaze(bytes))).toString('hex')).toBe(fix.hex);
    expect(PACKET_SIZE).toBe(92);
  });
  it('rejects corrupted packets', () => {
    const bad = bytes.slice();
    bad[40] ^= 1;
    expect(() => decodeGaze(bad)).toThrow(/CRC/);
    expect(() => decodeGaze(bytes.slice(0, 91))).toThrow(/size/);
  });
  it('crc32 matches the zlib check value', () => {
    expect(crc32(new TextEncoder().encode('123456789'))).toBe(0xcbf43926);
  });
});

describe('gaze filter', () => {
  it('1-euro filter smooths jitter during fixation and follows a saccade', () => {
    const f = new OneEuro(1.0, 0.02);
    let s = 0;
    let rnd = 1;
    const noise = () => ((rnd = (rnd * 16807) % 2147483647) / 2147483647 - 0.5) * 0.5;
    const out: number[] = [];
    for (let i = 0; i < 240; i++) out.push(f.filter(noise(), i / 240));
    for (let i = 0; i < 120; i++) out.push(f.filter(10 + noise(), 1 + i / 240));
    const fix = out.slice(100, 240);
    s = Math.sqrt(fix.reduce((a, v) => a + v * v, 0) / fix.length);
    expect(s).toBeLessThan(0.07); // raw jitter rms is 0.144
    expect(out[out.length - 1]).toBeGreaterThan(9.5);
  });
  it('flags saccades by velocity', () => {
    const g = new GazeFilter(100);
    g.push(0, 0, 0);
    expect(g.push(0.1, 0, 0.004).saccade).toBe(false);
    expect(g.push(3, 0, 0.008).saccade).toBe(true);
  });
});

describe('websocket relay framing', () => {
  it('computes the RFC 6455 example accept key', () => {
    expect(acceptKey('dGhlIHNhbXBsZSBub25jZQ==')).toBe('s3pPLMBiTxaQ9kYGzzhZRbK+xOo=');
  });
  it('frames a 92-byte packet with a 2-byte header', () => {
    const f = encodeFrame(new Uint8Array(92));
    expect(f.length).toBe(94);
    expect(f[0]).toBe(0x82);
    expect(f[1]).toBe(92);
    expect(encodeFrame(new Uint8Array(300))[1]).toBe(126);
  });
});

describe('metrics', () => {
  it('PSNR is infinite for identical images and finite otherwise; SSIM is 1 for identical', () => {
    const w = 32;
    const h = 24;
    const a = new Uint8Array(w * h * 4);
    for (let i = 0; i < a.length; i++) a[i] = (i * 37) % 256;
    const b = a.slice();
    b[100] ^= 16;
    const la = luma(a, w, h);
    const lb = luma(b, w, h);
    const reg = [{ name: 'all', r0: 0, r1: Infinity }];
    expect(regionalPsnr(la, la, w, h, [16, 12], reg)[0].psnr).toBe(Infinity);
    expect(regionalPsnr(la, lb, w, h, [16, 12], reg)[0].psnr).toBeGreaterThan(30);
    expect(regionalSsim(la, la, w, h, [16, 12], reg)[0].ssim).toBeCloseTo(1, 9);
  });
});
