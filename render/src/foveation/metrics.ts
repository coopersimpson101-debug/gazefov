/**
 * Image-quality metrics for foveated vs reference frames, split by eccentricity region.
 * Images are RGBA8, row-major, y down. Luma is BT.601 on the colour composited over a fixed
 * background (so premultiplied transparency is compared the way it is displayed).
 */
export interface Region {
  name: string;
  /** Inclusive lower / exclusive upper radius in pixels from the gaze point. */
  r0: number;
  r1: number;
}

export function luma(rgba: Uint8Array, w: number, h: number, bg: [number, number, number] = [13, 17, 23]): Float32Array {
  const out = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const a = rgba[4 * i + 3] / 255;
    const r = rgba[4 * i] + (1 - a) * bg[0];
    const g = rgba[4 * i + 1] + (1 - a) * bg[1];
    const b = rgba[4 * i + 2] + (1 - a) * bg[2];
    out[i] = 0.299 * r + 0.587 * g + 0.114 * b;
  }
  return out;
}

/** PSNR (dB, peak 255) per radial region around `gaze`. Infinity when identical. */
export function regionalPsnr(ref: Float32Array, test: Float32Array, w: number, h: number, gaze: [number, number], regions: Region[]) {
  const se = new Float64Array(regions.length);
  const n = new Float64Array(regions.length);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const r = Math.hypot(x + 0.5 - gaze[0], y + 0.5 - gaze[1]);
      const i = y * w + x;
      const e = ref[i] - test[i];
      for (let k = 0; k < regions.length; k++)
        if (r >= regions[k].r0 && r < regions[k].r1) {
          se[k] += e * e;
          n[k]++;
        }
    }
  return regions.map((reg, k) => {
    const mse = n[k] ? se[k] / n[k] : NaN;
    return { region: reg.name, pixels: n[k], psnr: mse === 0 ? Infinity : 10 * Math.log10((255 * 255) / mse) };
  });
}

function blur(src: Float32Array, w: number, h: number, k: Float32Array): Float32Array {
  const r = (k.length - 1) / 2;
  const tmp = new Float32Array(w * h);
  const out = new Float32Array(w * h);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = -r; j <= r; j++) s += k[j + r] * src[y * w + Math.min(w - 1, Math.max(0, x + j))];
      tmp[y * w + x] = s;
    }
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      let s = 0;
      for (let j = -r; j <= r; j++) s += k[j + r] * tmp[Math.min(h - 1, Math.max(0, y + j)) * w + x];
      out[y * w + x] = s;
    }
  return out;
}

/** SSIM map (Wang et al. 2004; 11-tap Gaussian, sigma 1.5, K1 = 0.01, K2 = 0.03), averaged per region. */
export function regionalSsim(ref: Float32Array, test: Float32Array, w: number, h: number, gaze: [number, number], regions: Region[]) {
  const k = new Float32Array(11);
  let ks = 0;
  for (let i = 0; i < 11; i++) {
    k[i] = Math.exp(-0.5 * ((i - 5) / 1.5) ** 2);
    ks += k[i];
  }
  for (let i = 0; i < 11; i++) k[i] /= ks;
  const xy = new Float32Array(w * h);
  const xx = new Float32Array(w * h);
  const yy = new Float32Array(w * h);
  for (let i = 0; i < w * h; i++) {
    xy[i] = ref[i] * test[i];
    xx[i] = ref[i] * ref[i];
    yy[i] = test[i] * test[i];
  }
  const mx = blur(ref, w, h, k);
  const my = blur(test, w, h, k);
  const sxx = blur(xx, w, h, k);
  const syy = blur(yy, w, h, k);
  const sxy = blur(xy, w, h, k);
  const C1 = (0.01 * 255) ** 2;
  const C2 = (0.03 * 255) ** 2;
  const sum = new Float64Array(regions.length);
  const n = new Float64Array(regions.length);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      const vx = sxx[i] - mx[i] * mx[i];
      const vy = syy[i] - my[i] * my[i];
      const cxy = sxy[i] - mx[i] * my[i];
      const s = ((2 * mx[i] * my[i] + C1) * (2 * cxy + C2)) / ((mx[i] ** 2 + my[i] ** 2 + C1) * (vx + vy + C2));
      const r = Math.hypot(x + 0.5 - gaze[0], y + 0.5 - gaze[1]);
      for (let q = 0; q < regions.length; q++)
        if (r >= regions[q].r0 && r < regions[q].r1) {
          sum[q] += s;
          n[q]++;
        }
    }
  return regions.map((reg, q) => ({ region: reg.name, ssim: n[q] ? sum[q] / n[q] : NaN }));
}
