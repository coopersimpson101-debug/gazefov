/**
 * Foveation layout: where the full-resolution inset goes and how big it is, given the gaze
 * direction and the foveal radii in degrees.
 *
 * Pixel conventions: `gazePx` and `insetRect` are in framebuffer pixels with y DOWN (row 0 at the
 * top), like WebGPU textures. splatweb's preprocess pass writes splat centres with y UP; the
 * raster pass converts with `insetOffsetYUp`.
 */
export interface FoveationConfig {
  /** Linear scale of the periphery layer (0 < s <= 1), e.g. 1/3 renders 1/9 of the pixels. */
  peripheryScale: number;
  /** Eccentricity (deg) inside which the output is 100 % full-resolution inset. */
  innerDeg: number;
  /** Width (deg) of the smooth blend band from inset to periphery. */
  blendDeg: number;
}

export interface Gaze {
  /** Normalised device coordinates of the gaze point on this view, x right, y up, in [-1, 1]. */
  ndc: [number, number];
}

export interface FoveationLayout {
  width: number;
  height: number;
  /** Periphery texture size. */
  periphery: [number, number];
  /** Inset rectangle [x, y, w, h], y down, clamped to the framebuffer. */
  insetRect: [number, number, number, number];
  /** Inset offset in splatweb's y-up pixel space (x, y of the rect's bottom-left corner). */
  insetOffsetYUp: [number, number];
  gazePx: [number, number];
  /** Blend radii in pixels around the gaze point (inner = 100 % inset, outer = 100 % periphery). */
  rInnerPx: number;
  rOuterPx: number;
}

/**
 * Radius in pixels on the image plane that subtends at least `deg` degrees around the gaze
 * direction for a symmetric perspective projection with focal length `f` px. The pixel density
 * of a flat image plane grows away from the centre, so the largest radial extent is used.
 */
export function radiusPx(f: number, gazeTanRadius: number, deg: number): number {
  const th = Math.atan(gazeTanRadius);
  const r = (deg * Math.PI) / 180;
  const outward = f * (Math.tan(Math.min(th + r, Math.PI / 2 - 1e-3)) - gazeTanRadius);
  const inward = f * (gazeTanRadius - Math.tan(th - r));
  return Math.max(outward, inward);
}

export function computeLayout(width: number, height: number, fovY: number, gaze: Gaze, cfg: FoveationConfig): FoveationLayout {
  const fy = height / 2 / Math.tan(fovY / 2);
  const gx = (gaze.ndc[0] * 0.5 + 0.5) * width;
  const gyDown = (0.5 - gaze.ndc[1] * 0.5) * height;
  const tanR = Math.hypot((gx - width / 2) / fy, (gyDown - height / 2) / fy);
  const rIn = radiusPx(fy, tanR, cfg.innerDeg);
  const rOut = radiusPx(fy, tanR, cfg.innerDeg + cfg.blendDeg);
  const half = Math.ceil(rOut) + 2;
  const x0 = Math.max(0, Math.floor(gx - half));
  const y0 = Math.max(0, Math.floor(gyDown - half));
  const x1 = Math.min(width, Math.ceil(gx + half));
  const y1 = Math.min(height, Math.ceil(gyDown + half));
  const w = Math.max(1, x1 - x0);
  const h = Math.max(1, y1 - y0);
  const pw = Math.max(1, Math.round(width * cfg.peripheryScale));
  const ph = Math.max(1, Math.round(height * cfg.peripheryScale));
  return {
    width,
    height,
    periphery: [pw, ph],
    insetRect: [x0, y0, w, h],
    insetOffsetYUp: [x0, height - y0 - h],
    gazePx: [gx, gyDown],
    rInnerPx: rIn,
    rOuterPx: rOut,
  };
}

/** Fraction of full-resolution pixels actually shaded (periphery + inset) — the ideal raster saving. */
export function shadedPixelFraction(l: FoveationLayout): number {
  const full = l.width * l.height;
  return (l.periphery[0] * l.periphery[1] + l.insetRect[2] * l.insetRect[3]) / full;
}
