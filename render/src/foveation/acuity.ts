/**
 * Perceptual acuity falloff used to size the foveal inset.
 *
 * Minimum angle of resolution grows roughly linearly with retinal eccentricity e (degrees):
 *
 *     MAR(e) = m * e + MAR0                         (Weymouth 1958; Guenter et al. 2012, eq. 1)
 *
 * with MAR0 = 1/48 deg (20/20 vision resolves 1 arc-minute per half cycle) and slope m in
 * [0.022, 0.034] deg/deg (Guenter et al. fit m from their user study inside that range). A
 * rendering layer whose pixel pitch is p degrees is "acuity matched" at eccentricities where
 * p <= MAR(e), i.e. beyond
 *
 *     e_min(p) = (p - MAR0) / m.
 *
 * gazefov uses the conservative end m = 0.022, so layers are switched later (larger fovea) than
 * a perceptually tuned system would.
 */
export const MAR0_DEG = 1 / 48;
export const MAR_SLOPE_CONSERVATIVE = 0.022;
export const MAR_SLOPE_AGGRESSIVE = 0.034;

export function marDeg(eccDeg: number, slope = MAR_SLOPE_CONSERVATIVE): number {
  return slope * Math.max(0, eccDeg) + MAR0_DEG;
}

/** Pixels per degree at the centre of a symmetric perspective view. */
export function centralPpd(heightPx: number, fovYRad: number): number {
  const f = heightPx / 2 / Math.tan(fovYRad / 2);
  return (f * Math.PI) / 180;
}

/** Smallest eccentricity (deg) at which a layer of pixel pitch `pitchDeg` is acuity-matched. */
export function minEccentricityForPitch(pitchDeg: number, slope = MAR_SLOPE_CONSERVATIVE): number {
  return Math.max(0, (pitchDeg - MAR0_DEG) / slope);
}

/**
 * Inner foveal radius (deg) for a periphery layer at `scale` (linear, e.g. 1/3) of a display with
 * `ppd` pixels per degree: acuity-matched eccentricity + a margin for gaze-tracking error and
 * end-to-end latency (both in degrees).
 */
export function fovealRadiusDeg(scale: number, ppd: number, trackingErrorDeg: number, latencyMarginDeg: number, slope = MAR_SLOPE_CONSERVATIVE): number {
  const pitch = 1 / (scale * ppd);
  return minEccentricityForPitch(pitch, slope) + trackingErrorDeg + latencyMarginDeg;
}
