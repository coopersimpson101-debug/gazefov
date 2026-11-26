/**
 * Gaze smoothing for foveation.
 *
 * The 1€ filter (Casiez, Roussel & Vogel, CHI 2012) is a first-order low-pass whose cutoff rises
 * with speed: heavy smoothing during fixations (kills tracker jitter that would make the inset
 * shimmer) and almost no lag during saccades. A simple velocity threshold (I-VT, Salvucci &
 * Goldberg 2000) additionally flags saccades so the host can widen the fovea while the eye is
 * moving (saccadic suppression hides the brief quality drop).
 */
export class OneEuro {
  private x: number | null = null;
  private dx = 0;
  private t: number | null = null;

  constructor(public minCutoff = 1.0, public beta = 0.02, public dCutoff = 1.0) {}

  private static alpha(cutoff: number, dt: number): number {
    const tau = 1 / (2 * Math.PI * cutoff);
    return 1 / (1 + tau / dt);
  }

  filter(value: number, tSec: number): number {
    if (this.x === null || this.t === null || tSec <= this.t) {
      this.x = value;
      this.t = tSec;
      return value;
    }
    const dt = tSec - this.t;
    this.t = tSec;
    const rawDx = (value - this.x) / dt;
    this.dx += OneEuro.alpha(this.dCutoff, dt) * (rawDx - this.dx);
    const cutoff = this.minCutoff + this.beta * Math.abs(this.dx);
    this.x += OneEuro.alpha(cutoff, dt) * (value - this.x);
    return this.x;
  }

  get velocity(): number {
    return this.dx;
  }
}

export interface FilteredGaze {
  yawDeg: number;
  pitchDeg: number;
  speedDegPerS: number;
  saccade: boolean;
}

export class GazeFilter {
  private readonly fy = new OneEuro(1.0, 0.02);
  private readonly fp = new OneEuro(1.0, 0.02);
  private lastRaw: [number, number, number] | null = null;

  constructor(public saccadeThresholdDegPerS = 100) {}

  push(yawDeg: number, pitchDeg: number, tSec: number): FilteredGaze {
    let speed = 0;
    if (this.lastRaw && tSec > this.lastRaw[2]) {
      speed = Math.hypot(yawDeg - this.lastRaw[0], pitchDeg - this.lastRaw[1]) / (tSec - this.lastRaw[2]);
    }
    this.lastRaw = [yawDeg, pitchDeg, tSec];
    return { yawDeg: this.fy.filter(yawDeg, tSec), pitchDeg: this.fp.filter(pitchDeg, tSec), speedDegPerS: speed, saccade: speed > this.saccadeThresholdDegPerS };
  }
}
