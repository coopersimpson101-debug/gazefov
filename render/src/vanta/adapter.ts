/**
 * Vanta adapter interface for foveated splat rendering.
 *
 * Vanta's API is not public and not part of this repository, so this file defines the *contract*
 * gazefov needs from a host engine, plus a host-agnostic driver (`FoveatedFrameDriver`) that
 * implements foveation on top of it. Whoever integrates Vanta implements `FoveationHost` (a thin
 * mapping of Vanta's device, per-eye views, render targets and frame timing) — see
 * docs/vanta-adapter.md. Nothing here guesses at Vanta's real types.
 */
import { computeLayout } from '../foveation/layout';
import type { FoveationConfig, FoveationLayout } from '../foveation/layout';
import { FoveatedSplatRenderer } from '../foveation/renderer';
import type { GazeSample, GazeSource } from '../gaze/source';
import type { Mat4 } from '../vendor/splatweb/core/math/mat4';
import type { SplatData } from '../vendor/splatweb/io/splatData';

/** One eye (or one mono view) of the host's frame. */
export interface HostEyeView {
  eye: 'left' | 'right' | 'mono';
  /** World -> view, column-major, right-handed, camera looks down -Z (WebXR/OpenGL convention). */
  view: Mat4;
  /** View -> clip, column-major, WebGPU depth [0, 1], may be asymmetric (XR). */
  proj: Mat4;
  /** Vertical field of view in radians for this eye (used to convert foveal degrees to pixels). */
  fovY: number;
  /** Colour target for this eye; must be RENDER_ATTACHMENT with format `colorFormat`. */
  colorView: GPUTextureView;
  width: number;
  height: number;
  near: number;
}

/** What gazefov needs from Vanta (or any engine). */
export interface FoveationHost {
  readonly device: GPUDevice;
  readonly colorFormat: GPUTextureFormat;
  /** Per-eye views for the frame being built. */
  eyeViews(): HostEyeView[];
  /** The host's command encoder for this frame (gazefov records passes; the host submits). */
  commandEncoder(): GPUCommandEncoder;
  /** Predicted display time of this frame (ms, performance.now() clock) for gaze prediction. */
  predictedDisplayTimeMs(): number;
}

export interface FoveationPolicy extends FoveationConfig {
  /** Fall back to full resolution when gaze is older than this (ms) or invalid. */
  staleAfterMs: number;
  /** Extra foveal radius (deg) while a saccade is flagged. */
  saccadeExtraDeg: number;
}

export const DEFAULT_POLICY: FoveationPolicy = { peripheryScale: 1 / 3, innerDeg: 10, blendDeg: 5, staleAfterMs: 100, saccadeExtraDeg: 5 };

/**
 * Host-agnostic per-frame driver: one FoveatedSplatRenderer per eye, gaze -> layout, fallbacks.
 * Binocular trackers send one combined gaze NDC; per-eye NDC differences (vergence) are below the
 * blend band for display distances > ~0.5 m and are ignored here.
 */
export class FoveatedFrameDriver {
  private readonly renderers = new Map<string, FoveatedSplatRenderer>();
  lastLayouts: (FoveationLayout | null)[] = [];

  constructor(private readonly host: FoveationHost, private readonly gaze: GazeSource, public policy: FoveationPolicy = DEFAULT_POLICY) {}

  setScene(data: SplatData): void {
    for (const r of this.renderers.values()) r.setScene(data);
    this.pending = data;
  }

  private pending: SplatData | null = null;

  private rendererFor(eye: string): FoveatedSplatRenderer {
    let r = this.renderers.get(eye);
    if (!r) {
      r = new FoveatedSplatRenderer({ device: this.host.device, colorFormat: this.host.colorFormat, timestamps: false });
      if (this.pending) r.setScene(this.pending);
      this.renderers.set(eye, r);
    }
    return r;
  }

  /** Decide the layout for one eye from the latest gaze sample (null = full resolution). */
  layoutFor(v: HostEyeView, g: GazeSample | null, nowMs: number): FoveationLayout | null {
    if (!g || !g.valid || nowMs - g.tMs > this.policy.staleAfterMs) return null;
    const extra = g.saccade ? this.policy.saccadeExtraDeg : 0;
    return computeLayout(v.width, v.height, v.fovY, { ndc: g.ndc }, {
      peripheryScale: this.policy.peripheryScale,
      innerDeg: this.policy.innerDeg + extra + Math.max(0, g.errorDeg),
      blendDeg: this.policy.blendDeg,
    });
  }

  encodeFrame(): void {
    const enc = this.host.commandEncoder();
    const g = this.gaze.latest();
    const now = this.host.predictedDisplayTimeMs();
    this.lastLayouts = this.host.eyeViews().map((v) => {
      const layout = this.layoutFor(v, g, now);
      this.rendererFor(v.eye).encode(enc, { colorView: v.colorView, width: v.width, height: v.height }, { view: v.view, proj: v.proj, near: v.near }, layout);
      return layout;
    });
  }

  destroy(): void {
    for (const r of this.renderers.values()) r.destroy();
    this.renderers.clear();
  }
}
