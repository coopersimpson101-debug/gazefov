/**
 * FoveatedSplatRenderer — gaze-contingent multi-resolution Gaussian-splat rendering.
 *
 * Per frame (all recorded into the host's command encoder):
 *
 *   1. preprocess  (splatweb's preprocess.wgsl, unmodified)   — once, at the full virtual resolution
 *   2. depth sort  (splatweb's GPU radix sort, unmodified)    — once
 *   3. raster → periphery layer, linear scale s (e.g. 1/3)    — foveated_raster.wgsl
 *   4. raster → foveal inset, full resolution, around gaze    — foveated_raster.wgsl
 *   5. composite: bilinear periphery + inset, smooth radial blend → host target  — composite.wgsl
 *
 * Steps 1–2 are shared by both layers because both see the same camera: the inset is an affine
 * crop of the full-resolution image and the periphery an affine down-scale, so neither the
 * projection nor the depth order changes. Only rasterisation — the dominant cost for splat
 * scenes (overdraw of large translucent splats) — is split by resolution.
 *
 * With `layout === null` the renderer draws one full-resolution layer straight into the target,
 * which is the like-for-like baseline used by the benchmark.
 */
import preprocessWGSL from '../vendor/splatweb/gpu/shaders/preprocess.wgsl?raw';
import rasterWGSL from './shaders/foveated_raster.wgsl?raw';
import compositeWGSL from './shaders/composite.wgsl?raw';
import { invert, multiply, create as mat4Create } from '../vendor/splatweb/core/math/mat4';
import type { Mat4 } from '../vendor/splatweb/core/math/mat4';
import { GpuRadixSorter } from '../vendor/splatweb/gpu/radixSort';
import {
  PREPROCESS_UNIFORM_BYTES, SPLAT2D_BYTES, SPLAT_BYTES, packSH, packSplats, shWordsPerSplat, writePreprocessUniforms,
} from '../vendor/splatweb/gpu/packing';
import type { SplatData } from '../vendor/splatweb/io/splatData';
import type { FoveationLayout } from './layout';

export interface FoveatedView {
  view: Mat4;
  proj: Mat4;
  near?: number;
}

export interface FoveatedTarget {
  colorView: GPUTextureView;
  width: number;
  height: number;
}

export interface FoveatedTimings {
  preprocessMs: number;
  sortMs: number;
  peripheryMs: number;
  insetMs: number;
  compositeMs: number;
  totalMs: number;
  visible: number;
}

const LOWPASS = 0.3; // splatweb's screen-space low-pass (px²), re-applied per layer

export class FoveatedSplatRenderer {
  readonly device: GPUDevice;
  readonly colorFormat: GPUTextureFormat;
  maxShDegree = 3;
  debugBlend = false;
  modelMatrix: Mat4 = mat4Create();

  private readonly preprocessLayout: GPUBindGroupLayout;
  private readonly rasterLayout: GPUBindGroupLayout;
  private readonly compositeLayout: GPUBindGroupLayout;
  private readonly preprocessPipeline: GPUComputePipeline;
  private readonly rasterPipeline: GPURenderPipeline;
  private readonly compositePipeline: GPURenderPipeline;
  private readonly sampler: GPUSampler;
  private readonly preUniforms: GPUBuffer;
  private readonly layerParams: GPUBuffer[];
  private readonly compositeParams: GPUBuffer;
  private readonly drawArgs: GPUBuffer;
  private readonly querySet: GPUQuerySet | null;
  private readonly queryResolve: GPUBuffer | null;
  private readonly readback: GPUBuffer;

  private count = 0;
  private shDegree = 0;
  private splatBuf: GPUBuffer | null = null;
  private shBuf: GPUBuffer | null = null;
  private out2d: GPUBuffer | null = null;
  private sorter: GpuRadixSorter | null = null;
  private preprocessBG: GPUBindGroup | null = null;
  private rasterBGs: GPUBindGroup[] = [];
  private periphery: GPUTexture | null = null;
  private inset: GPUTexture | null = null;
  private compositeBG: GPUBindGroup | null = null;
  private lastMode: 'full' | 'foveated' = 'full';

  constructor(opts: { device: GPUDevice; colorFormat: GPUTextureFormat; timestamps?: boolean }) {
    const d = opts.device;
    this.device = d;
    this.colorFormat = opts.colorFormat;
    const S = GPUShaderStage;
    const U = GPUBufferUsage;
    this.preprocessLayout = d.createBindGroupLayout({
      label: 'gazefov preprocess layout',
      entries: [
        { binding: 0, visibility: S.COMPUTE, buffer: { type: 'uniform' } },
        { binding: 1, visibility: S.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: S.COMPUTE, buffer: { type: 'read-only-storage' } },
        { binding: 3, visibility: S.COMPUTE, buffer: { type: 'storage' } },
        { binding: 4, visibility: S.COMPUTE, buffer: { type: 'storage' } },
        { binding: 5, visibility: S.COMPUTE, buffer: { type: 'storage' } },
        { binding: 6, visibility: S.COMPUTE, buffer: { type: 'storage' } },
      ],
    });
    this.rasterLayout = d.createBindGroupLayout({
      label: 'gazefov raster layout',
      entries: [
        { binding: 0, visibility: S.VERTEX, buffer: { type: 'uniform' } },
        { binding: 1, visibility: S.VERTEX, buffer: { type: 'read-only-storage' } },
        { binding: 2, visibility: S.VERTEX, buffer: { type: 'read-only-storage' } },
      ],
    });
    this.compositeLayout = d.createBindGroupLayout({
      label: 'gazefov composite layout',
      entries: [
        { binding: 0, visibility: S.FRAGMENT, buffer: { type: 'uniform' } },
        { binding: 1, visibility: S.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 2, visibility: S.FRAGMENT, texture: { sampleType: 'float' } },
        { binding: 3, visibility: S.FRAGMENT, sampler: { type: 'filtering' } },
      ],
    });
    this.preprocessPipeline = d.createComputePipeline({
      label: 'splatweb preprocess (shared)',
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.preprocessLayout] }),
      compute: { module: d.createShaderModule({ label: 'preprocess.wgsl', code: preprocessWGSL }), entryPoint: 'main' },
    });
    const rasterModule = d.createShaderModule({ label: 'foveated_raster.wgsl', code: rasterWGSL });
    const over: GPUBlendComponent = { srcFactor: 'one', dstFactor: 'one-minus-src-alpha', operation: 'add' };
    this.rasterPipeline = d.createRenderPipeline({
      label: 'gazefov layer raster',
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.rasterLayout] }),
      vertex: { module: rasterModule, entryPoint: 'vs_main' },
      fragment: { module: rasterModule, entryPoint: 'fs_main', targets: [{ format: this.colorFormat, blend: { color: over, alpha: over } }] },
      primitive: { topology: 'triangle-strip', cullMode: 'none' },
    });
    const compModule = d.createShaderModule({ label: 'composite.wgsl', code: compositeWGSL });
    this.compositePipeline = d.createRenderPipeline({
      label: 'gazefov composite',
      layout: d.createPipelineLayout({ bindGroupLayouts: [this.compositeLayout] }),
      vertex: { module: compModule, entryPoint: 'vs_main' },
      fragment: { module: compModule, entryPoint: 'fs_main', targets: [{ format: this.colorFormat }] },
      primitive: { topology: 'triangle-list' },
    });
    this.sampler = d.createSampler({ magFilter: 'linear', minFilter: 'linear', addressModeU: 'clamp-to-edge', addressModeV: 'clamp-to-edge' });
    this.preUniforms = d.createBuffer({ label: 'preprocess uniforms', size: PREPROCESS_UNIFORM_BYTES, usage: U.UNIFORM | U.COPY_DST });
    this.layerParams = [0, 1].map((i) => d.createBuffer({ label: `layer params ${i}`, size: 32, usage: U.UNIFORM | U.COPY_DST }));
    this.compositeParams = d.createBuffer({ label: 'composite params', size: 48, usage: U.UNIFORM | U.COPY_DST });
    this.drawArgs = d.createBuffer({ label: 'draw args', size: 16, usage: U.STORAGE | U.INDIRECT | U.COPY_DST | U.COPY_SRC });
    d.queue.writeBuffer(this.drawArgs, 0, new Uint32Array([4, 0, 0, 0]));
    const ts = (opts.timestamps ?? true) && d.features.has('timestamp-query');
    this.querySet = ts ? d.createQuerySet({ type: 'timestamp', count: 6 }) : null;
    this.queryResolve = ts ? d.createBuffer({ size: 48, usage: U.QUERY_RESOLVE | U.COPY_SRC }) : null;
    this.readback = d.createBuffer({ label: 'timing readback', size: 64, usage: U.MAP_READ | U.COPY_DST });
  }

  get splatCount(): number {
    return this.count;
  }

  setScene(data: SplatData): void {
    this.releaseScene();
    const d = this.device;
    const U = GPUBufferUsage;
    const n = Math.max(1, data.loaded);
    this.count = data.loaded;
    this.shDegree = data.shDegree;
    this.splatBuf = d.createBuffer({ label: 'splats', size: n * SPLAT_BYTES, usage: U.STORAGE | U.COPY_DST });
    this.shBuf = d.createBuffer({ label: 'splat SH', size: n * shWordsPerSplat(data.shDegree) * 4, usage: U.STORAGE | U.COPY_DST });
    this.out2d = d.createBuffer({ label: 'splats 2D', size: n * SPLAT2D_BYTES, usage: U.STORAGE | U.COPY_SRC });
    d.queue.writeBuffer(this.splatBuf, 0, packSplats(data, 0, data.loaded));
    d.queue.writeBuffer(this.shBuf, 0, packSH(data, 0, data.loaded));
    this.sorter = new GpuRadixSorter(d, n);
    this.sorter.setCount(this.count);
    this.preprocessBG = d.createBindGroup({
      layout: this.preprocessLayout,
      entries: [this.preUniforms, this.splatBuf, this.shBuf, this.out2d, this.sorter.keysA, this.sorter.valsA, this.drawArgs].map(
        (buffer, binding) => ({ binding, resource: { buffer } }),
      ),
    });
    this.rasterBGs = this.layerParams.map((p) =>
      d.createBindGroup({
        layout: this.rasterLayout,
        entries: [p, this.out2d!, this.sorter!.valsA].map((buffer, binding) => ({ binding, resource: { buffer } })),
      }),
    );
  }

  private ensureLayers(l: FoveationLayout): void {
    const d = this.device;
    const usage = GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_SRC;
    let changed = false;
    if (!this.periphery || this.periphery.width !== l.periphery[0] || this.periphery.height !== l.periphery[1]) {
      this.periphery?.destroy();
      this.periphery = d.createTexture({ label: 'periphery layer', size: l.periphery, format: this.colorFormat, usage });
      changed = true;
    }
    // The inset texture is sized for the unclamped square so it is not re-allocated as gaze moves.
    const side = Math.max(1, Math.min(Math.max(l.width, l.height), 2 * Math.ceil(l.rOuterPx) + 4));
    const need: [number, number] = [Math.max(side, l.insetRect[2]), Math.max(side, l.insetRect[3])];
    if (!this.inset || this.inset.width < need[0] || this.inset.height < need[1]) {
      this.inset?.destroy();
      this.inset = d.createTexture({ label: 'foveal inset layer', size: need, format: this.colorFormat, usage });
      changed = true;
    }
    if (changed || !this.compositeBG) {
      this.compositeBG = d.createBindGroup({
        layout: this.compositeLayout,
        entries: [
          { binding: 0, resource: { buffer: this.compositeParams } },
          { binding: 1, resource: this.periphery.createView() },
          { binding: 2, resource: this.inset.createView() },
          { binding: 3, resource: this.sampler },
        ],
      });
    }
  }

  /** Layer textures (for tests / figures). */
  get layers(): { periphery: GPUTexture | null; inset: GPUTexture | null } {
    return { periphery: this.periphery, inset: this.inset };
  }

  private writeLayer(i: number, viewport: [number, number], scale: [number, number], offset: [number, number], lowpass: number) {
    this.device.queue.writeBuffer(this.layerParams[i], 0, new Float32Array([viewport[0], viewport[1], scale[0], scale[1], offset[0], offset[1], lowpass, 0]));
  }

  private rasterPass(enc: GPUCommandEncoder, view: GPUTextureView, vp: [number, number, number, number], bg: number, tsIndex: number | null) {
    const rp = enc.beginRenderPass({
      label: 'gazefov layer',
      colorAttachments: [{ view, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
      timestampWrites: tsIndex !== null && this.querySet ? { querySet: this.querySet, endOfPassWriteIndex: tsIndex } : undefined,
    });
    rp.setViewport(vp[0], vp[1], vp[2], vp[3], 0, 1);
    rp.setScissorRect(vp[0], vp[1], vp[2], vp[3]);
    rp.setPipeline(this.rasterPipeline);
    rp.setBindGroup(0, this.rasterBGs[bg]);
    rp.drawIndirect(this.drawArgs, 0);
    rp.end();
  }

  /**
   * Record one frame. `layout === null` renders a single full-resolution layer into the target
   * (baseline); otherwise periphery + inset + composite.
   */
  encode(enc: GPUCommandEncoder, target: FoveatedTarget, view: FoveatedView, layout: FoveationLayout | null): void {
    if (!this.preprocessBG || !this.sorter || this.count === 0) return;
    const d = this.device;
    const W = target.width;
    const H = target.height;
    const modelView = multiply(view.view, this.modelMatrix);
    const inv = invert(modelView) ?? mat4Create();
    d.queue.writeBuffer(
      this.preUniforms, 0,
      writePreprocessUniforms({
        view: modelView, proj: view.proj, camPos: [inv[12], inv[13], inv[14]], viewport: [W, H], splatCount: this.count,
        shDegree: Math.min(this.maxShDegree, this.shDegree), shStride: shWordsPerSplat(this.shDegree), frontToBack: false,
        scaleModifier: 1, near: view.near ?? 0.05,
      }),
    );
    const qs = this.querySet;
    enc.clearBuffer(this.drawArgs, 4, 4);
    const pre = enc.beginComputePass({
      label: 'splat preprocess (shared)',
      timestampWrites: qs ? { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 } : undefined,
    });
    pre.setPipeline(this.preprocessPipeline);
    pre.setBindGroup(0, this.preprocessBG);
    const wgs = Math.ceil(this.count / 256);
    pre.dispatchWorkgroups(Math.min(wgs, 65535), Math.ceil(wgs / 65535));
    pre.end();
    const sp = enc.beginComputePass({ label: 'splat sort (shared)', timestampWrites: qs ? { querySet: qs, endOfPassWriteIndex: 2 } : undefined });
    this.sorter.encode(sp);
    sp.end();

    if (layout === null) {
      this.lastMode = 'full';
      this.writeLayer(0, [W, H], [1, 1], [0, 0], 0);
      this.rasterPass(enc, target.colorView, [0, 0, W, H], 0, 3);
    } else {
      this.lastMode = 'foveated';
      this.ensureLayers(layout);
      const [pw, ph] = layout.periphery;
      const sx = pw / W;
      const sy = ph / H;
      this.writeLayer(0, [pw, ph], [sx, sy], [0, 0], LOWPASS * (1 - sx * sy));
      const [, , iw, ih] = layout.insetRect;
      this.writeLayer(1, [iw, ih], [1, 1], layout.insetOffsetYUp, 0);
      this.rasterPass(enc, this.periphery!.createView(), [0, 0, pw, ph], 0, 3);
      this.rasterPass(enc, this.inset!.createView(), [0, 0, iw, ih], 1, 4);
      d.queue.writeBuffer(
        this.compositeParams, 0,
        new Float32Array([
          layout.gazePx[0], layout.gazePx[1], layout.rInnerPx, layout.rOuterPx, layout.insetRect[0], layout.insetRect[1], iw, ih, W, H,
          this.debugBlend ? 1 : 0, 0,
        ]),
      );
      const cp = enc.beginRenderPass({
        label: 'gazefov composite',
        colorAttachments: [{ view: target.colorView, loadOp: 'clear', clearValue: { r: 0, g: 0, b: 0, a: 0 }, storeOp: 'store' }],
        timestampWrites: qs ? { querySet: qs, endOfPassWriteIndex: 5 } : undefined,
      });
      cp.setPipeline(this.compositePipeline);
      cp.setBindGroup(0, this.compositeBG!);
      cp.draw(3);
      cp.end();
    }
    if (qs && this.queryResolve) {
      enc.resolveQuerySet(qs, 0, 6, this.queryResolve, 0);
      enc.copyBufferToBuffer(this.queryResolve, 0, this.readback, 0, 48);
    }
    enc.copyBufferToBuffer(this.drawArgs, 0, this.readback, 48, 16);
  }

  /** Await after the submit that contained encode(): GPU timings of that frame. */
  async readTimings(): Promise<FoveatedTimings> {
    await this.readback.mapAsync(GPUMapMode.READ);
    const buf = this.readback.getMappedRange();
    const ts = new BigUint64Array(buf.slice(0, 48));
    const visible = new Uint32Array(buf.slice(48, 64))[1];
    this.readback.unmap();
    const ms = (a: bigint, b: bigint) => (b > a ? Number(b - a) / 1e6 : 0);
    const full = this.lastMode === 'full';
    const end = full ? ts[3] : ts[5];
    return {
      preprocessMs: ms(ts[0], ts[1]),
      sortMs: ms(ts[1], ts[2]),
      peripheryMs: ms(ts[2], ts[3]),
      insetMs: full ? 0 : ms(ts[3], ts[4]),
      compositeMs: full ? 0 : ms(ts[4], ts[5]),
      totalMs: ms(ts[0], end),
      visible,
    };
  }

  private releaseScene() {
    for (const b of [this.splatBuf, this.shBuf, this.out2d]) b?.destroy();
    this.sorter?.destroy();
    this.splatBuf = this.shBuf = this.out2d = null;
    this.sorter = null;
    this.preprocessBG = null;
    this.rasterBGs = [];
    this.count = 0;
  }

  destroy(): void {
    this.releaseScene();
    this.periphery?.destroy();
    this.inset?.destroy();
    for (const b of [this.preUniforms, ...this.layerParams, this.compositeParams, this.drawArgs, this.readback, this.queryResolve]) b?.destroy();
    this.querySet?.destroy();
  }
}
