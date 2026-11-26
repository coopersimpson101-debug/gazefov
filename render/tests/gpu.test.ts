/**
 * GPU tests on Dawn (the `webgpu` npm package). Skipped — and reported as skipped — when no
 * adapter is available (e.g. hosted CI runners without a GPU).
 */
import { afterAll, beforeAll, describe, expect, it } from 'vitest';
import { OrbitCamera, projectionMatrix } from '../src/vendor/splatweb/camera/camera';
import { generateDemoScene } from '../src/vendor/splatweb/io/procedural';
import { computeLayout } from '../src/foveation/layout';
import { luma, regionalPsnr } from '../src/foveation/metrics';

let device: GPUDevice | null = null;
let keep: unknown = null;
try {
  const wg = await import('webgpu');
  Object.assign(globalThis, wg.globals);
  const gpu = wg.create([]);
  keep = gpu;
  const adapter = await gpu.requestAdapter();
  if (adapter) {
    device = await adapter.requestDevice({ requiredFeatures: adapter.features.has('timestamp-query') ? ['timestamp-query'] : [] });
  }
} catch {
  device = null;
}

const W = 320;
const H = 240;
const fmt: GPUTextureFormat = 'rgba8unorm';

describe.skipIf(!device)('foveated renderer on the GPU (Dawn)', () => {
  const scene = generateDemoScene({ count: 60000 });
  const cam = new OrbitCamera();
  cam.target = [0, 1.2, 0];
  cam.distance = 6.2;
  cam.yaw = 0.6;
  cam.pitch = 0.32;
  const view = { view: cam.viewMatrix(), proj: projectionMatrix(cam.lens, W / H), near: cam.lens.near };
  let tex: GPUTexture;

  const read = async (): Promise<Uint8Array> => {
    const d = device!;
    const bpr = Math.ceil((W * 4) / 256) * 256;
    const buf = d.createBuffer({ size: bpr * H, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = d.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [W, H]);
    d.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(W * H * 4);
    for (let y = 0; y < H; y++) out.set(src.subarray(y * bpr, y * bpr + W * 4), y * W * 4);
    buf.unmap();
    buf.destroy();
    return out;
  };

  beforeAll(() => {
    tex = device!.createTexture({ size: [W, H], format: fmt, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
  });
  afterAll(() => {
    tex?.destroy();
    void keep;
  });

  const renderFov = async (layout: ReturnType<typeof computeLayout> | null) => {
    const { FoveatedSplatRenderer } = await import('../src/foveation/renderer');
    const r = new FoveatedSplatRenderer({ device: device!, colorFormat: fmt });
    r.setScene(scene);
    const enc = device!.createCommandEncoder();
    r.encode(enc, { colorView: tex.createView(), width: W, height: H }, view, layout);
    device!.queue.submit([enc.finish()]);
    const t = await r.readTimings();
    const img = await read();
    r.destroy();
    return { img, t };
  };

  it('single full-resolution layer reproduces splatweb SplatRenderer', async () => {
    const { SplatRenderer } = await import('../src/vendor/splatweb/gpu/renderer');
    const s = new SplatRenderer({ device: device!, colorFormat: fmt, timestamps: false });
    s.setScene(scene);
    s.render({ colorView: tex.createView(), clear: { r: 0, g: 0, b: 0, a: 0 } }, { ...view, viewport: [W, H] });
    const ref = await read();
    s.destroy();
    const { img, t } = await renderFov(null);
    let maxd = 0;
    for (let i = 0; i < ref.length; i++) maxd = Math.max(maxd, Math.abs(ref[i] - img[i]));
    expect(maxd).toBeLessThanOrEqual(1);
    expect(t.visible).toBeGreaterThan(1000);
  });

  it('fovea matches the full-resolution reference; periphery is close', async () => {
    const ref = luma((await renderFov(null)).img, W, H);
    const layout = computeLayout(W, H, cam.lens.fovY, { ndc: [0.1, 0.05] }, { peripheryScale: 1 / 3, innerDeg: 6, blendDeg: 4 });
    const { img, t } = await renderFov(layout);
    const test = luma(img, W, H);
    const [fovea, , periphery] = regionalPsnr(ref, test, W, H, layout.gazePx, [
      { name: 'fovea', r0: 0, r1: layout.rInnerPx },
      { name: 'blend', r0: layout.rInnerPx, r1: layout.rOuterPx },
      { name: 'periphery', r0: layout.rOuterPx, r1: Infinity },
    ]);
    expect(fovea.psnr).toBeGreaterThan(55);
    expect(periphery.psnr).toBeGreaterThan(22);
    expect(periphery.psnr).toBeLessThan(fovea.psnr);
    if (device!.features.has('timestamp-query')) expect(t.totalMs).toBeGreaterThan(0);
  });

  it('a fovea covering the whole frame equals full resolution', async () => {
    const ref = luma((await renderFov(null)).img, W, H);
    const layout = computeLayout(W, H, cam.lens.fovY, { ndc: [0, 0] }, { peripheryScale: 0.25, innerDeg: 60, blendDeg: 1 });
    const test = luma((await renderFov(layout)).img, W, H);
    const [all] = regionalPsnr(ref, test, W, H, layout.gazePx, [{ name: 'all', r0: 0, r1: Infinity }]);
    expect(all.psnr).toBeGreaterThan(55);
  });
});

describe.skipIf(!!device)('GPU tests', () => {
  it.skip('skipped: no WebGPU adapter (Dawn) on this machine', () => {});
});
