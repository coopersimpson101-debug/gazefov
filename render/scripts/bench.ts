/**
 * Headless foveated-rendering benchmark + quality measurement + hero frame, on Dawn (the
 * `webgpu` npm package) — no browser needed.
 *
 *   npm run bench -- [--width 2160 --height 2160 --fov 90] [--counts 1000000] [--frames 120]
 *                    [--json ../results/foveation_bench.json]
 *   npm run hero                      # writes ../docs/assets/foveated_frame.png
 *
 * Timing = GPU timestamp queries around the whole frame (shared preprocess + sort, periphery
 * raster, inset raster, composite), median over `frames` frames of an orbiting camera after 10
 * warm-up frames. It measures GPU work only — not a browser or compositor frame time.
 */
import { mkdirSync, writeFileSync } from 'node:fs';
import { dirname } from 'node:path';
import { create, globals } from 'webgpu';
import { OrbitCamera, projectionMatrix } from '../src/vendor/splatweb/camera/camera';
import { generateDemoScene } from '../src/vendor/splatweb/io/procedural';
import { centralPpd, fovealRadiusDeg, minEccentricityForPitch } from '../src/foveation/acuity';
import { computeLayout, shadedPixelFraction } from '../src/foveation/layout';
import type { FoveationLayout } from '../src/foveation/layout';
import { luma, regionalPsnr, regionalSsim } from '../src/foveation/metrics';
import { drawCircle, drawCross, encodePng, overBackground } from './png';

Object.assign(globalThis, globals);

const args = process.argv.slice(2);
const flag = (n: string) => args.includes(`--${n}`);
const opt = (n: string, d: string) => {
  const i = args.indexOf(`--${n}`);
  return i >= 0 && i + 1 < args.length ? args[i + 1] : d;
};

const median = (v: number[]) => {
  const s = [...v].sort((a, b) => a - b);
  return s.length ? s[Math.floor(s.length / 2)] : NaN;
};

async function main() {
  const gpu = create([]);
  const adapter = await gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('No WebGPU adapter (Dawn) available');
  if (!adapter.features.has('timestamp-query')) throw new Error('timestamp-query not supported: cannot time GPU passes');
  const device = await adapter.requestDevice({
    requiredFeatures: ['timestamp-query'],
    requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize, maxBufferSize: adapter.limits.maxBufferSize },
  });
  const { FoveatedSplatRenderer } = await import('../src/foveation/renderer');
  const { SplatRenderer } = await import('../src/vendor/splatweb/gpu/renderer');
  const info = adapter.info;
  const gpuName = `${info.vendor} ${info.architecture} ${info.description}`.trim();
  const format: GPUTextureFormat = 'rgba8unorm';

  const readTexture = async (tex: GPUTexture, w: number, h: number): Promise<Uint8Array> => {
    const bpr = Math.ceil((w * 4) / 256) * 256;
    const buf = device.createBuffer({ size: bpr * h, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    const enc = device.createCommandEncoder();
    enc.copyTextureToBuffer({ texture: tex }, { buffer: buf, bytesPerRow: bpr }, [w, h]);
    device.queue.submit([enc.finish()]);
    await buf.mapAsync(GPUMapMode.READ);
    const src = new Uint8Array(buf.getMappedRange());
    const out = new Uint8Array(w * h * 4);
    for (let y = 0; y < h; y++) out.set(src.subarray(y * bpr, y * bpr + w * 4), y * w * 4);
    buf.unmap();
    buf.destroy();
    return out;
  };

  const makeCam = (fovDeg: number) => {
    const cam = new OrbitCamera();
    cam.target = [0, 1.2, 0];
    cam.distance = 6.2;
    cam.yaw = 0.6;
    cam.pitch = 0.32;
    cam.lens.fovY = (fovDeg * Math.PI) / 180;
    return cam;
  };

  // ------------------------------------------------------------------------------------ hero
  const heroOut = opt('hero', '');
  if (heroOut) {
    const W = parseInt(opt('hero-width', '1280'), 10);
    const H = parseInt(opt('hero-height', '800'), 10);
    const fov = parseFloat(opt('hero-fov', '70'));
    const scene = generateDemoScene({ count: parseInt(opt('hero-count', '1000000'), 10) });
    const r = new FoveatedSplatRenderer({ device, colorFormat: format });
    r.setScene(scene);
    const cam = makeCam(fov);
    const tex = device.createTexture({ size: [W, H], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const gaze: [number, number] = [parseFloat(opt('gaze-x', '0.22')), parseFloat(opt('gaze-y', '0.05'))];
    const cfg = { peripheryScale: 1 / 3, innerDeg: 8, blendDeg: 5 };
    const layout = computeLayout(W, H, cam.lens.fovY, { ndc: gaze }, cfg);
    const v = { view: cam.viewMatrix(), proj: projectionMatrix(cam.lens, W / H), near: cam.lens.near };
    const enc = device.createCommandEncoder();
    r.encode(enc, { colorView: tex.createView(), width: W, height: H }, v, layout);
    device.queue.submit([enc.finish()]);
    const t = await r.readTimings();
    const img = overBackground(await readTexture(tex, W, H), [13, 17, 23]);
    const accent: [number, number, number] = [103, 232, 249];
    const [gx, gy] = layout.gazePx;
    drawCircle(img, W, H, gx, gy, layout.rInnerPx, accent, 0.9, 6, 2);
    drawCircle(img, W, H, gx, gy, layout.rOuterPx, accent, 0.6, 3, 1);
    drawCross(img, W, H, gx, gy, 14, 5, accent, 2);
    mkdirSync(dirname(heroOut) || '.', { recursive: true });
    writeFileSync(heroOut, encodePng(W, H, img));
    const meta = {
      file: heroOut, width: W, height: H, fovYDeg: fov, splats: scene.loaded, gazeNdc: gaze, config: cfg,
      layout: { periphery: layout.periphery, insetRect: layout.insetRect, rInnerPx: layout.rInnerPx, rOuterPx: layout.rOuterPx },
      gpuMsThisFrame: t.totalMs, gpu: gpuName, date: new Date().toISOString().slice(0, 10),
    };
    writeFileSync(heroOut.replace(/\.png$/, '.json'), JSON.stringify(meta, null, 1));
    console.log(`wrote ${heroOut}`, JSON.stringify(meta));
    r.destroy();
  }

  // ------------------------------------------------------------------------------- benchmark
  const frames = parseInt(opt('frames', '120'), 10);
  if (frames > 0) {
    const W = parseInt(opt('width', '2160'), 10);
    const H = parseInt(opt('height', '2160'), 10);
    const fov = parseFloat(opt('fov', '90'));
    const counts = opt('counts', '1000000').split(',').map((s) => parseInt(s, 10));
    const scales = opt('scales', '0.5,0.3333,0.25').split(',').map(parseFloat);
    const inners = opt('inner', '5,10,15,20').split(',').map(parseFloat);
    const blendDeg = parseFloat(opt('blend', '5'));
    const trackErr = parseFloat(opt('track-err', '1.0'));
    const latencyDeg = parseFloat(opt('latency-deg', '2.0'));
    const gazes: [number, number][] = opt('gazes', '0,0;0.45,0.25').split(';').map((g) => g.split(',').map(parseFloat) as [number, number]);
    const ppd = centralPpd(H, (fov * Math.PI) / 180);
    const tex = device.createTexture({ size: [W, H], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const refTex = device.createTexture({ size: [W, H], format, usage: GPUTextureUsage.RENDER_ATTACHMENT | GPUTextureUsage.COPY_SRC });
    const results: Record<string, unknown>[] = [];
    const jsonOut = opt('json', '');
    const save = () => {
      if (!jsonOut) return;
      mkdirSync(dirname(jsonOut) || '.', { recursive: true });
      writeFileSync(jsonOut, JSON.stringify({
        meta: { gpu: gpuName, backend: 'Dawn via the webgpu npm package', width: W, height: H, fovYDeg: fov, ppd, frames, date: new Date().toISOString().slice(0, 10),
          scene: 'splatweb procedural demo scene, SH degree 3', trackingErrorDeg: trackErr, latencyMarginDeg: latencyDeg,
          note: 'GPU timestamp queries; full-res and foveated interleaved frame by frame; saving = 1 - median per-frame ratio over frames whose full-res time is within 25 % of the run minimum (keptFrames/allFrames); GPU shared with other processes' },
        results,
      }, null, 1));
    };
    console.log(`GPU: ${gpuName}\n${W}x${H}, vertical FOV ${fov} deg, ${ppd.toFixed(1)} px/deg at centre, median of ${frames} frames`);

    for (const n of counts) {
      const scene = generateDemoScene({ count: n });
      const fr = new FoveatedSplatRenderer({ device, colorFormat: format });
      fr.setScene(scene);
      const view = (cam: OrbitCamera) => ({ view: cam.viewMatrix(), proj: projectionMatrix(cam.lens, W / H), near: cam.lens.near });

      const once = async (cam: OrbitCamera, layout: FoveationLayout | null) => {
        const enc = device.createCommandEncoder();
        fr.encode(enc, { colorView: tex.createView(), width: W, height: H }, view(cam), layout);
        device.queue.submit([enc.finish()]);
        const t = await fr.readTimings();
        return [t.totalMs, t.preprocessMs, t.sortMs, t.peripheryMs, t.insetMs, t.compositeMs, t.visible];
      };
      const summarise = (samples: number[][]) => {
        const col = (k: number) => median(samples.map((s) => s[k]));
        return { total: col(0), preprocess: col(1), sort: col(2), periphery: col(3), inset: col(4), composite: col(5), visible: col(6) };
      };
      // Interleaved A/B timing: every frame renders the full-resolution baseline and the foveated
      // configuration back to back with the same camera, so load from other GPU clients hits both
      // alike; the saving is the median of the per-frame ratios.
      const timeConfig = async (layout: FoveationLayout | null) => {
        const cam = makeCam(fov);
        const a: number[][] = [];
        const b: number[][] = [];
        const ratio: number[] = [];
        for (let f = 0; f < frames + 10; f++) {
          cam.yaw += 0.01;
          const ta = await once(cam, null);
          const tb = layout ? await once(cam, layout) : ta;
          if (f >= 10) {
            a.push(ta);
            b.push(tb);
            ratio.push(tb[0] / ta[0]);
          }
        }
        // The GPU is shared with other clients: frames whose full-res time is > 25 % above the run's
        // minimum were slowed by someone else and are dropped (pairwise, so A and B stay matched).
        const minA = Math.min(...a.map((x) => x[0]));
        const keep = a.map((x) => x[0] <= 1.25 * minA);
        const ka = a.filter((_, i) => keep[i]);
        const kb = b.filter((_, i) => keep[i]);
        const kr = ratio.filter((_, i) => keep[i]);
        return { ...summarise(kb), fullTotal: summarise(ka).total, savingMedian: 1 - median(kr), keptFrames: kr.length,
          allFrames: ratio.length, savingAllFrames: 1 - median(ratio) };
      };

      // Baseline 1: splatweb's own renderer, unmodified (its per-stage timestamps).
      {
        const sr = new SplatRenderer({ device, colorFormat: format });
        sr.setScene(scene);
        const cam = makeCam(fov);
        const tot: number[] = [];
        for (let f = 0; f < frames + 10; f++) {
          cam.yaw += 0.01;
          sr.requestStats();
          sr.render({ colorView: tex.createView(), clear: { r: 0, g: 0, b: 0, a: 0 } }, { ...view(cam), viewport: [W, H] });
          await device.queue.onSubmittedWorkDone();
          await new Promise((res) => setTimeout(res, 1));
          const s = sr.stats;
          if (f >= 10 && s.gpuPreprocessMs !== undefined) tot.push(s.gpuPreprocessMs + (s.gpuSortMs ?? 0) + (s.gpuRenderMs ?? 0));
        }
        results.push({ splats: n, mode: 'splatweb SplatRenderer (unmodified)', totalMs: median(tot) });
        console.log(`\n${n.toLocaleString('en-US')} splats — splatweb baseline: ${median(tot).toFixed(2)} ms`);
        sr.destroy();
      }
      // Baseline 2: this renderer, one full-resolution layer.
      const base = await timeConfig(null);
      results.push({ splats: n, mode: 'full', ...base });
      console.log(`full-res (gazefov, 1 layer): ${base.total.toFixed(2)} ms (preprocess ${base.preprocess.toFixed(2)}, sort ${base.sort.toFixed(2)}, raster ${base.periphery.toFixed(2)})`);
      console.log('| gaze ndc | periphery scale | inner deg | blend deg | shaded px | GPU ms | full ms (same frames) | saving | periph ms | inset ms | comp ms | PSNR fovea | PSNR blend | PSNR periph | SSIM periph |');
      console.log('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|');

      // Reference frame for quality, static camera.
      const qcam = makeCam(fov);
      qcam.yaw += 0.3;
      {
        const enc = device.createCommandEncoder();
        fr.encode(enc, { colorView: refTex.createView(), width: W, height: H }, view(qcam), null);
        device.queue.submit([enc.finish()]);
        await fr.readTimings();
      }
      const refL = luma(await readTexture(refTex, W, H), W, H);

      const configs: { s: number; inner: number; tag: string }[] = [];
      for (const s of scales) for (const inner of inners) configs.push({ s, inner, tag: 'grid' });
      for (const s of scales) configs.push({ s, inner: Math.round(10 * fovealRadiusDeg(s, ppd, trackErr, latencyDeg)) / 10, tag: 'acuity' });

      for (const g of gazes)
        for (const c of configs) {
          const layout = computeLayout(W, H, (fov * Math.PI) / 180, { ndc: g }, { peripheryScale: c.s, innerDeg: c.inner, blendDeg });
          const t = await timeConfig(layout);
          const enc = device.createCommandEncoder();
          fr.encode(enc, { colorView: tex.createView(), width: W, height: H }, view(qcam), layout);
          device.queue.submit([enc.finish()]);
          await fr.readTimings();
          const testL = luma(await readTexture(tex, W, H), W, H);
          const regions = [
            { name: 'fovea', r0: 0, r1: layout.rInnerPx },
            { name: 'blend', r0: layout.rInnerPx, r1: layout.rOuterPx },
            { name: 'periphery', r0: layout.rOuterPx, r1: Infinity },
          ];
          const psnr = regionalPsnr(refL, testL, W, H, layout.gazePx, regions);
          const ssim = regionalSsim(refL, testL, W, H, layout.gazePx, regions);
          const shaded = shadedPixelFraction(layout);
          const saving = t.savingMedian;
          const fmt = (p: number) => (Number.isFinite(p) ? p.toFixed(1) : 'inf');
          console.log(
            `| ${g.join(',')} | ${c.s.toFixed(3)} | ${c.inner}${c.tag === 'acuity' ? ' (acuity)' : ''} | ${blendDeg} | ${(100 * shaded).toFixed(1)} % | ${t.total.toFixed(2)} | ${t.fullTotal.toFixed(2)} | ${(100 * saving).toFixed(1)} % (${t.keptFrames}/${t.allFrames}) | ` +
              `${t.periphery.toFixed(2)} | ${t.inset.toFixed(2)} | ${t.composite.toFixed(2)} | ${fmt(psnr[0].psnr)} | ${fmt(psnr[1].psnr)} | ${fmt(psnr[2].psnr)} | ${ssim[2].ssim.toFixed(3)} |`,
          );
          results.push({
            splats: n, mode: 'foveated', tag: c.tag, gazeNdc: g, peripheryScale: c.s, innerDeg: c.inner, blendDeg,
            minEccForScaleDeg: minEccentricityForPitch(1 / (c.s * ppd)), shadedFraction: shaded, ...t, saving,
            psnr: Object.fromEntries(psnr.map((p) => [p.region, Number.isFinite(p.psnr) ? p.psnr : 'inf'])),
            ssim: Object.fromEntries(ssim.map((p) => [p.region, p.ssim])),
            layout: { periphery: layout.periphery, insetRect: layout.insetRect, rInnerPx: layout.rInnerPx, rOuterPx: layout.rOuterPx },
          });
          save(); // incremental: a lost device keeps everything measured so far
        }
      fr.destroy();
    }
    save();
  }
  void flag;
  void gpu;
  device.destroy();
  process.exit(0);
}

try {
  await main();
} catch (e) {
  console.error(e);
  process.exit(1);
}
