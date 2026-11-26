/**
 * Browser demo: the splatweb procedural scene rendered foveated around the pointer (or a live
 * GZFV WebSocket stream with ?ws=ws://127.0.0.1:8766). F toggles full resolution, B tints the blend band.
 */
import { FoveatedSplatRenderer } from '../foveation/renderer';
import { computeLayout } from '../foveation/layout';
import { PointerGazeSource, WebSocketGazeSource } from '../gaze/source';
import type { GazeSource } from '../gaze/source';
import { OrbitCamera, projectionMatrix } from '../vendor/splatweb/camera/camera';
import { generateDemoScene } from '../vendor/splatweb/io/procedural';

async function main() {
  const canvas = document.getElementById('c') as HTMLCanvasElement;
  const stats = document.getElementById('stats')!;
  const modeEl = document.getElementById('mode')!;
  if (!navigator.gpu) {
    stats.textContent = 'WebGPU is not available in this browser';
    return;
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: 'high-performance' });
  if (!adapter) throw new Error('no adapter');
  const device = await adapter.requestDevice({ requiredLimits: { maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize } });
  const ctx = canvas.getContext('webgpu')!;
  const format: GPUTextureFormat = 'rgba8unorm';
  ctx.configure({ device, format, alphaMode: 'premultiplied' });
  const params = new URLSearchParams(location.search);
  const renderer = new FoveatedSplatRenderer({ device, colorFormat: format, timestamps: false });
  renderer.setScene(generateDemoScene({ count: parseInt(params.get('count') ?? '500000', 10) }));
  const ws = params.get('ws');
  const gaze: GazeSource = ws ? new WebSocketGazeSource(ws) : new PointerGazeSource(canvas);
  const cam = new OrbitCamera();
  cam.target = [0, 1.2, 0];
  cam.distance = 6.2;
  cam.pitch = 0.32;
  let foveated = true;
  addEventListener('keydown', (e) => {
    if (e.key === 'f' || e.key === 'F') foveated = !foveated;
    if (e.key === 'b' || e.key === 'B') renderer.debugBlend = !renderer.debugBlend;
    modeEl.textContent = foveated ? 'foveated' : 'full resolution';
  });
  let last = performance.now();
  let fps = 0;
  const frame = () => {
    const dpr = Math.min(devicePixelRatio, 2);
    const w = Math.max(1, Math.floor(canvas.clientWidth * dpr));
    const h = Math.max(1, Math.floor(canvas.clientHeight * dpr));
    if (canvas.width !== w || canvas.height !== h) {
      canvas.width = w;
      canvas.height = h;
    }
    cam.yaw += 0.002;
    const g = gaze.latest() ?? { ndc: [0, 0] as [number, number], valid: true };
    const layout = foveated && g.valid ? computeLayout(w, h, cam.lens.fovY, { ndc: g.ndc }, { peripheryScale: 1 / 3, innerDeg: 8, blendDeg: 5 }) : null;
    const enc = device.createCommandEncoder();
    renderer.encode(enc, { colorView: ctx.getCurrentTexture().createView(), width: w, height: h },
      { view: cam.viewMatrix(), proj: projectionMatrix(cam.lens, w / h), near: cam.lens.near }, layout);
    device.queue.submit([enc.finish()]);
    const now = performance.now();
    fps = 0.9 * fps + 0.1 * (1000 / (now - last));
    last = now;
    stats.textContent = `${w}x${h} · ${fps.toFixed(0)} fps (browser, not a benchmark)`;
    requestAnimationFrame(frame);
  };
  requestAnimationFrame(frame);
}

main().catch((e) => {
  document.getElementById('stats')!.textContent = String(e);
  console.error(e);
});
