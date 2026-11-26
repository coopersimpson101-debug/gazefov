# Vanta adapter interface

Vanta is the owner's engine; its API is not public and is **not** part of this repository. This
document defines what gazefov's foveated renderer needs from a host engine, and the small adapter
that someone with access to Vanta writes. gazefov targets WebGPU (splatweb's renderer core), so the
contract is expressed in WebGPU objects. Nothing here guesses at Vanta's real types.

Code: `render/src/vanta/adapter.ts` (`FoveationHost`, `HostEyeView`, `FoveationPolicy`,
`FoveatedFrameDriver`).

## The host contract

| # | gazefov needs | type | notes |
|---|---|---|---|
| 1 | GPU device | `GPUDevice` | with `maxStorageBufferBindingSize` ≥ 96 B × splats (SH degree 3) |
| 2 | per-eye view matrix | `Float32Array(16)` column-major | world → view, right-handed, −Z forward |
| 3 | per-eye projection | `Float32Array(16)` | WebGPU depth [0, 1]; asymmetric XR frusta are fine |
| 4 | per-eye vertical FOV | radians | converts foveal degrees to pixels |
| 5 | per-eye colour target | `GPUTextureView`, size, format | `RENDER_ATTACHMENT`; gazefov writes premultiplied colour |
| 6 | command encoder | `GPUCommandEncoder` | gazefov records passes; the host submits |
| 7 | predicted display time | ms, `performance.now()` clock | for gaze staleness / prediction |
| 8 | gaze | `GazeSource` | `WebSocketGazeSource` (GZFV stream) or the engine's own eye-tracking API |

```ts
interface FoveationHost {
  readonly device: GPUDevice;
  readonly colorFormat: GPUTextureFormat;
  eyeViews(): HostEyeView[];           // left/right (or mono)
  commandEncoder(): GPUCommandEncoder; // this frame's encoder
  predictedDisplayTimeMs(): number;
}
```

## What a Vanta adapter looks like

```ts
// vanta-foveation.ts — written by whoever has Vanta's API (pseudo-code, names are placeholders)
class VantaFoveationHost implements FoveationHost {
  constructor(private engine: VantaEngine /* Vanta type */) {}
  get device() { return this.engine./* gpu device */; }
  get colorFormat() { return this.engine./* swap-chain or XR layer format */; }
  eyeViews() { return this.engine./* per-eye cameras */.map((cam) => ({
    eye: cam./* 'left' | 'right' */, view: cam./* view matrix */, proj: cam./* proj matrix */,
    fovY: cam./* vertical fov */, colorView: cam./* eye render target view */,
    width: cam./* px */, height: cam./* px */, near: cam./* near */ })); }
  commandEncoder() { return this.engine./* current frame encoder */; }
  predictedDisplayTimeMs() { return this.engine./* XR predicted display time */; }
}

const driver = new FoveatedFrameDriver(new VantaFoveationHost(engine), new WebSocketGazeSource());
driver.setScene(splats);              // SplatData from splatweb's loaders
engine.onRenderSplats(() => driver.encodeFrame());   // inside Vanta's frame, before submit
```

## Behaviour of `FoveatedFrameDriver`

* One `FoveatedSplatRenderer` per eye (separate uniforms, shared scene upload pattern).
* Layout per eye from the latest gaze: `innerDeg = policy.innerDeg + errorDeg (+ saccadeExtraDeg)`.
* **Fallback to full resolution** when gaze is invalid or older than `staleAfterMs` (default 100 ms)
  — a lost tracker must never leave a blurry image where the user is looking.
* Output is premultiplied and cleared to transparent, so Vanta can composite it over or under its
  own passes. Depth testing against Vanta geometry is not wired in the foveated path yet (splatweb's
  baseline renderer supports a read-only host depth buffer); see the roadmap.

## Integration checklist (⏳ for the author)

- [ ] map Vanta's device / per-eye targets / matrices onto `FoveationHost`
- [ ] feed Vanta's eye-tracking API (or the GZFV relay) into a `GazeSource`
- [ ] check handedness: if Vanta is left-handed, convert view matrices before passing them
- [ ] measure the frame-time saving inside Vanta with its own GPU timers and fill the table:

| scene | resolution / eye | full-res ms | foveated ms | config |
|---|---|---:|---:|---|
| | | | | |
