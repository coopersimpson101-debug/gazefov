# Vendored: splatweb v0.1.0

These files are copied **unmodified** from
[splatweb](https://github.com/coopersimpson101-debug/splatweb) at tag `v0.1.0`
(commit `9ed182185ab852bfe3b36df1c22988bc73a9174b`), MIT licensed (see `LICENSE` in this folder).

| file | used for |
|---|---|
| `gpu/shaders/preprocess.wgsl`, `gpu/packing.ts` | per-splat EWA projection, run once per frame at the full virtual resolution |
| `gpu/radixSort.ts`, `gpu/shaders/radix_*.wgsl`, `core/sort.ts` | GPU depth sort, run once per frame |
| `gpu/renderer.ts`, `gpu/shaders/render.wgsl` | the unmodified full-resolution baseline in the benchmark |
| `io/procedural.ts`, `io/splatData.ts`, `core/*`, `camera/camera.ts` | demo scene, maths, camera |

gazefov's own foveation code lives in `render/src/foveation/` and only *imports* from here.
Vendoring (instead of an npm git dependency) keeps CI hermetic and pins the exact shader
layout the foveated rasteriser relies on (`Splat2D`, 40 bytes). To update, re-copy from a newer
splatweb tag and re-run `npm test` and `npm run bench`.
