# Contributing

Thanks for helping. Ground rules:

1. **Numbers come from code.** Every figure or table in the README or docs must be produced by a
   script in this repo (`scripts/eval_gaze.py`, `render/scripts/bench.ts`, `scripts/make_figures.py`)
   on stated hardware. Simulated results are labelled 🧪, hardware TODOs ⏳.
2. **Core algorithms stay hand-written** (numpy for the tracker, WGSL/TypeScript for the renderer).
   OpenCV is allowed only for camera I/O in `gazefov/camera.py`.
3. Run the tests before a PR:

```bash
pip install -e ".[test]" && pytest
cd render && npm ci && npm run typecheck && npm test   # GPU tests run if a WebGPU adapter exists
```

4. Keep PRs focused; describe how you verified the change (and on which GPU for renderer changes).
