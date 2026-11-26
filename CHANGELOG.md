# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[Semantic Versioning](https://semver.org/).

## [0.1.0] - 2026-11-26

### Added
- Synthetic near-eye IR renderer with exact ground truth: ray-traced eyelids, eyelashes, refracting
  cornea, pupil/iris, sclera; spherical-mirror corneal glints for an 8-LED ring; blur, shot/read
  noise, occluders, spurious reflections; headset slippage.
- Pupil detection (glint in-painting, MSER-like threshold sweep, morphology, connected components,
  Starburst edges, RANSAC direct-least-squares ellipse) and glint detection + LED association.
- Gaze estimation: PCCR + 9-point second-order polynomial, and a 3D model-based method (cornea
  centre from glints, refraction-aware pupil centre, kappa) with per-user calibration of kappa, R, K.
- Gaze-accuracy benchmark over 8 synthetic subjects, 5 noise levels and 0-3 mm slippage.
- Foveated Gaussian-splat renderer for WebGPU on top of splatweb v0.1.0 (vendored): shared
  preprocess + sort, low-res periphery and full-res inset layers, smooth radial composite; acuity
  model (Guenter et al. 2012) for sizing the fovea; Dawn headless benchmark with PSNR/SSIM by region.
- GZFV 92-byte UDP gaze packet (Python + TypeScript, shared golden vector), UDP-to-WebSocket relay,
  1-euro gaze filter with saccade flag, Vanta adapter interface and driver.
- Hardware notes: camera + 850 nm LED ring geometry, IEC 62471 exposure budget, camera driver hooks
  (UVC via OpenCV, libcamera via picamera2), latency budget.

[0.1.0]: https://github.com/coopersimpson101-debug/gazefov/releases/tag/v0.1.0
