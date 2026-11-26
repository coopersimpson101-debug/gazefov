# Hardware: IR camera, 850 nm LED ring, eye safety, drivers, latency

> Status ⏳ — design notes and calculations only. Nothing here has been built or measured on a
> headset; every number below is either an explicit assumption or computed by code in this repo
> (`gazefov/safety.py`, `gazefov/eye_model.py`). The synthetic rig in `eye_model.py` uses exactly
> this geometry, so the simulated accuracy corresponds to this design.

## Geometry (per eye, headset frame H, mm; origin = nominal eye rotation centre)

| item | value in the model | notes |
|---|---|---|
| cornea apex | z ≈ 13.1 | D + R |
| eyepiece / LED ring plane | z = 24 | ≈ 11 mm eye relief from the apex |
| LED ring | 8 LEDs, radius 19 mm, first at 22.5° | `led_ring()`; 4–8 glints visible over ±20° gaze |
| camera | (0, −17, 26), aimed at (0, 1.5, 11) | below the eyepiece, ~35° off-axis (like most VR trackers) |
| camera intrinsics | 320 × 240 px, f = 300 px | ≈ 56° × 44° FOV, ≈ 10 px/mm at the pupil |

Design rules the model makes visible:

* **Dark-pupil** illumination: LEDs well off the camera axis, so the retina does not retro-reflect.
* Glints must land on the **corneal cap** (inside the limbus) for the spherical model to hold; a
  ring radius ≈ 1.5× the eye relief keeps ≥ 2 glints on the cap over ±20° × ±15° (in the benchmark
  4–8 were visible; the lower LEDs are the first lost to the lower lid at down-gaze).
* LEDs at distinct angles make the glint pattern unambiguous for association (no symmetric ring
  mirrored by the cornea would confuse the similarity fit with a 180° rotation; the ±35° rotation
  gate rules that out).

## Camera

* Global-shutter monochrome sensor with good 850 nm QE (e.g. OV7251-class, 640 × 480 @ 120–200 fps,
  binned to 320 × 240), M8/M12 lens with 850 nm band-pass filter (rejects visible display light).
* Strobe output → LED driver, so LEDs are only on during exposure (duty cycle = exposure × fps).
* Exposure 1–2 ms limits motion blur during saccades (500°/s × 2 ms = 1°).

### Driver hooks (`gazefov/camera.py`)

| source | API | status |
|---|---|---|
| `SyntheticSource` | the ground-truth renderer | ✅ tested |
| `UVCSource` | `cv2.VideoCapture` (V4L2 / Media Foundation / AVFoundation): resolution, fps, manual exposure, gain | ⏳ untested on hardware (`GAZEFOV_UVC_DEVICE=0 pytest tests/test_camera.py`) |
| `LibcameraSource` | `picamera2` (libcamera) on Raspberry Pi CSI cameras: YUV420 → Y plane, fixed exposure/gain, sensor timestamps | ⏳ untested |

## Eye safety (IEC 62471, exempt group)

Computed by `python -m gazefov.safety` with these **assumptions**: 8 LEDs, each 4 mW/sr on-axis at
the drive current used (a typical small 850 nm SMD LED is a few mW/sr at 20 mA — check the
datasheet), all aimed at the eye from 30 mm, emitting area 1 mm diameter, 8-hour exposure.

| quantity | limit (exempt group) | 8 × 4 mW/sr, continuous | 8 × 4 mW/sr, 25 % strobe | 8 × 10 mW/sr, 25 % strobe |
|---|---:|---:|---:|---:|
| cornea/lens IR irradiance E_IR (780–3000 nm), t > 1000 s | 100 W/m² | 35.6 W/m² (×2.8 margin) | 8.9 W/m² (×11) | 22.2 W/m² (×4.5) |
| retinal thermal radiance L_R (weak visual stimulus), α = 0.033 rad | 6000/α = 180 000 W m⁻² sr⁻¹ | 2 553 (×71) | 638 (×282) | 1 595 (×113) |

Formulas: `E = duty · Σ I_i / d²`; `L_R = R(λ) · I / A_source · duty`, burn-hazard weighting
`R(λ) = 10^((700 − λ)/500)` → 0.50 at 850 nm; `α = source size / distance`, clamped ≥ 0.011 rad.

Notes: the corneal (thermal) limit, not the retinal one, is the binding constraint for near-eye IR.
Inverse-square at 30 mm is conservative only if the LED is far-field there; real compliance needs a
radiometer at the eye position, the worst-case (failed-on, 100 % duty) condition, and a lab
assessment. Add hardware current limiting and a watchdog that turns the LEDs off if the camera stops
strobing.

## Latency budget (gaze change → photons at the new fovea)

| stage | budget | basis |
|---|---:|---|
| exposure (mid-exposure timestamp) | 1 ms | 2 ms exposure |
| sensor readout + USB/CSI transfer | 3 ms | 320 × 240 × 8 bit at 200 fps |
| eye tracking (pupil, glints, 3D gaze) | 2 ms target | **measured here: 158 ms median for the numpy reference** (Ryzen 5 3600, single core, machine under load) — the Python code is a reference implementation, not real-time; a C/SIMD or GPU port is on the roadmap |
| UDP → relay → renderer | < 0.5 ms | localhost |
| render latch → scan-out | 11.1 + 5.6 ms | 90 Hz: one frame + half scan-out |
| **total** | **≈ 23 ms** | with a 2 ms tracker |

Albert et al. (2017) found foveated rendering tolerable up to ~50–70 ms of eye-to-image latency
thanks to saccadic suppression; the benchmark therefore adds a 2° latency margin and the measured p95
tracking error (≈ 1°) to the acuity-derived foveal radius (`fovealRadiusDeg`).

## Hardware validation — TODO for the author

1. Build one eye-cup with the geometry above; record a 9-point calibration + 20 validation targets.
2. `python -m gazefov.safety` with your LED's datasheet intensity; measure irradiance at 30 mm.
3. Fill in:

| measurement | value |
|---|---|
| camera / lens / filter | |
| LED part, drive current, duty | |
| measured irradiance at eye position (W/m²) | |
| glints visible over ±20° (min / median) | |
| real gaze error, deg (mean / median / p95) | |
| tracker processing time per frame (ms) | |
| end-to-end latency (photodiode test, ms) | |

## References

* IEC 62471:2006 / CIE S 009:2002, *Photobiological safety of lamps and lamp systems.*
* ICNIRP, *Guidelines on limits of exposure to incoherent visible and infrared radiation*, Health
  Physics 105(1), 2013.
* R. Albert, A. Patney, D. Luebke, J. Kim. *Latency requirements for foveated rendering in virtual
  reality.* ACM Trans. Applied Perception 14(4), 2017.
