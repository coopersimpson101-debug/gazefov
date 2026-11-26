# Eye tracking: models, algorithms and derivations

Everything here is implemented by hand in `gazefov/` (numpy only; OpenCV only for camera I/O in
`camera.py`). Units are millimetres and radians unless stated.

## 1. Frames and the eye model (`eye_model.py`, `geometry.py`)

**Headset frame H**: origin at the nominal centre of eye rotation, +X right, +Y up, +Z towards the
display. Gaze angles (yaw θ, pitch φ) map to `d = (sinθ cosφ, sinφ, cosθ cosφ)`. Gaze targets lie on
the virtual-image plane of the display, `z = 1000 mm`.

The eye follows the spherical-cornea model of Guestrin & Eizenman (2006):

| symbol | meaning | population value | synthetic subjects (`sample_subject`) |
|---|---|---|---|
| R | corneal radius of curvature | 7.8 | N(7.8, 0.25) |
| K | cornea centre → pupil centre | 4.2 | N(4.2, 0.25) |
| D | rotation centre → cornea centre | 5.3 | N(5.3, 0.3) |
| n | aqueous humour refractive index | 1.3375 | 1.3375 |
| α, β | kappa (visual − optical axis), horizontal / vertical | 5°, 1.5° | N(5°, 1°), N(1.5°, 0.8°) |

With optical axis **w**(θ, φ) the cornea centre is `c = e + D w`, the pupil centre `p = c + K w`, and
the visual axis `v = d(θ + α, φ + β)` passes through `c`. For a target T the eye pose is found by the
fixed-point iteration `θ, φ ← angles(T − c(θ, φ)) − (α, β)` (converges in a few steps because
D ≪ |T − c|). **Slippage** moves the rotation centre `e` in H (equivalently the headset moves on
the head); camera and LEDs stay fixed in H.

## 2. Synthetic IR renderer (`render_ir.py`)

Per pixel (2×2 super-sampled) a ray from the camera is traced through:

1. **Eyelid sphere** (radius r_eyeball + 1.6 mm, head-fixed). A hit with elevation above the upper-lid
   curve `el_up(az) = lid_up + 0.55 φ − 0.55 az²` or below the lower lid is skin. Eyelashes are thin
   slanted strands hanging below the upper-lid margin (in lid-sphere angular coordinates), so they
   occlude the pupil edge and glints consistently with the 3D geometry.
2. **Corneal cap** — the part of sphere (c, R) within the limbus radius r_iris of the optical axis.
   The ray is refracted with Snell's law, `t = η d + (η cos i − √(1 − η²(1 − cos² i))) n`,
   η = 1/n, and intersected with the iris plane `{x : (x − p)·w = 0}`; inside r_pupil it is pupil
   (dark-pupil illumination: the LEDs are off-axis), else iris with a procedural radial texture that
   rotates with the eye, plus a darker limbal ring.
3. **Sclera** — eyeball sphere (e, 12 mm), Lambert-shaded with faint vessel texture; else skin.

**Glints**: the reflection point q of LED l on the convex mirror (c, R) satisfies
`(q − c)/R = normalize(û(q→o) + û(q→l))` — the normal bisects the directions to camera and LED. It
is solved by the fixed-point iteration `q ← c + R · normalize(û(q→o) + û(q→l))` (tested against the
law of reflection to 1e-9). A glint is drawn (saturated Gaussian, σ = 0.8 px) only if q is on the
corneal cap, faces the camera and the camera ray to q passes the lids and lashes.

Camera effects: vignetting, optional dark/bright occluder blobs and spurious specular spots on the
sclera, defocus blur, shot noise (variance ∝ intensity) plus read noise, 8-bit quantisation. The
renderer also returns the label map (pupil centroid ground truth in px) and the exact glint
positions per LED, so detectors are scored in pixels and gaze in degrees.

Noise conditions used in the benchmark (`scripts/eval_gaze.py`):

| name | read noise σ (DN) | blur σ (px) | occluders | spurious glints |
|---|---:|---:|---:|---:|
| clean | 1 | 0.5 | 0 | 0 |
| low | 3 | 0.8 | 0 | 0 |
| medium | 6 | 1.1 | 0 | 1 |
| high | 10 | 1.5 | 1 | 1 |
| extreme | 16 | 2.0 | 2 | 2 |

## 3. Pupil detection (`pupil.py`, `imgproc.py`, `ellipse.py`)

1. **Glint in-painting.** Pixels brighter than their 11×11 mean by 40 DN (white top-hat) or above
   230 are dilated and replaced by a normalised-convolution average of the remaining neighbours,
   `fill = box(I · m̄) / box(m̄)`.
2. **Seed** = minimum of a box-blurred, 4×-downsampled image.
3. **Threshold sweep** ("MSER-lite"): `T_k = v_1% + f_k (v_60% − v_1%)`, f = 0.08 … 0.5. At each
   level: binary opening r = 2 (removes eyelashes) and closing r = 2 (hand-written shift-reduce
   morphology), 8-connected labelling (vectorised min-label propagation with pointer jumping). The
   region that contains the previous level's blob is followed until it becomes implausible (area >
   π·45², fill < 0.55) or its area jumps by more than 1.6× — that is the moment it leaks into a dark
   iris. This replaced a single threshold that failed on dark irises (`tests/test_tracking.py::
   test_dark_iris_does_not_capture_the_threshold`).
4. **Starburst** (Li, Winfield & Parkhurst 2005): 90 rays from the blob centroid; on each, the first
   sub-pixel crossing of `(v_pupil + v_iris)/2` on the σ = 1.2 smoothed image is an edge point.
   Points within 2 px of a glint, or whose outside (3 and 6 px further) is darker than the mid level
   (eyelash / lid), are rejected. The centroid of the edge points re-seeds a second pass.
5. **RANSAC ellipse.** Minimal 6-point subsets fitted with the numerically stable direct
   least-squares ellipse fit (Fitzgibbon, Pilu & Fisher 1999 in the Halir & Flusser 1998 block form,
   with Hartley-style normalisation), scored by the Sampson distance
   `|F(x)| / ‖∇F(x)‖` with a 0.9 px threshold, then refitted on all inliers.

## 4. Glints and LED association (`glints.py`)

Detection: top-hat (`I − box₁₁(I) > 45`, `I ≥ 140`) inside a disc of 3.2 pupil radii, connected
components of 1–40 px with a peak contrast ≥ 70, background-subtracted intensity-weighted centroid.

Association: to first order the glints of a ring of LEDs are a similarity transform of a **template**
— the glint pattern predicted by the model for a population eye looking straight ahead. Every
hypothesis "detection pair (i, j) ↔ ordered LED pair (a, b)" defines, in complex numbers,
`z_img = m z_tpl + t`, `m = (g_j − g_i)/(T_b − T_a)`. All hypotheses are evaluated at once
(vectorised): scale ∈ [0.6, 1.7], |rotation| ≤ 35°, inliers = template points within 0.3 × the
template spacing of a detection. The best hypothesis is refined by complex least squares
`m = Σ conj(a_k) b_k / Σ |a_k|²` on the one-to-one matches.

## 5. Gaze method (a): PCCR + 9-point polynomial (`gaze_poly.py`)

Feature: the pupil centre in the glint pattern's coordinates, `f = R(−θ)(p − t)/s − mean(T)`, which
is invariant to glint dropout and normalises camera–eye distance through s. Mapping: per output a
second-order bivariate polynomial `[1, x, y, xy, x², y²]` fitted by ridge least squares (λ = 1e-6)
on standardised features to the 9 calibration targets (3 × 3 grid, ±15° × ±11°), predicting
`(tan θ, tan φ)` of the target.

## 6. Gaze method (b): 3D model (`gaze_model.py`)

**Cornea centre.** For glint u_i of LED l_i with back-projected ray r_i from the camera centre o, the
law of reflection is coplanar: c lies in the plane through o spanned by (l_i − o) and r_i, with normal
`m_i = (l_i − o) × r_i`. With ≥ 2 glints, c is on the line `o + k b`, b = the null vector of the
stacked m_i (SVD). k is found by a 1-D scan of the residual below, then (c_x, c_y, c_z) are refined
by Levenberg–Marquardt on, per glint,

    q_i = o + t_i r_i (nearest intersection with sphere (c, R)),
    res_i = (q_i − c)/R − normalize(û(q_i→o) + û(q_i→l_i)),   + a miss penalty when r_i misses the sphere.

With exact glints this recovers c to < 1 µm (`test_cornea_centre_from_exact_glints`).

**Pupil centre.** The ray through the ellipse centre is refracted at the cornea and p is the point
on it with |p − c| = K. Because the image of the pupil centre is not the centre of the pupil's
image (perspective + refraction), the estimate is refined: every RANSAC-inlier edge point is
back-projected, refracted, and intersected with the current pupil plane `{c + K w, normal w}`; a
Kåsa circle fit in that plane gives the new p and `w = normalize(p − c)`; three iterations.

**Gaze.** `v = d(θ_w + α, φ_w + β)`, gaze point = intersection of `c + s v` with the display plane.

**Per-user calibration.** Given the 9 calibration frames: kappa in closed form as the median of
`angles(T_j − c_j) − angles(w_j)`; optionally (R, K) too by Levenberg–Marquardt on the angular
residuals with weak Gaussian priors (σ = 0.35 mm, weighted to 0.25° per σ), kappa re-solved inside.
The benchmark reports "kappa only", "kappa + R + K" and "no user calibration" (population values).

## 7. Why the 3D model survives slippage and the polynomial does not

PCCR maps an *image* vector to gaze. When the headset slips by δ, the cornea moves relative to
camera and LEDs, the pupil–glint vector changes although gaze did not, and the polynomial — fitted
at δ = 0 — reads that as a gaze change. The 3D method estimates c and w in the headset frame every
frame, so a translation of the eye only changes c, which is measured, and the visual axis is
still intersected correctly with the display. The benchmark shows exactly this: PCCR median error
grows by ~1.4° per mm of slip (0.33° → 4.56° from 0 to 3 mm); the calibrated model's stays at
0.08–0.12°.

## References

* E. D. Guestrin, M. Eizenman. *General theory of remote gaze estimation using the pupil center and
  corneal reflections.* IEEE Trans. Biomedical Engineering 53(6), 2006.
* C. Hennessey, B. Noureddin, P. Lawrence. *A single camera eye-gaze tracking system with free head
  motion.* ETRA 2006.
* D. Li, D. Winfield, D. Parkhurst. *Starburst: A hybrid algorithm for video-based eye tracking
  combining feature-based and model-based approaches.* CVPR Workshops 2005.
* A. Fitzgibbon, M. Pilu, R. Fisher. *Direct least squares fitting of ellipses.* IEEE TPAMI 21(5), 1999.
* R. Halir, J. Flusser. *Numerically stable direct least squares fitting of ellipses.* WSCG 1998.
* M. Fischler, R. Bolles. *Random sample consensus.* CACM 24(6), 1981.
* C. Morimoto, M. Mimica. *Eye gaze tracking techniques for interactive applications.* CVIU 98, 2005.
* J. J. Cerrolaza, A. Villanueva, R. Cabeza. *Taxonomic study of polynomial regressions applied to
  the calibration of video-oculographic systems.* ETRA 2008.
* Y. Le Grand, S. El Hage. *Physiological Optics.* Springer, 1980 (schematic-eye constants).
