"""Synthetic near-eye IR camera renderer with exact ground truth.

Each pixel is ray traced through a 3D scene made of

1. an **eyelid sphere** (head-fixed, radius ``r_eyeball + 1.6`` mm around the rotation centre):
   points above the upper-lid curve or below the lower-lid curve are skin; thin dark eyelash
   strands hang from the upper-lid margin;
2. the **corneal cap** (sphere ``c``, ``R``) — rays are refracted (Snell, n = 1.3375) and
   continue to the iris/pupil plane at ``c + K w``; the pupil is dark (off-axis "dark pupil"
   illumination), the iris carries a procedural radial texture that rotates with the eye;
3. the **eyeball/sclera sphere** (``e``, ``r_eyeball``);
4. a skin background.

Corneal **glints** use the spherical-mirror model: the reflection point of each LED is solved on
the cornea sphere (normal bisects the camera and LED directions), tested against the corneal cap,
the eyelids and the lashes, projected and drawn as a small saturated Gaussian.

Camera effects: supersampling (anti-aliasing), vignetting, defocus blur, shot + read noise,
8-bit quantisation, optional random occluders and spurious reflections.

The renderer also returns a label map and the exact projected glint positions so detectors can
be scored in pixels and gaze in degrees.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .eye_model import Camera, EyeParams, EyeState, Rig
from .geometry import normalize, ray_plane, ray_sphere, reflection_point_on_sphere, refract
from .imgproc import gaussian_blur

LABEL_SKIN, LABEL_SCLERA, LABEL_IRIS, LABEL_PUPIL, LABEL_LASH = 0, 1, 2, 3, 4
LID_MARGIN = 1.6  # mm, lid sphere radius above the eyeball


@dataclass
class RenderOptions:
    supersample: int = 2
    blur_sigma: float = 0.7  # px, defocus
    noise_sigma: float = 2.0  # grey levels, read noise
    shot_gain: float = 0.05  # variance per grey level (Poisson-like)
    vignette: float = 0.35
    occluders: int = 0  # random dark/bright blobs (dust, frame, mascara...)
    spurious_glints: int = 0  # extra specular spots on sclera/lids
    glint_sigma: float = 0.8  # px
    glint_peak: float = 600.0  # before clipping -> saturated core
    seed: int = 0


@dataclass
class Rendered:
    image: np.ndarray  # uint8 (H, W)
    labels: np.ndarray  # uint8 (H, W), from the centre sample of each pixel
    glints: np.ndarray  # (n_leds, 2) projected glint positions, NaN where not visible
    glint_visible: np.ndarray  # (n_leds,) bool
    pupil_center_px: np.ndarray  # centroid of the visible pupil label region (NaN if none)
    pupil_visible_frac: float
    state: EyeState
    extras: dict = field(default_factory=dict)


def _lid_curves(eye: EyeParams, az: np.ndarray, gaze_pitch: float):
    """Upper/lower lid elevation boundaries (radians) as a function of azimuth on the lid sphere.

    The upper lid follows vertical gaze (lid-eye coordination), the lower lid a little.
    """
    up = eye.lid_upper + 0.55 * gaze_pitch - 0.55 * az ** 2
    lo = eye.lid_lower + 0.20 * gaze_pitch + 0.45 * az ** 2
    return up, lo


def _lash_mask(eye: EyeParams, az, el, up, rng_seed: int):
    """Eyelash strands hanging below the upper-lid margin (in lid-sphere angular coordinates)."""
    r = np.random.default_rng(rng_seed)
    n = eye.lashes
    a0 = r.uniform(-0.75, 0.75, n)
    length = r.uniform(0.04, 0.13, n)
    slant = r.uniform(-0.6, 0.6, n)
    width = r.uniform(0.004, 0.009, n)
    below = up - el  # distance below the margin (radians)
    m = np.zeros(az.shape, bool)
    if n == 0:
        return m
    band = np.nonzero((below > -0.005) & (below < length.max()))[0]
    if band.size == 0:
        return m
    b, a = below[band], az[band]
    hit = np.zeros(band.size, bool)
    for k in range(n):
        centre = a0[k] + slant[k] * np.clip(b, 0, None)
        hit |= (b < length[k]) & (np.abs(a - centre) < width[k] * (1.0 - 0.6 * np.clip(b / length[k], 0, 1)))
    m[band] = hit
    return m


def _iris_texture(eye: EyeParams, local: np.ndarray, rr: np.ndarray):
    """Radial-fibre iris texture in eye coordinates (rotates with the eye)."""
    ang = np.arctan2(local[..., 1], local[..., 0])
    r = np.random.default_rng(eye.iris_seed)
    freqs = r.integers(8, 40, 5)
    phases = r.uniform(0, 2 * np.pi, 5)
    amps = r.uniform(0.03, 0.10, 5)
    t = np.zeros_like(ang)
    for f, ph, a in zip(freqs, phases, amps):
        t += a * np.sin(f * ang + ph + 3.0 * rr / eye.r_iris)
    collar = 0.12 * np.exp(-(((rr - 0.45 * eye.r_iris) / 0.5) ** 2))  # collarette ring
    limbal = -0.35 * np.clip((rr - 0.85 * eye.r_iris) / (0.15 * eye.r_iris), 0, 1)  # dark limbal ring
    return eye.iris_level * (1.0 + t + collar + limbal)


def _lid_hit(o, d, e, r_lid):
    """Lid-sphere entry point along rays from outside (o + t d)."""
    t, hit = ray_sphere(o, d, e, r_lid)
    return t, hit


def trace(eye: EyeParams, st: EyeState, cam: Camera, o: np.ndarray, d: np.ndarray, lash_seed: int):
    """Trace rays (N, 3). Returns (intensity (N,), label (N,))."""
    N = d.shape[0]
    val = np.full(N, eye.skin_level, float)
    lab = np.full(N, LABEL_SKIN, np.uint8)
    e, c, w = st.e, st.c, st.w
    r_lid = eye.r_eyeball + LID_MARGIN
    t_lid, hit_lid = _lid_hit(o, d, e, r_lid)
    # Skin background shading for rays that miss the lid sphere entirely.
    val *= 1.0 - 0.15 * np.clip(d[:, 1], -1, 1)

    P = o + np.nan_to_num(t_lid)[:, None] * d
    rel = P - e
    az = np.arctan2(rel[:, 0], rel[:, 2])
    el = np.arctan2(rel[:, 1], np.hypot(rel[:, 0], rel[:, 2]))
    up, lo = _lid_curves(eye, az, st.pitch)
    open_ = hit_lid & (el < up) & (el > lo) & (np.abs(az) < 0.95)
    lid = hit_lid & ~open_
    # Lid skin: Lambertian-ish with a soft crease shadow near the margin.
    n_lid = normalize(rel)
    lamb = np.clip(-np.sum(n_lid * d, axis=1), 0, 1)
    margin = np.minimum(np.abs(el - up), np.abs(el - lo))
    val[lid] = eye.skin_level * (0.85 + 0.15 * lamb[lid]) * (1 - 0.35 * np.exp(-((margin[lid] / 0.05) ** 2)))
    lashes = open_ & _lash_mask(eye, az, el, up, lash_seed)
    val[lashes] = 32.0
    lab[lashes] = LABEL_LASH
    inside = open_ & ~lashes

    if inside.any():
        ii = np.nonzero(inside)[0]
        oi, di = o if o.ndim == 1 else o[ii], d[ii]
        # Cornea cap
        tc, hc = ray_sphere(oi, di, c, eye.R)
        q = oi + np.nan_to_num(tc)[:, None] * di
        cap_cos = np.sqrt(max(eye.R ** 2 - eye.r_iris ** 2, 0.0)) / eye.R
        on_cap = hc & (np.sum((q - c) * w, axis=1) / eye.R >= cap_cos)
        # Sclera
        ts, hs = ray_sphere(oi, di, e, eye.r_eyeball)
        s_pt = oi + np.nan_to_num(ts)[:, None] * di
        sclera_ok = hs & (np.sum((s_pt - e) * w, axis=1) < eye.D + eye.R * cap_cos)
        use_cap = on_cap & (~sclera_ok | (np.nan_to_num(tc, nan=1e9) <= np.nan_to_num(ts, nan=1e9)))
        use_scl = ~use_cap & sclera_ok

        vals = np.full(len(ii), eye.skin_level * 0.8)
        labs = np.full(len(ii), LABEL_SKIN, np.uint8)
        # Sclera shading + faint vessels
        n_s = normalize(s_pt - e)
        lam_s = np.clip(-np.sum(n_s * di, axis=1), 0, 1)
        vessel = 0.06 * np.sin(23 * np.arctan2(n_s[:, 1], n_s[:, 0]) + 9 * n_s[:, 2])
        vals[use_scl] = eye.sclera_level * (0.55 + 0.45 * lam_s[use_scl]) * (1 + vessel[use_scl])
        labs[use_scl] = LABEL_SCLERA

        if use_cap.any():
            k = np.nonzero(use_cap)[0]
            qk = q[k]
            nk = (qk - c) / eye.R
            t_dir, ok = refract(di[k], nk, 1.0 / eye.n_aqueous)
            p0 = c + eye.K * w
            tp = ray_plane(qk, t_dir, p0, w)
            h = qk + tp[:, None] * t_dir
            rel_h = h - p0
            # Local eye coordinates of the hit (x, y in the iris plane)
            from .geometry import eye_rotation

            Re = eye_rotation(st.yaw, st.pitch)
            local = rel_h @ Re
            rr = np.hypot(local[:, 0], local[:, 1])
            pupil = ok & (rr < st.r_pupil)
            iris = ok & ~pupil & (rr < eye.r_iris)
            other = ~(pupil | iris)
            vk = np.empty(len(k))
            vk[pupil] = eye.pupil_level
            vk[iris] = _iris_texture(eye, local[iris], rr[iris])
            vk[other] = eye.sclera_level * 0.7
            lk = np.where(pupil, LABEL_PUPIL, np.where(iris, LABEL_IRIS, LABEL_SCLERA)).astype(np.uint8)
            # Corneal Fresnel-ish darkening towards grazing incidence
            cosi = np.clip(-np.sum(nk * di[k], axis=1), 0, 1)
            vk *= 0.85 + 0.15 * cosi
            vals[k] = vk
            labs[k] = lk
        val[ii] = vals
        lab[ii] = labs
    return val, lab


def glint_positions(eye: EyeParams, st: EyeState, rig: Rig, lash_seed: int):
    """Exact glint image positions (n, 2) and visibility (n,) for every LED."""
    cam = rig.camera
    o = cam.o
    q = reflection_point_on_sphere(st.c, eye.R, o, rig.leds)
    n = (q - st.c) / eye.R
    cap_cos = np.sqrt(max(eye.R ** 2 - eye.r_iris ** 2, 0.0)) / eye.R
    vis = np.sum(n * st.w, axis=1) >= cap_cos
    vis &= np.sum(n * normalize(o - q), axis=1) > 0.05
    uv = cam.project(q)
    inside_img = (uv[:, 0] >= 0) & (uv[:, 0] < cam.width) & (uv[:, 1] >= 0) & (uv[:, 1] < cam.height)
    vis &= inside_img
    # Lid / lash occlusion along the camera ray to q
    d = normalize(q - o)
    r_lid = eye.r_eyeball + LID_MARGIN
    t_lid, hit = ray_sphere(o[None, :], d, st.e, r_lid)
    P = o + np.nan_to_num(t_lid)[:, None] * d
    rel = P - st.e
    az = np.arctan2(rel[:, 0], rel[:, 2])
    el = np.arctan2(rel[:, 1], np.hypot(rel[:, 0], rel[:, 2]))
    up, lo = _lid_curves(eye, az, st.pitch)
    open_ = hit & (el < up) & (el > lo) & (np.abs(az) < 0.95)
    vis &= open_ & ~_lash_mask(eye, az, el, up, lash_seed)
    uv = np.where(vis[:, None], uv, np.nan)
    return uv, vis, q


def render(eye: EyeParams, st: EyeState, rig: Rig, opt: RenderOptions | None = None) -> Rendered:
    opt = opt or RenderOptions()
    cam = rig.camera
    rng = np.random.default_rng(opt.seed)
    lash_seed = eye.iris_seed + 17
    W, H, S = cam.width, cam.height, max(1, opt.supersample)
    # Sub-pixel sample grid (S x S per pixel); the label map uses the sample nearest the centre.
    offs = (np.arange(S) + 0.5) / S - 0.5
    ys, xs = np.mgrid[0:H, 0:W]
    acc = np.zeros((H, W))
    labels = None
    ci = S // 2
    for sy_i, sy in enumerate(offs):
        for sx_i, sx in enumerate(offs):
            uv = np.stack([xs + sx, ys + sy], axis=-1).reshape(-1, 2)
            d = cam.backproject(uv)
            v, lab = trace(eye, st, cam, cam.o, d, lash_seed)
            acc += v.reshape(H, W)
            if sy_i == ci and sx_i == ci:
                labels = lab.reshape(H, W)
    img = acc / (S * S)
    # Vignetting (cos^4-ish falloff of the IR illumination + lens)
    rr = ((xs - cam.cx) ** 2 + (ys - cam.cy) ** 2) / (0.5 * np.hypot(W, H)) ** 2
    img *= 1.0 - opt.vignette * rr

    # Occluders (random soft blobs)
    for _ in range(opt.occluders):
        x0, y0 = rng.uniform(0, W), rng.uniform(0, H)
        rad = rng.uniform(4, 14)
        lvl = rng.choice([25.0, 220.0])
        m = np.exp(-(((xs - x0) ** 2 + (ys - y0) ** 2) / (2 * rad ** 2)) ** 2)
        img = img * (1 - m) + lvl * m

    # Glints
    uvg, vis, q = glint_positions(eye, st, rig, lash_seed)
    gsig = opt.glint_sigma
    for (gx, gy), v in zip(uvg, vis):
        if not v:
            continue
        img += opt.glint_peak * np.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2 * gsig ** 2))
    spurious = []
    for _ in range(opt.spurious_glints):
        # A specular spot somewhere on the sclera / lid, away from the cornea
        for _try in range(50):
            gx, gy = rng.uniform(0, W), rng.uniform(0, H)
            if labels[int(gy), int(gx)] in (LABEL_SCLERA, LABEL_SKIN):
                break
        spurious.append((gx, gy))
        img += 0.8 * opt.glint_peak * np.exp(-((xs - gx) ** 2 + (ys - gy) ** 2) / (2 * (1.3 * gsig) ** 2))

    img = gaussian_blur(img, opt.blur_sigma)
    if opt.noise_sigma > 0 or opt.shot_gain > 0:
        var = opt.noise_sigma ** 2 + opt.shot_gain * np.clip(img, 0, None)
        img = img + rng.normal(0, 1, img.shape) * np.sqrt(var)
    image = np.clip(np.round(img), 0, 255).astype(np.uint8)

    pm = labels == LABEL_PUPIL
    if pm.any():
        py, px = np.nonzero(pm)
        pc = np.array([px.mean(), py.mean()])
    else:
        pc = np.array([np.nan, np.nan])
    # Fraction of the pupil disc that is visible: compare to an un-occluded pupil area estimate
    frac = float(pm.sum()) / max(1.0, _pupil_area_px(eye, st, cam))
    return Rendered(image=image, labels=labels, glints=uvg, glint_visible=vis, pupil_center_px=pc,
                    pupil_visible_frac=min(frac, 1.0), state=st, extras={"spurious": spurious, "q": q})


def _pupil_area_px(eye: EyeParams, st: EyeState, cam: Camera) -> float:
    """Approximate image area of the (refracted) pupil, ignoring occlusion: project a ring of
    pupil-boundary points along the line to the camera, magnified ~ n/(n - (n-1) K/R)."""
    from .geometry import eye_rotation

    Re = eye_rotation(st.yaw, st.pitch)
    a = np.linspace(0, 2 * np.pi, 64, endpoint=False)
    mag = 1.0 / (1.0 - (1 - 1 / eye.n_aqueous) * eye.K / eye.R)  # paraxial corneal magnification
    ring = st.p + (np.stack([np.cos(a), np.sin(a), np.zeros_like(a)], 1) * st.r_pupil * mag) @ Re.T
    uv = cam.project(ring)
    x, y = uv[:, 0], uv[:, 1]
    return 0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
