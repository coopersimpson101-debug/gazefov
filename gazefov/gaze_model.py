"""3D model-based gaze estimation (Guestrin & Eizenman 2006; Hennessey, Noureddin & Lawrence 2006).

Given a calibrated camera (intrinsics + pose in the headset frame H) and known LED positions:

1. **Cornea centre c** from >= 2 glints. Each glint u_i of LED l_i defines a plane through the
   camera centre o, l_i and the back-projected ray r_i; c lies in every such plane (law of
   reflection is coplanar), so c is on their common line o + k b. k is found by a 1-D search and
   then all three coordinates are refined by Levenberg-Marquardt on the reflection residual:
   at the ray/sphere intersection q_i the sphere normal must bisect (o - q_i) and (l_i - q_i).
2. **Pupil centre p**: the ray through the pupil-ellipse centre is refracted at the cornea
   (Snell, n = 1.3375) and p is the point on it at distance K from c. Because the image of the
   pupil centre is not the centre of the pupil image (perspective + refraction), the estimate is
   refined: every RANSAC-inlier pupil edge point is refracted onto the pupil plane
   {c + K w, normal w}, a circle is fitted in that plane, and w = normalize(p - c) is updated (3x).
3. **Optical axis** w = normalize(p - c) -> angles (yaw, pitch); **visual axis** = optical axis
   + kappa (alpha, beta). The gaze point is where the visual axis from c meets the display plane.

Per-user calibration fits kappa (and optionally R and K, with weak priors) to the 9 calibration
targets by Levenberg-Marquardt on the angular residuals.
"""
from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from .eye_model import D_DISPLAY, Camera
from .geometry import angles_to_dir, dir_to_angles, normalize, ray_plane, ray_sphere_both, refract
from .optim import levenberg_marquardt

DEG = np.pi / 180


@dataclass(frozen=True)
class UserModel:
    R: float = 7.8
    K: float = 4.2
    n: float = 1.3375
    alpha: float = 5.0 * DEG
    beta: float = 1.5 * DEG


def _reflection_residual(c, o, rays, leds, R):
    """Per glint: (sphere normal - bisector of the camera/LED directions) at the ray/sphere hit,
    plus a miss penalty when the ray passes beside the sphere. Vectorised over glints."""
    oc = o - c
    bq = rays @ oc
    disc = bq * bq - (oc @ oc - R * R)
    hit = disc >= 0
    t0 = -bq - np.sqrt(np.where(hit, disc, 0.0))
    t_close = -bq
    q0 = o + np.where(hit, t0, t_close)[:, None] * rays
    rel = q0 - c
    dist = np.linalg.norm(rel, axis=1)
    miss = np.where(hit, 0.0, (dist - R) / R)
    n = rel / np.maximum(dist, 1e-12)[:, None]
    q = c + R * n
    b = normalize(normalize(o - q) + normalize(leds - q))
    return np.concatenate([(n - b).ravel(), miss])


def cornea_center(cam: Camera, glint_uv: np.ndarray, leds: np.ndarray, R: float, c_prior: np.ndarray | None = None):
    """Estimate the cornea centre (H, mm) from >= 2 glints with known LEDs. Returns (c, rms)."""
    o = cam.o
    rays = cam.backproject(np.asarray(glint_uv, float))
    leds = np.asarray(leds, float)
    m = np.cross(leds - o, rays)
    m = normalize(m)
    _, _, vt = np.linalg.svd(m)
    b = vt[-1]
    if np.dot(b, rays.mean(0)) < 0:
        b = -b
    fun = lambda c: _reflection_residual(c, o, rays, leds, R)  # noqa: E731
    ks = np.linspace(15.0, 70.0, 45)
    costs = [float(np.sum(fun(o + k * b) ** 2)) for k in ks]
    k0 = ks[int(np.argmin(costs))]
    c0 = o + k0 * b
    if c_prior is not None and np.sum(fun(c_prior) ** 2) < np.min(costs):
        c0 = np.asarray(c_prior, float)
    c, r, cost = levenberg_marquardt(fun, c0, iters=40, h=1e-5)
    return c, float(np.sqrt(cost / max(1, len(r))))


def _refract_into_eye(cam: Camera, uv: np.ndarray, c: np.ndarray, R: float, n: float):
    """Back-project pixels, intersect the cornea sphere and refract. Returns (points, dirs, ok)."""
    o = cam.o
    d = cam.backproject(np.atleast_2d(uv))
    t0, _ = ray_sphere_both(o, d, c, R)
    ok = np.isfinite(t0) & (t0 > 0)
    t0 = np.where(ok, t0, np.sum((c - o) * d, axis=1))
    s = o + t0[:, None] * d
    nrm = normalize(s - c)
    td, ok2 = refract(d, nrm, 1.0 / n)
    return s, td, ok & ok2


def pupil_center_3d(cam: Camera, c: np.ndarray, um: UserModel, center_uv: np.ndarray,
                    edge_uv: np.ndarray | None = None, iters: int = 3):
    """Pupil centre p (H) and optical axis w. See module docstring, step 2."""
    s, td, ok = _refract_into_eye(cam, center_uv, c, um.R, um.n)
    s, td = s[0], td[0]
    sc = s - c
    bq = float(np.dot(td, sc))
    disc = bq * bq - (float(np.dot(sc, sc)) - um.K ** 2)
    t = -bq - np.sqrt(disc) if disc >= 0 else -bq
    p = s + t * td
    w = normalize(p - c)
    if edge_uv is not None and len(edge_uv) >= 6:
        se, tde, oke = _refract_into_eye(cam, edge_uv, c, um.R, um.n)
        se, tde = se[oke], tde[oke]
        if len(se) >= 6:
            for _ in range(iters):
                p0 = c + um.K * w
                tp = ray_plane(se, tde, p0, w)
                B = se + tp[:, None] * tde
                # in-plane orthonormal basis
                a1 = normalize(np.cross(w, [0.0, 1.0, 0.0]))
                a2 = np.cross(w, a1)
                X = np.stack([(B - p0) @ a1, (B - p0) @ a2], 1)
                # Kasa circle fit: x^2 + y^2 = 2 x0 x + 2 y0 y + k
                A = np.column_stack([2 * X[:, 0], 2 * X[:, 1], np.ones(len(X))])
                sol, *_ = np.linalg.lstsq(A, (X ** 2).sum(1), rcond=None)
                p = p0 + sol[0] * a1 + sol[1] * a2
                w = normalize(p - c)
    return p, w


@dataclass
class ModelGazeResult:
    ok: bool
    c: np.ndarray
    p: np.ndarray
    w: np.ndarray
    v: np.ndarray
    point: np.ndarray
    rms: float


def visual_axis(w: np.ndarray, um: UserModel) -> np.ndarray:
    yaw, pitch = dir_to_angles(w)
    return angles_to_dir(float(yaw) + um.alpha, float(pitch) + um.beta)


def display_point(c: np.ndarray, v: np.ndarray) -> np.ndarray:
    t = (D_DISPLAY - c[2]) / v[2]
    return c + t * v


def estimate(cam: Camera, leds: np.ndarray, um: UserModel, glint_uv: np.ndarray, led_ids: np.ndarray,
             pupil_uv: np.ndarray, edge_uv: np.ndarray | None = None, c_prior=None) -> ModelGazeResult:
    if len(led_ids) < 2:
        nan3 = np.full(3, np.nan)
        return ModelGazeResult(False, nan3, nan3, nan3, nan3, nan3, np.nan)
    c, rms = cornea_center(cam, glint_uv, leds[led_ids], um.R, c_prior)
    p, w = pupil_center_3d(cam, c, um, pupil_uv, edge_uv)
    v = visual_axis(w, um)
    return ModelGazeResult(True, c, p, w, v, display_point(c, v), rms)


def calibrate(cam: Camera, leds: np.ndarray, frames: list[dict], targets: np.ndarray, um0: UserModel | None = None,
              fit_shape: bool = True, prior_sd=(0.35, 0.35), prior_weight_deg: float = 0.25) -> UserModel:
    """Fit kappa (+ R, K when ``fit_shape``) to calibration frames.

    ``frames``: dicts with keys glint_uv, led_ids, pupil_uv, edge_uv (from tracker features).
    """
    um0 = um0 or UserModel()
    targets = np.asarray(targets, float)
    good = [i for i, f in enumerate(frames) if f is not None and len(f["led_ids"]) >= 2]
    if len(good) < 3:
        raise ValueError("need >= 3 calibration frames with >= 2 glints")

    # Pre-compute cornea centres for the prior R (and re-compute inside when R is fitted).
    def optical(um: UserModel):
        out = []
        for i in good:
            f = frames[i]
            c, _ = cornea_center(cam, f["glint_uv"], leds[f["led_ids"]], um.R)
            _, w = pupil_center_3d(cam, c, um, f["pupil_uv"], f.get("edge_uv"))
            out.append((c, w))
        return out

    def kappa_closed_form(cw):
        dy, dp = [], []
        for (c, w), i in zip(cw, good):
            yo, po = dir_to_angles(w)
            yt, pt = dir_to_angles(targets[i] - c)
            dy.append(float(yt - yo))
            dp.append(float(pt - po))
        return float(np.median(dy)), float(np.median(dp))

    cw = optical(um0)
    a, b = kappa_closed_form(cw)
    um = replace(um0, alpha=a, beta=b)
    if not fit_shape:
        return um

    def resid(x):
        R, K = x
        if not (6.0 < R < 9.5 and 2.5 < K < 6.0):
            return np.full(2 * len(good) + 2, 1e3)
        u = replace(um0, R=R, K=K)
        cw2 = optical(u)
        al, be = kappa_closed_form(cw2)
        u = replace(u, alpha=al, beta=be)
        r = []
        for (c, w), i in zip(cw2, good):
            v = visual_axis(w, u)
            yv, pv = dir_to_angles(v)
            yt, pt = dir_to_angles(targets[i] - c)
            r += [float(yv - yt) / DEG, float(pv - pt) / DEG]
        r += [prior_weight_deg * (R - um0.R) / prior_sd[0], prior_weight_deg * (K - um0.K) / prior_sd[1]]
        return np.array(r)

    x, _, _ = levenberg_marquardt(resid, np.array([um0.R, um0.K]), iters=15, h=1e-3)
    u = replace(um0, R=float(x[0]), K=float(x[1]))
    al, be = kappa_closed_form(optical(u))
    return replace(u, alpha=al, beta=be)
