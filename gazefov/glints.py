"""Corneal glint detection and glint-to-LED association.

Detection: bright threshold inside a region of interest around the pupil, 8-connected
components, size filter, background-subtracted intensity-weighted centroid (sub-pixel).

Association: the glints of an LED ring form (to first order) a scaled, rotated and translated
copy of a *template* - the glint pattern predicted by the eye model for a straight-ahead gaze.
RANSAC over pairs of correspondences estimates that similarity transform; the hypothesis with
the most template points landing on detections wins and detections are then matched one-to-one
to LEDs by distance. This is robust to missing glints (eyelid/eyelash occlusion, glints falling
off the corneal cap) and to spurious reflections on the sclera.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .eye_model import EyeParams, Rig, eye_state_for_target, target_point
from .imgproc import box_blur, bright_spots, component_stats, label
from .render_ir import glint_positions


@dataclass
class GlintParams:
    level: float = 140.0  # absolute floor
    contrast: float = 45.0  # above the local 11x11 mean
    peak_contrast: float = 70.0
    min_area: int = 1
    max_area: int = 40
    roi_radius_factor: float = 3.2  # x pupil semi-major axis
    roi_min: float = 40.0


@dataclass
class Association:
    ok: bool
    led_ids: np.ndarray = field(default_factory=lambda: np.zeros(0, int))  # per matched glint
    points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))  # matched glint positions
    scale: float = np.nan
    angle: float = np.nan
    translation: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan]))
    n_inliers: int = 0

    def transform(self, tpl: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.angle), np.sin(self.angle)
        Rm = np.array([[c, -s], [s, c]])
        return self.scale * tpl @ Rm.T + self.translation

    def inverse(self, pts: np.ndarray) -> np.ndarray:
        c, s = np.cos(self.angle), np.sin(self.angle)
        Rm = np.array([[c, -s], [s, c]])
        return (pts - self.translation) @ Rm / self.scale


def detect_glints(img: np.ndarray, centre: np.ndarray, radius: float, params: GlintParams | None = None) -> np.ndarray:
    """Return (N, 2) sub-pixel glint centres inside a disc ROI around ``centre``."""
    P = params or GlintParams()
    h, w = img.shape
    r = max(P.roi_min, P.roi_radius_factor * radius)
    x0, x1 = int(max(0, centre[0] - r)), int(min(w, centre[0] + r + 1))
    y0, y1 = int(max(0, centre[1] - r)), int(min(h, centre[1] + r + 1))
    if x1 <= x0 or y1 <= y0:
        return np.zeros((0, 2))
    win = img[y0:y1, x0:x1].astype(float)
    bgl = box_blur(win, 5)
    m = bright_spots(win, P.contrast, P.level)
    if not m.any():
        return np.zeros((0, 2))
    lab, n = label(m)
    # weights: intensity above a local background
    # sub-pixel centre: intensity above the local background, over the thresholded pixels
    st = component_stats(lab, n, weights=np.clip(win - bgl, 1, None))
    peak = np.zeros(n)
    np.maximum.at(peak, lab[lab > 0] - 1, (win - bgl)[lab > 0])
    keep = (st["area"] >= P.min_area) & (st["area"] <= P.max_area) & (peak >= P.peak_contrast)
    pts = np.stack([st["cx"][keep] + x0, st["cy"][keep] + y0], 1)
    d = np.hypot(pts[:, 0] - centre[0], pts[:, 1] - centre[1])
    return pts[d <= r]


def glint_template(rig: Rig, eye: EyeParams | None = None) -> np.ndarray:
    """Predicted glint image positions (n_leds, 2) for a population eye looking straight ahead,
    ignoring occlusion (used only as a shape template for association)."""
    from dataclasses import replace

    eye = eye or EyeParams()
    eye = replace(eye, lid_upper=1.5, lid_lower=-1.5, lashes=0, r_iris=7.7)
    st = eye_state_for_target(eye, target_point(0.0, 0.0))
    uv, vis, _ = glint_positions(eye, st, rig, 0)
    if not vis.all():  # fall back to the raw reflection points if any are off-image
        from .geometry import reflection_point_on_sphere

        q = reflection_point_on_sphere(st.c, eye.R, rig.camera.o, rig.leds)
        uv = rig.camera.project(q)
    return uv


def _similarity_from_pairs(a1, a2, b1, b2):
    """Similarity mapping template points a1->b1, a2->b2 (complex-number form)."""
    za = complex(*(a2 - a1))
    zb = complex(*(b2 - b1))
    if abs(za) < 1e-9:
        return None
    m = zb / za
    s, th = abs(m), np.angle(m)
    t = np.array(b1) - s * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]]) @ a1
    return s, th, t


def associate(glints: np.ndarray, template: np.ndarray, tol_frac: float = 0.3,
              scale_range=(0.6, 1.7), max_angle=np.deg2rad(35)) -> Association:
    """Match detected glints to LEDs. Needs >= 2 detections for a full similarity transform; a
    single detection is associated to the LED whose template glint is nearest after a pure
    translation guess (template centroid at the detection), flagged by ``n_inliers == 1``."""
    g = np.asarray(glints, float)
    n_t = len(template)
    tc = template.mean(0)
    spacing = np.min([np.linalg.norm(template[i] - template[j]) for i in range(n_t) for j in range(i + 1, n_t)])
    if len(g) == 0:
        return Association(ok=False)
    if len(g) == 1:
        return Association(ok=False, points=g, n_inliers=1)
    # Every hypothesis (detection pair i<j  <->  ordered template pair a!=b) at once, in complex form:
    # z_img = m * z_tpl + t  with  m = (g_j - g_i) / (T_b - T_a),  t = g_i - m T_a.
    zg = g[:, 0] + 1j * g[:, 1]
    zt = template[:, 0] + 1j * template[:, 1]
    ii, jj = np.triu_indices(len(g), 1)
    aa, bb = np.nonzero(~np.eye(n_t, dtype=bool))
    dz_t = (zt[bb] - zt[aa])[None, :]
    m = (zg[jj] - zg[ii])[:, None] / dz_t  # (P, Q)
    tr = zg[ii][:, None] - m * zt[aa][None, :]
    s_all, th_all = np.abs(m), np.angle(m)
    valid = (s_all >= scale_range[0]) & (s_all <= scale_range[1]) & (np.abs(th_all) <= max_angle)
    if not valid.any():
        return Association(ok=False, points=g)
    m, tr, s_all, th_all = m[valid], tr[valid], s_all[valid], th_all[valid]
    mapped_z = m[:, None] * zt[None, :] + tr[:, None]  # (H, n_t)
    d = np.abs(mapped_z[:, :, None] - zg[None, None, :])  # (H, n_t, n_g)
    tol_h = tol_frac * spacing * s_all
    dmin = d.min(2)
    inl = dmin < tol_h[:, None]
    n_in = inl.sum(1)
    resid = np.where(inl, dmin / tol_h[:, None], 0).sum(1) / np.maximum(n_in, 1)
    score = n_in - 0.5 * resid - 0.05 * np.abs(np.log(s_all)) - 0.05 * np.abs(th_all)
    h = int(np.argmax(score))
    mz = mapped_z[h]
    best = (float(s_all[h]), float(th_all[h]), np.array([tr[h].real, tr[h].imag]),
            np.stack([mz.real, mz.imag], 1), float(tol_h[h]))
    s, th, t, mapped, tol = best
    d = np.linalg.norm(mapped[:, None, :] - g[None, :, :], axis=2)
    pairs = sorted(((d[a, k], a, k) for a in range(n_t) for k in range(len(g)) if d[a, k] < tol))
    used_a, used_k, ids, pts = set(), set(), [], []
    for _, a, k in pairs:
        if a in used_a or k in used_k:
            continue
        used_a.add(a)
        used_k.add(k)
        ids.append(a)
        pts.append(g[k])
    if len(ids) < 2:
        return Association(ok=False, points=g)
    ids = np.array(ids)
    pts = np.array(pts)
    # Refine the similarity with all matched pairs (least squares, complex form)
    A = template[ids] - template[ids].mean(0)
    B = pts - pts.mean(0)
    za = A[:, 0] + 1j * A[:, 1]
    zb = B[:, 0] + 1j * B[:, 1]
    m = np.vdot(za, zb) / max(np.vdot(za, za).real, 1e-12)
    s2, th2 = abs(m), float(np.angle(m))
    Rm = np.array([[np.cos(th2), -np.sin(th2)], [np.sin(th2), np.cos(th2)]])
    t2 = pts.mean(0) - s2 * Rm @ template[ids].mean(0)
    order = np.argsort(ids)
    _ = tc
    return Association(ok=True, led_ids=ids[order], points=pts[order], scale=float(s2), angle=th2,
                       translation=t2, n_inliers=len(ids))
