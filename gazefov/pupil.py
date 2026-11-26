"""Dark-pupil detection: glint in-painting, adaptive threshold + morphology + connected
components for a coarse blob, Starburst-style radial edge search (Li, Winfield & Parkhurst
2005), and RANSAC ellipse fitting.

Pipeline (all hand-written, numpy only):

1. **Glint in-painting**: saturated pixels (dilated) are replaced by a normalised-convolution
   average of their surroundings, so corneal reflections on the pupil border do not punch holes.
2. **Seed**: minimum of a box-blurred, 4x down-sampled image (the pupil is the largest dark blob).
3. **Threshold sweep** (MSER-like): thresholds rising from the darkest level; at each one a
   binary opening (removes eyelashes), closing and connected components; the plausible blob
   near the seed is kept until its area jumps (it has leaked into a dark iris).
4. **Starburst**: rays from the blob centroid; on each ray the first sub-pixel crossing of the
   mid level between pupil and iris is an edge point. Points next to glints, or whose outside
   is not iris-bright (eyelashes, eyelids), are rejected. The centre is re-estimated from the
   edge points and the search repeated once.
5. **RANSAC ellipse** on the edge points (Sampson distance), then a least-squares refit.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .ellipse import Ellipse, ransac_ellipse
from .imgproc import bilinear, box_blur, bright_spots, closing, component_stats, dilate, gaussian_blur, label, opening


@dataclass
class PupilParams:
    glint_level: float = 230.0
    min_radius: float = 5.0
    max_radius: float = 45.0
    n_rays: int = 90
    ransac_iters: int = 150
    ransac_thresh: float = 0.9
    threshold_fracs: tuple = (0.08, 0.12, 0.16, 0.21, 0.27, 0.34, 0.42, 0.5)
    leak_ratio: float = 1.6
    seed: int = 0


@dataclass
class PupilResult:
    ok: bool
    ellipse: Ellipse | None = None
    center: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan]))
    edge_points: np.ndarray = field(default_factory=lambda: np.zeros((0, 2)))
    inliers: np.ndarray = field(default_factory=lambda: np.zeros(0, bool))
    threshold: float = np.nan
    blob_area: float = 0.0
    confidence: float = 0.0
    reason: str = ""


def inpaint_bright(img: np.ndarray, level: float, grow: int = 2, r: int = 4, contrast: float = 40.0) -> tuple[np.ndarray, np.ndarray]:
    m = dilate((img >= level) | bright_spots(img, contrast, 110.0), grow)
    if not m.any():
        return img.astype(float), m
    keep = (~m).astype(float)
    num = box_blur(img * keep, r)
    den = box_blur(keep, r)
    out = img.astype(float).copy()
    fill = num / np.maximum(den, 1e-6)
    out[m] = fill[m]
    # Pixels whose whole neighbourhood was masked: widen once more
    bad = m & (den < 1e-3)
    if bad.any():
        num2 = box_blur(img * keep, 3 * r)
        den2 = box_blur(keep, 3 * r)
        out[bad] = (num2 / np.maximum(den2, 1e-6))[bad]
    return out, m


def _seed(img_s: np.ndarray, min_radius: float):
    h, w = img_s.shape
    f = 4
    hh, ww = h // f, w // f
    small = img_s[: hh * f, : ww * f].reshape(hh, f, ww, f).mean(axis=(1, 3))
    r = max(1, int(round(min_radius / f)))
    sm = box_blur(small, r)
    # avoid the border where the box blur replicates edge pixels
    sm[:r, :] = sm[-r:, :] = np.inf
    sm[:, :r] = sm[:, -r:] = np.inf
    iy, ix = np.unravel_index(np.argmin(sm), sm.shape)
    return np.array([ix * f + f / 2 - 0.5, iy * f + f / 2 - 0.5])


def detect_pupil(img: np.ndarray, params: PupilParams | None = None) -> PupilResult:
    P = params or PupilParams()
    rng = np.random.default_rng(P.seed)
    h, w = img.shape
    clean, gmask = inpaint_bright(img, P.glint_level)
    sm = gaussian_blur(clean, 1.2)
    seed = _seed(sm, P.min_radius)

    # Local statistics in a window around the seed
    R = int(P.max_radius * 1.6)
    x0, x1 = int(max(0, seed[0] - R)), int(min(w, seed[0] + R))
    y0, y1 = int(max(0, seed[1] - R)), int(min(h, seed[1] + R))
    win = sm[y0:y1, x0:x1]
    v_p = float(np.percentile(win, 1.0))
    v_hi = float(np.percentile(win, 60.0))
    if v_hi - v_p < 15:
        return PupilResult(ok=False, reason="no contrast")
    # Threshold sweep ("MSER-lite"): grow the dark region around the seed with increasing
    # thresholds and keep the last plausible one before its area jumps (leaks into the iris).
    best = None
    prev_area = None
    for frac in P.threshold_fracs:
        T = v_p + frac * (v_hi - v_p)
        mask = closing(opening(win < T, 2), 2)
        lab, n = label(mask)
        if n == 0:
            continue
        st = component_stats(lab, n)
        area = st["area"]
        bb = st["bbox"]
        fill = area / np.maximum(np.pi / 4 * (bb[:, 2] - bb[:, 0] + 1) * (bb[:, 3] - bb[:, 1] + 1), 1)
        cxw, cyw = st["cx"] + x0, st["cy"] + y0
        dist = np.hypot(cxw - seed[0], cyw - seed[1])
        plausible = (area >= np.pi * P.min_radius ** 2 * 0.5) & (area <= np.pi * P.max_radius ** 2) & (fill > 0.55)
        if not plausible.any():
            if best is not None:
                break
            continue
        if best is None:
            score = np.where(plausible, area * fill ** 2 / (1.0 + dist / 20.0), -1)
            k = int(np.argmax(score))
        else:  # follow the same region: the component that contains the previous blob
            ov = lab[best[1]]
            ov = ov[ov > 0]
            if ov.size == 0:
                break
            k = int(np.bincount(ov).argmax()) - 1
            if not plausible[k] or area[k] > P.leak_ratio * prev_area:
                break
        best = (T, lab == (k + 1), np.array([cxw[k], cyw[k]]), float(area[k]))
        prev_area = float(area[k])
    if best is None:
        return PupilResult(ok=False, reason="no plausible dark blob")
    T, blob, centre, blob_area = best
    r_eq = np.sqrt(blob_area / np.pi)
    v_pupil = float(np.median(win[blob]))
    ring = dilate(blob, max(3, int(0.35 * r_eq))) & ~dilate(blob, 2)
    v_iris = float(np.median(win[ring])) if ring.any() else v_hi
    mid = 0.5 * (v_pupil + v_iris)

    edges = np.zeros((0, 2))
    ang = np.linspace(0, 2 * np.pi, P.n_rays, endpoint=False)
    dirs = np.stack([np.cos(ang), np.sin(ang)], 1)
    steps = np.arange(0.0, 2.2 * r_eq + 6, 0.5)
    for _ in range(2):
        xs = centre[0] + dirs[:, 0:1] * steps[None, :]
        ys = centre[1] + dirs[:, 1:2] * steps[None, :]
        vals = bilinear(sm, xs, ys)
        above = vals > mid
        # first crossing beyond a small inner radius (glint residue near the centre is ignored)
        above[:, : max(1, int(0.3 * r_eq / 0.5))] = False
        has = above.any(1)
        first = np.argmax(above, 1)
        pts, ok_list = [], []
        for i in range(P.n_rays):
            if not has[i] or first[i] == 0:
                continue
            j = first[i]
            v0, v1 = vals[i, j - 1], vals[i, j]
            t = steps[j - 1] + 0.5 * (mid - v0) / max(v1 - v0, 1e-6)
            px, py = centre[0] + dirs[i, 0] * t, centre[1] + dirs[i, 1] * t
            if not (1 <= px < w - 2 and 1 <= py < h - 2):
                continue
            # reject edges next to glints
            gx, gy = int(round(px)), int(round(py))
            if gmask[max(0, gy - 2):gy + 3, max(0, gx - 2):gx + 3].any():
                continue
            # outside must look like iris (not eyelash/eyelid dark band): sample 3..6 px further
            outside = bilinear(sm, np.array([px + dirs[i, 0] * 3, px + dirs[i, 0] * 6]),
                               np.array([py + dirs[i, 1] * 3, py + dirs[i, 1] * 6]))
            if outside.min() < mid:
                continue
            pts.append((px, py))
        if len(pts) < 8:
            return PupilResult(ok=False, threshold=T, blob_area=blob_area, reason="too few edge points")
        edges = np.array(pts)
        centre = edges.mean(0)
    e, inl = ransac_ellipse(edges, rng, iters=P.ransac_iters, thresh=P.ransac_thresh,
                            min_axis=P.min_radius * 0.6, max_axis=P.max_radius * 1.5)
    if e is None or inl is None or inl.sum() < 8:
        return PupilResult(ok=False, threshold=T, edge_points=edges, reason="ellipse fit failed")
    conf = float(inl.sum()) / P.n_rays
    return PupilResult(ok=True, ellipse=e, center=e.center, edge_points=edges, inliers=inl, threshold=T,
                       blob_area=blob_area, confidence=conf)
