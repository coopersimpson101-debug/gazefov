"""Ellipse fitting: numerically stable direct least squares (Fitzgibbon et al. 1999, in the
Halir & Flusser 1998 partitioned form) and a RANSAC wrapper scored by Sampson distance."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Ellipse:
    cx: float
    cy: float
    a: float  # semi-major (px)
    b: float  # semi-minor (px)
    theta: float  # major-axis angle (rad, image x towards y)
    conic: np.ndarray  # (A, B, C, D, E, F) with A x^2 + B xy + C y^2 + D x + E y + F = 0

    @property
    def center(self) -> np.ndarray:
        return np.array([self.cx, self.cy])

    def points(self, n: int = 64) -> np.ndarray:
        t = np.linspace(0, 2 * np.pi, n, endpoint=False)
        ct, st = np.cos(self.theta), np.sin(self.theta)
        x = self.a * np.cos(t)
        y = self.b * np.sin(t)
        return np.stack([self.cx + ct * x - st * y, self.cy + st * x + ct * y], axis=1)


def fit_conic(x: np.ndarray, y: np.ndarray) -> np.ndarray | None:
    """Direct least-squares ellipse fit. Coordinates should be roughly centred/scaled by the
    caller for conditioning (``fit_ellipse`` does this). Returns conic coefficients or None."""
    D1 = np.stack([x * x, x * y, y * y], axis=1)
    D2 = np.stack([x, y, np.ones_like(x)], axis=1)
    S1 = D1.T @ D1
    S2 = D1.T @ D2
    S3 = D2.T @ D2
    try:
        T = -np.linalg.solve(S3, S2.T)
    except np.linalg.LinAlgError:
        return None
    M = S1 + S2 @ T
    M = np.array([M[2] / 2.0, -M[1], M[0] / 2.0])  # premultiply by inv(C1)
    try:
        w, v = np.linalg.eig(M)
    except np.linalg.LinAlgError:
        return None
    v = np.real(v)
    cond = 4 * v[0] * v[2] - v[1] ** 2
    idx = np.nonzero(cond > 0)[0]
    if idx.size == 0:
        return None
    a1 = v[:, idx[0]]
    return np.concatenate([a1, T @ a1])


def conic_to_ellipse(c: np.ndarray) -> Ellipse | None:
    A, B, C, D, E, F = c
    den = B * B - 4 * A * C
    if den >= 0:
        return None
    cx = (2 * C * D - B * E) / den
    cy = (2 * A * E - B * D) / den
    # Value at the centre gives the scale
    F0 = A * cx * cx + B * cx * cy + C * cy * cy + D * cx + E * cy + F
    Mq = np.array([[A, B / 2], [B / 2, C]])
    w, v = np.linalg.eigh(Mq)
    if F0 == 0 or np.any(-F0 / w <= 0):
        return None
    axes = np.sqrt(-F0 / w)  # semi-axes along eigenvectors
    i_major = int(np.argmax(axes))
    vec = v[:, i_major]
    return Ellipse(cx=float(cx), cy=float(cy), a=float(axes[i_major]), b=float(axes[1 - i_major]),
                   theta=float(np.arctan2(vec[1], vec[0])), conic=np.asarray(c, float))


def fit_ellipse(pts: np.ndarray) -> Ellipse | None:
    """Fit an ellipse to (N, 2) points, N >= 5, with Hartley-style normalisation."""
    pts = np.asarray(pts, float)
    if len(pts) < 5:
        return None
    m = pts.mean(0)
    s = np.sqrt(2) / max(np.mean(np.linalg.norm(pts - m, axis=1)), 1e-9)
    q = (pts - m) * s
    c = fit_conic(q[:, 0], q[:, 1])
    if c is None:
        return None
    # Un-normalise: x' = s (x - m)
    A, B, C, D, E, F = c
    mx, my = m
    A2, B2, C2 = A * s * s, B * s * s, C * s * s
    D2 = D * s - 2 * A * s * s * mx - B * s * s * my
    E2 = E * s - 2 * C * s * s * my - B * s * s * mx
    F2 = A * s * s * mx * mx + B * s * s * mx * my + C * s * s * my * my - D * s * mx - E * s * my + F
    return conic_to_ellipse(np.array([A2, B2, C2, D2, E2, F2]))


def sampson_distance(c: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """First-order geometric distance of points to the conic (pixels)."""
    A, B, C, D, E, F = c
    x, y = pts[:, 0], pts[:, 1]
    f = A * x * x + B * x * y + C * y * y + D * x + E * y + F
    gx = 2 * A * x + B * y + D
    gy = B * x + 2 * C * y + E
    return np.abs(f) / np.maximum(np.hypot(gx, gy), 1e-12)


def ransac_ellipse(pts: np.ndarray, rng: np.random.Generator, iters: int = 120, thresh: float = 1.0,
                   min_axis: float = 2.0, max_axis: float = 200.0, max_ratio: float = 4.0):
    """RANSAC over 6-point subsets; returns (ellipse, inlier mask) or (None, None)."""
    pts = np.asarray(pts, float)
    n = len(pts)
    if n < 6:
        e = fit_ellipse(pts) if n >= 5 else None
        return e, (np.ones(n, bool) if e else None)
    best, best_in, best_score = None, None, -1.0
    for _ in range(iters):
        sub = pts[rng.choice(n, 6, replace=False)]
        e = fit_ellipse(sub)
        if e is None or not (min_axis <= e.b and e.a <= max_axis and e.a / max(e.b, 1e-9) <= max_ratio):
            continue
        d = sampson_distance(e.conic, pts)
        inl = d < thresh
        score = inl.sum() - 0.1 * np.minimum(d, thresh).sum() / thresh
        if score > best_score:
            best, best_in, best_score = e, inl, score
            if inl.mean() > 0.95:
                break
    if best is None:
        return None, None
    e = fit_ellipse(pts[best_in])
    if e is None:
        return best, best_in
    inl = sampson_distance(e.conic, pts) < thresh
    if inl.sum() >= 6:
        e2 = fit_ellipse(pts[inl])
        if e2 is not None:
            e = e2
    return e, inl
