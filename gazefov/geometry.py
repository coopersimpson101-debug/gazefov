"""Small 3D geometry kernel: rotations, gaze angles, ray/sphere/plane intersection, refraction.

Conventions (used everywhere in gazefov):

* **Headset frame H** (millimetres): origin at the *nominal* centre of rotation of the eye,
  +X to the wearer's right, +Y up, +Z forward (towards the display). This triple is
  left-handed (like Unity's); all formulas here are coordinate formulas, so that only matters
  when converting to OpenXR (``protocol.to_openxr`` flips Z).
* **Gaze angles** ``(yaw, pitch)`` in radians: ``dir = (sin(yaw)cos(pitch), sin(pitch),
  cos(yaw)cos(pitch))``. Positive yaw looks right, positive pitch looks up.
* **Camera frame** is OpenCV's: x right, y down, z forward (optical axis).

All functions are vectorised over leading dimensions where that is natural.
"""
from __future__ import annotations

import numpy as np

EPS = 1e-12


def normalize(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(n, EPS)


def angles_to_dir(yaw, pitch) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=float)
    pitch = np.asarray(pitch, dtype=float)
    cp = np.cos(pitch)
    return np.stack([np.sin(yaw) * cp, np.sin(pitch), np.cos(yaw) * cp], axis=-1)


def dir_to_angles(d: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    d = normalize(d)
    yaw = np.arctan2(d[..., 0], d[..., 2])
    pitch = np.arcsin(np.clip(d[..., 1], -1.0, 1.0))
    return yaw, pitch


def rot_x(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def rot_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def rot_z(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def eye_rotation(yaw: float, pitch: float) -> np.ndarray:
    """Rotation eye-frame -> H with R @ (0,0,1) == angles_to_dir(yaw, pitch) (Fick gimbal, no torsion)."""
    return rot_y(yaw) @ rot_x(-pitch)


def angle_between(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Angle (radians) between vectors, numerically robust for small angles."""
    a = normalize(a)
    b = normalize(b)
    cross = np.linalg.norm(np.cross(a, b), axis=-1)
    dot = np.sum(a * b, axis=-1)
    return np.arctan2(cross, dot)


def look_at(eye: np.ndarray, target: np.ndarray, up_hint=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Camera->H rotation for an OpenCV camera at ``eye`` looking at ``target``.

    Columns are the camera x (right), y (down), z (forward) axes expressed in H.
    """
    z = normalize(np.asarray(target, float) - np.asarray(eye, float))
    x = normalize(np.cross(np.asarray(up_hint, float), z))  # wearer's right
    y = np.cross(x, z)  # image down
    return np.stack([x, y, z], axis=1)


def ray_sphere(o: np.ndarray, d: np.ndarray, c: np.ndarray, r: float):
    """Nearest positive intersection of rays o + t d (d unit) with a sphere.

    Returns ``(t, hit)``; ``t`` is NaN where there is no hit in front of the origin.
    Works on broadcastable arrays of shape (..., 3).
    """
    oc = o - c
    b = np.sum(oc * d, axis=-1)
    cc = np.sum(oc * oc, axis=-1) - r * r
    disc = b * b - cc
    hit = disc >= 0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    t0 = -b - sq
    t1 = -b + sq
    t = np.where(t0 > 1e-9, t0, np.where(t1 > 1e-9, t1, np.nan))
    t = np.where(hit, t, np.nan)
    return t, np.isfinite(t)


def ray_sphere_both(o, d, c, r):
    """Both intersection distances (t_near, t_far), NaN where the line misses."""
    oc = o - c
    b = np.sum(oc * d, axis=-1)
    cc = np.sum(oc * oc, axis=-1) - r * r
    disc = b * b - cc
    hit = disc >= 0
    sq = np.sqrt(np.where(hit, disc, 0.0))
    return np.where(hit, -b - sq, np.nan), np.where(hit, -b + sq, np.nan)


def ray_plane(o, d, p0, n):
    """t such that o + t d lies on the plane through p0 with normal n (NaN if parallel)."""
    den = np.sum(d * n, axis=-1)
    num = np.sum((p0 - o) * n, axis=-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.where(np.abs(den) > 1e-12, num / den, np.nan)
    return t


def refract(d: np.ndarray, n: np.ndarray, eta: float):
    """Snell refraction of unit direction ``d`` at a surface with unit normal ``n`` facing the
    incoming ray (n . d < 0). ``eta`` = n_incident / n_transmitted. Returns (t_dir, ok)."""
    cosi = -np.sum(n * d, axis=-1, keepdims=True)
    k = 1.0 - eta * eta * (1.0 - cosi * cosi)
    ok = (k >= 0)[..., 0]
    t = eta * d + (eta * cosi - np.sqrt(np.maximum(k, 0.0))) * n
    return normalize(t), ok


def reflection_point_on_sphere(c: np.ndarray, r: float, cam: np.ndarray, light: np.ndarray, iters: int = 60):
    """Point q on a convex spherical mirror (centre c, radius r) where light from ``light`` is
    reflected towards ``cam``: the normal at q bisects the directions to cam and light.

    Fixed-point iteration q <- c + r * normalize(u(q->cam) + u(q->light)); converges quickly for
    a convex mirror when both points are well outside the sphere. ``light`` may be (k, 3).
    """
    light = np.atleast_2d(np.asarray(light, float))
    n = normalize(normalize(cam - c) + normalize(light - c))
    for _ in range(iters):
        q = c + r * n
        n_new = normalize(normalize(cam - q) + normalize(light - q))
        if np.max(np.abs(n_new - n)) < 1e-13:
            n = n_new
            break
        n = n_new
    return c + r * n
