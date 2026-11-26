"""Geometric eye model, IR camera and LED rig.

The eye follows the model used by Guestrin & Eizenman (2006) and Hennessey et al. (2006):

* the cornea is a sphere of radius ``R`` centred at ``c`` (the "centre of corneal curvature");
* the pupil is a disc of radius ``r_pupil`` in a plane perpendicular to the **optical axis**,
  centred at ``p = c + K * w`` where ``w`` is the optical-axis unit vector;
* the eye rotates about a fixed centre ``e`` with ``c = e + D * w``;
* the **visual axis** (what the user actually looks along) passes through ``c`` and is offset
  from the optical axis by the angle kappa = (alpha horizontal, beta vertical), expressed as
  offsets of the gaze angles: ``visual = (yaw_opt + alpha, pitch_opt + beta)``;
* the aqueous humour has refractive index ``n`` (1.3375, the keratometric index).

Typical adult values (Le Grand / Gullstrand schematic eyes, as tabulated by Guestrin &
Eizenman 2006): R = 7.8 mm, K = 4.2 mm, D = 5.3 mm, |alpha| = 4-6 deg, beta = 1-2 deg.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from .geometry import angles_to_dir, dir_to_angles, look_at, normalize

DEG = np.pi / 180.0

#: Virtual image distance of the headset display (mm). Gaze targets live on the plane z = D_DISPLAY.
D_DISPLAY = 1000.0


@dataclass(frozen=True)
class EyeParams:
    """Per-user anatomical parameters (mm / radians)."""

    R: float = 7.8  # corneal radius of curvature
    K: float = 4.2  # cornea centre -> pupil centre
    D: float = 5.3  # rotation centre -> cornea centre
    r_iris: float = 5.9  # limbus radius (edge of the corneal cap)
    r_eyeball: float = 12.0
    n_aqueous: float = 1.3375
    alpha: float = 5.0 * DEG  # kappa, horizontal (visual - optical yaw)
    beta: float = 1.5 * DEG  # kappa, vertical
    # Appearance (IR grey levels, 0..255 before noise)
    pupil_level: float = 18.0
    iris_level: float = 85.0
    sclera_level: float = 175.0
    skin_level: float = 135.0
    iris_seed: int = 0
    # Eyelid aperture (radians of elevation on the lid sphere, at azimuth 0)
    lid_upper: float = 0.40
    lid_lower: float = -0.46
    lashes: int = 45


@dataclass(frozen=True)
class Camera:
    """Pinhole IR camera with OpenCV conventions; pose is camera -> H."""

    width: int = 320
    height: int = 240
    f: float = 300.0
    cx: float = 159.5
    cy: float = 119.5
    position: tuple = (0.0, -17.0, 26.0)
    look: tuple = (0.0, 1.5, 11.0)

    @property
    def R_cw(self) -> np.ndarray:  # camera -> H
        return look_at(np.array(self.position), np.array(self.look))

    @property
    def o(self) -> np.ndarray:
        return np.array(self.position, float)

    @property
    def Kmat(self) -> np.ndarray:
        return np.array([[self.f, 0, self.cx], [0, self.f, self.cy], [0, 0, 1.0]])

    def project(self, X: np.ndarray) -> np.ndarray:
        """H points (..., 3) -> pixel coordinates (..., 2)."""
        Xc = (np.asarray(X, float) - self.o) @ self.R_cw  # R^T (X - o)
        z = Xc[..., 2:3]
        return np.concatenate([self.f * Xc[..., 0:1] / z + self.cx, self.f * Xc[..., 1:2] / z + self.cy], axis=-1)

    def backproject(self, uv: np.ndarray) -> np.ndarray:
        """Pixel coordinates (..., 2) -> unit ray directions in H (..., 3)."""
        uv = np.asarray(uv, float)
        d = np.stack([(uv[..., 0] - self.cx) / self.f, (uv[..., 1] - self.cy) / self.f, np.ones(uv.shape[:-1])], axis=-1)
        return normalize(d @ self.R_cw.T)


def led_ring(n: int = 8, radius: float = 19.0, z: float = 24.0, phase: float = np.pi / 8) -> np.ndarray:
    """``n`` 850 nm LEDs evenly spaced on a ring around the eyepiece (positions in H, mm)."""
    a = phase + 2 * np.pi * np.arange(n) / n
    return np.stack([radius * np.cos(a), radius * np.sin(a), np.full(n, z)], axis=1)


@dataclass(frozen=True)
class Rig:
    """Headset-fixed hardware: one camera and an LED ring (all in H)."""

    camera: Camera = field(default_factory=Camera)
    leds: np.ndarray = field(default_factory=led_ring)
    # Slippage is modelled as the eye moving in H, which is equivalent to the rig moving.


@dataclass
class EyeState:
    """Ground-truth eye pose for one frame (all in H)."""

    e: np.ndarray  # rotation centre
    c: np.ndarray  # cornea centre
    w: np.ndarray  # optical axis
    v: np.ndarray  # visual axis
    p: np.ndarray  # pupil centre
    yaw: float  # optical axis angles
    pitch: float
    r_pupil: float
    target: np.ndarray


def target_point(yaw_deg: float, pitch_deg: float) -> np.ndarray:
    """Display-plane point seen at (yaw, pitch) from the nominal eye position (H origin)."""
    return np.array([D_DISPLAY * np.tan(yaw_deg * DEG), D_DISPLAY * np.tan(pitch_deg * DEG), D_DISPLAY])


def target_angles(T: np.ndarray) -> tuple[float, float]:
    """Inverse of :func:`target_point` (degrees)."""
    T = np.asarray(T, float)
    return float(np.arctan(T[..., 0] / T[..., 2]) / DEG), float(np.arctan(T[..., 1] / T[..., 2]) / DEG)


def eye_state_for_target(eye: EyeParams, T: np.ndarray, slip: np.ndarray | None = None, r_pupil: float = 2.0) -> EyeState:
    """Rotate the eye so that its visual axis (through c) hits display point ``T``.

    ``slip`` is the headset slippage: the eye's rotation centre sits at ``slip`` in H instead of
    the origin. Solved by fixed-point iteration (c depends on the rotation, which depends on c).
    """
    e = np.zeros(3) if slip is None else np.asarray(slip, float)
    T = np.asarray(T, float)
    yaw_v, pitch_v = dir_to_angles(T - e)
    yaw, pitch = float(yaw_v) - eye.alpha, float(pitch_v) - eye.beta
    for _ in range(30):
        w = angles_to_dir(yaw, pitch)
        c = e + eye.D * w
        yv, pv = dir_to_angles(T - c)
        ny, np_ = float(yv) - eye.alpha, float(pv) - eye.beta
        if abs(ny - yaw) < 1e-12 and abs(np_ - pitch) < 1e-12:
            break
        yaw, pitch = ny, np_
    w = angles_to_dir(yaw, pitch)
    c = e + eye.D * w
    v = angles_to_dir(yaw + eye.alpha, pitch + eye.beta)
    return EyeState(e=e, c=c, w=w, v=v, p=c + eye.K * w, yaw=yaw, pitch=pitch, r_pupil=r_pupil, target=T)


def sample_subject(rng: np.random.Generator, idx: int = 0) -> EyeParams:
    """A plausible adult subject (parameter spreads roughly match published population data)."""
    return replace(
        EyeParams(),
        R=float(rng.normal(7.8, 0.25)),
        K=float(rng.normal(4.2, 0.25)),
        D=float(rng.normal(5.3, 0.3)),
        r_iris=float(rng.normal(5.9, 0.2)),
        alpha=float(rng.normal(5.0, 1.0)) * DEG,
        beta=float(rng.normal(1.5, 0.8)) * DEG,
        pupil_level=float(rng.uniform(12, 28)),
        iris_level=float(rng.uniform(65, 110)),
        sclera_level=float(rng.uniform(160, 195)),
        skin_level=float(rng.uniform(115, 150)),
        iris_seed=int(rng.integers(1 << 30)) + idx,
        lid_upper=float(rng.uniform(0.34, 0.46)),
        lid_lower=float(rng.uniform(-0.50, -0.40)),
        lashes=int(rng.integers(25, 70)),
    )
