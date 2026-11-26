"""End-to-end eye tracker: image -> features -> gaze, with per-user 9-point calibration.

    tracker = EyeTracker(rig)
    tracker.calibrate(images, targets)          # 9 frames + their display points (mm, H)
    g = tracker.track(image)                   # GazeEstimate with poly + model results
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import gaze_model as gm
from .eye_model import DEG, Rig, target_point
from .gaze_poly import PolyGaze, pccr_feature
from .glints import Association, GlintParams, associate, detect_glints, glint_template
from .pupil import PupilParams, PupilResult, detect_pupil


@dataclass
class Features:
    ok: bool
    pupil: PupilResult
    glints: np.ndarray
    assoc: Association
    pccr: np.ndarray = field(default_factory=lambda: np.array([np.nan, np.nan]))
    ms: float = 0.0

    def model_frame(self) -> dict | None:
        if not (self.pupil.ok and self.assoc.ok):
            return None
        e = self.pupil.edge_points[self.pupil.inliers] if self.pupil.inliers is not None else None
        return {"glint_uv": self.assoc.points, "led_ids": self.assoc.led_ids, "pupil_uv": self.pupil.center, "edge_uv": e}


def nine_point_targets(yaw_deg: float = 15.0, pitch_deg: float = 11.0) -> np.ndarray:
    return np.array([target_point(y, p) for p in (pitch_deg, 0.0, -pitch_deg) for y in (-yaw_deg, 0.0, yaw_deg)])


class EyeTracker:
    def __init__(self, rig: Rig, pupil_params: PupilParams | None = None, glint_params: GlintParams | None = None):
        self.rig = rig
        self.pp = pupil_params or PupilParams()
        self.gp = glint_params or GlintParams()
        self.template = glint_template(rig)
        self.poly: PolyGaze | None = None
        self.user = gm.UserModel()
        self.last_c: np.ndarray | None = None

    # -- features ---------------------------------------------------------------------------
    def features(self, img: np.ndarray) -> Features:
        t0 = time.perf_counter()
        pr = detect_pupil(img, self.pp)
        if not pr.ok:
            return Features(False, pr, np.zeros((0, 2)), Association(ok=False), ms=1e3 * (time.perf_counter() - t0))
        g = detect_glints(img, pr.center, pr.ellipse.a, self.gp)
        a = associate(g, self.template)
        f = Features(pr.ok and a.ok, pr, g, a)
        if a.ok:
            f.pccr = pccr_feature(pr.center, a, self.template)
        f.ms = 1e3 * (time.perf_counter() - t0)
        return f

    # -- calibration ------------------------------------------------------------------------
    def calibrate_features(self, feats: list[Features], targets: np.ndarray, fit_shape: bool = True):
        targets = np.asarray(targets, float)
        ok = [i for i, f in enumerate(feats) if f.ok]
        if len(ok) < 6:
            raise ValueError(f"only {len(ok)} usable calibration frames")
        tan = targets[:, :2] / targets[:, 2:3]
        self.poly = PolyGaze.fit(np.array([feats[i].pccr for i in ok]), tan[ok])
        self.user = gm.calibrate(self.rig.camera, self.rig.leds, [f.model_frame() for f in feats], targets,
                                 fit_shape=fit_shape)
        return self

    def calibrate(self, images, targets, fit_shape: bool = True):
        return self.calibrate_features([self.features(im) for im in images], targets, fit_shape)

    # -- tracking ---------------------------------------------------------------------------
    def gaze_poly(self, f: Features) -> np.ndarray | None:
        if not f.ok or self.poly is None:
            return None
        return self.poly.predict_point(f.pccr[None, :])[0]

    def gaze_model(self, f: Features) -> gm.ModelGazeResult | None:
        mf = f.model_frame()
        if mf is None:
            return None
        r = gm.estimate(self.rig.camera, self.rig.leds, self.user, mf["glint_uv"], mf["led_ids"], mf["pupil_uv"],
                        mf["edge_uv"], c_prior=self.last_c)
        if r.ok:
            self.last_c = r.c
        return r


def angular_error_deg(est_point: np.ndarray, true_point: np.ndarray, eye_c: np.ndarray) -> float:
    """Angle at the (true) cornea centre between estimated and true display points."""
    a = np.asarray(est_point, float) - eye_c
    b = np.asarray(true_point, float) - eye_c
    return float(np.arctan2(np.linalg.norm(np.cross(a, b)), np.dot(a, b)) / DEG)
