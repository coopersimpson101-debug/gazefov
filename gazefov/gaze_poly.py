"""Regression-based gaze: pupil-centre / corneal-reflection (PCCR) vector + polynomial mapping.

Feature: the pupil centre expressed in the coordinate frame of the associated glint pattern,

    f = R(-theta) (p - t) / s  -  mean(template)

where (s, theta, t) is the similarity that maps the LED glint template onto the detected glints
(see glints.associate). Normalising by the glint pattern (instead of a single glint) keeps the
feature stable when individual glints drop out and partly compensates for changes in camera-eye
distance (s), which is the classic weakness of PCCR (Morimoto & Mimica 2005; Cerrolaza et al. 2008).

Mapping: a second-order bivariate polynomial per output (6 terms: 1, x, y, xy, x^2, y^2) fitted
by ridge-regularised least squares to the 9 calibration targets, predicting (tan yaw, tan pitch)
of the target as seen from the nominal eye position.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .eye_model import D_DISPLAY


def design(f: np.ndarray, order: int = 2) -> np.ndarray:
    x, y = f[:, 0], f[:, 1]
    cols = [np.ones_like(x), x, y]
    if order >= 2:
        cols += [x * y, x * x, y * y]
    if order >= 3:
        cols += [x * x * y, x * y * y, x ** 3, y ** 3]
    return np.stack(cols, 1)


@dataclass
class PolyGaze:
    coef: np.ndarray  # (n_terms, 2)
    mu: np.ndarray
    sd: np.ndarray
    order: int = 2

    @staticmethod
    def fit(features: np.ndarray, targets_tan: np.ndarray, order: int = 2, ridge: float = 1e-6) -> "PolyGaze":
        f = np.asarray(features, float)
        mu = f.mean(0)
        sd = f.std(0) + 1e-9
        A = design((f - mu) / sd, order)
        coef = np.linalg.solve(A.T @ A + ridge * np.eye(A.shape[1]), A.T @ np.asarray(targets_tan, float))
        return PolyGaze(coef=coef, mu=mu, sd=sd, order=order)

    def predict_tan(self, features: np.ndarray) -> np.ndarray:
        f = np.atleast_2d(np.asarray(features, float))
        return design((f - self.mu) / self.sd, self.order) @ self.coef

    def predict_point(self, features: np.ndarray) -> np.ndarray:
        t = self.predict_tan(features)
        return np.stack([D_DISPLAY * t[:, 0], D_DISPLAY * t[:, 1], np.full(len(t), D_DISPLAY)], 1)


def pccr_feature(pupil_center: np.ndarray, assoc, template: np.ndarray) -> np.ndarray:
    """Pupil centre in glint-template coordinates, relative to the template centroid."""
    return assoc.inverse(np.asarray(pupil_center, float)[None, :])[0] - template.mean(0)
