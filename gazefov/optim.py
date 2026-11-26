"""Levenberg-Marquardt with a finite-difference Jacobian (small problems only)."""
from __future__ import annotations

import numpy as np


def numeric_jacobian(fun, x: np.ndarray, r0: np.ndarray, h: np.ndarray | float = 1e-6) -> np.ndarray:
    x = np.asarray(x, float)
    h = np.broadcast_to(np.asarray(h, float), x.shape)
    J = np.empty((len(r0), len(x)))
    for k in range(len(x)):
        dx = np.zeros_like(x)
        dx[k] = h[k]
        J[:, k] = (fun(x + dx) - fun(x - dx)) / (2 * h[k])
    return J


def levenberg_marquardt(fun, x0, iters: int = 30, lam: float = 1e-3, h=1e-6, tol: float = 1e-12):
    """Minimise 0.5 |fun(x)|^2. Returns (x, residual, cost)."""
    x = np.asarray(x0, float).copy()
    r = fun(x)
    cost = float(r @ r)
    for _ in range(iters):
        J = numeric_jacobian(fun, x, r, h)
        g = J.T @ r
        A = J.T @ J
        improved = False
        for _try in range(10):
            try:
                step = -np.linalg.solve(A + lam * np.diag(np.diag(A) + 1e-12), g)
            except np.linalg.LinAlgError:
                lam *= 10
                continue
            xn = x + step
            rn = fun(xn)
            cn = float(rn @ rn)
            if np.isfinite(cn) and cn < cost:
                x, r, improved = xn, rn, True
                done = cost - cn < tol * max(cost, 1e-30) or np.max(np.abs(step)) < 1e-12
                cost = cn
                lam = max(lam / 3, 1e-12)
                break
            lam *= 4
        if not improved or done:
            break
    return x, r, cost
