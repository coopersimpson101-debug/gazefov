"""Hand-written image primitives (numpy only): blur, gradients, morphology, connected components,
bilinear sampling. No OpenCV here: OpenCV is used only for camera I/O (see camera.py)."""
from __future__ import annotations

import numpy as np


def gaussian_kernel(sigma: float) -> np.ndarray:
    r = max(1, int(np.ceil(3.0 * sigma)))
    x = np.arange(-r, r + 1, dtype=float)
    k = np.exp(-0.5 * (x / sigma) ** 2)
    return k / k.sum()


def convolve1d(img: np.ndarray, k: np.ndarray, axis: int) -> np.ndarray:
    """Separable 1-D convolution with edge replication."""
    r = len(k) // 2
    pad = [(0, 0)] * img.ndim
    pad[axis] = (r, r)
    p = np.pad(img, pad, mode="edge")
    out = np.zeros_like(img, dtype=float)
    n = img.shape[axis]
    for i, w in enumerate(k):
        sl = [slice(None)] * img.ndim
        sl[axis] = slice(i, i + n)
        out += w * p[tuple(sl)]
    return out


def gaussian_blur(img: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return img.astype(float)
    k = gaussian_kernel(sigma)
    return convolve1d(convolve1d(img.astype(float), k, 0), k, 1)


def box_blur(img: np.ndarray, r: int) -> np.ndarray:
    """Mean filter of size (2r+1)^2 via an integral image."""
    img = img.astype(float)
    p = np.pad(img, r + 1, mode="edge")
    s = p.cumsum(0).cumsum(1)
    h, w = img.shape
    a = 2 * r + 1
    tot = s[a:a + h, a:a + w] - s[0:h, a:a + w] - s[a:a + h, 0:w] + s[0:h, 0:w]
    return tot / (a * a)


def sobel(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    p = np.pad(img.astype(float), 1, mode="edge")
    gx = (p[:-2, 2:] + 2 * p[1:-1, 2:] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[1:-1, :-2] + p[2:, :-2])
    gy = (p[2:, :-2] + 2 * p[2:, 1:-1] + p[2:, 2:]) - (p[:-2, :-2] + 2 * p[:-2, 1:-1] + p[:-2, 2:])
    return gx / 8.0, gy / 8.0


def _disk(r: int) -> list[tuple[int, int]]:
    return [(dy, dx) for dy in range(-r, r + 1) for dx in range(-r, r + 1) if dx * dx + dy * dy <= r * r + r]


def _shift_reduce(mask: np.ndarray, r: int, op) -> np.ndarray:
    h, w = mask.shape
    fill = op is np.logical_and  # erosion pads with True so borders don't erode
    p = np.full((h + 2 * r, w + 2 * r), fill, dtype=bool)
    p[r:r + h, r:r + w] = mask
    out = np.full(mask.shape, fill)
    for dy, dx in _disk(r):
        out = op(out, p[r + dy:r + dy + h, r + dx:r + dx + w])
    return out


def erode(mask: np.ndarray, r: int = 1) -> np.ndarray:
    return _shift_reduce(mask.astype(bool), r, np.logical_and)


def dilate(mask: np.ndarray, r: int = 1) -> np.ndarray:
    return _shift_reduce(mask.astype(bool), r, np.logical_or)


def opening(mask: np.ndarray, r: int = 1) -> np.ndarray:
    return dilate(erode(mask, r), r)


def closing(mask: np.ndarray, r: int = 1) -> np.ndarray:
    return erode(dilate(mask, r), r)


def label(mask: np.ndarray) -> tuple[np.ndarray, int]:
    """8-connected component labelling by min-label propagation + pointer jumping (vectorised).

    Returns (labels, n) with labels in 1..n and 0 for background.
    """
    mask = mask.astype(bool)
    h, w = mask.shape
    if not mask.any():
        return np.zeros((h, w), np.int32), 0
    big = h * w + 1
    idx = np.arange(h * w, dtype=np.int64).reshape(h, w)
    lab = np.where(mask, idx, big)
    p = np.full((h + 2, w + 2), big, dtype=np.int64)
    while True:
        p[1:-1, 1:-1] = lab
        m = lab
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy or dx:
                    m = np.minimum(m, p[1 + dy:1 + dy + h, 1 + dx:1 + dx + w])
        m = np.where(mask, m, big)
        # pointer jumping: follow label -> label-of-that-pixel until stable
        flat = m.ravel()
        fg = flat < big
        for _ in range(32):
            nxt = flat.copy()
            nxt[fg] = flat[flat[fg]]
            if np.array_equal(nxt, flat):
                break
            flat = nxt
        m = flat.reshape(h, w)
        if np.array_equal(m, lab):
            break
        lab = m
    roots, inv = np.unique(lab[mask], return_inverse=True)
    out = np.zeros((h, w), np.int32)
    out[mask] = inv.astype(np.int32) + 1
    return out, len(roots)


def component_stats(labels: np.ndarray, n: int, weights: np.ndarray | None = None):
    """Per-component area, centroid (x, y) and bounding box. Optional intensity weights for the
    centroid (sub-pixel glint centres)."""
    ys, xs = np.nonzero(labels)
    lab = labels[ys, xs] - 1
    area = np.bincount(lab, minlength=n).astype(float)
    wts = np.ones_like(xs, float) if weights is None else np.maximum(weights[ys, xs].astype(float), 1e-9)
    sw = np.bincount(lab, wts, minlength=n)
    cx = np.bincount(lab, wts * xs, minlength=n) / np.maximum(sw, 1e-12)
    cy = np.bincount(lab, wts * ys, minlength=n) / np.maximum(sw, 1e-12)
    x0 = np.full(n, np.inf)
    x1 = np.full(n, -np.inf)
    y0 = np.full(n, np.inf)
    y1 = np.full(n, -np.inf)
    np.minimum.at(x0, lab, xs)
    np.maximum.at(x1, lab, xs)
    np.minimum.at(y0, lab, ys)
    np.maximum.at(y1, lab, ys)
    return {"area": area, "cx": cx, "cy": cy, "bbox": np.stack([x0, y0, x1, y1], 1)}


def bilinear(img: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = img.shape
    x = np.clip(x, 0, w - 1.000001)
    y = np.clip(y, 0, h - 1.000001)
    x0 = np.floor(x).astype(int)
    y0 = np.floor(y).astype(int)
    fx = x - x0
    fy = y - y0
    a = img[y0, x0]
    b = img[y0, x0 + 1]
    c = img[y0 + 1, x0]
    d = img[y0 + 1, x0 + 1]
    return (a * (1 - fx) + b * fx) * (1 - fy) + (c * (1 - fx) + d * fx) * fy


def bright_spots(img: np.ndarray, contrast: float, level: float, r: int = 5) -> np.ndarray:
    """White top-hat style mask of small bright spots: pixels brighter than their (2r+1)^2 local
    mean by ``contrast`` and above an absolute ``level``."""
    f = img.astype(float)
    return ((f - box_blur(f, r)) > contrast) & (f >= level)
