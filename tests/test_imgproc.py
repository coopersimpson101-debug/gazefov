import numpy as np

from gazefov.ellipse import Ellipse, fit_ellipse, ransac_ellipse, sampson_distance
from gazefov.imgproc import bright_spots, closing, component_stats, dilate, erode, gaussian_blur, label, opening


def test_label_counts_components():
    m = np.zeros((40, 50), bool)
    m[2:8, 2:8] = True
    m[20:30, 10:12] = True
    m[5, 40:48] = True
    m[6, 48] = True  # 8-connected diagonal stays in the same component
    m[7, 49] = True
    lab, n = label(m)
    assert n == 3
    st = component_stats(lab, n)
    assert sorted(st["area"].astype(int).tolist()) == [10, 20, 36]


def test_label_spiral_long_component():
    m = np.zeros((30, 30), bool)
    m[1, 1:29] = True
    m[1:29, 28] = True
    m[28, 1:29] = True
    m[3:29, 1] = True
    m[3, 1:26] = True
    lab, n = label(m)
    assert n == 1


def test_morphology_removes_thin_lines():
    m = np.zeros((40, 40), bool)
    yy, xx = np.mgrid[:40, :40]
    m[(yy - 20) ** 2 + (xx - 20) ** 2 < 64] = True
    m[5, :] = True  # an eyelash
    o = opening(m, 2)
    assert not o[5].any() and o[20, 20]
    assert closing(m, 1)[20, 20]
    assert erode(m, 1).sum() < m.sum() < dilate(m, 1).sum()


def test_gaussian_blur_preserves_mean():
    rng = np.random.default_rng(1)
    img = rng.uniform(0, 255, (60, 80))
    assert abs(gaussian_blur(img, 1.5).mean() - img.mean()) < 0.5


def test_bright_spots_finds_small_peaks_only():
    img = np.full((50, 50), 120.0)
    img[10, 10] = 255
    img[30:45, 30:45] = 250  # large bright area is not a spot (except near its border)
    m = bright_spots(img, 45, 140)
    assert m[10, 10] and not m[37, 37]


def test_fit_ellipse_exact():
    t = Ellipse(cx=120.3, cy=80.7, a=30.0, b=18.0, theta=0.6, conic=np.zeros(6))
    pts = t.points(40)
    e = fit_ellipse(pts)
    assert abs(e.cx - 120.3) < 1e-6 and abs(e.cy - 80.7) < 1e-6
    assert abs(e.a - 30) < 1e-6 and abs(e.b - 18) < 1e-6
    assert np.max(sampson_distance(e.conic, pts)) < 1e-6


def test_ransac_rejects_outliers():
    rng = np.random.default_rng(3)
    t = Ellipse(cx=60, cy=50, a=20, b=15, theta=-0.3, conic=np.zeros(6))
    pts = t.points(60) + rng.normal(0, 0.2, (60, 2))
    out = rng.uniform(20, 100, (20, 2))
    e, inl = ransac_ellipse(np.vstack([pts, out]), rng, iters=200, thresh=1.0)
    assert np.hypot(e.cx - 60, e.cy - 50) < 0.2
    assert inl[:60].mean() > 0.9
