"""End-to-end tests on synthetic IR images with exact ground truth."""
from dataclasses import replace

import numpy as np

from gazefov.eye_model import DEG, EyeParams, Rig, eye_state_for_target, sample_subject, target_point
from gazefov.gaze_model import UserModel, cornea_center, pupil_center_3d
from gazefov.geometry import normalize
from gazefov.glints import associate, detect_glints, glint_template
from gazefov.pupil import detect_pupil
from gazefov.render_ir import LABEL_PUPIL, RenderOptions, glint_positions, render
from gazefov.tracker import EyeTracker, angular_error_deg, nine_point_targets

RIG = Rig()
FAST = RenderOptions(supersample=1, noise_sigma=2.0, blur_sigma=0.7)


def test_visual_axis_hits_target():
    eye = EyeParams()
    T = target_point(12.0, -7.0)
    st = eye_state_for_target(eye, T, slip=np.array([1.0, -0.5, 0.8]))
    assert np.allclose(normalize(T - st.c), st.v, atol=1e-9)
    assert np.isclose(np.linalg.norm(st.c - st.e), eye.D)


def test_renderer_ground_truth_consistency():
    eye = EyeParams()
    st = eye_state_for_target(eye, target_point(0, 0), r_pupil=2.0)
    r = render(eye, st, RIG, FAST)
    assert r.image.dtype == np.uint8 and r.image.shape == (240, 320)
    assert (r.labels == LABEL_PUPIL).sum() > 500
    assert r.glint_visible.sum() >= 4
    for (x, y), v in zip(r.glints, r.glint_visible):
        if v:  # glints are bright exactly where the model says they are
            assert r.image[int(round(y)), int(round(x))] > 200


def test_pupil_detection_subpixel_on_clean_images():
    eye = EyeParams()
    errs = []
    for y, p in [(0, 0), (12, 5), (-15, -6), (8, 10)]:
        st = eye_state_for_target(eye, target_point(y, p), r_pupil=2.2)
        r = render(eye, st, RIG, FAST)
        pr = detect_pupil(r.image)
        assert pr.ok, pr.reason
        errs.append(np.linalg.norm(pr.center - r.pupil_center_px))
    assert np.median(errs) < 0.6


def test_dark_iris_does_not_capture_the_threshold():
    eye = replace(EyeParams(), iris_level=60.0, pupil_level=22.0)
    st = eye_state_for_target(eye, target_point(-8, 4), r_pupil=2.2)
    r = render(eye, st, RIG, replace(FAST, noise_sigma=1.0))
    pr = detect_pupil(r.image)
    assert pr.ok and np.linalg.norm(pr.center - r.pupil_center_px) < 1.0
    assert pr.ellipse.a < 45


def test_glint_detection_and_association():
    eye = EyeParams()
    tpl = glint_template(RIG)
    st = eye_state_for_target(eye, target_point(6, 3), r_pupil=2.0)
    r = render(eye, st, RIG, replace(FAST, spurious_glints=2))
    pr = detect_pupil(r.image)
    g = detect_glints(r.image, pr.center, pr.ellipse.a)
    a = associate(g, tpl)
    assert a.ok and a.n_inliers >= 4
    err = np.linalg.norm(a.points - r.glints[a.led_ids], axis=1)
    assert np.all(np.isfinite(err)) and err.max() < 0.6


def test_cornea_centre_from_exact_glints():
    eye = EyeParams()
    st = eye_state_for_target(eye, target_point(10, -5), slip=np.array([0.7, 0.3, -1.0]))
    uv, vis, _ = glint_positions(replace(eye, lashes=0), st, RIG, 0)
    ids = np.nonzero(vis)[0]
    assert len(ids) >= 2
    c, _ = cornea_center(RIG.camera, uv[ids], RIG.leds[ids], eye.R)
    assert np.linalg.norm(c - st.c) < 1e-3


def test_cornea_centre_from_two_glints():
    eye = EyeParams()
    st = eye_state_for_target(eye, target_point(-4, 2))
    uv, vis, _ = glint_positions(replace(eye, lashes=0), st, RIG, 0)
    ids = np.nonzero(vis)[0][:2]
    c, _ = cornea_center(RIG.camera, uv[ids], RIG.leds[ids], eye.R)
    assert np.linalg.norm(c - st.c) < 1e-2


def test_model_gaze_end_to_end_and_slippage_robustness():
    rng = np.random.default_rng(11)
    eye = sample_subject(rng, 0)
    tr = EyeTracker(RIG)
    T9 = nine_point_targets()
    imgs = [render(eye, eye_state_for_target(eye, T, r_pupil=2.2), RIG, replace(FAST, seed=i)).image for i, T in enumerate(T9)]
    tr.calibrate(imgs, T9, fit_shape=True)
    assert abs(tr.user.alpha - eye.alpha) < 1.5 * DEG
    errs_m, errs_p = [], []
    for k in range(8):
        T = target_point(rng.uniform(-15, 15), rng.uniform(-10, 10))
        slip = normalize(rng.normal(size=3)) * 2.0
        st = eye_state_for_target(eye, T, slip=slip, r_pupil=2.0)
        f = tr.features(render(eye, st, RIG, replace(FAST, seed=100 + k)).image)
        assert f.ok
        errs_m.append(angular_error_deg(tr.gaze_model(f).point, T, st.c))
        errs_p.append(angular_error_deg(tr.gaze_poly(f), T, st.c))
    # The 3D model is slippage-invariant (sub-degree); the 2D polynomial is not.
    assert np.median(errs_m) < 1.0
    assert np.median(errs_p) > np.median(errs_m)


def test_pupil_centre_refinement_beats_centre_ray():
    """Refraction-aware edge refinement recovers the optical axis better than the ellipse-centre ray."""
    eye = EyeParams()
    st = eye_state_for_target(eye, target_point(14, 8), r_pupil=2.5)
    r = render(eye, st, RIG, replace(FAST, noise_sigma=0.5, blur_sigma=0.4, supersample=2))
    pr = detect_pupil(r.image)
    um = UserModel(R=eye.R, K=eye.K)
    _, w0 = pupil_center_3d(RIG.camera, st.c, um, pr.center, None)
    _, w1 = pupil_center_3d(RIG.camera, st.c, um, pr.center, pr.edge_points[pr.inliers])
    e0 = np.degrees(np.arccos(np.clip(w0 @ st.w, -1, 1)))
    e1 = np.degrees(np.arccos(np.clip(w1 @ st.w, -1, 1)))
    assert e1 < 0.5
    assert e1 <= e0 + 0.05


def test_association_needs_two_glints():
    a = associate(np.array([[10.0, 10.0]]), glint_template(RIG))
    assert not a.ok and a.n_inliers == 1
