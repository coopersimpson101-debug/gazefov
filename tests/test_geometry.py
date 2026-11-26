import numpy as np

from gazefov.geometry import (angle_between, angles_to_dir, dir_to_angles, eye_rotation, look_at, normalize, ray_sphere,
                              reflection_point_on_sphere, refract)


def test_angles_roundtrip():
    rng = np.random.default_rng(0)
    y, p = rng.uniform(-1, 1, 50), rng.uniform(-1.2, 1.2, 50)
    y2, p2 = dir_to_angles(angles_to_dir(y, p))
    assert np.allclose(y, y2) and np.allclose(p, p2)


def test_eye_rotation_maps_z_to_gaze():
    for y, p in [(0.3, -0.2), (-0.5, 0.4), (0, 0)]:
        R = eye_rotation(y, p)
        assert np.allclose(R @ [0, 0, 1], angles_to_dir(y, p))
        assert np.allclose(R.T @ R, np.eye(3))


def test_look_at_axes():
    R = look_at(np.array([0.0, 0, 0]), np.array([0.0, 0, 10]))
    assert np.allclose(R[:, 0], [1, 0, 0]) and np.allclose(R[:, 1], [0, -1, 0]) and np.allclose(R[:, 2], [0, 0, 1])


def test_ray_sphere():
    t, hit = ray_sphere(np.array([0.0, 0, -10]), np.array([0.0, 0, 1]), np.zeros(3), 2.0)
    assert hit and np.isclose(t, 8.0)
    t, hit = ray_sphere(np.array([0.0, 5, -10]), np.array([0.0, 0, 1]), np.zeros(3), 2.0)
    assert not hit


def test_snell_law():
    d = normalize(np.array([np.sin(0.5), 0, -np.cos(0.5)]))
    n = np.array([0.0, 0, 1])
    t, ok = refract(d, n, 1 / 1.3375)
    assert ok
    sin_t = np.linalg.norm(np.cross(t, -n))
    assert np.isclose(np.sin(0.5), 1.3375 * sin_t)


def test_reflection_point_obeys_law_of_reflection():
    c = np.array([1.0, -2.0, 3.0])
    cam = np.array([0.0, -20.0, 30.0])
    leds = np.array([[15.0, 5.0, 25.0], [-12.0, 8.0, 22.0]])
    q = reflection_point_on_sphere(c, 7.8, cam, leds)
    for qi, light in zip(q, leds):
        n = normalize(qi - c)
        assert np.isclose(np.linalg.norm(qi - c), 7.8)
        a_in = angle_between(light - qi, n)
        a_out = angle_between(cam - qi, n)
        assert abs(a_in - a_out) < 1e-9
        assert abs(np.dot(np.cross(light - qi, cam - qi), n)) < 1e-9
