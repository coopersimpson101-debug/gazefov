import json
import os
import socket

import numpy as np
import pytest

from gazefov import protocol as P
from gazefov.safety import LedSpec, check_exposure, irradiance_at_eye

FIX = os.path.join(os.path.dirname(__file__), "fixtures", "gaze_packet_v1.json")


def test_golden_vector():
    with open(FIX) as fh:
        g = json.load(fh)
    p = P.unpack(bytes.fromhex(g["hex"]))
    f = g["fields"]
    assert p.seq == f["seq"] and p.t_ns == int(f["t_ns"]) and p.flags == f["flags"]
    assert np.allclose(p.origin, f["origin"]) and np.allclose(p.direction, f["direction"], atol=1e-6)
    assert P.pack(p).hex() == g["hex"]


def test_roundtrip_and_crc():
    p = P.GazePacket(seq=1, t_ns=99, origin=np.zeros(3), direction=np.array([0, 0, 1.0]), ndc=np.array([0.5, -0.25]))
    b = P.pack(p)
    assert len(b) == P.PACKET_SIZE == 92
    assert np.allclose(P.unpack(b).ndc, [0.5, -0.25])
    bad = bytearray(b)
    bad[30] ^= 1
    with pytest.raises(ValueError):
        P.unpack(bytes(bad))


def test_udp_loopback():
    rx = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    rx.bind(("127.0.0.1", 0))
    rx.settimeout(2)
    s = P.GazeSender(port=rx.getsockname()[1])
    s.send(5, [0, 0, 0], [0, 0, 1], [0, 0])
    p = P.unpack(rx.recvfrom(256)[0])
    assert p.seq == 0 and p.t_ns == 5
    s.close()
    rx.close()


def test_openxr_flip_and_ndc():
    assert np.allclose(P.to_openxr([1, 2, 3]), [1, 2, -3])
    assert np.allclose(P.display_ndc([0, 0, 1], [0, 0, 0], 1.0, 1.0), 0)


def test_safety_numbers():
    led = LedSpec(radiant_intensity_mw_sr=4.0)
    e = irradiance_at_eye([led] * 8, distance_mm=30.0, duty=1.0)
    assert abs(e - 8 * 4e-3 / 0.03 ** 2) < 1e-9  # 8 x 4 mW/sr / (0.03 m)^2 = 35.6 W/m^2
    rep = check_exposure([led] * 8, distance_mm=30.0, duty=0.5, source_size_mm=1.0)
    assert rep["cornea_irradiance_W_m2"] < rep["cornea_limit_W_m2"]
    assert rep["exempt_cornea"] and rep["exempt_retina"]
    loud = check_exposure([LedSpec(radiant_intensity_mw_sr=60.0)] * 8, distance_mm=30.0, duty=1.0, source_size_mm=1.0)
    assert not loud["exempt_cornea"]
