import os

import numpy as np
import pytest

from gazefov.camera import SyntheticSource, UVCSource
from gazefov.eye_model import EyeParams, Rig, target_point
from gazefov.render_ir import RenderOptions


def test_synthetic_source_yields_frames():
    src = SyntheticSource(EyeParams(), Rig(), [target_point(0, 0), target_point(5, 5)], RenderOptions(supersample=1))
    frames = list(src.frames())
    assert len(frames) == 2 and frames[0].image.shape == (240, 320) and frames[1].index == 1
    assert frames[1].t_ns >= frames[0].t_ns


@pytest.mark.skipif(not os.environ.get("GAZEFOV_UVC_DEVICE"), reason="needs a real UVC IR camera (set GAZEFOV_UVC_DEVICE)")
def test_uvc_camera_smoke():
    dev = os.environ["GAZEFOV_UVC_DEVICE"]
    src = UVCSource(int(dev) if dev.isdigit() else dev)
    f = next(src.frames())
    src.close()
    assert f.image.dtype == np.uint8 and f.image.ndim == 2
