"""Camera driver hooks. OpenCV is used here — and only here — for camera I/O.

All sources yield ``Frame(image: uint8 HxW, t_ns: int, index: int)``:

* :class:`SyntheticSource` — the ground-truth renderer (tests, CI, demos).
* :class:`UVCSource` — any UVC IR camera through ``cv2.VideoCapture`` (V4L2 on Linux, Media
  Foundation / DirectShow on Windows, AVFoundation on macOS). Sets resolution, frame rate,
  manual exposure and gain where the driver allows it.
* :class:`LibcameraSource` — CSI cameras on Raspberry Pi / libcamera through ``picamera2``
  (optional import), with the sensor in raw/mono mode and fixed exposure.

LED strobing: the LED ring should be synchronised to the exposure (see docs/hardware.md); with
UVC that is done by the camera's strobe/flash output pin, so no software hook is needed here.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Iterator, Protocol

import numpy as np


@dataclass
class Frame:
    image: np.ndarray
    t_ns: int
    index: int


class FrameSource(Protocol):
    def frames(self) -> Iterator[Frame]: ...

    def close(self) -> None: ...


class SyntheticSource:
    """Renders an eye looking at a sequence of display targets (H frame, mm)."""

    def __init__(self, eye, rig, targets, options=None, pupil_mm: float = 2.2):
        self.eye, self.rig, self.targets, self.options, self.pupil = eye, rig, list(targets), options, pupil_mm

    def frames(self) -> Iterator[Frame]:
        from .eye_model import eye_state_for_target
        from .render_ir import render

        for i, T in enumerate(self.targets):
            st = eye_state_for_target(self.eye, T, r_pupil=self.pupil)
            yield Frame(render(self.eye, st, self.rig, self.options).image, time.monotonic_ns(), i)

    def close(self) -> None:
        pass


class UVCSource:
    """UVC IR camera via OpenCV. ``device`` is an index or a path such as /dev/video2."""

    def __init__(self, device: int | str = 0, width: int = 320, height: int = 240, fps: int = 120,
                 exposure: float | None = None, gain: float | None = None):
        import cv2  # camera I/O only

        self.cv2 = cv2
        self.cap = cv2.VideoCapture(device)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open camera {device!r}")
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        if exposure is not None:
            self.cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # V4L2 "manual" (driver dependent)
            self.cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
        if gain is not None:
            self.cap.set(cv2.CAP_PROP_GAIN, gain)
        self.i = 0

    def frames(self) -> Iterator[Frame]:
        while True:
            ok, img = self.cap.read()
            t = time.monotonic_ns()
            if not ok:
                return
            if img.ndim == 3:
                img = img[..., 1]  # IR sensors behind a Bayer/YUV pipeline: take one channel
            yield Frame(np.ascontiguousarray(img), t, self.i)
            self.i += 1

    def close(self) -> None:
        self.cap.release()


class LibcameraSource:
    """libcamera (Raspberry Pi CSI) via picamera2, mono 8-bit frames, fixed exposure in microseconds."""

    def __init__(self, width: int = 640, height: int = 480, fps: int = 120, exposure_us: int = 2000, gain: float = 2.0):
        from picamera2 import Picamera2  # optional dependency, only on the device

        self.cam = Picamera2()
        cfg = self.cam.create_video_configuration(main={"size": (width, height), "format": "YUV420"},
                                                  controls={"FrameRate": fps, "ExposureTime": exposure_us,
                                                            "AnalogueGain": gain, "AeEnable": False})
        self.cam.configure(cfg)
        self.cam.start()
        self.h = height
        self.i = 0

    def frames(self) -> Iterator[Frame]:
        while True:
            req = self.cam.capture_request()
            try:
                y = req.make_array("main")[: self.h]  # Y plane of YUV420 = luminance
                t = int(req.get_metadata().get("SensorTimestamp", time.monotonic_ns()))
            finally:
                req.release()
            yield Frame(np.ascontiguousarray(y), t, self.i)
            self.i += 1

    def close(self) -> None:
        self.cam.stop()
