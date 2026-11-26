"""Gaze output over UDP: a fixed 92-byte little-endian packet (see docs/protocol.md).

The layout mirrors insight-vio's pose packet (magic, version, flags, sequence, timestamp, payload,
CRC-32) so one receiver can demultiplex both by magic.

====== ====== ======== ==============================================================
offset size   type     field
====== ====== ======== ==============================================================
0      4      char[4]  magic ``b"GZFV"``
4      2      u16      version (=1)
6      2      u16      flags: bit0 gaze valid, bit1 left eye valid, bit2 right eye valid,
                       bit3 user-calibrated, bit4 saccade in progress, bit5 OpenXR axes
8      4      u32      sequence number
12     8      u64      timestamp, ns (middle of the camera exposure, camera clock)
20     12     f32[3]   combined gaze origin (mm, headset frame)
32     12     f32[3]   combined gaze direction (unit vector, headset frame)
44     8      f32[2]   gaze point on the display, NDC x right / y up in [-1, 1]
52     16     f32[4]   left eye: pupil diameter (mm), confidence (0..1), yaw (deg), pitch (deg)
68     16     f32[4]   right eye: same
84     4      f32      expected angular error (deg, p95 from calibration validation)
88     4      u32      CRC-32 (zlib polynomial) of bytes 0..87
====== ====== ======== ==============================================================

Headset frame: +X right, +Y up, +Z forward (towards the display). With ``FLAG_OPENXR`` the
vectors are re-expressed in OpenXR view space (+X right, +Y up, -Z forward).
"""
from __future__ import annotations

import socket
import struct
import zlib
from dataclasses import dataclass, field

import numpy as np

MAGIC = b"GZFV"
VERSION = 1
_FMT = "<4sHHIQ3f3f2f4f4ff"
_BODY = struct.calcsize(_FMT)
PACKET_SIZE = _BODY + 4
assert PACKET_SIZE == 92

FLAG_VALID = 1 << 0
FLAG_LEFT = 1 << 1
FLAG_RIGHT = 1 << 2
FLAG_CALIBRATED = 1 << 3
FLAG_SACCADE = 1 << 4
FLAG_OPENXR = 1 << 5


@dataclass
class EyeSample:
    pupil_mm: float = 0.0
    confidence: float = 0.0
    yaw_deg: float = 0.0
    pitch_deg: float = 0.0


@dataclass
class GazePacket:
    seq: int
    t_ns: int
    origin: np.ndarray
    direction: np.ndarray
    ndc: np.ndarray
    left: EyeSample = field(default_factory=EyeSample)
    right: EyeSample = field(default_factory=EyeSample)
    error_deg: float = 0.0
    flags: int = FLAG_VALID


def pack(p: GazePacket) -> bytes:
    body = struct.pack(
        _FMT, MAGIC, VERSION, p.flags & 0xFFFF, p.seq & 0xFFFFFFFF, int(p.t_ns) & 0xFFFFFFFFFFFFFFFF,
        *map(float, p.origin), *map(float, p.direction), *map(float, p.ndc),
        p.left.pupil_mm, p.left.confidence, p.left.yaw_deg, p.left.pitch_deg,
        p.right.pupil_mm, p.right.confidence, p.right.yaw_deg, p.right.pitch_deg, float(p.error_deg))
    return body + struct.pack("<I", zlib.crc32(body) & 0xFFFFFFFF)


def unpack(buf: bytes) -> GazePacket:
    if len(buf) != PACKET_SIZE:
        raise ValueError(f"bad packet size {len(buf)}")
    body, (crc,) = buf[:_BODY], struct.unpack("<I", buf[_BODY:])
    if zlib.crc32(body) & 0xFFFFFFFF != crc:
        raise ValueError("CRC mismatch")
    v = struct.unpack(_FMT, body)
    if v[0] != MAGIC or v[1] != VERSION:
        raise ValueError("bad magic/version")
    return GazePacket(seq=v[3], t_ns=v[4], origin=np.array(v[5:8]), direction=np.array(v[8:11]), ndc=np.array(v[11:13]),
                      left=EyeSample(*v[13:17]), right=EyeSample(*v[17:21]), error_deg=v[21], flags=v[2])


def to_openxr(vec: np.ndarray) -> np.ndarray:
    """Headset frame (Z forward) -> OpenXR view space (Z backward)."""
    v = np.asarray(vec, float).copy()
    v[..., 2] *= -1
    return v


def display_ndc(direction: np.ndarray, origin: np.ndarray, tan_half_fov_x: float, tan_half_fov_y: float,
                display_z_mm: float = 1000.0) -> np.ndarray:
    """Where a gaze ray hits a symmetric display frustum, in NDC (x right, y up)."""
    d = np.asarray(direction, float)
    o = np.asarray(origin, float)
    t = (display_z_mm - o[2]) / d[2]
    p = o + t * d
    return np.array([p[0] / (display_z_mm * tan_half_fov_x), p[1] / (display_z_mm * tan_half_fov_y)])


class GazeSender:
    """Fire-and-forget UDP gaze publisher. Defaults to localhost; use a LAN address only on a
    trusted network (packets are not authenticated)."""

    def __init__(self, host: str = "127.0.0.1", port: int = 5566, openxr_axes: bool = False):
        self.addr = (host, port)
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.seq = 0
        self.openxr = openxr_axes

    def send(self, t_ns: int, origin, direction, ndc, left: EyeSample | None = None, right: EyeSample | None = None,
             error_deg: float = 0.0, flags: int = FLAG_VALID) -> bytes:
        if self.openxr:
            origin, direction = to_openxr(origin), to_openxr(direction)
            flags |= FLAG_OPENXR
        pkt = pack(GazePacket(self.seq, t_ns, np.asarray(origin), np.asarray(direction), np.asarray(ndc),
                              left or EyeSample(), right or EyeSample(), error_deg, flags))
        self.sock.sendto(pkt, self.addr)
        self.seq += 1
        return pkt

    def close(self):
        self.sock.close()


def receive_forever(port: int = 5566, host: str = "127.0.0.1", callback=print):
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.bind((host, port))
    while True:
        buf, _ = s.recvfrom(256)
        try:
            callback(unpack(buf))
        except ValueError:
            continue
