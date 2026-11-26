"""Writes tests/fixtures/gaze_packet_v1.json: one GZFV packet + its fields, the shared golden
vector checked by both the Python and the TypeScript test suites."""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from gazefov.protocol import FLAG_CALIBRATED, FLAG_LEFT, FLAG_VALID, EyeSample, GazePacket, pack  # noqa: E402

p = GazePacket(seq=4242, t_ns=1_234_567_890_123, origin=np.array([31.5, -2.25, 0.5]),
               direction=np.array([0.1, -0.05, 0.99373]), ndc=np.array([0.125, -0.0625]),
               left=EyeSample(3.5, 0.97, 5.75, -2.875), right=EyeSample(0.0, 0.0, 0.0, 0.0),
               error_deg=0.75, flags=FLAG_VALID | FLAG_LEFT | FLAG_CALIBRATED)
b = pack(p)
out = {"hex": b.hex(), "fields": {"seq": p.seq, "t_ns": str(p.t_ns), "flags": p.flags, "origin": p.origin.tolist(),
                                  "direction": p.direction.tolist(), "ndc": p.ndc.tolist(),
                                  "left": [3.5, 0.97, 5.75, -2.875], "right": [0, 0, 0, 0], "error_deg": 0.75}}
path = os.path.join(os.path.dirname(__file__), "..", "tests", "fixtures", "gaze_packet_v1.json")
with open(path, "w") as fh:
    json.dump(out, fh, indent=1)
print(b.hex())
