"""Generates docs/assets/banner.svg. The motif on the right is real pipeline output: the pupil edge
points, fitted RANSAC ellipse and associated glints detected by gazefov on one synthetic IR frame,
inside the foveation rings (inner / outer blend radius) of the hero render.

    python scripts/make_banner.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from gazefov.eye_model import Rig, eye_state_for_target, sample_subject, target_point  # noqa: E402
from gazefov.render_ir import RenderOptions, render  # noqa: E402
from gazefov.tracker import EyeTracker  # noqa: E402

ACCENT = "#67E8F9"


def main():
    rng = np.random.default_rng(7)
    eye = sample_subject(rng, 3)
    rig = Rig()
    st = eye_state_for_target(eye, target_point(-6, 3), r_pupil=2.3)
    img = render(eye, st, rig, RenderOptions(supersample=2, noise_sigma=3, blur_sigma=0.8, seed=5)).image
    f = EyeTracker(rig).features(img)
    assert f.ok, "detection failed; banner needs real output"
    cx, cy = f.pupil.center
    s = 2.6  # px -> banner units
    ox, oy = 1010, 200
    tr = lambda x, y: (ox + s * (x - cx), oy + s * (y - cy))  # noqa: E731
    parts = []
    hero = os.path.join(ROOT, "docs", "assets", "foveated_frame.json")
    r_in, r_out = 88.4, 148.6
    if os.path.exists(hero):
        with open(hero) as fh:
            lay = json.load(fh)["layout"]
            r_in, r_out = lay["rInnerPx"], lay["rOuterPx"]
    k = 165 / r_out
    parts.append(f'<circle cx="{ox}" cy="{oy}" r="{r_out * k:.1f}" fill="none" stroke="{ACCENT}" stroke-opacity="0.25" stroke-dasharray="3 5"/>')
    parts.append(f'<circle cx="{ox}" cy="{oy}" r="{r_in * k:.1f}" fill="none" stroke="{ACCENT}" stroke-opacity="0.45" stroke-dasharray="8 6"/>')
    e = f.pupil.ellipse.points(160)
    d = "M" + " L".join(f"{tr(x, y)[0]:.1f},{tr(x, y)[1]:.1f}" for x, y in e) + " Z"
    parts.append(f'<path d="{d}" fill="{ACCENT}" fill-opacity="0.06" stroke="{ACCENT}" stroke-width="2"/>')
    for (x, y), inl in zip(f.pupil.edge_points, f.pupil.inliers):
        X, Y = tr(x, y)
        parts.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="{2.2 if inl else 1.6}" fill="{ACCENT if inl else "#8b949e"}" fill-opacity="{0.9 if inl else 0.6}"/>')
    for lid, (x, y) in zip(f.assoc.led_ids, f.assoc.points):
        X, Y = tr(x, y)
        parts.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="7" fill="none" stroke="{ACCENT}" stroke-width="1.5"/>')
        parts.append(f'<circle cx="{X:.1f}" cy="{Y:.1f}" r="2.5" fill="#e6edf3"/>')
    X, Y = tr(cx, cy)
    parts.append(f'<path d="M{X - 12:.1f},{Y:.1f} H{X - 4:.1f} M{X + 4:.1f},{Y:.1f} H{X + 12:.1f} M{X:.1f},{Y - 12:.1f} V{Y - 4:.1f} M{X:.1f},{Y + 4:.1f} V{Y + 12:.1f}" stroke="{ACCENT}" stroke-width="2"/>')
    mono = "ui-monospace,SFMono-Regular,Menlo,Consolas,monospace"
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 1280 400" width="1280" height="400">
  <defs>
    <pattern id="grid" width="32" height="32" patternUnits="userSpaceOnUse">
      <path d="M32 0H0V32" fill="none" stroke="#161b22" stroke-width="1"/>
    </pattern>
  </defs>
  <rect width="1280" height="400" fill="#0d1117"/>
  <rect width="1280" height="400" fill="url(#grid)"/>
  <text x="72" y="190" font-family="{mono}" font-size="84" font-weight="700" fill="#e6edf3">gazefov</text>
  <text x="76" y="238" font-family="{mono}" font-size="22" fill="#8b949e">IR eye tracking + foveated Gaussian-splat rendering</text>
  <text x="76" y="270" font-family="{mono}" font-size="16" fill="#8b949e">pupil &amp; glints → 3D gaze → full-res fovea, low-res periphery</text>
  <rect x="76" y="300" width="64" height="4" fill="{ACCENT}"/>
  {chr(10).join("  " + p for p in parts)}
</svg>
'''
    out = os.path.join(ROOT, "docs", "assets", "banner.svg")
    with open(out, "w", encoding="utf8") as fh:
        fh.write(svg)
    print("wrote", out, f"({len(f.pupil.edge_points)} edge points, {f.assoc.n_inliers} glints)")


if __name__ == "__main__":
    main()
