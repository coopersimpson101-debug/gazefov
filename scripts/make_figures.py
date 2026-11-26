"""Builds every figure in docs/assets from results/*.json and real pipeline output.

    python scripts/make_figures.py

Inputs: results/gaze_accuracy.json (scripts/eval_gaze.py), results/foveation_bench.json and
docs/assets/foveated_frame.png (render/scripts/bench.ts). Needs matplotlib.
"""
from __future__ import annotations

import json
import os
import sys
from dataclasses import replace

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.insert(0, ROOT)
from gazefov.eye_model import Rig, eye_state_for_target, sample_subject, target_point  # noqa: E402
from gazefov.render_ir import RenderOptions, render  # noqa: E402
from gazefov.tracker import EyeTracker  # noqa: E402

BG, FG, MUTED, GRID = "#0d1117", "#e6edf3", "#8b949e", "#30363d"
ACCENT, WARM, VIOLET = "#67E8F9", "#F59E0B", "#A78BFA"
A = os.path.join(ROOT, "docs", "assets")
plt.rcParams.update({"figure.facecolor": BG, "axes.facecolor": BG, "savefig.facecolor": BG, "text.color": FG,
                     "axes.labelcolor": FG, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": GRID,
                     "font.family": "monospace", "font.size": 9})


def load(name):
    with open(os.path.join(ROOT, "results", name)) as fh:
        return json.load(fh)


def eye_pipeline_figure():
    rng = np.random.default_rng(7)
    eye = sample_subject(rng, 3)
    rig = Rig()
    tr = EyeTracker(rig)
    fig, axs = plt.subplots(1, 3, figsize=(10.5, 2.9))
    for ax, (y, p, cond) in zip(axs, [(-12, 6, "low"), (8, -4, "medium"), (15, 9, "high")]):
        sig, blur, occ, spur = {"low": (3, 0.8, 0, 0), "medium": (6, 1.1, 0, 1), "high": (10, 1.5, 1, 1)}[cond]
        st = eye_state_for_target(eye, target_point(y, p), r_pupil=2.3)
        r = render(eye, st, rig, RenderOptions(supersample=2, noise_sigma=sig, blur_sigma=blur, occluders=occ, spurious_glints=spur, seed=3))
        f = tr.features(r.image)
        ax.imshow(r.image, cmap="gray", vmin=0, vmax=255)
        if f.pupil.ok:
            e = f.pupil.edge_points
            ax.scatter(e[~f.pupil.inliers, 0], e[~f.pupil.inliers, 1], s=3, c=WARM, lw=0)
            ax.scatter(e[f.pupil.inliers, 0], e[f.pupil.inliers, 1], s=2, c=ACCENT, lw=0)
            pts = f.pupil.ellipse.points(120)
            ax.plot(np.r_[pts[:, 0], pts[0, 0]], np.r_[pts[:, 1], pts[0, 1]], c=ACCENT, lw=0.8)
        for (gx, gy) in f.glints:
            ax.add_patch(plt.Circle((gx, gy), 4, fill=False, ec=MUTED, lw=0.6))
        if f.assoc.ok:
            for lid, (gx, gy) in zip(f.assoc.led_ids, f.assoc.points):
                ax.add_patch(plt.Circle((gx, gy), 4, fill=False, ec=VIOLET, lw=0.9))
                ax.text(gx + 5, gy - 4, str(lid), color=VIOLET, fontsize=6)
        ax.set_title(f"gaze ({y:+d}°, {p:+d}°) · noise '{cond}'", fontsize=8, color=FG)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("synthetic IR frames → pupil edges (cyan = RANSAC inliers, amber = rejected), ellipse, glints + LED ids (violet)",
                 fontsize=8, color=MUTED, y=0.03)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(os.path.join(A, "eye_pipeline.png"), dpi=130)
    plt.close(fig)


def accuracy_figure(acc):
    s = acc["summary"]
    keys = list(s.keys())
    labels = [k.replace("/slip0mm", "").replace("low/slip", "slip ").replace("mm", " mm") for k in keys]
    meths = [("poly", "PCCR polynomial", WARM), ("model_full", "3D model, per-user cal", ACCENT), ("model_uncal", "3D model, no user cal", MUTED)]
    x = np.arange(len(keys))
    fig, ax = plt.subplots(figsize=(9.5, 3.4))
    w = 0.26
    for i, (m, name, col) in enumerate(meths):
        med = [s[k][m]["median"] if s[k][m] else np.nan for k in keys]
        p95 = [s[k][m]["p95"] if s[k][m] else np.nan for k in keys]
        ax.bar(x + (i - 1) * w, med, w * 0.9, color=col, label=f"{name} (median)")
        ax.scatter(x + (i - 1) * w, p95, marker="_", s=120, color=col, lw=1.6)
    ax.set_yscale("log")
    ax.set_ylim(0.03, 100)
    ax.set_xticks(x, labels)
    ax.set_ylabel("angular error (deg, log)")
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=7, loc="upper left", ncol=3)
    ax.set_title("held-out gaze error on synthetic IR images (bar = median, tick = p95)", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(A, "accuracy.png"), dpi=130)
    plt.close(fig)


def frametime_figure(bench, ax=None):
    """Saving in GPU frame time (median of per-frame foveated/full ratios, interleaved A/B) vs
    foveal radius. Absolute ms are noisy on a shared GPU; the ratio is the robust quantity."""
    rs = bench["results"]
    own = ax is None
    if own:
        fig, ax = plt.subplots(figsize=(7.5, 3.4))
    fov = [r for r in rs if r["mode"] == "foveated" and r["tag"] == "grid"]
    scales = sorted({r["peripheryScale"] for r in fov}, reverse=True)
    inners = sorted({r["innerDeg"] for r in fov})
    cols = [VIOLET, ACCENT, "#F472B6"]
    for i, s in enumerate(scales):
        for gaze, ls, mk in [([0, 0], "-", "o"), ([0.3, 0.15], "--", "s")]:
            pts = [next((r for r in fov if r["peripheryScale"] == s and r["innerDeg"] == e and r["gazeNdc"] == gaze), None) for e in inners]
            y = [100 * p["saving"] if p else np.nan for p in pts]
            lab = f"1/{round(1 / s)} periphery" + (", gaze centre" if gaze == [0, 0] else ", gaze 18° off")
            ax.plot(inners, y, ls=ls, marker=mk, ms=4, color=cols[i % 3], lw=1.4, label=lab)
    ax.axhline(0, color=WARM, lw=1.0)
    ax.set_ylim(-55, 62)
    ax.text(inners[-1], 1.5, "full resolution", color=WARM, fontsize=7, ha="right", va="bottom")
    ax.set_xticks(inners, [f"{e:g}°" for e in inners])
    ax.set_xlabel("foveal radius (+ 5° blend)")
    ax.set_ylabel("GPU frame-time saving (%)")
    m = bench["meta"]
    ax.set_title(f"RTX 2060 / Dawn, {m['width']}x{m['height']} @ {m['fovYDeg']:g}°, 1M splats", fontsize=8)
    ax.grid(color=GRID, lw=0.5)
    ax.legend(frameon=False, fontsize=6, loc="lower left", ncol=2)
    if own:
        fig.tight_layout()
        fig.savefig(os.path.join(A, "frametime.png"), dpi=130)
        plt.close(fig)


def hero_figure(acc, bench):
    img = plt.imread(os.path.join(A, "foveated_frame.png"))
    fig = plt.figure(figsize=(12.8, 5.0))
    ax0 = fig.add_axes([0.0, 0.0, 0.56, 1.0])
    ax0.imshow(img)
    ax0.set_axis_off()
    ax0.text(12, 30, "foveated frame (Dawn, RTX 2060) · + = gaze · dashed = 100 % / 0 % inset", color=FG, fontsize=8)
    ax1 = fig.add_axes([0.6, 0.56, 0.39, 0.40])
    ax1.set_axis_off()
    s = acc["summary"]
    rows = [("condition", "PCCR poly", "3D model")]
    for k in ["low/slip0mm", "high/slip0mm", "low/slip2mm", "low/slip3mm"]:
        rows.append((k.replace("/slip0mm", " noise").replace("low/slip", "slip ").replace("mm", " mm"),
                     f"{s[k]['poly']['median']:.2f} / {s[k]['poly']['p95']:.2f}",
                     f"{s[k]['model_full']['median']:.2f} / {s[k]['model_full']['p95']:.2f}"))
    ax1.text(0, 1.0, "gaze error, deg (median / p95), synthetic", color=ACCENT, fontsize=9, va="top")
    for i, r in enumerate(rows):
        for j, c in enumerate(r):
            ax1.text([0, 0.42, 0.72][j], 0.82 - i * 0.16, c, color=FG if i else MUTED, fontsize=8.5, va="top")
    ax2 = fig.add_axes([0.64, 0.10, 0.34, 0.38])
    frametime_figure(bench, ax2)
    fig.savefig(os.path.join(A, "hero.png"), dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(A, exist_ok=True)
    acc = load("gaze_accuracy.json")
    bench = load("foveation_bench.json")
    eye_pipeline_figure()
    accuracy_figure(acc)
    frametime_figure(bench)
    hero_figure(acc, bench)
    _ = replace
    print("figures written to", A)
