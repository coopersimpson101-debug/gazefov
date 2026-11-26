"""Gaze-accuracy benchmark on synthetic IR eye images with exact ground truth.

    python scripts/eval_gaze.py                      # full run (8 subjects, ~3k frames)
    python scripts/eval_gaze.py --subjects 2 --test 10 --quick

For every synthetic subject (random corneal radius, K, kappa, iris, lids, lashes) and every
condition (camera noise level / headset slippage):

* render a 9-point calibration (3 x 3 grid, +-15 deg x +-11 deg) at slip 0,
* calibrate both gaze methods per user,
* render held-out test targets uniformly in +-20 deg x +-15 deg (so some extrapolate beyond the
  calibration grid) with a random pupil size each frame,
* score angular error (deg) at the true cornea centre.

Writes results/gaze_accuracy.json (every frame) and results/gaze_accuracy.md (summary tables).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from dataclasses import replace
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from gazefov import gaze_model as gm  # noqa: E402
from gazefov.eye_model import Rig, eye_state_for_target, sample_subject, target_point  # noqa: E402
from gazefov.geometry import normalize  # noqa: E402
from gazefov.render_ir import RenderOptions, render  # noqa: E402
from gazefov.tracker import EyeTracker, angular_error_deg, nine_point_targets  # noqa: E402

NOISE = {
    # name: (read-noise sigma, blur sigma px, occluders, spurious glints)
    "clean": (1.0, 0.5, 0, 0),
    "low": (3.0, 0.8, 0, 0),
    "medium": (6.0, 1.1, 0, 1),
    "high": (10.0, 1.5, 1, 1),
    "extreme": (16.0, 2.0, 2, 2),
}
SLIPS = [0.0, 1.0, 2.0, 3.0]  # mm, at "low" noise
METHODS = ["poly", "model_kappa", "model_full", "model_uncal"]


def conditions():
    out = [(n, 0.0) for n in NOISE]
    out += [("low", s) for s in SLIPS if s > 0]
    return out


def run_subject(args):
    sid, n_test, ss, seed = args
    rng = np.random.default_rng(seed + 1000 * sid)
    eye = sample_subject(rng, sid)
    rig = Rig()
    T9 = nine_point_targets()
    recs = []
    calib_cache = {}
    for cond, slip in conditions():
        sig, blur, occ, spur = NOISE[cond]
        opt = RenderOptions(supersample=ss, blur_sigma=blur, noise_sigma=sig, occluders=occ, spurious_glints=spur)
        if cond not in calib_cache:
            tr = EyeTracker(rig)
            feats = []
            for i, T in enumerate(T9):
                st = eye_state_for_target(eye, T, r_pupil=float(rng.uniform(1.6, 2.6)))
                r = render(eye, st, rig, replace(opt, seed=int(rng.integers(1 << 30))))
                feats.append(tr.features(r.image))
            tr_shape = EyeTracker(rig)
            try:
                tr.calibrate_features(feats, T9, fit_shape=False)
                tr_shape.calibrate_features(feats, T9, fit_shape=True)
                cal_ok = True
            except ValueError:
                cal_ok = False
            calib_cache[cond] = (tr, tr_shape, cal_ok, sum(f.ok for f in feats))
        tr, tr_shape, cal_ok, n_cal = calib_cache[cond]
        uncal = gm.UserModel()
        for k in range(n_test):
            yaw, pitch = float(rng.uniform(-20, 20)), float(rng.uniform(-15, 15))
            T = target_point(yaw, pitch)
            s = normalize(rng.normal(size=3)) * slip if slip > 0 else None
            st = eye_state_for_target(eye, T, slip=s, r_pupil=float(rng.uniform(1.4, 3.0)))
            r = render(eye, st, rig, replace(opt, seed=int(rng.integers(1 << 30))))
            f = tr.features(r.image)
            rec = {"subject": sid, "cond": cond, "slip": slip, "yaw": yaw, "pitch": pitch, "ok": bool(f.ok and cal_ok),
                   "ms_features": f.ms, "n_glints_vis": int(r.glint_visible.sum()), "n_glints_assoc": int(f.assoc.n_inliers),
                   "pupil_vis": r.pupil_visible_frac, "n_cal_ok": n_cal}
            if f.pupil.ok and np.all(np.isfinite(r.pupil_center_px)):
                rec["pupil_px_err"] = float(np.linalg.norm(f.pupil.center - r.pupil_center_px))
            if f.assoc.ok:
                gt = r.glints[f.assoc.led_ids]
                good = np.all(np.isfinite(gt), 1)
                rec["glint_px_err"] = [float(x) for x in np.linalg.norm(f.assoc.points[good] - gt[good], axis=1)]
                rec["glint_wrong"] = int((~good).sum())
            if rec["ok"]:
                t0 = time.perf_counter()
                rec["err_poly"] = angular_error_deg(tr.gaze_poly(f), T, st.c)
                t1 = time.perf_counter()
                g = tr.gaze_model(f)
                rec["ms_model"] = 1e3 * (time.perf_counter() - t1)
                rec["ms_poly"] = 1e3 * (t1 - t0)
                rec["err_model_kappa"] = angular_error_deg(g.point, T, st.c)
                rec["cornea_mm_err"] = float(np.linalg.norm(g.c - st.c))
                g2 = tr_shape.gaze_model(f)
                rec["err_model_full"] = angular_error_deg(g2.point, T, st.c)
                mf = f.model_frame()
                g3 = gm.estimate(rig.camera, rig.leds, uncal, mf["glint_uv"], mf["led_ids"], mf["pupil_uv"], mf["edge_uv"])
                rec["err_model_uncal"] = angular_error_deg(g3.point, T, st.c)
            recs.append(rec)
    subj = {"id": sid, "R": eye.R, "K": eye.K, "alpha_deg": np.degrees(eye.alpha), "beta_deg": np.degrees(eye.beta),
            "fitted": {c: {"R": v[1].user.R, "K": v[1].user.K, "alpha_deg": np.degrees(v[1].user.alpha),
                           "beta_deg": np.degrees(v[1].user.beta)} for c, v in calib_cache.items()}}
    return recs, subj


def stats(x):
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)])
    if len(x) == 0:
        return None
    return {"n": int(len(x)), "mean": float(x.mean()), "median": float(np.median(x)), "p95": float(np.percentile(x, 95))}


def summarise(recs):
    out = {}
    for cond, slip in conditions():
        rs = [r for r in recs if r["cond"] == cond and r["slip"] == slip]
        key = f"{cond}/slip{slip:g}mm"
        ok = [r for r in rs if r["ok"]]
        d = {"frames": len(rs), "valid": len(ok)}
        for m in METHODS:
            d[m] = stats([r.get(f"err_{m}") for r in ok])
        d["pupil_px"] = stats([r.get("pupil_px_err") for r in rs])
        d["glint_px"] = stats([e for r in rs for e in r.get("glint_px_err", [])])
        d["cornea_mm"] = stats([r.get("cornea_mm_err") for r in ok])
        out[key] = d
    return out


def to_markdown(summary, meta):
    lines = [f"Synthetic IR eye images, {meta['subjects']} subjects x {meta['test_per_condition']} held-out targets per "
             f"condition; {meta['machine']}; {meta['date']}.", ""]
    lines.append("| condition | valid | PCCR poly mean / med / p95 (deg) | model, kappa-cal | model, kappa+R+K cal | model, no user cal | pupil err px (med) | glint err px (med) |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for k, d in summary.items():
        def f(m):
            s = d[m]
            return "-" if s is None else f"{s['mean']:.2f} / {s['median']:.2f} / {s['p95']:.2f}"
        pp = "-" if d["pupil_px"] is None else "%.2f" % d["pupil_px"]["median"]
        gp = "-" if d["glint_px"] is None else "%.2f" % d["glint_px"]["median"]
        lines.append(f"| {k} | {d['valid']}/{d['frames']} | {f('poly')} | {f('model_kappa')} | {f('model_full')} | {f('model_uncal')} | "
                     f"{pp} | {gp} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", type=int, default=8)
    ap.add_argument("--test", type=int, default=40)
    ap.add_argument("--supersample", type=int, default=2)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=min(6, os.cpu_count() or 1))
    ap.add_argument("--out", default="results")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    jobs = [(s, a.test, a.supersample, a.seed) for s in range(a.subjects)]
    if a.workers > 1:
        with Pool(a.workers) as p:
            res = p.map(run_subject, jobs)
    else:
        res = [run_subject(j) for j in jobs]
    recs = [r for rr, _ in res for r in rr]
    subjects = [s for _, s in res]
    meta = {"subjects": a.subjects, "test_per_condition": a.test, "supersample": a.supersample, "seed": a.seed,
            "machine": f"{platform.processor() or platform.machine()}, Python {platform.python_version()}, numpy {np.__version__}",
            "date": time.strftime("%Y-%m-%d"), "wall_s": time.time() - t0, "noise": NOISE, "slips_mm": SLIPS}
    summary = summarise(recs)
    ms = [r["ms_features"] for r in recs if r["ok"]]
    mm = [r["ms_model"] for r in recs if "ms_model" in r]
    meta["timing_ms"] = {"features_median": float(np.median(ms)), "model_median": float(np.median(mm)), "note": "single CPU core, while other jobs ran"}
    os.makedirs(a.out, exist_ok=True)
    if not a.quick:
        with open(os.path.join(a.out, "gaze_accuracy.json"), "w") as fh:
            json.dump({"meta": meta, "summary": summary, "subjects": subjects, "frames": recs}, fh, indent=1)
        with open(os.path.join(a.out, "gaze_accuracy.md"), "w") as fh:
            fh.write(to_markdown(summary, meta))
    print(to_markdown(summary, meta))
    print(json.dumps(meta["timing_ms"]), f"wall {meta['wall_s']:.0f}s")


if __name__ == "__main__":
    main()
