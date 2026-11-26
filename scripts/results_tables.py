"""Prints the README result tables (markdown) from results/*.json.

    python scripts/results_tables.py
"""
import json
import os

ROOT = os.path.join(os.path.dirname(__file__), "..")


def acc_table():
    with open(os.path.join(ROOT, "results", "gaze_accuracy.json")) as fh:
        d = json.load(fh)
    print("| condition | valid frames | PCCR + polynomial | 3D model, kappa+R+K cal | 3D model, kappa-only cal | 3D model, no user cal |")
    print("|---|---:|---:|---:|---:|---:|")
    for k, s in d["summary"].items():
        name = k.replace("/slip0mm", " noise, no slip").replace("low/slip", "low noise, slip ").replace("mm", " mm")

        def f(m):
            x = s[m]
            return "-" if x is None else f"{x['mean']:.2f} / {x['median']:.2f} / {x['p95']:.2f}"
        print(f"| {name} | {s['valid']}/{s['frames']} | {f('poly')} | **{f('model_full')}** | {f('model_kappa')} | {f('model_uncal')} |")
    print()
    print("| condition | pupil centre error px (median / p95) | glint error px (median / p95) | cornea centre error mm (median) |")
    print("|---|---:|---:|---:|")
    for k, s in d["summary"].items():
        p, g, c = s["pupil_px"], s["glint_px"], s["cornea_mm"]
        print(f"| {k} | {p['median']:.2f} / {p['p95']:.2f} | {g['median']:.2f} / {g['p95']:.2f} | {'-' if c is None else '%.2f' % c['median']} |")
    print("\nmeta:", json.dumps(d["meta"]["timing_ms"]), d["meta"]["date"], d["meta"]["machine"])


def bench_table():
    with open(os.path.join(ROOT, "results", "foveation_bench.json")) as fh:
        d = json.load(fh)
    rs = d["results"]
    m = d["meta"]
    full = next(r for r in rs if r["mode"] == "full")
    sw = next(r for r in rs if r["mode"].startswith("splatweb"))
    print(f"{m['width']}x{m['height']}, {m['fovYDeg']} deg, {m['ppd']:.1f} ppd, {m['frames']} frames, {m['gpu']}, {m['date']}")
    print(f"full: {full['total']:.2f} ms (pre {full['preprocess']:.2f}, sort {full['sort']:.2f}, raster {full['periphery']:.2f}); splatweb {sw['totalMs']:.2f} ms")
    print("| gaze | periphery | inner radius | shaded px | foveated ms | full-res ms (same frames) | saving (frames kept) | periphery / inset / composite ms | PSNR fovea / blend / periphery (dB) | SSIM periphery |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for r in rs:
        if r["mode"] != "foveated":
            continue
        g = "centre" if r["gazeNdc"] == [0, 0] else f"({r['gazeNdc'][0]}, {r['gazeNdc'][1]}) NDC"
        inner = f"{r['innerDeg']}°" + (" (acuity)" if r["tag"] == "acuity" else "")
        ps = r["psnr"]
        fmt = lambda v: "∞" if v == "inf" else f"{v:.1f}"  # noqa: E731
        print(f"| {g} | 1/{round(1 / r['peripheryScale'])} | {inner} | {100 * r['shadedFraction']:.0f} % | {r['total']:.2f} | {r['fullTotal']:.2f} | "
              f"**{100 * r['saving']:.0f} %** ({r.get('keptFrames', '?')}/{r.get('allFrames', '?')}) | {r['periphery']:.2f} / {r['inset']:.2f} / {r['composite']:.2f} | "
              f"{fmt(ps['fovea'])} / {fmt(ps['blend'])} / {fmt(ps['periphery'])} | {r['ssim']['periphery']:.3f} |")


if __name__ == "__main__":
    acc_table()
    print()
    bench_table()
