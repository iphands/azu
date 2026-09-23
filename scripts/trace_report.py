#!/usr/bin/env python3
"""Turn azu_replay outputs into one self-contained HTML report (stdlib only).

usage: scripts/trace_report.py RUN_DIR [RUN_DIR ...] [-o report.html]

Each RUN_DIR is an azu_replay --out directory (summary.json, frames.csv,
trace.csv). Several runs are overlaid on the same charts, so an A/B (before and
after a fix, CPU vs CUDA, two volume settings) reads directly.

Charts: cumulative yaw about gravity, tilt error vs the accelerometer, tracking
grade per frame, inlier and overlap ratios, fraction of live points outside the
volume, fit RMS, model staleness, and per-stage times.
"""
import argparse
import csv
import html
import json
import os
import sys

COLORS = ["#2f6fdf", "#d9502b", "#2a9d4b", "#8a44c4", "#c49a12", "#1a9aa6"]


def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def num(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return default


def load_run(d):
    summary = {}
    if os.path.exists(os.path.join(d, "summary.json")):
        with open(os.path.join(d, "summary.json")) as f:
            summary = json.load(f)
    frames = read_csv(os.path.join(d, "frames.csv"))
    trace = read_csv(os.path.join(d, "trace.csv"))
    track = [r for r in trace if r.get("kind") == "track" and num(r, "quality", -1) >= 0]
    integ = [r for r in trace if r.get("kind") == "integ"]
    return {"name": os.path.basename(os.path.normpath(d)), "summary": summary,
            "frames": frames, "track": track, "integ": integ}


def line_chart(title, series, y_label, width=900, height=240, y_min=None, y_max=None):
    """series: list of (label, color, [(x, y), ...])"""
    pts = [p for _, _, s in series for p in s]
    if not pts:
        return f"<h3>{html.escape(title)}</h3><p>(no data)</p>"
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    x0, x1 = min(xs), max(xs)
    y0 = min(ys) if y_min is None else y_min
    y1 = max(ys) if y_max is None else y_max
    if x1 == x0:
        x1 = x0 + 1
    if y1 == y0:
        y1 = y0 + 1
    L, R, T, B = 60, 10, 10, 30
    W, H = width - L - R, height - T - B

    def sx(x):
        return L + (x - x0) / (x1 - x0) * W

    def sy(y):
        return T + H - (min(max(y, y0), y1) - y0) / (y1 - y0) * H

    out = [f'<h3>{html.escape(title)}</h3><svg width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}" class="chart">']
    for k in range(5):
        yv = y0 + (y1 - y0) * k / 4
        yy = sy(yv)
        out.append(f'<line x1="{L}" y1="{yy:.1f}" x2="{L + W}" y2="{yy:.1f}" class="grid"/>')
        out.append(f'<text x="{L - 6}" y="{yy + 4:.1f}" class="tick" text-anchor="end">{yv:.3g}</text>')
    for k in range(6):
        xv = x0 + (x1 - x0) * k / 5
        out.append(f'<text x="{sx(xv):.1f}" y="{height - 8}" class="tick" text-anchor="middle">{xv:.0f}</text>')
    out.append(f'<text x="12" y="{T + H / 2:.0f}" class="tick" transform="rotate(-90 12 {T + H / 2:.0f})" '
               f'text-anchor="middle">{html.escape(y_label)}</text>')
    for label, color, s in series:
        if not s:
            continue
        d = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in s)
        out.append(f'<polyline points="{d}" fill="none" stroke="{color}" stroke-width="1.5">'
                   f'<title>{html.escape(label)}</title></polyline>')
    out.append("</svg>")
    out.append('<div class="legend">' + " ".join(
        f'<span><i style="background:{c}"></i>{html.escape(l)}</span>' for l, c, _ in series) + "</div>")
    return "\n".join(out)


def grade_strip(run, color_map, width=900, height=26):
    frames = run["frames"]
    if not frames:
        return ""
    n = len(frames)
    w = max(1.0, width / n)
    cells = []
    for i, r in enumerate(frames):
        q = int(num(r, "quality", -1))
        lost = int(num(r, "state", 0)) == 2
        c = color_map["lost"] if lost else color_map.get(q, "#999")
        cells.append(f'<rect x="{i * width / n:.1f}" y="0" width="{w:.2f}" height="{height}" fill="{c}"/>')
    return (f'<div class="strip-label">{html.escape(run["name"])}</div>'
            f'<svg width="{width}" height="{height}" class="strip">' + "".join(cells) + "</svg>")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("-o", "--output", default="report.html")
    args = ap.parse_args()
    runs = [load_run(d) for d in args.runs]
    for i, r in enumerate(runs):
        r["color"] = COLORS[i % len(COLORS)]

    rows = []
    keys = ["backend", "volume", "resolution", "voxel_size", "frames", "good", "poor", "failed",
            "lost_frames", "first_lost", "yaw_tracked_deg", "tilt_mean_deg", "tilt_max_deg",
            "loop_trans_mm", "loop_rot_deg", "loop_reliable", "wall_seconds"]
    head = "<tr><th>run</th>" + "".join(f"<th>{k}</th>" for k in keys) + "</tr>"
    for r in runs:
        cells = []
        for k in keys:
            v = r["summary"].get(k, "")
            cells.append(f"<td>{v:.4g}</td>" if isinstance(v, float) else f"<td>{html.escape(str(v))}</td>")
        rows.append(f'<tr><td style="color:{r["color"]}">{html.escape(r["name"])}</td>' + "".join(cells) + "</tr>")

    def per_frame(field, source="frames", scale=1.0, fn=None):
        out = []
        for r in runs:
            data = r[source]
            if fn:
                pts = [(i, fn(row)) for i, row in enumerate(data)]
            else:
                pts = [(i, num(row, field) * scale) for i, row in enumerate(data)]
            out.append((r["name"], r["color"], pts))
        return out

    def ratio(a, b):
        return lambda row: num(row, a) / num(row, b) if num(row, b) > 0 else 0.0

    grade_colors = {-1: "#bbb", 0: "#2a9d4b", 1: "#e0b020", 2: "#d9502b", "lost": "#6b1010"}
    charts = [
        line_chart("Cumulative yaw about gravity (deg)", per_frame("yaw_total_deg"), "deg"),
        line_chart("Tilt error vs accelerometer (deg)", per_frame("tilt_err_deg"), "deg", y_min=0),
        line_chart("Inliers / valid live points (policy v1 ratio)",
                   per_frame(None, "track", fn=ratio("inliers", "valid_live")), "ratio", y_min=0, y_max=1),
        line_chart("Overlap: valid model / valid live", per_frame(None, "track", fn=ratio("valid_model", "valid_live")),
                   "ratio", y_min=0, y_max=1),
        line_chart("Live points outside the TSDF volume", per_frame("outside_volume", "track"), "fraction",
                   y_min=0, y_max=1),
        line_chart("Fit RMS (mm)", per_frame("rms_m", "track", scale=1000.0), "mm", y_min=0),
        line_chart("Model staleness (frames since the model was raycast)",
                   per_frame(None, "track", fn=lambda row: num(row, "frame_id") - num(row, "model_frame_id")),
                   "frames", y_min=0),
        line_chart("Tracking time per frame (ms)", per_frame("ms_track", "track"), "ms", y_min=0),
        line_chart("Integrate + raycast time (ms)",
                   per_frame(None, "integ", fn=lambda row: num(row, "ms_integrate") + num(row, "ms_raycast")),
                   "ms", y_min=0),
    ]
    strips = "".join(grade_strip(r, grade_colors) for r in runs)

    doc = f"""<!doctype html><html><head><meta charset="utf-8"><title>azu replay report</title>
<style>
:root {{ --bg:#fff; --fg:#1d1d1f; --grid:#e3e3e8; --muted:#666; }}
@media (prefers-color-scheme: dark) {{ :root {{ --bg:#16171a; --fg:#e8e8ea; --grid:#2c2e33; --muted:#9a9aa2; }} }}
body {{ background:var(--bg); color:var(--fg); font:14px/1.45 system-ui,sans-serif; margin:16px; max-width:960px; }}
table {{ border-collapse:collapse; font-size:12px; display:block; overflow-x:auto; }}
td,th {{ border:1px solid var(--grid); padding:3px 6px; text-align:right; white-space:nowrap; }}
.chart .grid {{ stroke:var(--grid); }} .tick {{ fill:var(--muted); font-size:11px; }}
.legend span {{ margin-right:14px; font-size:12px; }} .legend i {{ display:inline-block; width:10px; height:10px; margin-right:4px; }}
.strip-label {{ font-size:12px; color:var(--muted); margin-top:6px; }}
svg {{ max-width:100%; height:auto; }}
</style></head><body>
<h1>azu replay report</h1>
<table>{head}{''.join(rows)}</table>
<h3>Tracking grade per frame</h3>
<p style="font-size:12px"><span style="color:#2a9d4b">&#9632; good</span> &nbsp; <span style="color:#e0b020">&#9632; poor</span> &nbsp;
<span style="color:#d9502b">&#9632; failed</span> &nbsp; <span style="color:#6b1010">&#9632; lost</span></p>
{strips}
{''.join(charts)}
</body></html>"""
    with open(args.output, "w") as f:
        f.write(doc)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
