#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Build a v1-vs-v2 benchmark report (markdown table + PDF with charts) from the
JSON produced by compare_v1_v2.py.

Reads results_v1v2/{v1,v2}/**/<bench>.json, matches states by axis values, and
compares GPU mean times. v2 is the candidate, v1 the baseline:
  %diff   = (t_v2 - t_v1) / t_v1 * 100   (negative => v2 faster)
  speedup = t_v1 / t_v2                   (>1 => v2 faster)

No third-party deps -- the PDF is written by hand -- so it runs even where
matplotlib/PyPI are unavailable.

Usage:
  python make_report.py [--results DIR] [--out report.pdf]
"""

import argparse
import datetime
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent
GPU_MEAN_TAG = "nv/cold/time/gpu/mean"
NOISE_TAG = "nv/cold/time/gpu/stdev/relative"


# --------------------------------------------------------------------------- #
# Data extraction
# --------------------------------------------------------------------------- #


def _summary(state: dict, tag: str):
    for s in state["summaries"]:
        if s["tag"] == tag:
            return float(s["data"][0]["value"])
    return None


def _clean_axis(name: str) -> str:
    return name.replace("{ct}", "").replace("{io}", "")


def _axis_str(axis_values: list) -> str:
    parts = []
    for a in axis_values:
        val = a["value"]
        # Render large pow2 element counts compactly.
        if a.get("type") == "int64":
            try:
                n = int(val)
                if n >= 1024 and (n & (n - 1)) == 0:
                    val = f"2^{n.bit_length() - 1}"
            except (TypeError, ValueError):
                pass
        parts.append(f"{_clean_axis(a['name'])}={val}")
    return " ".join(parts)


def _axis_key(axis_values: list) -> tuple:
    return tuple((a["name"], str(a["value"])) for a in axis_values)


def _states_by_key(path: Path) -> tuple[dict, str]:
    """Map (subbench_name, axis_key) -> state for every state in a result file."""
    data = json.load(open(path))
    device = ""
    if data.get("devices"):
        device = data["devices"][0].get("name", "")
    out = {}
    for bench in data["benchmarks"]:
        bname = bench["name"]
        for st in bench["states"]:
            if st.get("is_skipped"):
                continue
            out[(bname, _axis_key(st["axis_values"]))] = st
    return out, device


def collect_rows(results_dir: Path) -> tuple[list, str]:
    v1_dir = results_dir / "v1"
    v2_dir = results_dir / "v2"
    rows = []
    device = ""
    for v1_json in sorted(v1_dir.rglob("*.json")):
        rel = v1_json.relative_to(v1_dir)
        v2_json = v2_dir / rel
        if not v2_json.exists():
            continue
        bench = str(rel.with_suffix(""))
        v1_states, dev = _states_by_key(v1_json)
        v2_states, _ = _states_by_key(v2_json)
        device = device or dev
        for key, s1 in v1_states.items():
            s2 = v2_states.get(key)
            if s2 is None:
                continue
            t1 = _summary(s1, GPU_MEAN_TAG)
            t2 = _summary(s2, GPU_MEAN_TAG)
            if not t1 or not t2:
                continue
            subbench, axis_key = key
            label = bench if subbench == "base" else f"{bench} [{subbench}]"
            n1 = _summary(s1, NOISE_TAG) or 0.0
            n2 = _summary(s2, NOISE_TAG) or 0.0
            pct = (t2 - t1) / t1 * 100.0
            # Mirror nvbench-compare: within the larger noise band => "same".
            noise_band = max(n1, n2) * 100.0
            if abs(pct) <= noise_band:
                status = "same"
            elif pct > 0:
                status = "slower"
            else:
                status = "faster"
            rows.append(
                {
                    "label": label,
                    "config": _axis_str(s1["axis_values"]),
                    "t1_us": t1 * 1e6,
                    "t2_us": t2 * 1e6,
                    "n1": n1 * 100.0,
                    "n2": n2 * 100.0,
                    "pct": pct,
                    "speedup": t1 / t2,
                    "status": status,
                }
            )
    return rows, device


# --------------------------------------------------------------------------- #
# Markdown
# --------------------------------------------------------------------------- #

_STATUS_MD = {"faster": "🟢 faster", "slower": "🔴 slower", "same": "🔵 same"}


def markdown_table(rows: list, device: str) -> str:
    out = []
    out.append(
        f"**v1 vs v2 — {device}**  (baseline=v1, candidate=v2; "
        f"negative %diff = v2 faster)\n"
    )
    out.append("| Benchmark | Config | v1 (µs) | v2 (µs) | %diff | Speedup | Status |")
    out.append("|---|---|--:|--:|--:|--:|:--|")
    for r in rows:
        out.append(
            f"| {r['label']} | {r['config']} | {r['t1_us']:.2f} | {r['t2_us']:.2f} "
            f"| {r['pct']:+.1f}% | {r['speedup']:.3f}× | {_STATUS_MD[r['status']]} |"
        )
    # Aggregate summary.
    n = len(rows)
    faster = sum(1 for r in rows if r["status"] == "faster")
    slower = sum(1 for r in rows if r["status"] == "slower")
    same = sum(1 for r in rows if r["status"] == "same")
    gmean = _gmean([r["speedup"] for r in rows]) if rows else 0.0
    out.append("")
    out.append(
        f"**{n} configs** — 🟢 {faster} faster · 🔵 {same} same · "
        f"🔴 {slower} slower · geomean speedup (v1/v2) **{gmean:.3f}×**"
    )
    return "\n".join(out)


def _gmean(xs: list) -> float:
    import math

    xs = [x for x in xs if x > 0]
    if not xs:
        return 0.0
    return math.exp(sum(math.log(x) for x in xs) / len(xs))


# --------------------------------------------------------------------------- #
# Minimal hand-rolled PDF writer (vector graphics + Helvetica text, no deps)
# --------------------------------------------------------------------------- #


class PDF:
    def __init__(self, width=612, height=792):
        self.w, self.h = width, height
        self.pages = []  # list of content-stream strings
        self._buf = []

    def new_page(self):
        if self._buf:
            self.pages.append("\n".join(self._buf))
        self._buf = []

    def _esc(self, s: str) -> str:
        # Built-in Helvetica is Latin-1; drop anything outside it (e.g. emoji).
        s = s.encode("latin-1", "replace").decode("latin-1")
        return s.replace("\\", r"\\").replace("(", r"\(").replace(")", r"\)")

    def text(self, x, y, s, size=10, bold=False, color=(0, 0, 0)):
        font = "F2" if bold else "F1"
        r, g, b = color
        self._buf.append(
            f"BT /{font} {size} Tf {r} {g} {b} rg "
            f"{x:.2f} {y:.2f} Td ({self._esc(s)}) Tj ET"
        )

    def text_right(self, x, y, s, size=10, bold=False, color=(0, 0, 0)):
        self.text(x - len(s) * size * 0.5, y, s, size, bold, color)  # approx width

    def rect(self, x, y, w, h, color):
        r, g, b = color
        self._buf.append(f"{r} {g} {b} rg {x:.2f} {y:.2f} {w:.2f} {h:.2f} re f")

    def line(self, x1, y1, x2, y2, width=0.5, color=(0.6, 0.6, 0.6)):
        r, g, b = color
        self._buf.append(
            f"{r} {g} {b} RG {width} w {x1:.2f} {y1:.2f} m {x2:.2f} {y2:.2f} l S"
        )

    def save(self, path: Path):
        self.new_page()
        objs = []  # object bodies (without "N 0 obj"/"endobj")

        font1 = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>"
        font2 = "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold >>"

        # Object layout: 1=catalog, 2=pages, then per page: page + content.
        n_pages = len(self.pages)
        page_obj_ids = [3 + 2 * i for i in range(n_pages)]
        content_obj_ids = [4 + 2 * i for i in range(n_pages)]
        font1_id = 3 + 2 * n_pages
        font2_id = font1_id + 1

        objs.append((1, "<< /Type /Catalog /Pages 2 0 R >>"))
        kids = " ".join(f"{pid} 0 R" for pid in page_obj_ids)
        objs.append((2, f"<< /Type /Pages /Count {n_pages} /Kids [{kids}] >>"))
        for i, content in enumerate(self.pages):
            res = f"<< /Font << /F1 {font1_id} 0 R /F2 {font2_id} 0 R >> >>"
            objs.append(
                (
                    page_obj_ids[i],
                    f"<< /Type /Page /Parent 2 0 R "
                    f"/MediaBox [0 0 {self.w} {self.h}] /Resources {res} "
                    f"/Contents {content_obj_ids[i]} 0 R >>",
                )
            )
            stream = content.encode("latin-1", "replace")
            objs.append(
                (
                    content_obj_ids[i],
                    f"<< /Length {len(stream)} >>\nstream\n" + content + "\nendstream",
                )
            )
        objs.append((font1_id, font1))
        objs.append((font2_id, font2))

        objs.sort()
        out = b"%PDF-1.4\n"
        offsets = {}
        for num, body in objs:
            offsets[num] = len(out)
            out += f"{num} 0 obj\n{body}\nendobj\n".encode("latin-1", "replace")
        xref_pos = len(out)
        max_id = max(offsets)
        out += f"xref\n0 {max_id + 1}\n".encode()
        out += b"0000000000 65535 f \n"
        for i in range(1, max_id + 1):
            out += f"{offsets.get(i, 0):010d} 00000 n \n".encode()
        out += (
            f"trailer\n<< /Size {max_id + 1} /Root 1 0 R >>\n"
            f"startxref\n{xref_pos}\n%%EOF\n"
        ).encode()
        path.write_bytes(out)


# Colors
GREEN = (0.13, 0.55, 0.13)
RED = (0.80, 0.16, 0.16)
GRAY = (0.45, 0.45, 0.45)
DARK = (0.15, 0.15, 0.15)
LIGHT = (0.85, 0.85, 0.85)
HEADER = (0.10, 0.30, 0.55)
_COLOR = {"faster": GREEN, "slower": RED, "same": GRAY}


def _truncate(s: str, max_chars: int) -> str:
    return s if len(s) <= max_chars else s[: max_chars - 2] + ".."


def _nice_ceil(x: float) -> float:
    import math

    if x <= 0:
        return 1.0
    e = 10 ** math.floor(math.log10(x))
    for m in (1, 2, 2.5, 5, 10):
        if m * e >= x:
            return m * e
    return 10 * e


def _percentile(sorted_xs: list, q: float) -> float:
    if not sorted_xs:
        return 0.0
    i = min(len(sorted_xs) - 1, int(q * len(sorted_xs)))
    return sorted_xs[i]


def render_pdf(rows: list, device: str, out_path: Path) -> None:
    pdf = PDF()
    W, H = pdf.w, pdf.h
    margin = 45
    date = datetime.date.today().isoformat()

    # ---- Page 1: title + diverging bar chart of %diff ----
    pdf.text(
        margin, H - 55, "cuda.compute Benchmarks: v1 vs v2", 20, bold=True, color=HEADER
    )
    pdf.text(margin, H - 75, f"{device}   ·   {date}", 11, color=GRAY)

    faster = sum(1 for r in rows if r["status"] == "faster")
    slower = sum(1 for r in rows if r["status"] == "slower")
    same = sum(1 for r in rows if r["status"] == "same")
    gmean = _gmean([r["speedup"] for r in rows])
    pdf.text(
        margin,
        H - 95,
        f"{len(rows)} configs   ·   {faster} faster   ·   {same} same   ·   "
        f"{slower} slower   ·   geomean speedup (v1/v2) {gmean:.3f}x",
        11,
        bold=True,
        color=DARK,
    )
    # Cap the axis so one outlier (e.g. +476%) doesn't squash every other bar.
    abs_sorted = sorted(abs(r["pct"]) for r in rows)
    actual_max = abs_sorted[-1] if abs_sorted else 1.0
    cap = max(_nice_ceil(_percentile(abs_sorted, 0.90)), 20.0)
    clipped = actual_max > cap
    note = (
        "Bars show % change in GPU time, v2 vs v1. Left/green = v2 faster, "
        "right/red = v2 slower."
    )
    if clipped:
        note += f"  Axis clipped at +/-{cap:.0f}% (true value labeled)."
    pdf.text(margin, H - 112, note, 9, color=GRAY)

    # Chart geometry
    label_w = 165
    chart_x0 = margin + label_w
    chart_x1 = W - margin - 28  # leave room for value labels at the edge
    zero_x = (chart_x0 + chart_x1) / 2
    top = H - 135
    bottom = margin + 30
    row_h = (top - bottom) / max(len(rows), 1)
    row_h = min(row_h, 22)
    bar_h = row_h * 0.6

    half = chart_x1 - zero_x
    scale = half / cap

    for frac in (-1.0, -0.5, 0.0, 0.5, 1.0):
        gx = zero_x + frac * cap * scale
        pdf.line(gx, bottom, gx, top, 0.4, LIGHT if frac else GRAY)
        pdf.text(gx - 10, top + 4, f"{frac * cap:+.0f}%", 7, color=GRAY)

    y = top - row_h
    for r in rows:
        cy = y + (row_h - bar_h) / 2
        col = _COLOR[r["status"]]
        clamped = max(-cap, min(cap, r["pct"]))
        bar_len = clamped * scale
        if bar_len >= 0:
            pdf.rect(zero_x, cy, max(bar_len, 0.5), bar_h, col)
            pdf.text(
                zero_x + bar_len + 3,
                cy + bar_h * 0.2,
                f"{r['pct']:+.1f}%",
                7,
                color=DARK,
            )
        else:
            pdf.rect(zero_x + bar_len, cy, -bar_len, bar_h, col)
            pdf.text_right(
                zero_x + bar_len - 3,
                cy + bar_h * 0.2,
                f"{r['pct']:+.1f}%",
                7,
                color=DARK,
            )
        pdf.text(margin, cy + bar_h * 0.2, _truncate(r["label"], 33), 8, color=DARK)
        y -= row_h

    # ---- Following pages: data table ----
    _render_table(pdf, rows, device, date)
    pdf.save(out_path)


def _render_table(pdf: PDF, rows: list, device: str, date: str) -> None:
    W, H = pdf.w, pdf.h
    margin = 45
    # Columns: label, config, v1, v2, %diff, speedup, status
    cols = [
        ("Benchmark", margin, "l"),
        ("Config", margin + 135, "l"),
        ("v1 us", margin + 300, "r"),
        ("v2 us", margin + 360, "r"),
        ("%diff", margin + 415, "r"),
        ("speedup", margin + 470, "r"),
        ("status", margin + 478, "l"),
    ]
    row_h = 16
    top = H - 70

    def header():
        pdf.text(margin, H - 50, "Detailed results", 15, bold=True, color=HEADER)
        for name, x, align in cols:
            if align == "r":
                pdf.text_right(x, top, name, 8, bold=True, color=GRAY)
            else:
                pdf.text(x, top, name, 8, bold=True, color=GRAY)
        pdf.line(margin, top - 4, W - margin, top - 4, 0.6, GRAY)

    pdf.new_page()
    header()
    y = top - row_h
    for i, r in enumerate(rows):
        if y < margin + 20:
            pdf.new_page()
            header()
            y = top - row_h
        if i % 2 == 0:
            pdf.rect(margin - 3, y - 3, W - 2 * margin + 6, row_h, (0.96, 0.96, 0.96))
        vals = [
            (_truncate(r["label"], 24), cols[0][1], "l", DARK),
            (_truncate(r["config"], 28), cols[1][1], "l", GRAY),
            (f"{r['t1_us']:.2f}", cols[2][1], "r", DARK),
            (f"{r['t2_us']:.2f}", cols[3][1], "r", DARK),
            (f"{r['pct']:+.1f}%", cols[4][1], "r", _COLOR[r["status"]]),
            (f"{r['speedup']:.3f}x", cols[5][1], "r", DARK),
            (r["status"], cols[6][1], "l", _COLOR[r["status"]]),
        ]
        for s, x, align, color in vals:
            if align == "r":
                pdf.text_right(x, y, s, 8, color=color)
            else:
                pdf.text(x, y, s, 8, color=color)
        y -= row_h


# --------------------------------------------------------------------------- #


def main() -> None:
    ap = argparse.ArgumentParser(description="v1-vs-v2 benchmark report")
    ap.add_argument(
        "--results",
        default=str(SCRIPT_DIR / "results_v1v2"),
        help="results dir with v1/ and v2/ subdirs",
    )
    ap.add_argument(
        "--out",
        default=str(SCRIPT_DIR / "results_v1v2" / "report.pdf"),
        help="output PDF path",
    )
    args = ap.parse_args()

    results_dir = Path(args.results)
    rows, device = collect_rows(results_dir)
    if not rows:
        sys.exit(f"No comparable results found under {results_dir}")

    # Stable, readable order: by benchmark label.
    rows.sort(key=lambda r: r["label"])

    print(markdown_table(rows, device))
    out_path = Path(args.out)
    render_pdf(rows, device, out_path)
    print(f"\nPDF written to {out_path}")


if __name__ == "__main__":
    main()
