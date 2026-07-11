#!/usr/bin/env python3
"""Render a resource-timeline figure from monitor_resources.py output — for the AMD
performance report.

  python bench/plot_resources.py run1.csv            # -> run1.png
  python bench/plot_resources.py run1.csv -o out.png --title "Grasp task on Strix Halo"

Design: stacked panels sharing one time axis (no dual-axis) —
  1) Utilisation %  : GPU + CPU
  2) Power (W)      : APU socket package power  (the perf/W story)
  3) Memory (GB)    : VRAM + GTT (unified) + system RAM
A headline strip on top carries the peak/efficiency numbers so the figure reads at a
glance. Reads <base>.summary.json when present for those numbers.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# CVD-safe hues, assigned by identity (never cycled). Text stays in ink tokens.
C_GPU = "#2563eb"   # blue
C_CPU = "#ea7317"   # orange  (blue/orange = the classic colourblind-safe pair)
C_PWR = "#dc2626"   # red
C_VRAM = "#0d9488"  # teal
C_GTT = "#7c3aed"   # violet
C_RAM = "#94a3b8"   # muted gray (system RAM — recessive; the GPU-side story is the colour)
INK = "#0f172a"
MUTED = "#64748b"
GRID = "#e2e8f0"


def _load(csv_path):
    cols = {}
    with open(csv_path) as f:
        r = csv.DictReader(f)
        for k in r.fieldnames:
            cols[k] = []
        for row in r:
            for k, v in row.items():
                if k == "wall":
                    cols[k].append(v)
                else:
                    cols[k].append(float(v) if v not in ("", None) else None)
    return cols


def _series(cols, name):
    t, y = [], []
    for ti, yi in zip(cols["t_sec"], cols[name]):
        if yi is not None:
            t.append(ti)
            y.append(yi)
    return t, y


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="samples CSV from monitor_resources.py")
    ap.add_argument("-o", "--out", default=None, help="output PNG (default: alongside CSV)")
    ap.add_argument("--title", default="Resource usage during task run — AMD APU", help="figure title")
    args = ap.parse_args()

    cols = _load(args.csv)
    base = os.path.splitext(args.csv)[0]
    out = args.out or f"{base}.png"
    summary = {}
    sp = f"{base}.summary.json"
    if os.path.exists(sp):
        with open(sp) as f:
            summary = json.load(f)

    plt.rcParams.update({
        "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8,
        "figure.facecolor": "white", "axes.facecolor": "white",
    })
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True,
                                        gridspec_kw={"hspace": 0.18})

    def peak(name):
        s = summary.get(name)
        return s.get("peak") if isinstance(s, dict) else None

    # headline strip
    bits = []
    if peak("gpu_util_pct") is not None: bits.append(f"peak GPU {peak('gpu_util_pct'):.0f}%")
    if peak("vram_used_mb") is not None: bits.append(f"peak VRAM {peak('vram_used_mb')/1024:.1f} GB")
    if peak("gtt_used_mb") is not None: bits.append(f"peak GTT {peak('gtt_used_mb')/1024:.1f} GB")
    if peak("ram_used_mb") is not None: bits.append(f"peak RAM {peak('ram_used_mb')/1024:.1f} GB")
    if summary.get("avg_power_w") is not None: bits.append(f"avg {summary['avg_power_w']:.0f} W")
    if summary.get("energy_joules") is not None: bits.append(f"energy {summary['energy_joules']:.0f} J")
    bits.append("NPU 0% (unused)")
    fig.suptitle(args.title, x=0.5, y=0.975, fontsize=14, fontweight="bold", ha="center")
    if bits:
        fig.text(0.5, 0.938, "   ·   ".join(bits), ha="center", fontsize=9.5, color=MUTED)

    # panel 1 — utilisation
    for name, c, lab in ((("gpu_util_pct"), C_GPU, "GPU"), ("cpu_util_pct", C_CPU, "CPU")):
        t, y = _series(cols, name)
        ax1.plot(t, y, color=c, linewidth=1.8, label=lab)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("Utilisation (%)")
    ax1.legend(loc="upper right", frameon=False, ncol=2)

    # panel 2 — power
    t, y = _series(cols, "power_w")
    ax2.plot(t, y, color=C_PWR, linewidth=1.8, label="APU package")
    ax2.fill_between(t, y, color=C_PWR, alpha=0.08)
    ax2.set_ylabel("Power (W)")
    if y:
        ax2.set_ylim(0, max(y) * 1.25)
    ax2.legend(loc="upper right", frameon=False)

    # panel 3 — memory (GB)
    for name, c, lab in (("ram_used_mb", C_RAM, "System RAM"),
                         ("vram_used_mb", C_VRAM, "VRAM (dedicated)"),
                         ("gtt_used_mb", C_GTT, "GTT (unified)")):
        t, y = _series(cols, name)
        if y:
            ax3.plot(t, [v / 1024 for v in y], color=c, linewidth=1.8, label=lab)
    ax3.set_ylabel("Memory (GB)")
    ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right", frameon=False, ncol=3)

    for ax in (ax1, ax2, ax3):
        ax.margins(x=0)
        for sp_ in ("top", "right"):
            ax.spines[sp_].set_visible(False)

    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
