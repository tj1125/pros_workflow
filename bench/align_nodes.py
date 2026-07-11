#!/usr/bin/env python3
"""Align a resource capture with the workflow's per-node timeline and report, for each
node, the CPU / GPU / power / memory state during its execution.

It reconstructs each node's wall-clock window from the workflow result
(`started_at` + cumulative `node_latency_sec` in `node_sequence` order) and matches it
to the monitor samples via their `epoch` column, so the monitor and the Commander only
need to share the machine clock (they do).

  python bench/align_nodes.py bench/results/grasp_full.csv \
      --result workflow/result/result.json          # default: last run in the file
      [--run <index|experiment_id>] [-o out.png] [--title "..."]

Outputs an annotated figure (node bands over the utilisation/power/memory panels) and a
per-node table (printed + <out>.per_node.csv).
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_resources import C_CPU, C_GPU, C_GTT, C_PWR, C_RAM, C_VRAM, GRID, INK, MUTED, _load, _series  # noqa: E402


def _pick_run(runs, sel):
    if sel is None:
        return runs[-1]
    try:
        return runs[int(sel)]
    except (ValueError, IndexError):
        for r in runs:
            if r.get("experiment_id") == sel:
                return r
    raise SystemExit(f"run '{sel}' not found (have {len(runs)} runs)")


def _node_windows(run):
    """[(short_name, start_epoch, end_epoch, dur), …] from started_at + cumulative latency."""
    t0 = dt.datetime.fromisoformat(run["started_at"]).timestamp()
    lat = run.get("node_latency_sec") or {}
    seq = run.get("node_sequence") or list(lat.keys())
    out, cum = [], 0.0
    for node in seq:
        d = float(lat.get(node, 0.0) or 0.0)
        if d <= 0:
            continue
        short = node[:-5] if node.endswith("_node") else node
        out.append((short, t0 + cum, t0 + cum + d, d))
        cum += d
    return out


def _agg(cols, epoch0, name, s, e):
    """mean & peak of `name` for samples whose epoch is within [s, e]."""
    ep = cols["epoch"]
    vals = [v for v, x in zip(cols[name], ep) if v is not None and x is not None and s <= x <= e]
    if not vals:
        return None, None
    return sum(vals) / len(vals), max(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="monitor CSV (must have an epoch column)")
    ap.add_argument("--result", required=True, help="workflow result.json")
    ap.add_argument("--run", default=None, help="run index or experiment_id (default: last)")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default="Per-node resource usage — AMD APU")
    args = ap.parse_args()

    cols = _load(args.csv)
    if "epoch" not in cols or not any(v is not None for v in cols["epoch"]):
        raise SystemExit("this CSV has no epoch column — recapture with the current monitor_resources.py")
    runs = json.loads(open(args.result).read())
    run = _pick_run(runs if isinstance(runs, list) else [runs], args.run)
    windows = _node_windows(run)
    if not windows:
        raise SystemExit("no node windows (missing node_latency_sec/started_at in the run)")

    epoch0 = next(v for v in cols["epoch"] if v is not None)
    base = os.path.splitext(args.csv)[0]
    out = args.out or f"{base}.per_node.png"

    # --- figure: 3 shared-x panels + node bands ---
    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                                        gridspec_kw={"hspace": 0.16})
    fig.suptitle(args.title, x=0.5, y=0.99, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.955, f"run {run.get('experiment_id','?')} · \"{run.get('task_instruction','')}\" · "
             f"{run.get('wall_time_sec','?')}s · success={run.get('task_success')}",
             ha="center", fontsize=9, color=MUTED)

    def band(ax, top_labels=False):
        for i, (name, s, e, d) in enumerate(windows):
            rs, re = s - epoch0, e - epoch0
            ax.axvspan(rs, re, color=(GRID if i % 2 == 0 else "#f1f5f9"), alpha=0.7, lw=0)
            if top_labels:
                ax.axvline(rs, color="#cbd5e1", lw=0.6)
                ax.text((rs + re) / 2, 104, name, rotation=45, ha="left", va="bottom",
                        fontsize=7.5, color=MUTED)

    band(ax1, top_labels=True)
    for name, c, lab in (("gpu_util_pct", C_GPU, "GPU"), ("cpu_util_pct", C_CPU, "CPU")):
        t, y = _series(cols, name)
        ax1.plot(t, y, color=c, linewidth=1.8, label=lab)
    ax1.set_ylim(0, 100); ax1.set_ylabel("Utilisation (%)")
    ax1.legend(loc="upper right", frameon=False, ncol=2)

    band(ax2)
    t, y = _series(cols, "power_w")
    ax2.plot(t, y, color=C_PWR, linewidth=1.8, label="APU package")
    ax2.fill_between(t, y, color=C_PWR, alpha=0.08)
    ax2.set_ylabel("Power (W)")
    if y:
        ax2.set_ylim(0, max(y) * 1.25)
    ax2.legend(loc="upper right", frameon=False)

    band(ax3)
    for name, c, lab in (("ram_used_mb", C_RAM, "System RAM"), ("vram_used_mb", C_VRAM, "VRAM"),
                         ("gtt_used_mb", C_GTT, "GTT (unified)")):
        t, y = _series(cols, name)
        if y:
            ax3.plot(t, [v / 1024 for v in y], color=c, linewidth=1.8, label=lab)
    ax3.set_ylabel("Memory (GB)"); ax3.set_xlabel("Time (s)")
    ax3.legend(loc="upper right", frameon=False, ncol=3)

    for ax in (ax1, ax2, ax3):
        ax.margins(x=0)
        for spn in ("top", "right"):
            ax.spines[spn].set_visible(False)
    fig.savefig(out, dpi=150, bbox_inches="tight")

    # --- per-node table ---
    rows = []
    for name, s, e, d in windows:
        gpu_m, gpu_p = _agg(cols, epoch0, "gpu_util_pct", s, e)
        cpu_m, _ = _agg(cols, epoch0, "cpu_util_pct", s, e)
        pw_m, pw_p = _agg(cols, epoch0, "power_w", s, e)
        vr_m, vr_p = _agg(cols, epoch0, "vram_used_mb", s, e)
        energy = (pw_m or 0) * d
        rows.append({"node": name, "dur_s": round(d, 1),
                     "gpu_mean_%": _r(gpu_m), "gpu_peak_%": _r(gpu_p),
                     "cpu_mean_%": _r(cpu_m), "power_mean_W": _r(pw_m), "power_peak_W": _r(pw_p),
                     "vram_peak_GB": _r((vr_p or 0) / 1024, 2), "energy_J": _r(energy, 0)})

    csv_out = f"{base}.per_node.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    hdr = f"{'node':<26}{'dur':>6}{'GPU%avg':>9}{'GPU%pk':>8}{'CPU%avg':>9}{'W avg':>7}{'W pk':>7}{'VRAMpk':>8}{'energy':>9}"
    print("\n" + hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['node']:<26}{r['dur_s']:>6}{_s(r['gpu_mean_%']):>9}{_s(r['gpu_peak_%']):>8}"
              f"{_s(r['cpu_mean_%']):>9}{_s(r['power_mean_W']):>7}{_s(r['power_peak_W']):>7}"
              f"{_s(r['vram_peak_GB']):>8}{_s(r['energy_J']):>9}")
    print(f"\nwrote {out}\nwrote {csv_out}")
    return 0


def _r(v, nd=1):
    return None if v is None else round(v, nd)


def _s(v):
    return "-" if v is None else str(v)


if __name__ == "__main__":
    sys.exit(main())
