#!/usr/bin/env python3
"""Align a resource capture with the workflow's per-node timeline and report, for each
node, the CPU / GPU / power / memory state during its execution.

Preferred source is the trace log (`workflow/logs/trace_logger.jsonl`): every entry has
an **absolute** `unix_timestamp` (node end) + `decision/execution_latency_sec`, so each
node's window is [ts - dur, ts] with no drift. Entries are filtered to the capture's
epoch range, which auto-selects the run that overlaps the monitoring. Both the monitor
and the Commander stamp the same machine clock, so no clock sync is needed.

Fallback source is `result.json` (`started_at` + cumulative `node_latency_sec`), which
reconstructs windows from durations and can drift if nodes don't run back-to-back.

  # preferred — absolute timestamps:
  python bench/align_nodes.py bench/results/grasp_full.csv \
      --trace workflow/logs/trace_logger.jsonl --title "Per-node resources — AMD Strix Halo"

  # fallback:
  python bench/align_nodes.py bench/results/grasp_full.csv --result workflow/result/result.json

Outputs an annotated figure (node bands over the panels) and a per-node table
(printed + <out>.per_node.csv).
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


def _short(node: str) -> str:
    return node[:-5] if node.endswith("_node") else node


def _windows_from_trace(path, lo, hi):
    """[(short, start_epoch, end_epoch, dur), …] from trace_logger.jsonl, using absolute
    unix_timestamp (node end) and latency (node duration). Kept only if it overlaps
    [lo, hi] — the capture window — which selects the right run automatically."""
    out = []
    for line in open(path):
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
        except Exception:
            continue
        ts = r.get("unix_timestamp")
        node = r.get("agent_called")
        if ts is None or not node:
            continue
        dur = float(r.get("decision_latency_sec") or 0) + float(r.get("execution_latency_sec") or 0)
        start, end = ts - dur, ts
        if end < lo or start > hi:  # outside the capture — different run
            continue
        out.append((_short(node), start, end, dur))
    out.sort(key=lambda w: w[1])
    return out


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


def _windows_from_result(run):
    t0 = dt.datetime.fromisoformat(run["started_at"]).timestamp()
    lat = run.get("node_latency_sec") or {}
    seq = run.get("node_sequence") or list(lat.keys())
    out, cum = [], 0.0
    for node in seq:
        d = float(lat.get(node, 0.0) or 0.0)
        if d <= 0:
            continue
        out.append((_short(node), t0 + cum, t0 + cum + d, d))
        cum += d
    return out


def _agg(cols, name, s, e):
    ep = cols["epoch"]
    vals = [v for v, x in zip(cols[name], ep) if v is not None and x is not None and s <= x <= e]
    if not vals:
        return None, None
    return sum(vals) / len(vals), max(vals)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", help="monitor CSV (must have an epoch column)")
    ap.add_argument("--trace", default=None, help="workflow trace_logger.jsonl (absolute timestamps — preferred)")
    ap.add_argument("--result", default=None, help="workflow result.json (fallback)")
    ap.add_argument("--run", default=None, help="result.json run index or experiment_id (default: last)")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--title", default="Per-node resource usage — AMD APU")
    ap.add_argument("--min-node-sec", type=float, default=1.0,
                    help="hide nodes shorter than this from the chart + table (default 1.0)")
    args = ap.parse_args()
    if not args.trace and not args.result:
        raise SystemExit("give --trace (preferred) or --result")

    cols = _load(args.csv)
    epochs = [v for v in cols["epoch"] if v is not None] if "epoch" in cols else []
    if not epochs:
        raise SystemExit("this CSV has no epoch column — recapture with the current monitor_resources.py")
    epoch0, epochN = epochs[0], epochs[-1]

    if args.trace:
        windows = _windows_from_trace(args.trace, epoch0, epochN)
        src = "aligned by absolute trace timestamps"
    else:
        runs = json.loads(open(args.result).read())
        run = _pick_run(runs if isinstance(runs, list) else [runs], args.run)
        windows = _windows_from_result(run)
        src = f"run {run.get('experiment_id','?')} — reconstructed from durations"
    if not windows:
        raise SystemExit("no node windows overlap this capture — was the workflow running during it?")
    windows = [w for w in windows if w[3] >= args.min_node_sec]  # drop sub-second noise nodes
    if not windows:
        raise SystemExit(f"no nodes ≥ {args.min_node_sec}s")
    subtitle = f"{src} · {len(windows)} nodes ≥ {args.min_node_sec:g}s"

    base = os.path.splitext(args.csv)[0]
    out = args.out or f"{base}.per_node.png"

    plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "text.color": INK,
                         "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                         "grid.color": GRID, "grid.linewidth": 0.8,
                         "figure.facecolor": "white", "axes.facecolor": "white"})
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(11, 8.5), sharex=True,
                                        gridspec_kw={"hspace": 0.16})
    fig.subplots_adjust(top=0.85)
    fig.suptitle(args.title, x=0.5, y=0.995, fontsize=14, fontweight="bold")
    fig.text(0.5, 0.945, subtitle, ha="center", fontsize=9, color=MUTED)

    def _disp(n):
        return n.replace("get_item_info_no_sam3d", "get_item_info").replace("update_item_info_2", "update_item")

    # numbered legend under the subtitle — the chart carries only the numbers, so labels
    # never collide however many/short the nodes are.
    legend = "    ".join(f"{i}·{_disp(n)}" for i, (n, *_) in enumerate(windows, 1))
    fig.text(0.5, 0.925, legend, ha="center", fontsize=8, color=MUTED, wrap=True)

    def band(ax, numbers=False):
        for i, (name, s, e, d) in enumerate(windows, 1):
            rs, re = s - epoch0, e - epoch0
            ax.axvspan(rs, re, color=(GRID if i % 2 else "#f1f5f9"), alpha=0.7, lw=0)
            if numbers:
                ax.axvline(rs, color="#cbd5e1", lw=0.5)
                ax.text((rs + re) / 2, 102, str(i), ha="center", va="bottom",
                        fontsize=8.5, fontweight="bold", color=MUTED)

    band(ax1, numbers=True)
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

    rows = []
    for name, s, e, d in windows:
        gpu_m, gpu_p = _agg(cols, "gpu_util_pct", s, e)
        cpu_m, _ = _agg(cols, "cpu_util_pct", s, e)
        pw_m, pw_p = _agg(cols, "power_w", s, e)
        _, vr_p = _agg(cols, "vram_used_mb", s, e)
        _, ram_p = _agg(cols, "ram_used_mb", s, e)
        rows.append({"node": name, "dur_s": round(d, 1),
                     "gpu_mean_%": _r(gpu_m), "gpu_peak_%": _r(gpu_p), "cpu_mean_%": _r(cpu_m),
                     "power_mean_W": _r(pw_m), "power_peak_W": _r(pw_p),
                     "vram_peak_GB": _r((vr_p or 0) / 1024, 2), "ram_peak_GB": _r((ram_p or 0) / 1024, 2),
                     "energy_J": _r((pw_m or 0) * d, 0)})

    csv_out = f"{base}.per_node.csv"
    with open(csv_out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)

    hdr = (f"{'node':<26}{'dur':>6}{'GPU%avg':>9}{'GPU%pk':>8}{'CPU%avg':>9}{'W avg':>7}{'W pk':>7}"
           f"{'VRAMpk':>8}{'RAMpk':>8}{'energy':>9}")
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for r in rows:
        print(f"{r['node']:<26}{r['dur_s']:>6}{_s(r['gpu_mean_%']):>9}{_s(r['gpu_peak_%']):>8}"
              f"{_s(r['cpu_mean_%']):>9}{_s(r['power_mean_W']):>7}{_s(r['power_peak_W']):>7}"
              f"{_s(r['vram_peak_GB']):>8}{_s(r['ram_peak_GB']):>8}{_s(r['energy_J']):>9}")
    print(f"\nwrote {out}\nwrote {csv_out}")
    return 0


def _r(v, nd=1):
    return None if v is None else round(v, nd)


def _s(v):
    return "-" if v is None else str(v)


if __name__ == "__main__":
    sys.exit(main())
