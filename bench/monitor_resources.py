#!/usr/bin/env python3
"""Sample GPU / CPU / memory / power on an AMD APU while a task runs — for the AMD
performance report. Pure stdlib (no deps); reads sysfs + /proc directly.

Usage
-----
  # Monitor while you run the task in another terminal; Ctrl+C to stop and write results:
  python bench/monitor_resources.py -o run1

  # Wrap a command — monitors until it exits:
  python bench/monitor_resources.py -o run1 --command "python -m grasp_agent"

  # Fixed duration (seconds):
  python bench/monitor_resources.py -o run1 --duration 120

Writes  <out>.csv  (one row per sample) and  <out>.summary.json  (peaks / means /
energy). Plot with:  python bench/plot_resources.py <out>.csv

Metrics (all best-effort; missing sources are recorded blank):
  gpu_util_pct  amdgpu gpu_busy_percent
  vram_used_mb  amdgpu mem_info_vram_used            (dedicated VRAM carveout)
  gtt_used_mb   amdgpu mem_info_gtt_used             (unified memory borrowed from RAM)
  power_w       amdgpu hwmon power1_average          (APU socket package power)
  cpu_util_pct  /proc/stat aggregate busy delta
  ram_used_mb   /proc/meminfo  MemTotal - MemAvailable
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time


def _find_amdgpu_card() -> str | None:
    for dev in sorted(glob.glob("/sys/class/drm/card*/device")):
        if os.path.exists(os.path.join(dev, "gpu_busy_percent")):
            return dev
    return None


def _find_power_file(card: str | None) -> str | None:
    if not card:
        return None
    for pat in ("power1_average", "power1_input"):
        hits = glob.glob(os.path.join(card, "hwmon", "hwmon*", pat))
        if hits:
            return hits[0]
    return None


def _read_int(path: str) -> int | None:
    try:
        with open(path) as f:
            return int(f.read().strip())
    except Exception:
        return None


def _cpu_busy_total() -> tuple[int, int] | None:
    """Return (busy, total) jiffies from /proc/stat's aggregate cpu line."""
    try:
        with open("/proc/stat") as f:
            parts = f.readline().split()
        vals = [int(x) for x in parts[1:]]
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
        total = sum(vals)
        return total - idle, total
    except Exception:
        return None


def _ram_used_mb() -> float | None:
    try:
        info = {}
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, v = line.partition(":")
                info[k] = int(v.split()[0])  # kB
        return (info["MemTotal"] - info["MemAvailable"]) / 1024.0
    except Exception:
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-o", "--out", default="run", help="output basename (writes <out>.csv / <out>.summary.json)")
    ap.add_argument("-i", "--interval", type=float, default=1.0, help="sample interval seconds (default 1.0)")
    ap.add_argument("--duration", type=float, default=None, help="stop after N seconds")
    ap.add_argument("--command", default=None, help="run this command and monitor until it exits")
    ap.add_argument("--label", default=None, help="free-text label stored in the summary")
    args = ap.parse_args()

    card = _find_amdgpu_card()
    power_file = _find_power_file(card)
    if card is None:
        print("WARN: no amdgpu card found (gpu_busy_percent); GPU columns will be blank.", file=sys.stderr)
    else:
        print(f"amdgpu card : {card}", file=sys.stderr)
        print(f"power source: {power_file or '(none — power will be blank)'}", file=sys.stderr)

    vram_total_mb = None
    if card:
        vt = _read_int(os.path.join(card, "mem_info_vram_total"))
        vram_total_mb = vt / 1024 / 1024 if vt else None

    proc = None
    if args.command:
        print(f"launching: {args.command}", file=sys.stderr)
        proc = subprocess.Popen(args.command, shell=True)

    stop = {"flag": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(flag=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(flag=True))

    csv_path = f"{args.out}.csv"
    # `epoch` (wall-clock seconds) lets a run be aligned to the workflow's per-node
    # timestamps (result.json started_at + node_latency) — see bench/align_nodes.py.
    cols = ["t_sec", "epoch", "wall", "gpu_util_pct", "vram_used_mb", "gtt_used_mb", "power_w", "cpu_util_pct", "ram_used_mb"]
    rows: list[list] = []

    prev_cpu = _cpu_busy_total()
    t0 = time.monotonic()
    energy_j = 0.0
    prev_t = t0
    prev_p = None

    print("sampling… (Ctrl+C to stop)", file=sys.stderr)
    while not stop["flag"]:
        now = time.monotonic()
        t = now - t0

        gpu = _read_int(os.path.join(card, "gpu_busy_percent")) if card else None
        vram = _read_int(os.path.join(card, "mem_info_vram_used")) if card else None
        gtt = _read_int(os.path.join(card, "mem_info_gtt_used")) if card else None
        vram_mb = vram / 1024 / 1024 if vram is not None else None
        gtt_mb = gtt / 1024 / 1024 if gtt is not None else None
        pw = _read_int(power_file) if power_file else None
        power_w = pw / 1_000_000 if pw is not None else None  # µW -> W

        cur_cpu = _cpu_busy_total()
        cpu_pct = None
        if prev_cpu and cur_cpu and cur_cpu[1] != prev_cpu[1]:
            cpu_pct = 100.0 * (cur_cpu[0] - prev_cpu[0]) / (cur_cpu[1] - prev_cpu[1])
        prev_cpu = cur_cpu

        ram_mb = _ram_used_mb()

        # integrate power (trapezoid) for energy
        if power_w is not None:
            if prev_p is not None:
                energy_j += (power_w + prev_p) / 2 * (now - prev_t)
            prev_p = power_w
            prev_t = now

        rows.append([round(t, 3), round(time.time(), 3), time.strftime("%H:%M:%S"), gpu,
                     _r(vram_mb), _r(gtt_mb), _r(power_w, 2), _r(cpu_pct, 1), _r(ram_mb)])

        if args.duration and t >= args.duration:
            break
        if proc is not None and proc.poll() is not None:
            break
        time.sleep(max(0.0, args.interval - (time.monotonic() - now)))

    if proc is not None and proc.poll() is None:
        proc.terminate()

    # write csv
    with open(csv_path, "w") as f:
        f.write(",".join(cols) + "\n")
        for r in rows:
            f.write(",".join("" if v is None else str(v) for v in r) + "\n")

    # summary
    def col(name):
        i = cols.index(name)
        return [r[i] for r in rows if r[i] is not None]

    def stat(name):
        vals = col(name)
        if not vals:
            return None
        return {"mean": round(sum(vals) / len(vals), 2), "peak": round(max(vals), 2)}

    dur = rows[-1][0] if rows else 0.0
    summary = {
        "label": args.label,
        "samples": len(rows),
        "duration_sec": round(dur, 1),
        "vram_total_mb": _r(vram_total_mb),
        "energy_joules": round(energy_j, 1),
        "energy_wh": round(energy_j / 3600, 3),
        "avg_power_w": round(energy_j / dur, 2) if dur else None,
        "gpu_util_pct": stat("gpu_util_pct"),
        "cpu_util_pct": stat("cpu_util_pct"),
        "vram_used_mb": stat("vram_used_mb"),
        "gtt_used_mb": stat("gtt_used_mb"),
        "ram_used_mb": stat("ram_used_mb"),
        "power_w": stat("power_w"),
        "npu_util_pct": {"mean": 0, "peak": 0, "note": "XDNA2 NPU not used by this pipeline"},
    }
    with open(f"{args.out}.summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # print table
    print(f"\n=== {args.out} — {summary['samples']} samples over {summary['duration_sec']}s ===", file=sys.stderr)
    _p("peak GPU util", summary["gpu_util_pct"], "%")
    _p("peak CPU util", summary["cpu_util_pct"], "%")
    _p("peak VRAM", summary["vram_used_mb"], "MB", total=vram_total_mb)
    _p("peak GTT (unified)", summary["gtt_used_mb"], "MB")
    _p("peak RAM", summary["ram_used_mb"], "MB")
    _p("avg / peak power", summary["power_w"], "W")
    print(f"  energy this run   : {summary['energy_joules']} J  ({summary['energy_wh']} Wh)  avg {summary['avg_power_w']} W", file=sys.stderr)
    print(f"  NPU util          : 0 % (not used — roadmap)", file=sys.stderr)
    print(f"\nwrote {csv_path} and {args.out}.summary.json", file=sys.stderr)
    print(f"plot: python bench/plot_resources.py {csv_path}", file=sys.stderr)
    return 0


def _r(v, nd=1):
    return None if v is None else round(v, nd)


def _p(label, s, unit, total=None):
    if not s:
        return
    extra = f" / {total/1024:.1f} GB" if (total and unit == "MB") else ""
    tail = f"{total:.0f}{unit}" if total else ""
    print(f"  {label:<18}: mean {s['mean']} {unit} | peak {s['peak']} {unit}{extra}", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
