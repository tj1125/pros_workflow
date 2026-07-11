# bench — resource & power monitoring (for the AMD performance report)

Two small tools that capture what AMD cares about when a workload runs on a Strix Halo
APU: **GPU/CPU utilisation, VRAM + unified (GTT) + system RAM, APU package power, and
energy per run** — then render it as one at-a-glance figure. Pure stdlib for sampling;
matplotlib only for the plot. NPU (XDNA2) is reported as 0 % because this pipeline does
not use it (roadmap item).

## Capture a run

```bash
# A) monitor while you run the task in another terminal; Ctrl+C to stop:
python bench/monitor_resources.py -o results/grasp_run1

# B) wrap a command — monitors until it exits:
python bench/monitor_resources.py -o results/grasp_run1 \
    --command "python -m grasp_agent" --label "grasp :8007"

# C) fixed duration:
python bench/monitor_resources.py -o results/grasp_run1 --duration 120
```

Writes `grasp_run1.csv` (one row per sample) and `grasp_run1.summary.json` (peaks / means
/ energy). Prints a peak/efficiency table on stop.

## Plot

```bash
python bench/plot_resources.py results/grasp_run1.csv \
    --title "Grasp task — AMD Strix Halo (Radeon 8060S)"
# -> results/grasp_run1.png : stacked panels (utilisation / power / memory) sharing the
#    time axis, with a headline strip of peak VRAM, avg power, energy, NPU=0%.
```

## What to actually show AMD

Raw latency vs an RTX 3090 is the wrong headline (a small iGPU loses on raw speed). Lead
with AMD's differentiators instead:

1. **Runs end-to-end on one APU** — full perception → VLM → grasp pipeline on the iGPU +
   unified memory, no discrete GPU. (functional parity)
2. **Perf-per-watt / energy per task** — `avg_power_w` and `energy_joules` from the
   summary. This is where an APU beats a discrete GPU; make it a headline number.
3. **Unified memory** — the memory panel: a big LLM (gemma3:12b ≈ 8 GB) coexists with
   perception in one shared pool, zero-copy CPU↔GPU. Show peak VRAM + GTT + RAM together.
4. **Honest bottleneck + roadmap** — the VLM `reason_node` is memory-bandwidth-bound on
   the iGPU (slower than a discrete card); note it and the mitigations (smaller model, or
   moving inference to the **NPU**, currently 0 %). Owning the limitation reads as credible.

### Rigor
- **Run each scenario ≥ 5 times** and report mean ± std — a single run (n = 1) will not
  convince anyone. Capture one CSV per run (`-o run1 … run5`).
- Keep the CSVs — they are the raw evidence behind every number in the figure.
- Pair this with the existing task-latency comparison (`workflow/result/`), annotated with
  *which hardware each node ran on*.
