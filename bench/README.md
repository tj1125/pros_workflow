# bench — resource & power monitoring (for the AMD performance report)

Two small tools that capture what AMD cares about when a workload runs on a Strix Halo
APU: **GPU/CPU utilisation, VRAM + unified (GTT) + system RAM, APU package power, and
energy per run** — then render it as one at-a-glance figure. Pure stdlib for sampling;
matplotlib only for the plot. NPU (XDNA2) is reported as 0 % because this pipeline does
not use it (roadmap item).

## Capture a full run (recommended)

`record.sh` records the **whole machine for the entire task** — system-wide, so it covers
the workflow, A2A servers and Ollama together — then plots automatically. Start it,
trigger your task, press Enter when the task finishes:

```bash
bash bench/record.sh grasp_full "Full grasp task — AMD Strix Halo"
# -> monitoring… run the task… press Enter to stop
# writes bench/results/grasp_full.{csv,summary.json,png}
```

### Other capture modes (`monitor_resources.py` directly)

```bash
# A) background start / stop by killing it (record.sh wraps this):
python bench/monitor_resources.py -o results/run1 &
MON=$!;  # ... run the task ...  ;  kill $MON

# B) wrap a single command — monitors until it exits:
python bench/monitor_resources.py -o results/run1 --command "python -m grasp_agent"

# C) fixed duration:
python bench/monitor_resources.py -o results/run1 --duration 120
```

Writes `<out>.csv` (one row per sample) and `<out>.summary.json` (peaks / means / energy).
Prints a peak/efficiency table on stop.

## Plot

```bash
python bench/plot_resources.py results/grasp_run1.csv \
    --title "Grasp task — AMD Strix Halo (Radeon 8060S)"
# -> results/grasp_run1.png : stacked panels (utilisation / power / memory) sharing the
#    time axis, with a headline strip of peak VRAM, avg power, energy, NPU=0%.
```

## Per-node breakdown (align to the workflow nodes)

To see the CPU/GPU/power/memory state **during each workflow node**, align the capture
with the Commander's `result.json`. Each node's window is reconstructed from `started_at`
+ cumulative `node_latency_sec`, matched to the samples by their `epoch` column (both use
the machine clock). Capture with the current `monitor_resources.py` (it records `epoch`)
and the workflow run must overlap the capture in time.

```bash
python bench/align_nodes.py results/grasp_run1.csv \
    --result workflow/result/result.json          # default: last run in the file
    --title "Per-node resources — AMD Strix Halo"
```

Outputs an annotated figure (node bands over the panels) and a per-node table
(`…​.per_node.csv` + printed), e.g.:

```
node                         dur  GPU%avg  GPU%pk  CPU%avg  W avg   W pk  VRAMpk   energy
reason                       8.0     99.9   100.0     16.6  130.9  150.1   21.24   1047.0
car_grasp                    6.0    100.0   100.0     14.6  129.3  149.1   21.24    776.0
```

so you can point at, say, `reason_node` and state its GPU %, average watts and energy.

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
