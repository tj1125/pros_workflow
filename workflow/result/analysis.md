# Performance Analysis — Autonomous Grasping Pipeline on AMD Strix Halo

One full task run (`pick doll`) of the complete system executed **entirely on a single
AMD Ryzen AI MAX+ (Strix Halo) APU** — "All-on-Halo". Resource + power captured with
`bench/monitor_resources.py` and aligned to the LangGraph nodes with `bench/align_nodes.py`
(absolute trace timestamps). Figure: `bench/results/grasp_full.per_node.png`.

> **Terminology — there is no discrete GPU.** This machine has exactly one GPU: the
> **integrated Radeon 8060S (iGPU)** built into the APU. Every mention of "GPU" below means
> that iGPU. Nothing here runs on a discrete graphics card — the whole stack runs on the
> iGPU, which is the point of the "All-on-Halo" result.

## What runs where (all on one APU)

| Component | Engine | Role |
|---|---|---|
| **Unity** | iGPU (render) | Simulates the scene/robot; renders continuously |
| **pros_cameraapi** (`run_unity.sh`, YOLO26-nano) | **CPU** | Bridges Unity cameras to ROS2; runs YOLO26-nano on CPU |
| **Foxglove** | CPU / iGPU (light) | Visualisation |
| **OtherServer A2A** (`:8006`/`:8007`) | iGPU (ROCm) | SAM / YOLO / GraspGen perception + 6-DoF grasp |
| **Commander** (`workflow/web_main.py`) + **Ollama** | CPU + iGPU | LangGraph orchestration; VLM (gemma3:12b) decisions |
| **NPU (XDNA2)** | — | **Not used (0%)** |

Everything shares **one iGPU and one unified-memory pool**.

## Baseline (always-on)

For most of the run the traces sit on a flat baseline — **~75 W, GPU-busy pegged at 100 %,
CPU ~13 %, VRAM ~22 GB**. That baseline is the sum of the always-on components:

- **Unity rendering** pegs `gpu_busy_percent` at 100 % and draws the ~75 W floor.
- **camera YOLO26-nano on CPU** is the ~13 % CPU floor.
- **A2A services + Ollama** keep their models resident → ~22 GB VRAM floor.

> **Key caveat: `gpu_busy_percent` is contaminated by Unity.** It reads 100 % whenever
> Unity renders, regardless of what the workflow is doing. The honest per-node signals are
> **power** and **VRAM**, not GPU-busy %.

## Per-node breakdown

| # | Node | dur (s) | GPU avg/pk % | CPU avg % | Power avg/pk (W) | VRAM pk (GB) | What it is |
|---|---|--:|--:|--:|--:|--:|---|
| 1 | task_classification | 1.6 | 24 / 40 | 15 | 68 / 84 | 22.4 | small classifier LLM (qwen2.5-coder:3b) |
| 2 | find | 6.6 | 18 / 53 | 18 | 67 / 99 | 22.4 | object find |
| 3 | get_item_info | 10.4 | 48 / 99 | 23 | **88 / 137** | **25.8** | **A2A perception (SAM/GraspGen on iGPU)** |
| 4 | nav_move | 63.9 | 100 / 100 | 13 | 76 / 95 | 23.0 | navigation (sim-bound) |
| 5 | observe | 1.7 | 100 / 100 | 19 | 78 / 79 | 23.0 | brief observation |
| 6 | **reason** | 29.7 | 100 / 100 | **60** | **126 / 143** | 23.0 | **VLM decision (Ollama gemma3:12b)** |
| 7 | car_grasp | 6.6 | 100 / 100 | 24 | 99 / 115 | **26.4** | **A2A grasp (GraspGen on iGPU)** |
| 8 | car_approach | 39.3 | 100 / 100 | 14 | 76 / 98 | 24.2 | approach (sim-bound) |
| 9 | nav_home | 50.7 | 100 / 100 | 14 | 74 / 81 | 24.2 | navigation home (sim-bound) |

**Only two things lift the traces off the baseline — and both show up in power, not GPU %:**

- **Node 6 `reason` (VLM)** — the single heaviest compute event: **126 W, CPU 60 %, ~30 s.**
  The gemma3:12b LLM is memory-bandwidth-bound on the iGPU (an integrated-GPU weakness),
  and Ollama also drives the CPU hard.
- **Nodes 3 & 7 (perception / grasp)** — power to 88 W / 99 W, **VRAM jumps to 25.8 / 26.4 GB**
  as SAM + GraspGen load and run on the iGPU.

The early ramp (nodes 1–3) is the only place GPU-busy % is meaningful — before Unity's
render load dominates you can see the real perception work; after node 4 the iGPU pegs at
100 % for the rest (Unity).

## Key findings

**1. ~73 % of the time is navigation, not computation.**
`nav_move (64 s) + car_approach (39 s) + nav_home (51 s) ≈ 154 s of ~210 s`. Those stretches
sit on the power baseline (74–76 W, CPU ~13 %) — the APU is largely **idle-waiting for the
robot to move in the Unity sim**, not compute-bound.

**2. Real AI compute is ~22 % of the run and the VLM dominates it.**
`get_item (10 s) + reason (30 s) + car_grasp (7 s) ≈ 47 s`. Of that, **`reason` (the VLM) is
by far the heaviest** — 126 W and 60 % CPU for 30 s. It is the pipeline's compute bottleneck.

**3. Unified memory has plenty of headroom.**
Peak **VRAM 26.4 GB / 64 GB (~41 %)**, **GTT ≈ 0.3 GB** (the iGPU never had to spill into
system RAM), RAM peak 24 GB. A 12 B LLM, perception models and Unity textures **coexist in
one shared pool with zero-copy CPU↔GPU** and ~60 % of the pool still free.

**4. Efficiency.** Whole capture: **avg 78 W, peak 143 W** on an integrated APU running the
entire autonomous-grasping stack — no discrete GPU.

## NPU (XDNA2) — unused (0 %): the clearest optimisation left

This chip has three engines — CPU, iGPU and a **~50-TOPS XDNA2 NPU** — but the run uses
only two. **The NPU sits at 0 %.** The NPU is built for **sustained, low-power inference of
quantised (INT8/INT4) neural networks**, especially **CNN vision models** — its edge is
**perf-per-watt**.

**Which of this project's inference tasks fit the NPU:**

| Task | Now on | NPU fit |
|---|---|---|
| **camera YOLO26-nano** | **CPU** | ✅✅ best target — small CNN, per-frame, quantisable |
| **A2A YOLO** (yolov26_best) | iGPU | ✅ CNN detector, INT8-friendly |
| SAM segmentation (ViT-B) | iGPU | 🟡 partial — ViT support still maturing |
| classifier LLM (gemma3:1b / qwen-3b) | iGPU/CPU | 🟡 possible — small LLMs are becoming NPU-viable |
| GraspGen (pointnet2 point-cloud) | iGPU | ❌ custom point-cloud ops, not NPU-portable |
| VLM gemma3:12b | iGPU | ❌ too large for the NPU |

**Why leaving it idle is a waste, and what moving the detectors to the NPU would gain
(directly visible in this run's data):**

1. **Frees the CPU** — the camera YOLO26-nano *is* the ~13 % CPU baseline; on the NPU that
   CPU returns to the Commander / ROS2 control loop.
2. **Frees the iGPU and cuts contention with Unity** — today Unity render + perception + VLM
   all fight over the one iGPU. Moving the detectors to the NPU takes them off the iGPU queue,
   so perception (node 3) no longer competes with Unity rendering and the heavy VLM has more
   iGPU to itself → lower perception latency.
3. **Much lower inference power** — NPU vision inference draws a fraction of the GPU/CPU
   power, improving the perf-per-watt story that matters most for a **mobile robot** (battery,
   thermals) and for AMD.
4. **It is the AMD differentiator** — a robotics demo that uses **CPU + iGPU + NPU together**
   is exactly the Ryzen AI story; NPU at 0 % is the most conspicuous gap.

The hardware path is already in place: the `amdxdna` driver is loaded and `/dev/accel/accel0`
exists. The remaining work is software — quantise YOLO to INT8 and run it through **ONNX
Runtime with the Ryzen AI (VitisAI) execution provider**.

## One-line summary (for AMD)

> The full autonomous-grasping pipeline runs end-to-end on a single Strix Halo APU at **avg
> 78 W**, using only **~41 % of the 64 GB unified memory** (zero GTT spill). AI compute is
> ~22 % of the run — dominated by the **VLM decision (126 W, 30 s)**, the iGPU's
> memory-bandwidth-bound weakness; the rest is navigation waiting. The **XDNA2 NPU is unused
> (0 %)** and is the clearest next win: offloading the **YOLO detectors** to it would free the
> CPU, cut iGPU contention with Unity, and cut inference power — turning this from a
> two-engine into a true three-engine deployment.
