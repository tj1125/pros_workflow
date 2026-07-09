# 3090 Server — pros_workflow A2A Services

This folder holds the A2A Agent Servers that run on the RTX 3090. The Commander runs LangGraph, VLM decisions, and ROS2 control on the local machine; the 3090 only handles GPU/deep-learning perception and grasp inference. This document explains how to bring those services up on the 3090.

## Services

| Service | Port | Runtime status | Function |
|---|---:|---|---|
| `get_item_info_agent_no_sam3d` | `8006` (hardcoded) | **required** | Multi-view RGB + `/world_position_data` + SAM/geometry fusion (no SAM3D); outputs the target center and ranked goal poses. |
| `grasp_agent` | `8007` (hardcoded) | **required** | `Camera_Car` RGBD + YOLO + SAM + GraspGen; outputs 6-DoF grasp poses. |
| `get_item_info_agent` | `8008` (env `GET_ITEM_INFO_LEGACY_PORT`) | legacy | Full SAM3D pipeline (YOLO→SAM→Triangulation→DepthAnything→SAM3D→GraspGen). Kept for experiments that need mesh reconstruction. |

## Choose your platform

Pick the one matching your GPU — each is a single command that creates a conda env,
installs PyTorch + all deps, builds the GraspGen kernel, and verifies the GPU:

| GPU | Setup (one command) | Env | Deps file |
|---|---|---|---|
| **NVIDIA** — RTX 3090, CUDA 12.1 | `bash setup_nv.sh` | `pros_workflow` | [`requirements-nv.txt`](./requirements-nv.txt) |
| **AMD** — Radeon iGPU, ROCm (gfx1151 / gfx1150) | `bash setup_rocm.sh` | `pros_workflow` | [`requirements-rocm.txt`](./requirements-rocm.txt) |

After setup, the launch step and the Commander `.env` are the same for both — see
[Launch](#launch) and [Commander `.env` mapping](#commander-env-mapping) below.

## Quick start (NVIDIA CUDA)

Prerequisites: an NVIDIA GPU with the CUDA 12.1 toolkit (`nvcc` on PATH, needed to build
the kernel) and `conda`/Miniconda.

**1. One command to build the env**

```bash
bash setup_nv.sh
```

Creates the `pros_workflow` conda env (override with `ENV_NAME=...`/`PYVER=...`) and, in order:

1. `conda create -n pros_workflow python=3.11`
2. Installs `torch==2.5.1+cu121` (+ matching torchvision/torchaudio) from the CUDA index.
3. Installs the full perception + GraspGen stack from [`requirements-nv.txt`](./requirements-nv.txt).
   Unlike the ROCm build this **includes** the SAM3D packages (`xformers`/`spconv`/
   `torch_scatter`/`pytorch3d`/`moge`), so the legacy `:8008` service also works on CUDA.
4. Builds GraspGen's `pointnet2_ops` kernel with `nvcc`.
5. Verifies the GPU and that `furthest_point_sample` runs on it.

**2. Add model weights** — drop the checkpoints (YOLO / SAM / DepthAnything / SAM3D /
GraspGen) into `models/` (only `.gitkeep` is committed). See
[Model weights](#model-weights--download-sources--paths) for every link and its exact
relative path; the same paths are set in each service's `configs/*.yaml`.

Then jump to [Launch](#launch).

## Quick start (AMD ROCm — Radeon iGPU: gfx1151 Ryzen AI MAX+ / gfx1150 Ryzen AI 300)

The two required services (`:8006`, `:8007`) run on the AMD Radeon integrated GPU via
ROCm instead of CUDA. PyTorch's ROCm build exposes the AMD GPU through the same
`torch.cuda` API, so the application code and every `configs/*.yaml` `device: "cuda"`
work **unchanged** — the only work is swapping the CUDA-pinned packages for ROCm ones
and compiling GraspGen's custom kernel for the AMD GPU. The legacy SAM3D service
(`:8008`) is **not ported** (it needs `spconv`/`pytorch3d`/`xformers`, which have no
ROCm build here) and the robot does not use it.

Prerequisites: a ROCm-supported AMD iGPU (verified on gfx1151 = Radeon 8060S; gfx1150 =
Radeon 860M/890M uses the same recipe) and `conda`/Miniconda.

**1. One command to build the env**

```bash
bash setup_rocm.sh
```

This creates the `pros_workflow` conda env and does everything below. It **auto-detects the
GPU arch** (`rocminfo`), so the same command works on a gfx1151 or gfx1150 box; override
with `GFX=gfx1150 bash setup_rocm.sh` (or `ENV_NAME=...`, `PYVER=...`) if needed. What it
does, in order:

1. `conda create -n pros_workflow python=3.12`
2. Installs `torch`/`torchvision`/`torchaudio` + the pip-packaged ROCm SDK
   (`rocm[devel]`, which bundles the HIP headers/compiler) from AMD's per-GPU wheel
   index `https://repo.amd.com/rocm/whl/<gfx>/`, then `rocm-sdk init`.
3. Installs the perception + GraspGen deps from [`requirements-rocm.txt`](./requirements-rocm.txt)
   (no CUDA-only packages: `xformers`/`spconv`/`torch_scatter`/`pytorch3d`/`moge`/`sam3d`
   are skipped — only the unported `:8008` needs them).
4. Installs the `hipcc` wrapper (embedded in `setup_rocm.sh`) and builds GraspGen's
   `pointnet2_ops` kernel for the detected arch.
5. Verifies the GPU is visible and `furthest_point_sample` runs on it.

> **Why the hipcc wrapper?** GraspGen's kernel is HIPified and compiled by `hipcc`.
> On these boxes PyTorch's default invocation makes `hipcc` fall back to a stale system
> HIP header set under `/usr/include/hip`, which clashes with the SDK headers (duplicate
> `abort`/`__assert_fail`, unresolved `hipsolver`/fp8 types). The wrapper normalizes the
> invocation (single `-x hip` before the source, SDK include first, ROCm env vars
> cleared) so one consistent header set is used. `pointnet2_ops/setup.py` also drops the
> NVCC-only `-Xfatbin/-compress-all` flags on HIP builds.
>
> **Why no `torch_scatter`?** It is only imported by the PTv3 backbone, and the default
> **Robotiq 2F-140** gripper uses the `pointnet` backbone. Its ROCm build also fails on
> these iGPUs (wave64 vs 32-bit warp masks), so it is omitted unless you switch to a
> PTv3 checkpoint.

**2. Add model weights** — same as the CUDA path (see
[Model weights](#model-weights--download-sources--paths)). Then jump to [Launch](#launch).

## Launch

Same for both platforms — activate the `pros_workflow` env you built and start the two
required services. Each server advertises `http://$EXTERNAL_IP:<port>/` in its A2A
AgentCard, which is the address the Commander POSTs tasks back to. `EXTERNAL_IP` is read
from the project `.env` (see [Commander `.env` mapping](#commander-env-mapping)); a shell
`EXTERNAL_IP=...` still overrides it.

```bash
conda activate pros_workflow
python -m get_item_info_agent_no_sam3d   # :8006 required
python -m grasp_agent                    # :8007 required
# CUDA only, optional (legacy SAM3D pipeline):
python -m get_item_info_agent            # :8008
```

> **`EXTERNAL_IP` must be an address the Commander can actually reach.** If the Commander
> runs in a container, set it (in `.env`) to the host's LAN IP or the docker-bridge
> gateway, **not** `127.0.0.1` — that would resolve to the container itself and tasks
> would never arrive. Default when unset is `127.0.0.1` (fine only for same-host tests).

## Directory layout

```text
3090server/pros_workflow/
├── a2a_utils/                        # A2A success/error response helpers
├── models/                           # shared model-weights dir (only .gitkeep; add weights at deploy time)
├── tool/                             # shared inference code (see below)
│   ├── grasp/graspgen.py             # GraspGen bridge (point-cloud/collision filtering)
│   ├── graspgen_runtime/             # vendored GraspGen runtime + pointnet2_ops kernel (shared)
│   ├── vision/yolo.py                # YOLO detection helper
│   ├── vision/sam.py                 # SAM segmentation helper
│   └── runtime/memory.py             # CUDA memory release
├── get_item_info_agent_no_sam3d/     # current item-info server (:8006)
├── get_item_info_agent/              # legacy SAM3D item-info server (:8008)
└── grasp_agent/                      # GraspGen grasp server (:8007)
```

`tool/` holds all shared inference code and must not be duplicated. The GraspGen runtime
lives at `tool/graspgen_runtime` (the canonical source both `grasp_agent` and `no_sam3d`
point to); the two required services depend only on `tool/` and `models/`, **not** on the
legacy `get_item_info_agent/` directory. Shared model weights all live under `models/`.

## Service details

- All three servers bind `0.0.0.0` on their fixed ports; `get_item_info_agent`'s port can be overridden with `GET_ITEM_INFO_LEGACY_PORT`.
- `EXTERNAL_IP` is only written into the A2A AgentCard `url`; it is read from the project `.env` (default `127.0.0.1` if unset), and a shell `EXTERNAL_IP=...` overrides it. Set it to an address the Commander can reach.
- One conda env (`pros_workflow`) serves the services — built via `requirements-nv.txt` on CUDA or `requirements-rocm.txt` on ROCm — and they share the same GraspGen runtime under `tool/graspgen_runtime`. On ROCm only the two required services (`:8006`, `:8007`) are supported; the legacy `:8008` SAM3D service is CUDA-only.

## Commander `.env` mapping

The local Commander calls the 3090 services through these URLs (actual values live in the project-root `.env`):

```env
INF_GET_ITEM_INFO_NO_SAM3D_URL=http://<gpu-host>:8006
INF_GRASP_URL=http://<gpu-host>:8007
INF_GET_ITEM_INFO_URL=                                   # empty = legacy :8008 disabled
```

## Model weights — download sources & paths

`models/` keeps only `.gitkeep` in git; the real weights are added at deploy time. All paths below are **relative to this folder** (`3090server/pros_workflow/`) and match the values in each service's `configs/*.yaml`.

| Model | Used by | Relative path | Direct download |
|---|---|---|---|
| **YOLOv26** (custom) | `grasp_agent`, `get_item_info_agent` | `models/yolov26/yolov26_best.pt` | [Google Drive folder](https://drive.google.com/drive/folders/1pcecS0mu3Sr0DWqiGN2IbM6aMMz1WrnK?usp=share_link) |
| **SAM ViT-B** | all three services | `models/segmentation/sam_vit_b_01ec64.pth` | [https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth](https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth) (public) |
| **Depth Anything V2 (Base)** | `get_item_info_agent` (legacy) | `models/depth/depth_anything_v2_vitb.pth` | [https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth](https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth) (public) |
| **GraspGen — Robotiq 2F-140** (default gripper) | `grasp_agent`, `get_item_info_agent*` | `models/graspgen_checkpoints/graspgen_robotiq_2f_140_dis.pth`, `..._gen.pth`, `..._140.yml` | [nvidia/GraspGen](https://huggingface.co/nvidia/GraspGen) — **gated**, needs `hf download` (see below) |
| **GraspGen — Franka Panda** | optional (alt gripper) | `models/graspgen_checkpoints/graspgen_franka_panda_dis.pth`, `..._gen.pth`, `..._panda.yml` | [nvidia/GraspGen](https://huggingface.co/nvidia/GraspGen) — **gated** |
| **GraspGen — Suction 30 mm** | optional (alt gripper) | `models/graspgen_checkpoints/graspgen_single_suction_cup_30mm_dis.pth`, `..._gen.pth`, `..._30mm.yml` | [nvidia/GraspGen](https://huggingface.co/nvidia/GraspGen) — **gated** |
| **SAM 3D Objects** (7 `.ckpt` + `.yaml`) | `get_item_info_agent` (legacy SAM3D pipeline) | `models/sam3d/hf/` (`ss_generator.ckpt`, `slat_generator.ckpt`, `ss_decoder.ckpt`, `slat_decoder_gs.ckpt`, `slat_decoder_gs_4.ckpt`, `slat_decoder_mesh.ckpt`, `pipeline.yaml`, …) | [facebook/sam-3d-objects](https://huggingface.co/facebook/sam-3d-objects) — **gated**, needs `hf download` (see below) |

### Download commands

```bash
# run from this folder: 3090server/pros_workflow/

# --- Public direct downloads ---
# SAM ViT-B
wget -O models/segmentation/sam_vit_b_01ec64.pth \
  https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth
# Depth Anything V2 (Base)
wget -O models/depth/depth_anything_v2_vitb.pth \
  https://huggingface.co/depth-anything/Depth-Anything-V2-Base/resolve/main/depth_anything_v2_vitb.pth

# --- YOLOv26 (custom) ---
# Download yolov26_best.pt from the Google Drive folder and place it here:
#   models/yolov26/yolov26_best.pt
#   https://drive.google.com/drive/folders/1pcecS0mu3Sr0DWqiGN2IbM6aMMz1WrnK?usp=share_link

# --- Gated Hugging Face repos (accept the license, then `hf auth login`) ---
pip install 'huggingface-hub[cli]<1.0'

# GraspGen checkpoints -> models/graspgen_checkpoints/
hf download nvidia/GraspGen --repo-type model --local-dir models/graspgen_checkpoints

# SAM 3D Objects checkpoints -> models/sam3d/hf/
hf download facebook/sam-3d-objects --repo-type model --local-dir models/sam3d/download
mv models/sam3d/download/checkpoints models/sam3d/hf && rm -rf models/sam3d/download
```

> The `nvidia/GraspGen` and `facebook/sam-3d-objects` repos are **gated**: anonymous requests return HTTP 401, so there is no plain direct file URL. Request access on each repo page, authenticate with `hf auth login`, then run the `hf download` commands above. After downloading GraspGen, make sure the `.pth`/`.yml` files land directly in `models/graspgen_checkpoints/` (flatten any subfolders if the repo nests them).

Notes:
- The default gripper is **Robotiq 2F-140** — it is the only GraspGen set referenced by the configs (`gripper_config: models/graspgen_checkpoints/graspgen_robotiq_2f_140.yml`). The Franka Panda and Suction 30 mm sets are only needed if you switch grippers.
- Each GraspGen gripper needs both the discriminator (`*_dis.pth`) and generator (`*_gen.pth`) checkpoints plus its `.yml`; the `.yml` names the two `.pth` files.
- **SAM 3D Objects** is gated: request access on the Hugging Face repo, run `hf auth login`, then `hf download facebook/sam-3d-objects` and place the `checkpoints/` contents under `models/sam3d/hf/`. Only the legacy `:8008` SAM3D pipeline needs these; the two required services (`:8006`, `:8007`) do not.

## Environment variables

- `models/` keeps only `.gitkeep`; the real weights (YOLO / SAM / DepthAnything / SAM3D / GraspGen checkpoints) are added at deploy time (see the table above).
- `grasp_agent` model/camera paths can override the config via env vars: `GRASP_YOLO_WEIGHTS`, `GRASP_SAM_CHECKPOINT`, `GRASP_GRASPGEN_ROOT`, `GRASP_GRIPPER_CONFIG`, `GRASP_CAMERA_INTRINSICS`.
- `get_item_info_agent_no_sam3d` reads no env vars; all settings come from `configs/scene.default.yaml`.
- Each service's `configs/*.yaml` points to camera parameters, the SAM checkpoint, and the GraspGen runtime; the item-info services also reference the Nav2 keepout map.

## Per-service docs

- [get_item_info_agent_no_sam3d/README.md](./get_item_info_agent_no_sam3d/README.md)
- [grasp_agent/README.md](./grasp_agent/README.md)
- [get_item_info_agent/README.md](./get_item_info_agent/README.md) (legacy)
