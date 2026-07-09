# 3090 Server — pros_workflow A2A Services

This folder holds the A2A Agent Servers that run on the RTX 3090. The Commander runs LangGraph, VLM decisions, and ROS2 control on the local machine; the 3090 only handles GPU/deep-learning perception and grasp inference. This document explains how to bring those services up on the 3090.

## Services

| Service | Port | Runtime status | Function |
|---|---:|---|---|
| `get_item_info_agent_no_sam3d` | `8006` (hardcoded) | **required** | Multi-view RGB + `/world_position_data` + SAM/geometry fusion (no SAM3D); outputs the target center and ranked goal poses. |
| `grasp_agent` | `8007` (hardcoded) | **required** | `Camera_Car` RGBD + YOLO + SAM + GraspGen; outputs 6-DoF grasp poses. |
| `get_item_info_agent` | `8008` (env `GET_ITEM_INFO_LEGACY_PORT`) | legacy | Full SAM3D pipeline (YOLO→SAM→Triangulation→DepthAnything→SAM3D→GraspGen). Kept for experiments that need mesh reconstruction. |

## Quick start

Prerequisites: an RTX 3090 (CUDA 12.1) with `conda` installed. The three services share one root `requirements.txt` and one GraspGen runtime under `tool/`, so **a single conda env runs all of them** and none of them depends on another service's folder.

**1. Enter the folder**

```bash
cd /path/to/pros_workflow/3090server/pros_workflow
```

**2. Create the conda env and install deps**

```bash
conda create -n pros_workflow python=3.11 -y
conda activate pros_workflow

# PyTorch (CUDA 12.1) — install first, with the CUDA index
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# Shared perception + GraspGen stack (YOLO / SAM / DepthAnything / SAM3D / GraspGen)
pip install -r requirements.txt

# GraspGen CUDA ops (editable, no build isolation)
pip install --no-build-isolation -e tool/graspgen_runtime/pointnet2_ops
```

**3. Add model weights**

Drop the checkpoints (YOLO / SAM / DepthAnything / SAM3D / GraspGen) into `models/` — only `.gitkeep` is committed. See [Model weights — download sources & paths](#model-weights--download-sources--paths) for every download link and its exact relative path; the same paths are set in each service's `configs/*.yaml`.

**4. Set the GPU host in the project-root `.env`**

The servers auto-load the unified `pros_workflow/.env` (the same project-root file the Commander uses — see [Server `.env`](#server-env)). On the GPU host, set `EXTERNAL_IP` to this machine's address; it goes into the A2A AgentCard `url` the Commander connects to:

```env
EXTERNAL_IP=<gpu-host>
INF_GET_ITEM_INFO_NO_SAM3D_URL=http://${EXTERNAL_IP}:8006
INF_GRASP_URL=http://${EXTERNAL_IP}:8007
```

**5. Launch the two required services**

Each service auto-loads the `.env` above, so no inline env prefix is needed:

```bash
python -m get_item_info_agent_no_sam3d   # :8006 required
python -m grasp_agent                    # :8007 required
# optional (legacy SAM3D pipeline):
python -m get_item_info_agent            # :8008
```

A command-line prefix still overrides the file for a one-off host, e.g. `EXTERNAL_IP=1.2.3.4 python -m grasp_agent`.

**6. Point the Commander at this host**

On the Commander machine, set the project-root `.env` (the same `INF_*` URLs as above):

```env
INF_GET_ITEM_INFO_NO_SAM3D_URL=http://<gpu-host>:8006
INF_GRASP_URL=http://<gpu-host>:8007
```

## Directory layout

```text
3090server/pros_workflow/
├── requirements.txt                  # shared deps for all three services
├── a2a_utils/                        # A2A success/error response helpers
├── models/                           # shared model-weights dir (only .gitkeep; add weights at deploy time)
├── tool/                             # shared inference helpers + vendored runtime (see below)
│   ├── grasp/graspgen.py             # GraspGen wrapper + point-cloud/collision filtering
│   ├── graspgen_runtime/             # vendored GraspGen runtime (grasp_gen + pointnet2 CUDA ops), shared by all services
│   ├── vision/yolo.py                # YOLO detection helper
│   ├── vision/sam.py                 # SAM segmentation helper
│   ├── runtime/env.py                # zero-dependency .env loader
│   └── runtime/memory.py             # CUDA memory release
├── get_item_info_agent_no_sam3d/     # current item-info server (:8006)
├── get_item_info_agent/              # legacy SAM3D item-info server (:8008)
└── grasp_agent/                      # GraspGen grasp server (:8007)
```

`tool/` is the runtime shared by the three servers and should not be duplicated. The GraspGen runtime lives at `tool/graspgen_runtime` (canonical source); the configs of `grasp_agent`, `no_sam3d`, and the legacy server all point there, so no service depends on another's folder.

## Service details

- All three servers bind `0.0.0.0` on their fixed ports; `get_item_info_agent`'s port can be overridden with `GET_ITEM_INFO_LEGACY_PORT`.
- `EXTERNAL_IP` is only written into the A2A AgentCard `url` (defaults to `140.116.82.226` in code); set it in `.env` (or as a command-line prefix) so the Commander can reach the service.
- One conda env serves all three services — they share the root `requirements.txt` and the same GraspGen runtime under `tool/graspgen_runtime`.

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

## Server `.env`

Each service auto-loads the unified **project-root `pros_workflow/.env`** at startup via `tool/runtime/env.py` — the same file the Commander uses, so the GPU host is defined once instead of prefixing every launch command. The loader walks up from this folder to the repo root to find it (a standalone copy of `3090server/pros_workflow/` can instead drop its own `.env` here). Values already set in the real environment win, so a `EXTERNAL_IP=... python -m ...` prefix still overrides the file (useful for a one-off host). `${VAR}` references inside the file are expanded.

```env
EXTERNAL_IP=140.116.82.226
INF_GET_ITEM_INFO_NO_SAM3D_URL=http://${EXTERNAL_IP}:8006
INF_GRASP_URL=http://${EXTERNAL_IP}:8007
```

Only `EXTERNAL_IP` is consumed by the servers (for the AgentCard `url`); the two `INF_*` URLs are consumed by the Commander. Keeping all three in one `.env` means the GPU host is written once.

## Environment variables

- `models/` keeps only `.gitkeep`; the real weights (YOLO / SAM / DepthAnything / SAM3D / GraspGen checkpoints) are added at deploy time (see the table above).
- `EXTERNAL_IP` (from `.env` or a command-line prefix) sets the AgentCard `url` for all three servers.
- `grasp_agent` model/camera paths can override the config via env vars: `GRASP_YOLO_WEIGHTS`, `GRASP_SAM_CHECKPOINT`, `GRASP_GRASPGEN_ROOT`, `GRASP_GRIPPER_CONFIG`, `GRASP_CAMERA_INTRINSICS`.
- `get_item_info_agent_no_sam3d` reads only `EXTERNAL_IP`; all other settings come from `configs/scene.default.yaml`.
- Each service's `configs/*.yaml` points to camera parameters, the SAM checkpoint, and the GraspGen runtime (`tool/graspgen_runtime`); the item-info services also reference the Nav2 keepout map.

## Per-service docs

- [get_item_info_agent_no_sam3d/README.md](./get_item_info_agent_no_sam3d/README.md)
- [grasp_agent/README.md](./grasp_agent/README.md)
- [get_item_info_agent/README.md](./get_item_info_agent/README.md) (legacy)
