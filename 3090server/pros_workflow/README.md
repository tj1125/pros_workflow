# 3090 Server — pros_workflow A2A Services

This folder holds the A2A Agent Servers that run on the RTX 3090. The Commander runs LangGraph, VLM decisions, and ROS2 control on the local machine; the 3090 only handles GPU/deep-learning perception and grasp inference. This document explains how to bring those services up on the 3090.

## Services

| Service | Port | Runtime status | Function |
|---|---:|---|---|
| `get_item_info_agent_no_sam3d` | `8006` (hardcoded) | **required** | Multi-view RGB + `/world_position_data` + SAM/geometry fusion (no SAM3D); outputs the target center and ranked goal poses. |
| `grasp_agent` | `8007` (hardcoded) | **required** | `Camera_Car` RGBD + YOLO + SAM + GraspGen; outputs 6-DoF grasp poses. |
| `get_item_info_agent` | `8008` (env `GET_ITEM_INFO_LEGACY_PORT`) | legacy | Full SAM3D pipeline (YOLO→SAM→Triangulation→DepthAnything→SAM3D→GraspGen). Kept for experiments that need mesh reconstruction. |

## Directory layout

```text
3090server/pros_workflow/
├── a2a_utils/                        # A2A success/error response helpers
├── models/                           # shared model-weights dir (only .gitkeep; add weights at deploy time)
├── tool/                             # shared inference helpers (see below)
│   ├── grasp/graspgen.py             # GraspGen runtime + point-cloud/collision filtering
│   ├── vision/yolo.py                # YOLO detection helper
│   ├── vision/sam.py                 # SAM segmentation helper
│   └── runtime/memory.py             # CUDA memory release
├── get_item_info_agent_no_sam3d/     # current item-info server (:8006)
├── get_item_info_agent/              # legacy SAM3D item-info server (:8008)
└── grasp_agent/                      # GraspGen grasp server (:8007)
```

`tool/` is the runtime shared by the three servers and should not be duplicated. The GraspGen runtime uses `get_item_info_agent/vendor/graspgen_runtime` as the canonical source; the configs of both `grasp_agent` and `no_sam3d` point to it.

## Running the services

On the 3090 machine, enter this folder and launch each server as a module:

```bash
cd /path/to/pros_workflow/3090server/pros_workflow

python -m get_item_info_agent_no_sam3d   # :8006 required
python -m grasp_agent                    # :8007 required
python -m get_item_info_agent            # :8008 legacy (only when needed)
```

`EXTERNAL_IP` is written into the A2A AgentCard `url`; it defaults to `140.116.82.226` in code. Set it explicitly when deploying to another machine:

```bash
EXTERNAL_IP=<gpu-host> python -m grasp_agent
```

Each server binds `0.0.0.0` on the port shown above (`get_item_info_agent` can be overridden with `GET_ITEM_INFO_LEGACY_PORT`).

## Commander `.env` mapping

The local Commander calls the 3090 services through these URLs (actual values live in the project-root `.env`):

```env
INF_GET_ITEM_INFO_NO_SAM3D_URL=http://<gpu-host>:8006
INF_GRASP_URL=http://<gpu-host>:8007
INF_GET_ITEM_INFO_URL=                                   # empty = legacy :8008 disabled
```

## Models and environment variables

- `models/` keeps only `.gitkeep`; the real weights (YOLO / SAM / DepthAnything / SAM3D / GraspGen checkpoints) are added at deploy time.
- `grasp_agent` model/camera paths can override the config via env vars: `GRASP_YOLO_WEIGHTS`, `GRASP_SAM_CHECKPOINT`, `GRASP_GRASPGEN_ROOT`, `GRASP_GRIPPER_CONFIG`, `GRASP_CAMERA_INTRINSICS`.
- `get_item_info_agent_no_sam3d` reads no env vars; all settings come from `configs/scene.default.yaml`.
- Each service's `configs/*.yaml` points to camera parameters, the SAM checkpoint, and the GraspGen runtime; the item-info services also reference the Nav2 keepout map.

## Per-service docs

- [get_item_info_agent_no_sam3d/README.md](./get_item_info_agent_no_sam3d/README.md)
- [grasp_agent/README.md](./grasp_agent/README.md)
- [get_item_info_agent/README.md](./get_item_info_agent/README.md) (legacy)
