# pros_workflow

An active-perception grasping system deployed in a Unity/ROS2 environment. A local Commander maintains task state with LangGraph and makes high-level decisions with a local Ollama VLM. It drives the local ROS2 navigation/control stack and the A2A inference services on an RTX 3090 to complete: find object → navigate → grasp-pose generation → base approach and arm/gripper finish.

![System architecture](docs/system_architecture.png)

## Code layout

| Path | Contents |
|---|---|
| `src/` | ROS2 packages: `workflow_bringup` (runtime launch aggregator), `nav_goal_bridge_pkg` (Nav2 + goal bridge), `car_control_pkg`, `arm_control_pkg`, `action_interface`, `robot_description`. |
| `workflow/` | The non-ROS Commander: `web_main.py`/`main.py` entry points, `commander/` (LangGraph orchestrator, flows, brain, state/contracts, camera/nav/perception/storage/web), `agents/` (A2A clients and the car-approach runtime), `config/` (cameras/objects/runtime YAML). |
| `3090server/pros_workflow/` | RTX 3090 A2A inference services (item-info, grasp). See its [README](3090server/pros_workflow/README.md). |
| `docker/` | Dockerfile used to build the workflow image. |
| `scripts/` | In-container shell helpers (`env.sh` defines `run`/`web`) and the ROS runtime launcher (`start.sh`). |
| `docs/` | Architecture diagrams and deployment spec. |

## Runtime flow

The Commander is a LangGraph state machine (entry `greeting_node`). The main grasp path chains: `find_node` → `get_item_info_no_sam3d_node` → `nav_move_node` → `observe_node` → `reason_node` (VLM decision) → `major_nav_node` / `car_grasp_node` → `car_approach_node`. Each turn the VLM emits a `BrainDecision` (`call_module` ∈ `major_nav_node` / `grasp_agent` / `car_approach_agent` / `DONE`).

![LangGraph flow](docs/langgraph_flow.png)

## Environment file (`.env`)

The Commander loads `.env` from the project root at startup. Required and common keys:

```env
# ROS
ROS_DOMAIN_ID=1

# Ollama (VLM brain)
OLLAMA_BASE_URL=http://<ollama-host>:11434
OLLAMA_MODEL=gemma3:12b
OLLAMA_CLASSIFIER_MODEL=gemma3:1b
OLLAMA_CHAT_MODEL=gemma3:12b

# RTX 3090 A2A inference services (IP:Port of the GPU servers)
EXTERNAL_IP=<gpu-host>
INF_GET_ITEM_INFO_NO_SAM3D_URL=http://${EXTERNAL_IP}:8006
INF_GRASP_URL=http://${EXTERNAL_IP}:8007
```

Set `OLLAMA_BASE_URL` to point at your Ollama server and `EXTERNAL_IP` to your RTX 3090 host; the two `INF_*` URLs derive from it. This same `.env` is auto-loaded by the 3090 servers (they read `EXTERNAL_IP` for their AgentCard `url`), so the GPU host is defined in one place.

## Quick start (local Commander)

Build the image:

```bash
cd pros_workflow
docker build -t pros_workflow_image:latest -f docker/Dockerfile .
```

Terminal 1 — start the ROS runtime:

```bash
cd pros_workflow
./run.sh            # enter the pros_workflow container
r                   # build the ROS workspace (PROS image alias for rebuild_colcon.rc)
scripts/start.sh    # launch Nav2 / car / arm / rosbridge
# then play Unity.
```

Terminal 2 — start the web workflow:

```bash
cd pros_workflow
./run.sh
web 8080            # start workflow/web_main.py
# then open http://localhost:8080.
```

By default the `logs/` folder (trace log + session data) is deleted when the web server stops or is interrupted. Add `--keep-logs` (`web 8080 --keep-logs`) to keep it.

`./run.sh` only enters Docker (it does not build the image); it publishes `8080:8080` (web) and `9090:9090` (rosbridge), mounts the repo at `/workspace/pros_workflow`, and uses the venv at `/opt/pros_workflow_venv`. Rebuild the image after changing the Dockerfile or `workflow/pyproject.toml`; run `r` after changing ROS code.

## RTX 3090 setup

The perception/grasp inference runs on a separate RTX 3090 machine as A2A servers. On that machine:

1. Copy the `3090server/pros_workflow/` folder to the GPU host.
2. Create one conda env and install the shared `requirements.txt` (per-service instructions in the [3090server README](3090server/pros_workflow/README.md)).
3. Drop the model weights (YOLO / SAM / DepthAnything / SAM3D / GraspGen checkpoints) into `3090server/pros_workflow/models/` — only `.gitkeep` is committed.
4. Set the GPU host in the project-root `pros_workflow/.env` (auto-loaded by the servers; written into the A2A AgentCard `url`):

   ```env
   EXTERNAL_IP=<gpu-host>
   INF_GET_ITEM_INFO_NO_SAM3D_URL=http://${EXTERNAL_IP}:8006
   INF_GRASP_URL=http://${EXTERNAL_IP}:8007
   ```

5. Launch the required services (no inline env prefix needed — the `.env` is loaded automatically):

   ```bash
   cd /path/to/pros_workflow/3090server/pros_workflow
   python -m get_item_info_agent_no_sam3d   # :8006 required
   python -m grasp_agent                    # :8007 required
   ```

6. Point the Commander `.env` at them: `INF_GET_ITEM_INFO_NO_SAM3D_URL=http://<gpu-host>:8006` and `INF_GRASP_URL=http://<gpu-host>:8007`.

See [3090server/pros_workflow/README.md](3090server/pros_workflow/README.md) for the full service list, ports, configs, and env overrides.
