# agents

`agents/` holds the adapters the Commander calls: A2A clients and a local subprocess runtime. The main flow is defined by `commander/orchestrator.py`, and the state contract by `commander/contracts.py`.

`flows/pick.py` lazy-imports these adapters inside node methods using the top-level `agents.*` path, so the workflow must run from the `workflow/` directory.

## Adapters used by the main flow

| File/dir | Role | Upstream node | Downstream/deps |
|---|---|---|---|
| `a2a_adapter.py` | A2A Message/Task/Artifact normalization helpers (`data_part`, `file_part`, `extract_result_payload`, `require_agent_card_modes`). | other A2A clients | `a2a-sdk` |
| `get_item_info_agent_no_sam3d.py` | `GetItemInfoNoSam3DAgent`: the current item-info A2A client. Metadata as `DataPart`, room-camera images as `FilePart`; parses `Task.artifacts` / a direct `Message`. | `get_item_info_no_sam3d_node` | `INF_GET_ITEM_INFO_NO_SAM3D_URL` |
| `grasp_agent.py` | `GraspAgent`: the GraspGen A2A client. `Camera_Car` RGB/depth as `FilePart`. | `car_grasp_node` | `INF_GRASP_URL` |
| `car_approach_agent.py` | `CarApproachAgent`: Commander wrapper. In real mode it runs `python -m agents.car_approach.subprocess_entry` as a subprocess (a mock path also exists). | `car_approach_node` | ROS2, PyBullet/OMPL, `src/` |
| `car_approach/` | Base-approach and arm/gripper-finish runtime (base_sampler, sample_logic, pipeline, runner, move_arm/move_car, arm_ik, joint_sequence, subprocess_entry, configs, debug). | `car_approach_agent.py` | ROS2 action/topic, `src/` |

Only `get_item_info_agent_no_sam3d`, `grasp_agent`, and `car_approach_agent` are actually imported by `flows/pick.py`.

## Call contract

The graph state never stores raw images/base64/raw responses. An A2A client may read artifact bytes/base64 into local variables to build a `FilePart`, but when a node returns state it may only write typed payloads and `ArtifactRef`s.

Main `CommanderState` fields (`commander/state.py`, a `TypedDict`): `context_id`, `task`, `requested_object`, `selected_instance`, `world_position`, `room_cameras`, `item_info`, `navigation`, `observation`, `decision`, `module_params`, `grasp_result`, `approach_result`, `last_execution`, `session_summary`, `history_buffer`. The payload models written into these fields are defined in `commander/contracts.py` (`ItemInfoResult`, `GraspResult`, `ApproachResult`, `NavigationState`, `ArtifactRef`, etc., all `extra="forbid"`).
