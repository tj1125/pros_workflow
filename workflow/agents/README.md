# agents

`agents/` 放的是 Commander 端可呼叫的 adapter：A2A client 與本機 subprocess runtime。主流程以 `commander/orchestrator.py` 為準，狀態契約以 `commander/contracts.py` 為準。

`flows/pick.py` 在 node 內以 top-level `agents.*` 路徑 lazy import 這些 adapter，所以 workflow 必須在 `workflow/` 目錄下執行。

## 主流程使用的 adapter

| 檔案/目錄 | 角色 | 上游節點 | 下游/依賴 |
|---|---|---|---|
| `a2a_adapter.py` | A2A Message/Task/Artifact 正規化 helper（`data_part`、`file_part`、`extract_result_payload`、`require_agent_card_modes`）。 | 其他 A2A client | `a2a-sdk` |
| `get_item_info_agent_no_sam3d.py` | `GetItemInfoNoSam3DAgent`：現行物體資訊 A2A client。metadata 用 `DataPart`、room camera image 用 `FilePart`，解析 `Task.artifacts` / direct `Message`。 | `get_item_info_no_sam3d_node` | `INF_GET_ITEM_INFO_NO_SAM3D_URL` |
| `grasp_agent.py` | `GraspAgent`：GraspGen A2A client。`Camera_Car` RGB/depth 用 `FilePart`。 | `car_grasp_node` | `INF_GRASP_URL` |
| `car_approach_agent.py` | `CarApproachAgent`：Commander wrapper。real mode 以 subprocess 執行 `python -m agents.car_approach.subprocess_entry`（另有 mock 路徑）。 | `car_approach_node` | ROS2、PyBullet/OMPL、`src/` |
| `car_approach/` | 底盤靠近與手臂/夾爪收尾 runtime（base_sampler、sample_logic、pipeline、runner、move_arm/move_car、arm_ik、joint_sequence、subprocess_entry、configs、debug）。 | `car_approach_agent.py` | ROS2 action/topic、`src/` |

只有 `get_item_info_agent_no_sam3d`、`grasp_agent`、`car_approach_agent` 三個模組會被 `flows/pick.py` 實際 import。

## 呼叫契約

主流程 graph state 不存 raw image/base64/raw response。A2A client 可以用 local 變數讀 artifact bytes/base64 來組 `FilePart`，但 node 回傳 state 時只能寫 typed payload 和 `ArtifactRef`。

`CommanderState`（`commander/state.py`，`TypedDict`）主要欄位：`context_id`、`task`、`requested_object`、`selected_instance`、`world_position`、`room_cameras`、`item_info`、`navigation`、`observation`、`decision`、`module_params`、`grasp_result`、`approach_result`、`last_execution`、`session_summary`、`history_buffer`。寫入這些欄位的 payload model 定義在 `commander/contracts.py`（`ItemInfoResult`、`GraspResult`、`ApproachResult`、`NavigationState`、`ArtifactRef` 等，皆 `extra="forbid"`）。
