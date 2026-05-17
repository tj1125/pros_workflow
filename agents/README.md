# agents

`agents/` 放的是 Commander 端可呼叫的 adapter。這一層現在只保留主流程使用的 local wrapper / A2A client；舊實驗 adapter 放在 `agents/legacy/`。

主流程以 `commander/orchestrator.py` 為準，狀態契約以 `commander/contracts.py` 為準。

## 主流程使用的 adapter

| 檔案/目錄 | 角色 | 上游節點 | 下游/依賴 |
|---|---|---|---|
| `a2a_adapter.py` | A2A Message/Task/Artifact normalization helper。 | A2A clients | `a2a-sdk` |
| `get_item_info_agent_no_sam3d.py` | 現行物體資訊 A2A client。metadata 用 `DataPart`，room camera image 用 `FilePart`，回傳解析 `Task.artifacts` / direct `Message`。 | `get_item_info_no_sam3d_node` | `INF_GET_ITEM_INFO_NO_SAM3D_URL` |
| `grasp_agent.py` | GraspGen A2A client。metadata 用 `DataPart`，`Camera_Car` RGB/depth 用 `FilePart`。 | `car_grasp_node` | `INF_GRASP_URL` |
| `car_approach_agent.py` | Commander wrapper。real mode 以 subprocess 執行 `agents.car_approach.subprocess_entry`。 | `car_approach_node` | ROS2、PyBullet/OMPL、`tools/car_control` |
| `car_approach/` | 底盤靠近與手臂/夾爪收尾 runtime。 | `car_approach_agent.py` | ROS2 action/topic、`tools/car_control` |

## Legacy

| 檔案/目錄 | 狀態 | 說明 |
|---|---|---|
| `legacy/find_agent.py` | legacy | 舊 A2A YOLO find client。現行 `find_node` 直接讀 `/world_position_data`，並以 artifact refs 保存 room camera snapshots。 |
| `legacy/get_item_info_agent.py` | legacy | SAM3D/full 3D pipeline 的舊 item-info client。主流程使用 no-SAM3D 版本。 |
| `arm_approach_agent.py` / `arm_approach/` | 備援 | 目前主流程由 `car_approach_agent.py` 負責接近與手臂收尾。 |

## 呼叫契約

主流程 graph state 不存 raw image/base64/raw response。A2A client 可以用 local 變數讀 artifact bytes/base64 來組 `FilePart`，但 node 回傳 state 時只能寫 typed slices 和 `ArtifactRef`。

目前主要 state slices：`task`, `requested_object`, `selected_instance`, `world_position`, `room_cameras`, `item_info`, `navigation`, `observation`, `decision`, `grasp_result`, `approach_result`, `last_execution`, `session_summary`, `history_buffer`。
