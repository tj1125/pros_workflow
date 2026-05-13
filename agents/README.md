# agents

`agents/` 放的是 Commander 端可呼叫的 adapter。這些檔案大多不是獨立 server，而是 LangGraph 節點使用的本機 client/wrapper：有些透過 A2A 呼叫 RTX 3090，有些啟動 ROS2/PyBullet 子程序。

主流程以 `docs/langgraph_flow.png` 和 `commander/orchestrator.py` 為準。

## 主流程使用的 adapter

| 檔案/目錄 | 角色 | 上游節點 | 下游/依賴 |
|---|---|---|---|
| `get_item_info_agent_no_sam3d.py` | 現行物體資訊 A2A client。送多相機影像、bbox 與 `/world_position_data`，回傳目標中心與 ranked goal poses。 | `get_item_info_no_sam3d_node` | `INF_GET_ITEM_INFO_NO_SAM3D_URL`，fallback `INF_GET_ITEM_INFO_URL` |
| `grasp_agent.py` | GraspGen A2A client。送 `Camera_Car` RGBD 與 `object_id`，回傳 6-DoF grasp poses。 | `car_grasp_node` | `INF_GRASP_URL` |
| `car_approach_agent.py` | Commander wrapper。real mode 以 subprocess 執行 `agents.car_approach.subprocess_entry`。 | `car_approach_node` | ROS2、PyBullet/OMPL、`tools/car_control` |
| `car_approach/` | 底盤靠近與手臂/夾爪收尾 runtime。包含 base pose sampling、rule navigation、arm finish sequence。 | `car_approach_agent.py` | ROS2 action/topic、`tools/car_control` |
| `schemas.py` | 共用 Pydantic response schema。 | 多個 adapter | Pydantic |

## 保留但非主流程

| 檔案/目錄 | 狀態 | 說明 |
|---|---|---|
| `find_agent.py` | 保留 | A2A YOLO find client。現行 `find_node` 主要直接讀 `/world_position_data` 與 room cameras，但此 adapter 仍可支援獨立 YOLO 找物服務。 |
| `get_item_info_agent.py` | legacy | SAM3D/full 3D pipeline 的 item-info client。主流程改用 no-SAM3D 版本。 |
| `arm_approach_agent.py` | 備援 | 獨立手臂接近 wrapper；目前 prompt 與 graph 禁止 Brain 直接輸出 `arm_approach_agent`。 |
| `arm_approach/` | 備援 | 手臂 IK 與軌跡執行 runtime，可供未來拆回獨立手臂 approach 階段。 |

## 呼叫契約

所有 Commander-facing adapter 都盡量回傳同一類結構：

```python
{
    "result": {...},
    "success": True,
}
```

`commander/orchestrator.py` 會把結果整理為：

- `agent_result`
- `agent_success`
- `latest_nav_result`
- `latest_grasp_result`
- `latest_approach_result`
- `history_buffer`

## Real mode 注意事項

- `MOCK_MODE=false` 時，A2A client 會要求對應的 `INF_*_URL` 已設定且 server 可取得 AgentCard。
- `car_approach_agent.py` 會清理 uv/virtualenv 相關環境變數，改用 `ROS_PYTHON_BIN` 與 ROS setup script 啟動 subprocess。
- `car_approach/` 會寫 debug output 到自己的 `outputs/` 或 `/tmp`，部署前不應把大型輸出加入版本控制。
- `arm_approach_agent.py` 目前不是主流程入口，修改時請同步確認 `commander/prompts.py` 和 `commander/orchestrator.py` 是否要重新開啟路由。
