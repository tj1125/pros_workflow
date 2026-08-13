# Experiment A2 — 消融實驗：ranked goal poses → single nearest goal pose

## 1. 目的

證明「**多候選停靠點 + 排序 + 失敗換點**」對觀察／抓取成功率的貢獻。

- **完整系統（基線）**：`get_item_info_no_sam3d_node` 會回傳多個 orientation-group
  候選停靠點（`group_ranking`），workflow 依排名導航；若 grasp/approach 失敗，
  `major_nav_node` 會換下一個 ranked goal pose 再試（見 `_should_force_major_nav_after_failed_attempt`）。
- **A2 消融**：只保留**距離目標最近的單一 goal pose**，丟掉其餘候選，且不換點。

兩個條件**只差這一個變因**：感知（item-info A2A 呼叫）完全不變，差別只在 workflow 是否
擁有多候選與排序能力。

## 2. 開關與設定（`.env`）

| 變數 | 預設 | 說明 |
|---|---|---|
| `ABLATION_SINGLE_NEAREST_GOAL` | `true`（A2） | `true`＝只用最近單一候選；`false`＝完整基線（做對照組）。 |
| `EXPERIMENT_RECORD_ENABLED` | `true` | 是否輸出 JSON 執行報告。 |
| `EXPERIMENT_A2_RESULT_DIR` | `result` | 報告輸出資料夾（單一 `result.json` 存放處）。 |
| `EXPERIMENT_SCENE_ID` | （空） | 場景／回合代號，會寫進報告，例如 `S3_occlusion_01`。 |
| `EXPERIMENT_ID_PREFIX` | `A2` | `experiment_id` 前綴（→ `A2_001`, `A2_002`, …）。 |

> 做對照組（完整系統）時，把 `ABLATION_SINGLE_NEAREST_GOAL=false`、`EXPERIMENT_ID_PREFIX=A2base`
> 即可，報告會自動標記 `"ablation.mode": "full_ranked_candidates"`。

## 3. 實作位置

| 檔案 | 角色 |
|---|---|
| `commander/ablation.py` | `single_nearest_goal_enabled()`＋`collapse_to_nearest_goal()`：把 `group_ranking` 收斂成「docking pose 在 ROS map XY 上離目標最近」的單一候選，並重新標為 rank 1。 |
| `commander/flows/pick.py` | `_get_item_info_no_sam3d_node` 在建 `item_info` 時套用收斂；把消融 metadata 放進 `navigation.ablation`。 |
| `commander/contracts.py` | `NavigationState` 新增 `ablation` 欄位。 |
| `commander/experiment_recorder.py` | `ExperimentRecorder`：觀察每個 node update，於 `goodbye_node`（即 graph END）輸出報告。 |
| `commander/storage/session_store.py` | 建立並餵食 recorder（`main.py` 與 web 都經由 `SessionMemoryStore.record_event`）。 |

因為收斂後只剩 1 個候選，現有的換點機制會自然失效：`goal_pose_for_rank(rank=2)`
回傳 out of range → `major_nav_node` 標記 `MAJOR_NAV_EXHAUSTED` → 回 home。**不需要**
另外改動路由或 safety guard。

## 4. 報告位置與格式

- 所有抓取任務（`get_item_info_no_sam3d_node → END`）累積在同一份檔案：
  `result/result.json`（JSON 陣列，每筆一個 `A2_NNN` entry；id 依檔內現有最大值 +1 遞增）。

報告涵蓋使用者要求的 10 個必記指標（前段）＋補充指標：

```json
{
  "experiment_id": "A2_001",
  "experiment_name": "ranked_goal_poses_ablation_single_nearest",
  "ablation": {
    "variable": "ranked_goal_poses",
    "mode": "single_nearest_goal_pose",
    "applied": true,
    "perception_group_count": 5,
    "selected_perception_rank": 3,
    "selected_distance_m": 0.42,
    "dropped_group_count": 4
  },
  "scene_id": "S3_occlusion_01",
  "context_id": "…",
  "task_instruction": "幫我拿桌子旁邊的熊",
  "start_node": "get_item_info_no_sam3d_node",
  "end_node": "END",

  "task_success": true,
  "total_time_from_get_item_info_to_END": 42.31,
  "node_sequence": ["get_item_info_no_sam3d_node", "nav_move_node", "observe_node", "reason_node", "car_grasp_node", "car_approach_node", "nav_home_node", "goodbye_node", "END"],
  "node_latency_sec": {"get_item_info_no_sam3d_node": 8.42, "nav_move_node": 15.73, "...": 0},
  "node_visit_count": {"observe_node": 1, "reason_node": 1, "major_nav_node": 0, "update_memory_node": 0},
  "reason_actions": ["grasp_agent"],
  "ranked_goal_count": 1,
  "tried_goal_count": 1,
  "success_goal_rank": 1,
  "failure_reason": "",

  "final_status": "SUCCESS",
  "exhausted_all_ranks": false,
  "vlm_latency_sec": 3.85,
  "a2a_latency_sec": {"get_item_info_agent": 8.42, "grasp_agent": 9.64},
  "nav_success": true,
  "nav_time_sec": 15.73,
  "grasp_success": true,
  "grasp_pose_ready": true,
  "grasp_candidate_count": 20,
  "grasp_valid_count": 5,
  "approach_success": true,
  "arm_success": true,
  "memory_update_count": 0,
  "wall_clock_sec": 43.10,
  "node_timeline": [{"step": 1, "node": "get_item_info_no_sam3d_node", "status": "ITEM_INFO_NO_SAM3D_READY", "success": true, "latency_sec": 8.42}]
}
```

### 欄位語意重點

| 欄位 | 說明 |
|---|---|
| `task_success` / `final_status` | 以 `car_approach_node` 是否成功為準。`final_status` 進一步分類：`SUCCESS` / `GRASP_FAILED` / `APPROACH_FAILED` / `NAV_FAILED` / `ITEM_INFO_FAILED` / `TARGET_LOST` / `NO_VALID_GOAL` / `FAILED`。 |
| `total_time_from_get_item_info_to_END` | 從 item-info 到 END 各 node latency 的總和（`wall_clock_sec` 為實際牆鐘時間，供對照）。 |
| `ranked_goal_count` | workflow **實際可用**的候選數。A2 恆為 `1`；基線為感知產生的候選數 N。 |
| `tried_goal_count` | 真正導航嘗試過的候選數（`nav_move_node` 上出現過的不同 rank）。A2 恆為 `1`。 |
| `success_goal_rank` | 成功時是第幾次嘗試的候選（A2 恆為 1；基線可能是 1、2、3…）。 |
| `ablation.selected_perception_rank` | A2 選到的「最近候選」原本在感知排名的第幾名（佐證最近 ≠ 排名第一）。 |
| `failure_reason` | `navigation_failed` / `target_not_visible` / `grasp_failed` / `ik_failed` / `arm_failed` / `approach_failed:<phase>` / `all_ranked_goals_exhausted` / `no_valid_goal_pose` / `item_info_failed`。 |
| `exhausted_all_ranks` | 是否把所有 ranked 候選都試完仍失敗（基線分析用）。 |

## 5. 怎麼寫實驗結果

`result/result.json` 是一個 JSON 陣列（每筆一次執行），可直接彙整：

- **成功率**：`mean(task_success)`，A2 vs 基線。
- **平均完成時間**：`mean(total_time_sec)`（只算成功，或全部分開列）。
- **排序是否有效**：基線的 `success_goal_rank` 分佈（若常常 > 1，代表「換點」確實救回任務）；
  對照 A2 因為不能換點而失敗的比例（`final_status` 落在 grasp/approach/exhausted）。
- **最近 ≠ 最佳**：A2 的 `ablation.selected_perception_rank` 分佈（若常常不是 1，說明
  純距離選點不等於排序選點）。
- **瓶頸 node**：彙整各報告的 `node_latency_sec`，找最慢的 node。
- **失敗階段**：彙整 `failure_reason`，看失敗主要落在導航 / 抓取 / 接近 / IK。

## 6. 測試

```bash
cd workflow
PYTHONPATH="$PWD" python3 tests/test_ablation_a2.py   # 消融收斂 + 報告（不需 langgraph）
PYTHONPATH="$PWD" python3 tests/test_refactor_contracts.py   # 既有合約/路由（需完整環境）
```
