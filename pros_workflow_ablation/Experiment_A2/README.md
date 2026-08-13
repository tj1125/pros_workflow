# Experiment A2 — 消融實驗：ranked goal poses

本資料夾是主動感知抓取系統的 **A2 消融版**：移除「多候選停靠點 + 排序 + 失敗換點」
的能力，只保留**距離目標最近的單一 goal pose**，用來證明多候選與排序能提高觀察／
抓取成功率。

- 切換開關：`.env` 的 `ABLATION_SINGLE_NEAREST_GOAL`（A2 預設 `true`，設 `false`
  即還原成完整基線版本做對照）。
- 實作：`workflow/commander/ablation.py`；只在 `get_item_info_no_sam3d_node` 把
  `group_ranking` 收斂成最近的單一候選（感知服務本身不變），因此 `major_nav_node`
  在第一次嘗試後就會 `MAJOR_NAV_EXHAUSTED`，不再換點。

部署與執行方式同基線系統（見 `../../VLM_RL`）。
