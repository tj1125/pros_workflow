# Experiment A3 — 消融實驗：移除物體資訊估計（w/o get_item_info_no_sam3d_node）

本資料夾是主動感知抓取系統的 **A3 消融版**。

完整系統會呼叫 no-SAM3D item-info A2A 服務估計物體尺寸、yaw、grasp direction
分群與抓取可行性的 ranked goal poses。A3 把 `get_item_info_no_sam3d_node` 改成
**只用目標物中心 + ROS keepout 地圖**產生候選導航點（8 個方位、固定 stand-off、
只留可走的點，朝向一律面向目標中心），不做尺寸/yaw/抓取方向分群、不做品質排序與
觀察角度選擇，也不呼叫 A2A。節點名稱、輸出契約與圖結構維持不變。

用來證明物體資訊估計對候選 goal pose 產生、觀察角度選擇、抓取可行性與整體任務
成功率的幫助。候選點產生在 `workflow/commander/nav/ablation_goals.py`，可用
`ABLATION_GOAL_*` 環境變數調整（細節見 `docs/spec.md`）。

部署與執行方式同基線系統（見 `../../VLM_RL`）。
