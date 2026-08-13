# Experiment A1 — 消融實驗：移除 VLM observe-reason（高層決策）

本資料夾是主動感知抓取系統的 **A1 消融版**。

完整系統每一輪都先由 VLM 看當前 RGBD 畫面做高層決策（`observe_node` +
`reason_node`），判斷要從目前視角抓取（`car_approach`）還是換到下一個視角
（`major_nav`）。A1 **移除整個 VLM observe-reason 步驟**，改成固定規則流程：

```
get_item_info → nav_move → car_grasp → car_approach
  car_approach 成功 → nav_home（END）
  car_approach 失敗 → major_nav 換下一個 ranked goal pose → 再試
```

用來證明「VLM 主動感知 / 視角決策」對任務成功率的貢獻。

部署與執行方式同基線系統（見 `../../VLM_RL`）。
