# View Agent — 強化學習與預訓練系統

這是 VLM_RL 專案中 **獨立的強化學習訓練模組**。本系統負責收集 Unity 模擬數據，並透過 Teacher-Student 架構訓練出能夠自動調整觀察視角的 SAC Policy。

## 目錄架構

本訓練專案實體分為以下三個獨立的執行環境，以確保訓練、推論與模擬徹底分離：

### 1. `docker/` (本機 Docker — 資料採集與標注)
在本機執行，負責透過 Rosbridge 收集 Unity 的軌跡影像、關節狀態等，並利用 Teacher VLM (Gemini/Ollama) 進行離線動作標注。
- `collector/`: 負責軌跡採集、HDF5 存檔
- `labeler/`: 負責與預訓練大模型 API 溝通，進行評分與行為克隆 (BC) 標注
- `scripts/`: 提供收集資料與標注的主程式

### 2. `3090server/` (RTX 3090 — 預訓練與 RL 訓練)
在配備 GPU 的伺服器執行，負責訓練 SAC Actor-Critic。完成訓練後，將產生的 checkpoint 提供給主專案的 A2A Server 進行即時推論。
- `models/`: 特徵提取、時序融合與 Policy 網路定義
- `pretrain/`: Behavior Cloning 訓練迴圈
- `rl/`: 包含 Temporal Replay Buffer 及主 SAC 訓練迴圈
- `env/`: 包裝 Unity 為 Gym Environment
- `scripts/`: 提供預訓練與正式 RL 訓練的主程式

### 3. `unity/` (Unity C# 腳本)
運行於 Unity 模擬器中，提供資料發布與環境控制的 API。
- `Sensors/`: 負責將相機與關節數據發布至 ROS
- `Environment/`: 計算物理真值遮擋率，處理環境隨機化請求

## 如何開始

1. 將 `unity/` 中的腳本匯入 Unity 專案，結合 Unity Robotics Hub 提供 ROS2 endpoint。
2. 啟動 `docker/scripts/run_collect.py` 收集 HDF5 訓練軌跡。
3. 啟動 `docker/scripts/run_label.py` 使用 Gemini 批次標注。
4. 將 HDF5 標注檔複製到 `3090server/`。
5. 啟動 `3090server/scripts/run_pretrain.py` 獲得初始權重。
6. 啟動 `3090server/scripts/run_train.py` 進行 SAC 微調。
