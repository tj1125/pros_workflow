# View Agent — RTX 3090 端（預訓練與強化學習）

本目錄包含在配備強大 GPU 的伺服器上執行的程式碼，主要負責訓練 SAC Policy 網路並將其推論提供出來。由於 SAC 訓練會直接呼叫 Unity（Gym Env），因此 `collector/` 工具也一併被複製過來共用。

## 模型架構

*   **`models/feature_extractor.py`**: CLIP + DINOv2 融合特徵 (1280-dim)。
*   **`models/temporal_encoder.py`**: 將連續 3 幀的影像特徵融合成為時序狀態 (512-dim)。
*   **`models/sac_policy.py`**: SAC Actor-Critic。

## 執行流程

1. 將 `docker/` 端採集並標注好的 HDF5 資料複製到 `data/labeled`。
2. 在 3090 上安裝套件：`pip install -r requirements.txt`
3. 執行 Behavior Cloning 預訓練（讓策略具備良好的初始狀態，節省 RL 時間）：
   ```bash
   python scripts/run_pretrain.py --config configs/train_config.yaml
   ```
4. 啟動 Unity（如果尚未啟動），執行 SAC RL 強化訓練：
   ```bash
   python scripts/run_train.py --config configs/train_config.yaml
   ```
