# View Agent — Docker 端（資料採集與標注）

本目錄包含在開發本機（Docker）上執行的程式碼，主要負責透過 Rosbridge 收集 Unity 端的大量軌跡，並透過呼叫遠端 VLM (Gemini / Ollama) API 進行批次自動標注。

## 功能

*   **`collector/`**：持續訂閱相機、關節角度與物理指標，定時發送「場景隨機化」給 Unity，將整段軌跡以 HDF5 儲存。
*   **`labeler/`**：讀取存好的 HDF5，將影片格擷取出來餵給 Teacher VLM，然後將 Expert Actions (`a_expert`) 與 Reward 寫回原來檔案裡供後續 RL 訓練。

## 執行流程

1. 啟動 Unity 及 Rosbridge (確保 9090 port 可連線)。
2. 安裝套件：`pip install -r requirements.txt`
3. 執行資料採集：
   ```bash
   python scripts/run_collect.py --config configs/train_config.yaml --n 500
   ```
4. 執行 Teacher 標注：
   ```bash
   export TEACHER_VLM_PROVIDER=gemini
   export GEMINI_API_KEY=your_api_key
   python scripts/run_label.py --config configs/train_config.yaml
   ```
