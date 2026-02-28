# 3090 Server — VLM_RL Agent Services

這個資料夾包含所有在 **RTX 3090 伺服器**上執行的 A2A Agent Server 程式碼。

## 目錄結構

```
3090server/VLM_RL/
├── README.md
├── a2a_utils/             # A2A 回應生成工具
│   ├── __init__.py
│   └── response.py
│
├── models/                # 模型權重存放區 (.pt, .onnx 等)
│
├── find_agent/            # Find Server (Port 8005)
│   ├── requirements.txt   # (包含 YOLO 等依賴)
│   ├── __init__.py
│   ├── __main__.py        # uvicorn API 啟動點
│   ├── agent_executor.py  # A2A 執行緒
│   └── yolo_service.py    # YOLO 推論封裝
│
└── get_item_info_agent/   # Get Item Info Server (Port 8006)
    ├── requirements.txt   # (基礎 A2A 依賴)
    ├── __init__.py
    ├── __main__.py
    └── agent_executor.py
```

## 虛擬環境與啟動方式

強烈建議**為每一個 Agent 建立專屬的 Conda 虛擬環境**，避免套件衝突。
我們統一採用的命名規則為 `vlm_env_<agent_name>`。

### 1. 啟動 Find Agent (環境：vlm_env_find)
負責運行 YOLO 推論，並將標記的圖片回傳。

```bash
# 1. 建立並進入專屬環境 (只需執行一次)
conda create -n vlm_env_find python=3.10 -y
conda activate vlm_env_find

# 2. 安裝套件
cd 3090server/VLM_RL
pip install -r find_agent/requirements.txt

# 3. 啟動服務 (需維持環境在 vlm_env_find)
python -m find_agent
```

### 2. 啟動 Get Item Info Agent (環境：vlm_env_get_item_info)
負責計算目標物的 3D 空間位置。

```bash
# 1. 建立並進入專屬環境 (只需執行一次)
conda create -n vlm_env_get_item_info python=3.10 -y
conda activate vlm_env_get_item_info

# 2. 安裝套件
cd 3090server/VLM_RL
pip install -r get_item_info_agent/requirements.txt

# 3. 啟動服務 (需維持環境在 vlm_env_get_item_info)
python -m get_item_info_agent
```

## 環境需求

- Python 3.10+
- PyTorch + Ultralytics YOLO
- a2a-sdk（從 PyPI 安裝）
