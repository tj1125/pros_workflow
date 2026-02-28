# 3090 Server — VLM_RL Agent Services

這個資料夾包含所有在 **RTX 3090 伺服器**上執行的 A2A Agent Server 程式碼。

## 目錄結構

```
3090server/VLM_RL/
├── requirements.txt       # 共用依賴套件 (a2a-sdk, YOLO 等)
├── README.md
├── a2a_utils/             # A2A 回應生成工具
│   ├── __init__.py
│   └── response.py
│
├── models/                # 模型權重存放區 (.pt, .onnx 等)
│
├── find_agent/            # Find Server (Port 8005)
│   ├── __init__.py
│   ├── __main__.py        # uvicorn API 啟動點
│   ├── agent_executor.py  # A2A 執行緒
│   └── yolo_service.py    # YOLO 推論封裝
│
└── get_item_info_agent/   # Get Item Info Server (Port 8006)
    ├── __init__.py
    ├── __main__.py
    └── agent_executor.py
```

## 虛擬環境與啟動方式

對於依賴高度重疊且 Python 版本相同的 Agent（例如目前的 `find_agent` 與 `get_item_info_agent`），
我們建立一個共用的 Conda 虛擬環境 `vlm_a2a`。如果未來有依賴完全衝突的新 Agent，再另外建專屬環境。

### 1. 建立並安裝共用環境 (只需執行一次)
```bash
# 建立共用虛擬環境
conda create -n vlm_a2a python=3.10 -y
conda activate vlm_a2a

# 安裝所有 Agent 共用的依賴套件
cd 3090server/VLM_RL
pip install -r requirements.txt
```

### 2. 啟動服務
開啟不同的終端機視窗，皆確保處於 `vlm_a2a` 環境中：

**視窗 A: Find Agent Server (Port 8005)**
```bash
cd 3090server/VLM_RL
conda activate vlm_a2a
python -m find_agent
```

**視窗 B: Get Item Info Agent Server (Port 8006)**
```bash
cd 3090server/VLM_RL
conda activate vlm_a2a
python -m get_item_info_agent
```

## 環境需求

- Python 3.10+
- PyTorch + Ultralytics YOLO
- a2a-sdk（從 PyPI 安裝）
