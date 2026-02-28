# 3090 Server — VLM_RL Agent Services

這個資料夾包含所有在 **RTX 3090 伺服器**上執行的 A2A Agent Server 程式碼。

## 目錄結構

```
3090server/VLM_RL/
├── pyproject.toml         # 依賴管理
├── README.md
├── a2a_utils/             # A2A 回應生成工具
│   ├── __init__.py
│   └── response.py
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

## 啟動方式

嚴格遵照 Python package 規範，利用 `-m` 執行模組的 `__main__.py`：

```bash
cd 3090server/VLM_RL

# 啟動 Find Agent Server (Port 8005)
python -m find_agent

# 啟動 Get Item Info Agent Server (Port 8006)
python -m get_item_info_agent
```

## 環境需求

- Python 3.10+
- PyTorch + Ultralytics YOLO
- a2a-sdk（從 PyPI 安裝）
