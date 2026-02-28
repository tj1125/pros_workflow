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

為了與系統其他專案區分，並為未來的各個 Agent 保留擴展性，我們採用 `a2a_vlm_<agent_name>` 的命名規則。以下是本專案 3090 伺服器端的 Agent 對應表：

| A2A 代理人模組 (`python -m`) | Conda 虛擬環境名稱 | 負責功能 / 推論內容 | 主要特定依賴 |
|-----------------------------|-------------------|------------------|------------|
| `find_agent` | **`a2a_vlm_find`** | 接收多相機影像，進行 YOLO 目標辨識並畫框 | `ultralytics`, `Pillow` |
| `get_item_info_agent` | **`a2a_vlm_find`** | 針對所選目標，推算 3D 空間位置與大小 | 目前與 find 共用依賴，故使用相同的環境 |
| `nav_agent` *(未來規劃)* | **`a2a_vlm_nav`** | 接收避障與相機資訊，推論底盤移動點 | *(待定)* |
| `grasp_agent` *(未來規劃)* | **`a2a_vlm_grasp`**| 接收點雲，生成 6D 抓取姿態 (GraspGen) | PointNet 等 3D 庫 |
| `approach_agent` *(未來規劃)* | **`a2a_vlm_approach`**| 接收抓取姿態，產生最後靠近的手臂控制策略 | *(待定)* |

> **💡 實務提醒：** 
> 由於目前 `find_agent` 與 `get_item_info_agent` 的 Python 版本需求 (3.10+) 相同，且依賴完全相容，為了省事，你**可以直接共用 `a2a_vlm_find` 這個環境**來跑這兩個 Agent。

### 1. 建立環境與安裝 (只需執行一次)
```bash
# 建立專屬虛擬環境 (加上 a2a_vlm_ 前綴以利辨識本專案)
conda create -n a2a_vlm_find python=3.10 -y
conda activate a2a_vlm_find

# 安裝所需依賴套件
cd 3090server/VLM_RL
pip install -r requirements.txt
```

### 2. 啟動服務
開啟兩個終端機視窗，都確認在 `a2a_vlm_find` 環境下：

**視窗 A: Find Agent Server (Port 8005)**
```bash
cd 3090server/VLM_RL
conda activate a2a_vlm_find
python -m find_agent
```

**視窗 B: Get Item Info Agent Server (Port 8006)**
```bash
cd 3090server/VLM_RL
conda activate a2a_vlm_find
python -m get_item_info_agent
```

## 環境需求

- Python 3.10+
- PyTorch + Ultralytics YOLO
- a2a-sdk（從 PyPI 安裝）
