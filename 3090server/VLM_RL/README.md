# 3090 Server — VLM_RL Agent Services

這個資料夾包含所有在 **RTX 3090 伺服器**上執行的 A2A Agent Server 程式碼。

## 目錄結構

```
3090server/VLM_RL/
├── README.md
├── a2a_utils/                   # A2A 回應生成工具
│   ├── __init__.py
│   └── response.py
│
├── models/                      # 模型權重存放區 (.pt, .onnx 等)
│
├── find_agent/                  # Find Server (Port 8005)
│   ├── requirements.txt
│   ├── __init__.py
│   ├── __main__.py
│   ├── agent_executor.py
│   └── yolo_service.py
│
└── get_item_info_agent/         # Get Item Info Server (Port 8006)
    ├── requirements.txt
    ├── __init__.py
    ├── __main__.py
    ├── agent_executor.py
    └── pipeline/                # 感知 pipeline（複製自 get_item_info/app）
        ├── __init__.py
        ├── constants.py         # GET_ITEM_INFO_ROOT → 原始專案路徑
        ├── config.py
        ├── types.py
        ├── pipeline.py          # 主流程
        ├── steps/
        │   ├── detect_and_triangulate.py
        │   ├── sam_and_depth.py
        │   ├── mesh_align.py
        │   └── goal_pose.py
        └── adapters/
            ├── grasp_format.py
            ├── sam3d_adapter.py
            └── graspgen_adapter.py
```

> **原始 pipeline 程式碼位於** `/home/tjchen/workspace/get_item_info`  
> 不要直接修改那個目錄；`pipeline/` 是獨立的副本。

---

## A2A 訊息格式

### 輸入（3 parts）

| Part | 類型 | 內容 |
|------|------|------|
| `[0]` | text (JSON) | `{"yolo_class": "doll", "scene_config": "<path>"}` |
| `[1]` | data (bytes) | camera-A 影像（PNG/JPG） |
| `[2]` | data (bytes) | camera-B 影像（PNG/JPG） |

### 輸出（text JSON）

```json
{
  "center_world": [x, y, z],
  "group_ranking": [
    {
      "rank": 1,
      "best_confidence": 0.87,
      "best_goal_pose_ros_map": [map_x, map_y],
      "best_pose_unity": [...],
      "best_goal_pose_unity": [...]
    }
  ],
  "goal_pose_path": "/tmp/get_item_info_xxx/goal_pose.json"
}
```

---

## 虛擬環境與啟動方式

| Agent | Conda 環境 | Port |
|-------|------------|------|
| `find_agent` | `a2a_vlm_find` | 8005 |
| `get_item_info_agent` | `get_item_info_agent` | 8006 |

### 建立 get_item_info_agent 環境（只需一次）

```bash
conda create -n get_item_info_agent python=3.11 -y
conda activate get_item_info_agent

# 1. PyTorch + CUDA
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 2. 其他依賴
cd 3090server/VLM_RL
pip install -r get_item_info_agent/requirements.txt

# 3. GraspGen pointnet2 local CUDA extension
pip install --no-build-isolation \
    -e /home/tjchen/workspace/get_item_info/vendor/graspgen_runtime/pointnet2_ops
```

### 啟動服務

**視窗 A: Find Agent (Port 8005)**
```bash
conda activate a2a_vlm_find
cd 3090server/VLM_RL
python -m find_agent
```

**視窗 B: Get Item Info Agent (Port 8006)**
```bash
conda activate get_item_info_agent
cd 3090server/VLM_RL
python -m get_item_info_agent
```

---

## 環境需求

- Python 3.11
- CUDA 12.1
- PyTorch 2.5.1
- a2a-sdk（PyPI）
