# Get Item Info Agent Server

**Get Item Info Agent** 是一個 A2A Agent Server，負責接收雙目立體相機（Stereo RGB）的影像，針對指定的物件類別進行一連串 3D 感知與推論，最後輸出物件的 3D 世界座標以及可用於機器人抓取的目標姿態 (Goal Pose)。

## 執行環境

- **Conda 環境名稱**：`a2a_vlm_find`
- **主要依賴**：`torch`, `segment_anything`, `ultralytics`, `pytorch3d`, `GraspGen` 相關依賴
- **服務 Port**：8006

### 環境建置 (只需一次)

此 Agent 的依賴較為龐大，涵蓋了 2D 到 3D 的完整感知管線：

```bash
conda create -n a2a_vlm_find python=3.11 -y
conda activate a2a_vlm_find

# 1. PyTorch + CUDA
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 torchaudio==2.5.1+cu121 \
    --extra-index-url https://download.pytorch.org/whl/cu121

# 2. 安裝本資料夾內的所有依賴
pip install -r requirements.txt

# 3. GraspGen pointnet2 local CUDA extension
pip install --no-build-isolation -e vendor/graspgen_runtime/pointnet2_ops
```

## 啟動服務

確認在 `a2a_vlm_find` 環境下，於 `VLM_RL` 目錄執行：

```bash
conda activate a2a_vlm_find
python -m a2a_vlm_find
```

---

## 核心感知 Pipeline

本專案將原始的 `get_item_info` 工具包裝成 Agent。  
流程：`Stereo RGB -> YOLO -> SAM -> Triangulation -> DepthAnything -> SAM3D -> Pose Alignment -> GraspGen -> Goal Pose`

> 本 Agent 已經與外部原始工具完全解耦，所有必備的 `models/`, `vendor/`, `configs/`, `data/` 已被搬移至本目錄 `a2a_vlm_find` 內，可完全獨立運行。

---

## A2A 訊息格式

### 輸入（Multi-part Request）

必須提供至少三個 components：

| Part | 類型 | 內容描述 |
|------|------|----------|
| `[0]` | text (JSON) | 設定檔，例如：`{"yolo_class": "doll", "scene_config": "<path>", "selected_camera": "Camera_Room1_1", "camera_names": ["Camera_Room1_1", "Camera_Room1_2", "Camera_Room1_3"]}` |
| `[1..N]` | data / text(base64) | 與 `camera_names` 同順序的 RGB 原圖（PNG/JPG 格式） |

Server 會優先以 `selected_camera` 為主視角，從同組上傳影像中挑出可用的 stereo pair 來跑既有 3D pipeline。

### 輸出（Response - JSON text）

傳回運算的結果以及產生的 `goal_pose.json` 本地路徑：

```json
{
  "center_world": [1.23, -0.45, 0.89],
  "group_ranking": [
    {
      "rank": 1,
      "best_confidence": 0.87,
      "best_goal_pose_ros_map": [2.5, 3.1],
      "best_pose_unity": [...],
      "best_pose_matrix_unity": [...],
      "best_goal_pose_unity": [...]
    }
  ],
  "goal_pose_path": "/tmp/get_item_info_xxxxx/goal_pose.json"
}
```
