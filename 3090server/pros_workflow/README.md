# 3090 Server — pros_workflow A2A Services

這個資料夾是 RTX 3090 端執行的 A2A Agent Server。Commander 在本機負責 LangGraph、VLM 決策與 ROS2 控制；3090 端只處理 GPU/深度學習的感知與抓取推論。本文件說明如何在 3090 上啟動這些服務。

## 服務列表

| 服務 | Port | 主流程狀態 | 功能 |
|---|---:|---|---|
| `get_item_info_agent_no_sam3d` | `8006`（hardcoded） | **required** | 多視角 RGB + `/world_position_data` + SAM/幾何融合（不跑 SAM3D），輸出目標中心與 ranked goal poses。 |
| `grasp_agent` | `8007`（hardcoded） | **required** | `Camera_Car` RGBD + YOLO + SAM + GraspGen，輸出 6-DoF grasp poses。 |
| `get_item_info_agent` | `8008`（env `GET_ITEM_INFO_LEGACY_PORT`） | legacy | 完整 SAM3D pipeline（YOLO→SAM→Triangulation→DepthAnything→SAM3D→GraspGen）。保留給需要 mesh reconstruction 的實驗。 |

## 目錄結構

```text
3090server/pros_workflow/
├── a2a_utils/                        # A2A success/error response helpers
├── models/                           # 共用模型權重目錄（只留 .gitkeep，權重部署時放入）
├── tool/                             # 共用推論 helper（見下）
│   ├── grasp/graspgen.py             # GraspGen runtime + 點雲/碰撞過濾
│   ├── vision/yolo.py                # YOLO 偵測 helper
│   ├── vision/sam.py                 # SAM 分割 helper
│   └── runtime/memory.py            # CUDA 記憶體釋放
├── get_item_info_agent_no_sam3d/     # 現行 item-info server（:8006）
├── get_item_info_agent/              # legacy SAM3D item-info server（:8008）
└── grasp_agent/                      # GraspGen grasp server（:8007）
```

`tool/` 是三個 server 共用的 runtime，不應複製多份。GraspGen runtime 以 `get_item_info_agent/vendor/graspgen_runtime` 為 canonical source，`grasp_agent` 與 `no_sam3d` 的 config 都指向它。

## 啟動

在 3090 server 上進入本目錄，以 module 方式啟動：

```bash
cd /path/to/pros_workflow/3090server/pros_workflow

python -m get_item_info_agent_no_sam3d   # :8006 required
python -m grasp_agent                    # :8007 required
python -m get_item_info_agent            # :8008 legacy（需要時才開）
```

`EXTERNAL_IP` 只用來寫入 A2A AgentCard 的 `url`，程式內預設 `140.116.82.226`；部署到別的機器時明確指定：

```bash
EXTERNAL_IP=140.116.82.226 python -m grasp_agent
```

各 server 綁定 `0.0.0.0`，port 如上表（`get_item_info_agent` 可用 `GET_ITEM_INFO_LEGACY_PORT` 覆蓋）。

## Commander `.env` 對應

本機 Commander 透過這些 URL 呼叫 3090 服務（實際值見根目錄 `.env`）：

```env
INF_GET_ITEM_INFO_NO_SAM3D_URL=http://140.116.82.226:8006
INF_GRASP_URL=http://140.116.82.226:8007
INF_GET_ITEM_INFO_URL=                                   # 空值＝不啟用 legacy :8008
```

## 模型與環境變數

- `models/` 只保留 `.gitkeep`，實際權重（YOLO / SAM / DepthAnything / SAM3D / GraspGen checkpoints）部署時放入。
- `grasp_agent` 的模型/相機路徑可用 env 覆蓋 config：`GRASP_YOLO_WEIGHTS`、`GRASP_SAM_CHECKPOINT`、`GRASP_GRASPGEN_ROOT`、`GRASP_GRIPPER_CONFIG`、`GRASP_CAMERA_INTRINSICS`。
- `get_item_info_agent_no_sam3d` 不讀任何 env，設定全部來自 `configs/scene.default.yaml`。
- 每個服務的 `configs/*.yaml` 指向 camera parameters、SAM checkpoint、GraspGen runtime，item-info 服務另含 Nav2 keepout map。

## 個別文件

- [get_item_info_agent_no_sam3d/README.md](./get_item_info_agent_no_sam3d/README.md)
- [grasp_agent/README.md](./grasp_agent/README.md)
- [get_item_info_agent/README.md](./get_item_info_agent/README.md)（legacy）
