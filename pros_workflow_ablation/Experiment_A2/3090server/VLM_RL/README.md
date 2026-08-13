# 3090 Server — VLM_RL A2A Services

這個資料夾包含 RTX 3090 端執行的 A2A Agent Server。Commander 在本機負責 LangGraph、VLM 決策與 ROS2 控制；3090 端只處理 GPU/深度學習或高成本感知推論。

## 服務列表

| 服務 | Port | 主流程狀態 | 功能 |
|---|---:|---|---|
| `find_agent` | 8005 | optional | 多相機 YOLO detection 與標註圖輸出。 |
| `get_item_info_agent_no_sam3d` | 8006 | required | 現行主流程 item-info。使用多視角 RGB、bbox、`/world_position_data` 與 geometry/SAM，輸出目標中心與 ranked goal poses。 |
| `grasp_agent` | 8007 | required | 使用 `Camera_Car` RGBD、YOLO、SAM、GraspGen 產生 6-DoF grasp poses。 |
| `get_item_info_agent` | 8008 | legacy | SAM3D/full 3D pipeline。預設移出 8006，避免佔用現行 no-SAM3D 服務。 |

## 目錄結構

```text
3090server/VLM_RL/
├── a2a_utils/                         # A2A success/error response helpers
├── models/                            # 共用模型權重目錄，實際權重不入庫
├── tool/                              # 共用 YOLO/SAM/GraspGen/runtime helpers
├── find_agent/                        # YOLO detection server
├── get_item_info_agent_no_sam3d/      # 現行 item-info server
├── get_item_info_agent/               # legacy SAM3D item-info server
└── grasp_agent/                       # GraspGen server
```

## 啟動方式

在 3090 server 上進入本目錄：

```bash
cd /path/to/VLM_RL/3090server/VLM_RL
```

依服務使用對應環境後啟動：

```bash
python -m find_agent
python -m get_item_info_agent_no_sam3d
python -m grasp_agent
```

`EXTERNAL_IP` 會寫入 A2A AgentCard 的 `url`，預設為程式內設定值；部署時建議明確指定：

```bash
EXTERNAL_IP=192.168.1.10 python -m grasp_agent
```

Commander `.env` 對應：

```env
INF_FIND_URL=http://192.168.1.10:8005
INF_GET_ITEM_INFO_NO_SAM3D_URL=http://192.168.1.10:8006
INF_GRASP_URL=http://192.168.1.10:8007
```

## 模型與資料

- `models/` 只保留 `.gitkeep`，模型權重需部署時放入。
- `get_item_info_agent_no_sam3d/configs/scene.default.yaml` 指向 camera parameters、SAM checkpoint、GraspGen runtime 與 Nav2 keepout map。
- `grasp_agent/configs/runtime.default.yaml` 可由 `GRASP_*` 環境變數覆蓋。
- `tool/` 是共用 runtime，不應複製多份到各 agent。

## 個別文件

- [find_agent/README.md](./find_agent/README.md)
- [get_item_info_agent_no_sam3d/README.md](./get_item_info_agent_no_sam3d/README.md)
- [get_item_info_agent/README.md](./get_item_info_agent/README.md)
- [grasp_agent/README.md](./grasp_agent/README.md)
