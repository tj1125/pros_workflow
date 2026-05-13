# VLM 多代理人主動感知抓取系統

本專案是部署於 Unity/ROS2 環境的主動感知抓取系統。主控端使用 LangGraph 維護任務狀態，透過 VLM 做高層決策，並呼叫本機 ROS2 控制流程與 RTX 3090 上的 A2A 推論服務完成找物、導航、抓取姿態生成與底盤/手臂收尾。

目前部署文件以 [docs/langgraph_flow.png](docs/langgraph_flow.png) 的流程為準。

## 現行主流程

```text
greeting_node
  -> human_reply_node
  -> task_classification_node
      -> ai_reply_node -> chat_memory_node -> human_reply_node
      -> input_node
          -> find_node
          -> get_item_info_no_sam3d_node
          -> nav_move_node
          -> observe_node
          -> reason_node
              -> update_item_info_1_node -> major_nav_node -> nav_move_node
              -> update_item_info_2_node -> car_grasp_node -> car_approach_node
              -> nav_home_node -> goodbye_node
          -> update_memory_node -> observe_node
```

重點行為：

- `find_node` 以 `/world_position_data` 與 room camera 影像列出目標 instance，並讓使用者確認。
- `get_item_info_no_sam3d_node` 呼叫 RTX 3090 的 no-SAM3D 物體資訊服務，取得 `center_world`、`group_ranking` 與 Nav2 可用的 goal pose。
- `nav_move_node` 透過 ROS2 子程序發布 `/goal_pose`，等待 Nav2 plan 與 AMCL 到位。
- `reason_node` 只允許 VLM 輸出 `major_nav_node`、`grasp_agent`、`car_approach_agent` 或 `DONE`。
- `car_approach_agent` 會執行底盤靠近，並在到位後直接完成手臂與夾爪收尾。

`arm_approach_agent` 保留為備援 adapter，但目前主圖不直接呼叫；主線由 `car_approach_agent` 完成底盤靠近與手臂/夾爪收尾。

## 快速開始

```bash
pip install uv
uv sync
cp .env.example .env
```

Mock 模式不需要 GPU、VLM API 或 ROS2：

```bash
uv run python main.py --mock
```

Real 模式會使用 `.env` 中的 VLM、A2A 與 ROS2 設定：

```bash
uv run python main.py --no-mock
```

Web 介面：

```bash
uv run python web_main.py --host 0.0.0.0 --port 8080
```

瀏覽器開 `http://localhost:8080`。

## Nav2 與容器

先啟動 Nav2 導航容器：

```bash
./launch_nav.sh --rebuild
./launch_nav.sh
```

背景啟動：

```bash
./launch_nav.sh -d
```

停止 Nav2：

```bash
./launch_nav.sh --stop
```

進入開發容器：

```bash
./enter_docker.sh
```

若需要從主機瀏覽器連 web UI：

```bash
./enter_docker.sh --web-port 8080
```

容器內常用命令：

```bash
r
ros2 run car_control_pkg car_control_node
ros2 run arm_control_pkg arm_control_node
ros2 run keyboard_mode_interface_pkg keyboard_control_node
ros2 launch nav_goal_bridge_pkg navigation.launch.py
run
```

需要 PyBullet GUI 或 X11 視窗時使用：

```bash
./enter_docker_x11.sh
```

## RTX 3090 A2A 服務

主流程 real mode 需要下列服務：

| 服務 | 目錄 | 預設 Port | Commander env |
|---|---|---:|---|
| Find Agent | `3090server/VLM_RL/find_agent` | 8005 | `INF_FIND_URL` |
| Get Item Info No SAM3D | `3090server/VLM_RL/get_item_info_agent_no_sam3d` | 8006 | `INF_GET_ITEM_INFO_NO_SAM3D_URL` |
| Grasp Agent | `3090server/VLM_RL/grasp_agent` | 8007 | `INF_GRASP_URL` |

保留服務：

| 服務 | 目錄 | 預設 Port | 狀態 |
|---|---|---:|---|
| Get Item Info SAM3D | `3090server/VLM_RL/get_item_info_agent` | 8006 | legacy / heavy 3D pipeline |
3090 端總說明在 [3090server/VLM_RL/README.md](3090server/VLM_RL/README.md)。

## 目錄結構

```text
VLM_RL/
├── main.py                         # CLI 入口
├── web_main.py                     # FastAPI/SSE web 入口
├── commander/                      # LangGraph 主控、VLM、ROS bridge、session/log
├── agents/                         # Commander-facing agent adapters
├── 3090server/VLM_RL/              # RTX 3090 A2A server services
├── tools/car_control/src/          # ROS2 car/arm/nav packages
├── config/                         # cameras.yaml / objects.yaml
├── docs/                           # 主流程圖與系統規格
├── thesis/                         # 論文總整理與研究筆記
├── Dockerfile
├── Dockerfile.nav2
├── enter_docker.sh
└── launch_nav.sh
```

## 環境變數

核心設定：

| 變數 | 說明 | 預設/範例 |
|---|---|---|
| `MOCK_MODE` | `true` 時不呼叫 VLM/GPU/ROS 實體服務 | `true` |
| `VLM_PROVIDER` | `google` 或 `ollama` | `google` |
| `GOOGLE_API_KEY` | Gemini API key，`VLM_PROVIDER=google` 時必填 | - |
| `GEMINI_MODEL` | Gemini model | `gemini-2.0-flash` |
| `OLLAMA_BASE_URL` | Ollama OpenAI-compatible endpoint | `http://localhost:11434` |
| `OLLAMA_MODEL` | Brain 使用的 Ollama 模型 | `gemma4:26b` |
| `OLLAMA_CLASSIFIER_MODEL` | 任務分類模型 | `gemma3:1b` |
| `OLLAMA_CHAT_MODEL` | 一般聊天模型 | `gemma4:26b` |
| `TRACE_LOG_FILE` | JSONL trace log | `logs/trace_logger.jsonl` |

A2A 服務：

| 變數 | 說明 |
|---|---|
| `INF_FIND_URL` | Find Agent A2A server，例如 `http://192.168.1.10:8005` |
| `INF_GET_ITEM_INFO_NO_SAM3D_URL` | 現行主流程 item-info server，例如 `http://192.168.1.10:8006` |
| `INF_GET_ITEM_INFO_URL` | legacy SAM3D item-info fallback |
| `INF_GRASP_URL` | Grasp Agent server，例如 `http://192.168.1.10:8007` |

ROS2/導航：

| 變數 | 說明 | 預設 |
|---|---|---|
| `ROS_DOMAIN_ID` | ROS2 DDS domain | `1` |
| `ROS_PYTHON_BIN` | 可載入 `rclpy` 的 Python | `/usr/bin/python3` |
| `ROS_SETUP_BASH` | ROS2 base setup | `/opt/ros/humble/setup.bash` |
| `ROS_OVERLAY_SETUP_BASH` | workspace overlay setup | `/workspaces/install/setup.bash` |
| `NAV_PLAN_TIMEOUT_SEC` | 等待 Nav2 plan 秒數 | `8` |
| `NAV_ARRIVAL_TIMEOUT_SEC` | 等待到達秒數 | `180` |
| `NAV_GOAL_TOLERANCE_M` | 到點距離容忍 | 由 `commander/nav_settings.py` 決定 |
| `WORLD_POSITION_UPDATE_THRESHOLD_M` | 目標移動超過此距離時重算 item info | `0.3` |
| `APPROACH_AGENT_TIMEOUT_SEC` | car approach 子程序 timeout | `420` |

Web：

| 變數 | 說明 | 預設 |
|---|---|---|
| `WEB_HOST` | Web server host | `0.0.0.0` |
| `WEB_PORT` | Web server port | `8080` |

## 驗證

語法檢查：

```bash
python3 -m py_compile main.py web_main.py commander/*.py agents/*.py
```

Mock 端到端測試：

```bash
uv run python test_client.py --mock-only
```

A2A 連線測試會讀取 `.env` 中已設定的 `INF_*_URL`：

```bash
uv run python test_client.py
```

## 文件

- [docs/spec.md](docs/spec.md)：部署版系統規格與狀態流程。
- [agents/README.md](agents/README.md)：本機 agent adapter 說明。
- [3090server/VLM_RL/README.md](3090server/VLM_RL/README.md)：GPU 端 A2A service 說明。
- [thesis/thesis.tex](thesis/thesis.tex)：論文總整理與研究敘述主檔。
