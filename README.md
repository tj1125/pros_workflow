# VLM-RL 多代理人主動感知抓取系統

基於 **LangGraph** 與 **A2A 協議**的具身智慧機器人抓取系統。

## 系統架構

```
🐳 Docker (本機)
  └── LangGraph 單一進程
        ├── Brain (Gemini/Ollama VLM) — 觀察 → 推理 → 決策
        ├── Nav Agent Node    (A2A Client → RTX3090 InfNAV)
        ├── GraspGen Agent    (A2A Client → RTX3090 InfGrasp)
        ├── Approach Agent    (本地控制 / ROS)
        └── View Agent Node   (A2A Client → RTX3090 InfView)

⚡ RTX 3090 推論伺服器 (A2A Servers)
  ├── Inference NAV   :9001
  ├── Inference Grasp :8007
  └── Inference View  :9003

🤖 Unity 數位孿生 (ROS/Rosbridge)
  ├── Unity Env. Image → Brain
  └── InfoBus ↔ Agent Nodes
```

## 快速開始

### 1. 安裝依賴

```bash
pip install uv
uv sync
```

### 2. 設定環境變數

```bash
cp .env.example .env
# 編輯 .env 填入 GOOGLE_API_KEY 等設定
```

### 3. Mock 模式執行（無需 VLM 或 GPU）

```bash
uv run python main.py --mock
```

### 4. 真實模式執行

```bash
# 確保 .env 中已設定 GOOGLE_API_KEY、INF_NAV_URL 等
uv run python main.py --no-mock
```

### 5. 執行測試

```bash
uv run python test_client.py            # Mock + A2A 連線測試
uv run python test_client.py --mock-only # 僅 Mock 閉環測試
```

## Docker 部署 & 導航系統啟動

### 啟動 Nav 導航系統（先於 `run` 啟動）

```bash
# 第一次或 Dockerfile.nav2 有改時，先重建本地 Nav image
./launch_nav.sh --rebuild

# 前景模式（建議除錯時使用，可直接看 Nav2 log）
./launch_nav.sh

# 背景模式（啟動後可繼續在同一終端執行 run）
./launch_nav.sh -d

# 停止 Nav2
./launch_nav.sh --stop
```

### 進入 VLM-RL 開發容器

```bash
# 第一次進入會自動 build 本地 image
./enter_docker.sh

# Dockerfile 有改時強制重建 image
./enter_docker.sh --rebuild
```

如果你要在 Linux 桌面主機上開 PyBullet GUI 或其他 X11 視窗，請用獨立入口：

```bash
./enter_docker_x11.sh

# Dockerfile 有改時強制重建 image
./enter_docker_x11.sh --rebuild
```

`enter_docker_x11.sh` 會做這些事：

- 檢查主機是 Linux 且有 `DISPLAY`
- 檢查 `/tmp/.X11-unix` 是否存在
- 以本地 `vlm-rl-env:latest` 為底，建立一個含 `OMPL Python bindings` 的專用 X11 image
- 額外補齊 `Pillow` / `PyYAML`，讓 Camera_Car 深度 PNG 與 YAML 內參可直接在容器內解析
- 透過 `xhost +SI:localuser:root` 暫時授權本機 root container 使用 X11
- 把 `DISPLAY` 和 X11 socket 傳進容器

離開容器後，它會自動收回剛剛加的 X11 權限。

離開 `enter_docker.sh` 後，dev container 的 `/workspaces/build`、`/workspaces/install`、`/workspaces/log` 會自動清空；下次重新進容器時請重新執行 `r`。

容器內的 ROS 工作流：

```bash
# 建置並載入 ROS workspace
r

# 直接啟動控制節點
ros2 run car_control_pkg car_control_node
ros2 run arm_control_pkg arm_control_node
ros2 run keyboard_mode_interface_pkg keyboard_control_node

# 啟動導航 launch
ros2 launch nav_goal_bridge_pkg navigation.launch.py
```

同一輪 dev container 內如果你已經跑過 `r`，overlay 會自動載入；真的要手動重載才用：

```bash
ros_ws_source
```

## 目錄結構

```
VLM_RL/
├── main.py                # 系統進入點 (Click CLI)
├── pyproject.toml         # uv 套件管理
├── Dockerfile             # VLM-RL 開發容器建置
├── Dockerfile.nav2        # Nav2 專用 image（含 laser_scan_matcher）
├── enter_docker.sh        # 進入開發容器
├── container_env.sh       # 容器 shell rc + ROS workspace helper
├── launch_nav.sh          # 一鍵啟動 Nav 導航系統
├── docker-compose-nav2.yml # Nav2 容器編排
├── .env.example           # 環境變數範本
├── test_client.py         # 端對端測試
├── commander/
│   ├── brain.py           # VLM 推理中樞 (Gemini/Ollama)
│   ├── orchestrator.py    # LangGraph 狀態圖 (5 節點)
│   ├── nav_move_runner.py # ROS 2 Nav2 Action Client
│   ├── state.py           # CommanderState TypedDict
│   └── logger.py          # JSONL 追蹤日誌
├── agents/
│   ├── nav_agent.py       # 導航 Node (A2A Client)
│   ├── grasp_agent.py     # 抓取生成 Node (A2A Client)
│   ├── approach_agent.py  # 靠近 Node (本地控制)
│   ├── view_agent.py      # 視野調整 Node (A2A Client)
│   └── schemas.py         # Pydantic 資料模型
├── tools/
│   └── car_control/src/
│       ├── nav_goal_bridge_pkg/      # Nav2 launch/config/map + /goal_pose bridge
│       ├── car_control_pkg/          # 車體控制與自動導航 action server
│       └── action_interface/         # ROS 2 action 定義
├── config/                # 其他設定檔 (camera, objects)
└── logs/                  # JSONL 追蹤日誌輸出
```

## 環境變數

| 變數 | 說明 | 預設值 |
|---|---|---|
| `VLM_PROVIDER` | `google` 或 `ollama` | `google` |
| `GOOGLE_API_KEY` | Gemini API Key | — |
| `MOCK_MODE` | `true`/`false` | `true` |
| `INF_NAV_URL` | RTX3090 Nav 推論伺服器地址 | — |
| `INF_GRASP_URL` | RTX3090 Grasp 推論伺服器地址 | — |
| `INF_VIEW_URL` | RTX3090 View 推論伺服器地址 | — |
| `ROSBRIDGE_URL` | Unity Rosbridge WebSocket | `ws://localhost:9090` |
| `ROS_DOMAIN_ID` | ROS 2 DDS Domain，必須和相機 / Nav2 同步 | `1` |
| `ROS_PYTHON_BIN` | 具有 `rclpy` 的 Python 執行檔 | `/usr/bin/python3` |
| `ROS_SETUP_BASH` | ROS 2 base setup script | `/opt/ros/humble/setup.bash` |
| `ROS_OVERLAY_SETUP_BASH` | ROS overlay setup script | `/workspaces/install/setup.bash` |
