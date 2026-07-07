# pros_workflow

部署於 Unity/ROS2 環境的主動感知抓取系統。本機 Commander 用 LangGraph 維護任務狀態、透過 VLM（預設 Gemini）做高層決策，呼叫本機 ROS2 導航/控制流程與 RTX 3090 上的 A2A 推論服務，完成找物 → 導航 → 抓取姿態生成 → 底盤靠近與手臂/夾爪收尾。

![System architecture](docs/system_architecture.png)

## 程式碼結構

| 路徑 | 內容 |
|---|---|
| `src/` | ROS2 packages。`workflow_bringup`（runtime launch 匯總）、`nav_goal_bridge_pkg`（Nav2 + goal bridge）、`car_control_pkg`、`arm_control_pkg`、`action_interface`、`robot_description`。 |
| `workflow/` | 非 ROS 的 Commander。`web_main.py`/`main.py` 進入點、`commander/`（LangGraph orchestrator、flows、brain、state/contracts、camera/nav/perception/storage/web）、`agents/`（A2A client 與 car approach runtime）、`config/`（cameras/objects/runtime yaml）。 |
| `3090server/pros_workflow/` | RTX 3090 端 A2A 推論服務（item-info、grasp）。見該目錄 [README](3090server/pros_workflow/README.md)。 |
| `docker/` | build workflow image 用的 Dockerfile。 |
| `scripts/` | 容器內 shell helper（`env.sh` 定義 `run`/`web`）與 ROS runtime 啟動（`start.sh`）。 |
| `docs/` | 架構圖與部署規格。 |

## 執行流程

Commander 是一個 LangGraph 狀態機（entry `greeting_node`），主要抓取節點串接：`find_node` → `get_item_info_no_sam3d_node` → `nav_move_node` → `observe_node` → `reason_node`（VLM 決策）→ `major_nav_node` / `car_grasp_node` → `car_approach_node`。VLM 每回合輸出 `BrainDecision`（`call_module` ∈ `major_nav_node` / `grasp_agent` / `car_approach_agent` / `DONE`）。

![LangGraph flow](docs/langgraph_flow.png)

## Quick Start

先確認根目錄有 `.env`（VLM provider、`INF_*` 服務 URL 等）。

Build image：

```bash
cd /home/scream/TJ/pros_workflow
docker build -t pros_workflow_image:latest -f docker/Dockerfile .
```

Terminal 1 — 啟動 ROS runtime：

```bash
cd /home/scream/TJ/pros_workflow
./run.sh            # 進入 pros_workflow 容器
r                   # build ROS workspace（PROS image alias）
scripts/start.sh    # launch Nav2 / car / arm / rosbridge
# then play unity.
```

Terminal 2 — 開 web 跑 workflow：

```bash
cd /home/scream/TJ/pros_workflow
./run.sh
web 8080            # 啟動 workflow/web_main.py
# then open http://localhost:8080.
```

3090 端另外啟動 A2A 服務（見 [3090server/pros_workflow/README.md](3090server/pros_workflow/README.md)）。

## Notes

- `./run.sh` 只負責進 Docker，不會自動 build image；容器內工作目錄是 `/workspace/pros_workflow`，venv 在 `/opt/pros_workflow_venv`。
- `./run.sh` 會 publish `8080:8080`（web）和 `9090:9090`（rosbridge）。
- `r` 是 PROS image 內建 alias：`source /workspaces/rebuild_colcon.rc`。
- `scripts/start.sh` 會啟動 Nav2、car control、arm control、rosbridge。
- `run` 執行 `workflow/main.py --no-mock`（CLI）；`web 8080` 執行 `workflow/web_main.py`。
- 改 ROS code 後跑 `r`；改 Dockerfile 或 `workflow/pyproject.toml` 後重新 build image。
