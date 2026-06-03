# VLM_RL

本專案是部署於 Unity/ROS2 環境的主動感知抓取系統。主控端使用 LangGraph 維護任務狀態，透過 VLM 做高層決策，並呼叫本機 ROS2 控制流程與 RTX 3090 上的 A2A 推論服務完成找物、導航、抓取姿態生成與底盤/手臂收尾。

ROS 節點放在 `src/`，非 ROS 的 Commander workflow 放在 `workflow/`，RTX 3090 端 A2A 推論服務放在 `3090server/VLM_RL/`。目前統一用 Docker 進入環境，透過 `r` build ROS，`scripts/start.sh` 啟動 ROS runtime，`run` 或 `web 8080` 啟動 workflow。

## Workflow

![LangGraph flow](docs/langgraph_flow.png)

## Quick Start

先確認根目錄有 `.env`。

Build image：

```bash
cd /home/scream/TJ/VLM_RL
docker build -t pros_workflow_image:latest -f docker/Dockerfile .
```

Terminal 1：啟動 ROS runtime

```bash
cd /home/scream/TJ/VLM_RL
./run.sh
r
scripts/start.sh
# then play unity.
```

Terminal 2：開 web 跑 workflow

```bash
cd /home/scream/TJ/VLM_RL
./run.sh
web 8080
# then open http://localhost:8080.
```

## Notes

- `./run.sh` 只負責進 Docker，不會自動 build image。
- `./run.sh` 會 publish `8080:8080` 和 `9090:9090`。
- `r` 是 PROS image 內建 alias：`source /workspaces/rebuild_colcon.rc`。
- `scripts/start.sh` 會啟動 Nav2、car control、arm control、rosbridge。
- `run` 會執行 `workflow/main.py --no-mock`。
- `web 8080` 會啟動 `workflow/web_main.py`。
- 改 ROS code 後跑 `r`。
- 改 Dockerfile 或 `workflow/pyproject.toml` 後重新 build image。
- 不需要本機 `.venv_linux`、`.uv_python`、`uv`。
