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
  ├── Inference Grasp :9002
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

## Docker 部署

```bash
# 複製並填寫設定
cp .env.example .env

# 建置並啟動
docker compose up --build

# 或使用 podman
podman build . -f Containerfile -t vlm-rl-system
podman run --env-file .env vlm-rl-system
```

## 目錄結構

```
VLM_RL/
├── main.py                # 系統進入點 (Click CLI)
├── pyproject.toml         # uv 套件管理
├── Containerfile          # Docker 建置
├── docker-compose.yml     # 容器編排
├── .env.example           # 環境變數範本
├── test_client.py         # 端對端測試
├── commander/
│   ├── brain.py           # VLM 推理中樞 (Gemini/Ollama)
│   ├── orchestrator.py    # LangGraph 狀態圖 (5 節點)
│   ├── state.py           # CommanderState TypedDict
│   └── logger.py          # JSONL 追蹤日誌
├── agents/
│   ├── nav_agent.py       # 導航 Node (A2A Client)
│   ├── grasp_agent.py     # 抓取生成 Node (A2A Client)
│   ├── approach_agent.py  # 靠近 Node (本地控制)
│   ├── view_agent.py      # 視野調整 Node (A2A Client)
│   └── schemas.py         # Pydantic 資料模型
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
