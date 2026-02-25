# 多代理人主動感知抓取系統 (VLM-RL) 規格文件

本文件描述基於 A2A (Act-to-Reason-to-Act) 架構與 LangGraph 狀態機框架的具身智慧 (Embodied AI) 機器人系統架構。針對 Day 1 實作，以 Mock 代理人鏈路為主。

## 1. 架構與選型

系統採用單機集中式設計，將 **RTX 3090 Server** 作為唯一運行平台，同時執行高層次的 LangGraph 狀態管理決策中樞，以及底層實際感知與動作任務的 Agent 服務。

```mermaid
graph TD
  subgraph Commander [3090 Commander]
    LG[LangGraph Orchestrator]
    Memory[滑動窗口緩衝記憶體 x3]
    Brain[Brain: Gemini/Ollama]
    Logger[TraceLogger: JSONL]
  end
  
  subgraph Executors [RTX 3090 Server]
    Nav[Nav Agent: 8001]
    Grasp[GraspGen Agent: 8002]
    Approach[Approach Agent: 8003]
    View[View Agent: 8004]
  end

  LG --> Brain
  LG <--> Memory
  LG --> Logger
  LG <--> Nav
  LG <--> Grasp
  LG <--> Approach
  LG <--> View
```

## 2. 資料模型

定義 LangGraph 的全局狀態以及 A2A 協議的資料傳輸契約。

```mermaid
classDiagram
  class CommanderState {
    +list history_buffer
    +string current_status
    +dict current_observation
    +int retry_count
  }
  class A2AAgentExecutor {
    +execute(RequestContext, EventQueue)
    +cancel(RequestContext, EventQueue)
  }
  class TaskState {
    +working
    +input_required
    +completed
  }
  class TraceLogEntry {
    +string timestamp
    +string agent_called
    +float decision_latency
    +float execution_latency
    +string reasoning
  }
```

## 3. 關鍵流程

描述從取得影像、推理決策到代理人執行的主要閉迴路系統 (Observe -> Reason -> Act)。

```mermaid
sequenceDiagram
  participant ENV as Environment/Camera
  participant C as Commander (3090)
  participant V as VLM Brain
  participant A as Agents (3090 Server)
  participant L as TraceLogger

  C->>ENV: 取得觀測視角影像 (Observe)
  C->>V: 提供 Context 與歷史動作紀錄
  V-->>C: 回傳 JSON: reasoning & call_module (Reason)
  C->>C: 記錄 Decision Latency
  C->>A: 發送 Task Context 與 EventQueue (Async)
  A-->>C: 更新 TaskState (working/completed) (Act)
  C->>C: 記錄 Execution Latency
  C->>L: 儲存單次追蹤日誌 (JSONL)
  C->>C: 更新最近 3 次的 Memory
```

## 4. 系統脈絡圖

定義整體系統的外部依賴與互動關係。

```mermaid
C4Context
  Person(user, "User / 測試者")
  System(vlm_rl, "VLM RL 抓取系統", "以 LangGraph 進行狀態管理的核心控制平台")
  
  System_Ext(gemini_api, "Gemini API", "提供多模態邏輯推理能力")
  System_Ext(agent_server, "Agent Server (RTX3090)", "提供手臂與底盤控制、GraspNet 辨識服務")

  Rel(user, vlm_rl, "啟動主控迴圈、查看 Log")
  Rel(vlm_rl, gemini_api, "提供影像與提示詞以換取決策", "HTTPS")
  Rel(vlm_rl, agent_server, "雙向 A2A 通訊 (下達命令與獲得反饋)", "HTTPS / JSON")
```

## 5. 容器/部署概觀

Day 1 開發階段中的網路端口部署對應關係全數運行於 3090 主機。

```mermaid
graph TD
  subgraph "Commander Process"
    Main[main.py: 系統進入點]
    LangGraph[LangGraph 狀態機]
  end

  subgraph "Agent Services (Localhost: 8001-8004)"
    S1[FastAPI: 8001 Nav]
    S2[FastAPI: 8002 GraspGen]
    S3[FastAPI: 8003 Approach]
    S4[FastAPI: 8004 View]
  end

  Main --> LangGraph
  LangGraph <--> S1
  LangGraph <--> S2
  LangGraph <--> S3
  LangGraph <--> S4
```

## 6. 模組關係圖

程式內碼元件的解耦設計結構。

```mermaid
graph LR
  Orchestrator[Orchestrator] --> Brain[Brain Node]
  Orchestrator --> Executor[Executor Node]
  Executor --> HTTPClient[Async HTTP Client]
  HTTPClient --> APIs[Agent APIs]
  Orchestrator -. 寫入 .-> ContextMem[Memory/State]
  Orchestrator -. 寫入 .-> TraceLog[TraceLogger]
```

## 7. 流程圖

Day 1 模擬測試流程的詳細步驟。

```mermaid
flowchart TD
  Start[啟動系統] --> SpinMocks[啟動四個 FastAPI Mock 服務]
  SpinMocks --> Loop[啟動 LangGraph Loop]
  Loop --> Obs[讀取影像 / 假影像]
  Obs --> Reason[Brain: 推理出 JSON]
  Reason --> Parse[解析目標代理人與參數]
  Parse --> Execute[呼叫代理人 API]
  Execute --> Wait[等待回傳、計算 Latency]
  Wait --> Log[寫入 JSONL 日誌]
  Log --> MemoryUpdate[更新 State 快取區]
  MemoryUpdate --> TaskCheck{任務是否完成?}
  TaskCheck -- 否 --> Loop
  TaskCheck -- 是 --> Done[結束工作]
```

## 8. 狀態圖

LangGraph Node 轉換之狀態機拓撲。

```mermaid
stateDiagram-v2
  [*] --> OBSERVE
  OBSERVE --> REASON : 影像資料整合完畢
  REASON --> ROUTE_EXECUTION : 產出 JSON 行動計畫
  ROUTE_EXECUTION --> NAV_AGENT : 選擇移動
  ROUTE_EXECUTION --> GRASP_AGENT : 選擇找尋抓取點
  ROUTE_EXECUTION --> APPROACH_AGENT : 選擇逼近物體
  ROUTE_EXECUTION --> VIEW_AGENT : 選擇重整視角
  
  NAV_AGENT --> UPDATE_MEMORY
  GRASP_AGENT --> UPDATE_MEMORY
  APPROACH_AGENT --> UPDATE_MEMORY
  VIEW_AGENT --> UPDATE_MEMORY
  
  UPDATE_MEMORY --> OBSERVE : 繼續迴圈
  UPDATE_MEMORY --> [*] : 抓取成功/終止
```
