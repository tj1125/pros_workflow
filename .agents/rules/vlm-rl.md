---
trigger: manual
---

## 專案名稱：基於 LangGraph 與 A2A 協議之多代理人主動感知抓取系統

本專案開發一套基於 **LangGraph 與 A2A 通訊架構** 的具身智慧 (Embodied AI) 機器人系統。核心特色在於導入 **LangGraph 狀態機框架**，以 **Gemini VLM 或 Ollama VLM** 為高層推理中樞，透過結構化的 **A2A 通訊協議** 與 GPU 推論伺服器協同運作，並透過 Rosbridge 與 Unity 數位孿生環境進行控制與感知資料交換。本系統旨在解決複雜環境中，因動態遮擋或視角不佳導致的抓取失敗問題。

### 1. 系統架構：LangGraph 狀態驅動中樞

系統用 **LangGraph** 構建具備循環與記憶能力的決策狀態圖：

* **狀態化決策 (Stateful Orchestration)**：利用 LangGraph 的 `State` 機制維護全局變數（如場景資訊、機器人狀態與嘗試歷史），確保 VLM 能基於當前觀察與歷史資訊進行連續推理，避免無效的重複嘗試。

* **A2A 推論通訊協議**：定義標準化的 HTTPS JSON 通訊契約。運行於本機 Docker 中的 Agent Node 作為 A2A Client，向 NVIDIA RTX 3090 推論伺服器發送推論請求（如 grasp pose 生成或 policy inference），並接收推論結果後轉換為機器人控制指令，形成 **觀察 (Observe) -> 推理 (Reason) -> 行動 (Act)** 的閉環循環。

* **分散式部署**：系統透過網路協議將決策、推論與模擬環境解耦。LangGraph 與 Agent 運行於本機 Docker，GPU 密集推論任務部署於 NVIDIA RTX 3090 伺服器，而機器人與環境模擬運行於 Unity 數位孿生環境，並透過 Rosbridge 與本機進行資料交換。

### 2. 四大代理人模組 (Agent Nodes)

在 LangGraph 中，各個功能模組被封裝為獨立的「節點 (Nodes)」，並透過 A2A 協議向推論伺服器發送請求：

* **Nav Agent (導航節點)**：負責控制機器人底盤移動至新的觀察位置。當 VLM 判斷當前視角不足時，透過導航調整機器人位置以改善觀察條件。

* **GraspGen Agent (抓取生成節點)**：負責生成抓取位姿。Agent 會將感知資料透過 A2A 協議傳送至 RTX 3090 推論伺服器，由 grasp generation 模型進行推論並回傳抓取位姿與相關資訊。

* **Approach Agent (靠近節點)**：負責引導夾爪接近目標物。透過向推論伺服器請求控制策略推論結果，逐步調整機械手臂位置以接近預抓取點。

* **View Agent (視野調整節點)**：負責調整機械手臂或機器人姿態以改善觀察視角，使系統能獲得更清晰的感知資訊，提升後續抓取成功率。

所有 Agent Node 均運行於本機 Docker 環境中，並作為 A2A Client 與 RTX 3090 推論伺服器通訊。

### 3. 核心技術價值

* **自主邏輯恢復 (Self-Recovery)**：透過 LangGraph 的循環邊（Cyclic Edges），系統可根據最新觀察結果重新進行決策，持續調整觀察與動作策略，而非僅依賴單次推理結果。

* **高度解耦之架構設計**：系統將決策（LangGraph）、推論（RTX 3090 Server）與模擬環境（Unity）分離，透過 Rosbridge 與 A2A 協議進行通訊，使各模組可獨立開發與替換。

* **支援多步驟閉環任務執行**：系統可透過持續的觀察、推理與動作更新，逐步完成複雜抓取任務，而非僅依賴單次感知與動作。

### 4. 專案願景

本研究致力於開發具備**閉環決策能力**的機器人系統。透過 LangGraph 結合 Vision-Language Model 與分散式推論架構，使機器人能根據環境持續調整其行為。本系統驗證了 LangGraph 與分散式推論架構在具身智慧任務中的可行性，並為未來 VLM 驅動之機器人系統提供一個具備可擴展性與模組化的開發架構。