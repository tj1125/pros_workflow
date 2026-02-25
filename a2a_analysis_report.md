# A2A SDK 官方範例對比分析報告

根據研讀 `a2a-samples/samples/python/agents/langgraph` 以及 `helloworld` 範例，並對齊目前專案的 `vlm_rl` 架構，總結出以下三個我們專案目前 **尚未實作/有待加強** 的關鍵要件：

## 1. 缺乏標準化的 A2A Client 呼叫 (最核心的差異)

**目前實作：**
我們在 `Commander` 的 `Orchestrator._node_execute` 中，只使用了原生的 `httpx.get(url)` 進行簡單的 API 存活（Ping）測試。

**官方範例：**
官方要求 Commander 必須使用 `a2a.client.A2AClient` 發送基於 **JSON-RPC 2.0** 格式的載荷，而非單純的 REST API。
通訊流程應為：
1. 透過 `A2ACardResolver` 獲取目標 Agent 的 `AgentCard`。
2. 實例化 `A2AClient(httpx_client, agent_card)`。
3. 把指令透過 `MessageSendParams` 封裝，給予唯一的 `message_id` 與 `role`。
4. 使用 `client.send_message(request)` 或 `client.send_message_streaming(request)` 發送請求。

*如果我們不實踐這一步，底層四個 Agent 上的 `A2AStarletteApplication` 其實沒有真正在「工作」，因為它只聽得懂標準格式的 JSON-RPC 封包。*

## 2. 缺乏 Context ID 與 Task ID 的追蹤機制

**目前實作：**
在 `TraceLogger` 與 LangGraph 的 `Memory Buffer` 內，我們是用自定義的 String 參數（如 `"call_module": "nav_agent"`）來辨識歷史紀錄。

**官方範例：**
A2A 的狀態管理高度依賴 **Task / Context 生命週期**。
* **Task ID**: 單次動作的身份證。
* **Context ID**: 整個連續對話或作業環境的身份證。
當我們送出動作給 `Approach Agent` 失敗後，要切回 `Nav Agent`，如果沒有維持同一個 `Context ID`，各路 Agent 會視為全新的任務。在未來要實作循環推理時，導入原生的上下文 ID 將能避免狀態錯亂。

## 3. 未正確處理 Agent 的回傳型態 (Artifacts vs Status)

**目前實作：**
`BaseMockAgentExecutor` 直接粗暴地回傳一段文字 `f"[{self.agent_name}] 執行完畢... "`。

**官方範例：**
A2A 規範代理人的回傳包含兩種主要資訊：
* **Status Updates**: 例如 `working`, `input-required`, `completed`。
* **Artifacts**: 執行後產生的具體數據。
對於我們具身機器人的情境，
- `Grasp Agent` 的推論結果 (6DoF 位姿) 理應被包裝為 **Artifact** 送回 Commander。
- 當 `View Agent` 發現完全看不到目標，需要 Commander 重新決策時，它應該回傳 **`input-required` (或是自定義 Exception)**，而不是單純的 `success=False` 字串。

---

## 總結與修復建議 (Day 2 提頭)

若要讓我們的系統符合正統的 A2A Agentic 系統規範，我們需要在 `Orchestrator` 內實作下方的 `call_a2a_agent` 核心邏輯：

```python
from a2a.client import A2AClient, A2ACardResolver
from a2a.types import SendMessageRequest, MessageSendParams
import uuid

async def call_a2a_agent(http_client, url, task_text, context_id=None):
    # 1. 解析 Agent Card
    resolver = A2ACardResolver(httpx_client=http_client, base_url=url)
    agent_card = await resolver.get_agent_card()
    
    # 2. 建立標準客戶端
    client = A2AClient(httpx_client=http_client, agent_card=agent_card)
    
    # 3. 組裝 JSON-RPC 請求
    payload = {
        'message': {
            'role': 'user',
            'parts': [{'kind': 'text', 'text': task_text}],
            'message_id': uuid.uuid4().hex,
        }
    }
    if context_id:
        payload['message']['context_id'] = context_id
        
    request = SendMessageRequest(
        id=str(uuid.uuid4()), 
        params=MessageSendParams(**payload)
    )
    
    # 4. 發送並等待 Result Artifacts
    response = await client.send_message(request)
    return response
```

## 4. 關於 LangGraph 的部署位置 (回覆使用者的疑問)

**使用者的見解完全正確：LangGraph 真的只需要在 Commander 端！**

在 `a2a-samples/samples/python/agents/langgraph` 這個官方範例中，官方是把「LangGraph」包在一個 A2A Agent 裡面對外當作 Service。但在我們 `VLM_RL` 這種主從架構 (Orchestrator-Agents) 中，我們設計的「四大 Agent (Nav, Grasp, Approach, View)」其實只是**單一動作執行者 (Tools/Executors)**，它們內部通常只需要：
1. Rule-based 邏輯 (Nav)
2. 呼叫 PyTorch 推論 (Grasp)
3. 執行 RL 演算法 (Approach/View)

這類 Agent **不需要** 自己跑複雜的多回合狀態機，所以它們單純使用 `A2AStarletteApplication` 加上我們寫的 `BaseMockAgentExecutor` 直接處理請求就夠了（頂多內部加上一些非同步處理）。

**結論：**
LangGraph 這個「狀態圖引擎」確實應該專職待在 `Commander/Orchestrator` 裡面。由 Commander 利用 LangGraph 決定下一步要找哪個 Agent，然後 Commander 扮演 `A2AClient` 透過網路對底層的 FastAPI (A2A Server) 發送標準化的操作指令（就像上述的 `call_a2a_agent` 函式那樣）。我們的 `spec.md` 設計本身就是走這套中樞神經系統架構，非常正確。
