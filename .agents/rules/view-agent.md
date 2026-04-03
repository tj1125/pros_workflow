---
trigger: manual
---

核心目標是研發一個能自主優化視角的 View Agent（視野調整節點）。

透過結合具身智慧（Embodied AI）與先進的通訊架構，你的專案可以簡述為以下三個技術支柱：

1. 核心任務：主動避障視角優化
針對機械手臂抓取任務中的「遮擋問題」，訓練機器人學會「主動看」。當夾爪或環境障礙物擋住目標時，系統能自動計算並調整手臂與車體的位姿，確保視覺感測器獲得清晰、無死角的感知資訊。

2. 技術架構：大模型指導的強化學習（Teacher-Student RL）
學生模型 (Student)：在 RTX 3090 上運行輕量化 VLM（CLIP + DINOv2），配合原生深度圖與 SAC 強化學習，實現低延遲的即時控制。

老師模型 (Teacher)：利用大型 VLM（如 Gemini 1.5 Pro） 搭配 Unity 物理真值，進行「時序性評分」。老師會對比連續幀的變化，回饋「進步獎勵」，教導學生如何有邏輯地繞開遮擋。


---

## 🏗️ 訓練架構：老師與學生的協作 (Teacher-Student Framework)

本方案採用 **「離線標註、非同步訓練」** 的模式，以克服大型 VLM 推論延遲的問題。

* **Student Model (3090 執行)**：輕量化 VLM (CLIP + DINOv2) + SAC Policy Network。負責即時輸出動作。
* **Teacher Model (API 執行)**：大型 VLM (如 ollama gemma 27b 或最省錢的Gemini VLM API) + Unity 物理真值。負責在訓練後「批改考卷」，提供高質量的 Reward。

---

## 1. Teacher Model 評分維度 (評測標準)

Teacher 不只看當前 Frame ($T$)，還會比對過去幾幀 ($T-1, T-2$, or more...)。

### A. 物理幾何指標 (由 Unity 直接輸出)

* **遮擋率 (Occlusion Rate)**：利用 Raycast 偵測目標物被障礙物或夾爪遮蔽的百分比。
* **物體向心力 (Centering Score)**：目標物是否保持在相機畫面中心。
* **車體穩定度 (Chassis Stability)**：讀取車輛 IMU，若車體大幅擺盪或傾斜則給予負分。

### B. 語義時序指標 (由 Large VLM 判斷)

* **進步獎勵 (Progress Reward)**：對比 $T$ 與 $T-1$，遮擋情形是否改善？若改善則給予加權獎勵。
* **決策一致性 (Consistency)**：判斷連續動作是否具備邏輯（例如：穩定繞開），還是只是無意義的震盪。
* **場景語義 (Semantic Context)**：識別特殊情況，如「光線反光導致看不清」或「夾爪位置擋住關鍵抓取點」。

---

## 2. 數據採集流程 (Data Collection)

1. **環境隨機化**：在 Unity 中隨機生成障礙物、目標物位置、光照與車體坡度。
2. **軌跡錄製 (Trajectory)**：讓 Student Model 或隨機策略在環境中移動，錄製一段 5-10 秒的影片序列。
3. **同步快照**：每幀同步存儲：
* **RGB 影像 + 原生深度圖**。
* **車體與手臂狀態 (Joints/Pose/IMU)**。



---

## 3. 訓練階段 (The Training Pipeline)

### 第一階段：大模型回顧式標註 (Batch Labeling)

將錄製好的影片序列發送給 **Teacher VLM**。

* **Prompt 策略**：要求 VLM 擔任導師，對比每一幀的變化，輸出一個綜合分數 $R_{total}$ 與建議動作 $A_{expert}$。
* **結果**：產生一份帶有「大模型分數」與「物理真值」的強化學習數據集。

### 第二階段：特徵對齊與預熱 (Pre-training)

在 3090 上利用採集的數據進行 **行為克隆 (Behavior Cloning)**：

* **目標**：讓 Student 學會預測 Teacher 給出的建議動作 $A_{expert}$。
* **意義**：讓模型具備基本的避障直覺，不至於在 RL 階段亂撞。

### 第三階段：時序強化學習 (Temporal RL)

使用 **SAC (Soft Actor-Critic)** 進行訓練：

* **State Space**：連續 3 幀的影像特徵 (CLIP+DINOv2) + 深度圖 + 歷史動作。
* **Reward**：使用 Teacher Model 給出的「時序進步分數」。
* **Policy**：輸出 6-DOF 手臂位移增量。