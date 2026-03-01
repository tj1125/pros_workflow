# 3090 Server — VLM_RL Agent Services

這個資料夾包含所有在 **RTX 3090 伺服器**上執行的 A2A Agent Server 程式碼。

## 目錄結構

此伺服器預期將運行多個專注於不同感知或決策任務的 Agent。

```
3090server/VLM_RL/
├── README.md                    # 本文件 (總體架構概觀)
├── a2a_utils/                   # 共用 A2A 回應與工具
│   ├── __init__.py
│   └── response.py
│
├── models/                      # 共用的模型權重存放區 (.pt, .onnx 等)
│
├── find_agent/                  # Find Server (YOLO 目標辨識)
│   ├── README.md                # 專屬環境與啟動說明
│   ├── requirements.txt
│   └── ...
│
└── get_item_info_agent/         # Get Item Info Server (Stereo 3D 感知)
    ├── README.md                # 專屬環境與啟動說明、A2A 訊息格式
    ├── requirements.txt
    └── pipeline/                # 獨立的感知流程
```

---

## 虛擬環境與 Agent 列表

每個 Agent 擁有獨立的運作邏輯，有些 Agent 需要極為特定的深度學習套件（例如 CUDA extensions）。為了確保升級與維護的穩定性，部分 Agent 會被分配到**專屬的 Conda 環境**。

| A2A Agent 模組名稱 | Conda 虛擬環境名稱 | 通訊 Port | 負責功能 / 推論內容 |
|--------------------|--------------------|-----------|--------------------|
| `find_agent` | **`a2a_vlm_find`** | 8005 | 接收多相機影像，進行 YOLO 目標辨識並畫框 |
| `get_item_info_agent` | **`get_item_info_agent`** | 8006 | 針對所選目標，推算 3D 空間位置、邊界及抓取姿態 |
| `nav_agent` *(未來規劃)* | **`a2a_vlm_nav`** | 待定 | 接收避障與相機資訊，推論底盤移動點 |
| `grasp_agent` *(未來規劃)* | **`a2a_vlm_grasp`**| 待定 | 接收點雲，生成 6D 抓取姿態 |
| `approach_agent` *(未來規劃)* | **`a2a_vlm_approach`**| 待定 | 接收抓取姿態，產生最後靠近的手臂控制策略 |

## 個別 Agent 說明

有關各個 Agent 的**環境安裝步驟**、**啟動方式**與 **A2A 傳入/傳出參數格式**的詳細資訊，請參見各個資料夾內的 `README.md`：

- [Find Agent 說明文件](./find_agent/README.md)
- [Get Item Info Agent 說明文件](./get_item_info_agent/README.md)

