# Find Agent Server

**Find Agent** 是一個 A2A Agent Server，負責接收相機影像，進行 YOLO 目標辨識並回傳畫框與辨識結果。

## 執行環境

- **Conda 環境名稱**：`a2a_vlm_find`
- **主要依賴**：`ultralytics`, `Pillow`, `a2a-sdk`
- **服務 Port**：8005

### 環境建置

```bash
# 建立專屬虛擬環境 (Python 3.11)
conda create -n a2a_vlm_find python=3.11 -y
conda activate a2a_vlm_find

# 安裝所需依賴套件 (在 VLM_RL 目錄下)
pip install -r find_agent/requirements.txt
```

## 啟動服務

確認在 `a2a_vlm_find` 環境下，於 `VLM_RL` 目錄執行：

```bash
conda activate a2a_vlm_find
python -m find_agent
```

## 功能與 A2A 介面

- **收到訊息**：多機影像。
- **處理邏輯**：使用 `yolo_service.py` 封裝 YOLO 模型進行推論。
- **回傳訊息**：目標的 2D Bounding Box 及畫框結果。
