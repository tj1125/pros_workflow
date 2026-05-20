# Find Agent Server

`find_agent` 是 RTX 3090 端的 A2A YOLO detection server。它接收多相機 RGB 影像，對每張圖做 YOLO detection，回傳全域編號的 bbox metadata 與標註影像。

目前 Commander 主流程主要直接使用 `/world_position_data` 做候選 instance 確認；此服務仍保留給需要純 YOLO 掃描或除錯的流程。

## 執行環境

- Conda 環境建議：`a2a_vlm_find`
- 主要依賴：`ultralytics`, `Pillow`, `a2a-sdk`
- Port：8005

```bash
conda create -n a2a_vlm_find python=3.11 -y
conda activate a2a_vlm_find
pip install -r find_agent/requirements.txt
```

## 啟動

在 `3090server/VLM_RL` 下：

```bash
conda activate a2a_vlm_find
EXTERNAL_IP=192.168.1.10 python -m find_agent
```

Commander `.env`：

```env
INF_FIND_URL=http://192.168.1.10:8005
```

## A2A Request

`parts[0].text` 是 JSON：

```json
{
  "camera_images": {
    "Camera_Room1_12": "...base64...",
    "Camera_Room1_13": "...base64..."
  },
  "task_description": "抓取紅色蘋果",
  "target_object": {"id": "apple", "label": "紅色蘋果"}
}
```

## A2A Response

回傳 JSON text，主要欄位：

```json
{
  "yolo_detections": {
    "1": {
      "camera": "Camera_Room1_12",
      "bbox": [100, 100, 300, 300],
      "label": "apple",
      "conf": 0.92,
      "annotated_image_base64": "..."
    }
  }
}
```
