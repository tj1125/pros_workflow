import os
# 啟用 MPS fallback 機制，必須在 import torch 之前
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
import psycopg2
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
import roslibpy
import base64
import json
import re
import torch
import ollama  # Ollama Python SDK for embeddings

# ======== 路徑設定 ========
LOCAL_FOLDER = "/Users/chentingjie/self-hosted-ai-starter-kit/shared/pics_db"
DB_IMAGE_PREFIX = "/data/shared/pics_db"
IP = "192.168.75.62"

# ======== 初始化 BLIP ========
device = "mps" if torch.backends.mps.is_available() else "cpu"
processor = BlipProcessor.from_pretrained('Salesforce/blip-image-captioning-base')
model = BlipForConditionalGeneration.from_pretrained('Salesforce/blip-image-captioning-base').to(device)
print(f"🔧 使用裝置: {device}")

# ======== PostgreSQL 設定 ========
conn = psycopg2.connect(
    dbname="item_in_house_db",
    user="item_in_house_user",
    password="Aa101201301401",
    host="localhost"
)
conn.autocommit = True
cursor = conn.cursor()

# ======== 座標暫存區 ========
coords_dict = {}

# ======== 資料處理函數（UPSERT，含 embedding + meta） ========
def save_to_db(name, coord, caption, embedding):
    image_path = f"{DB_IMAGE_PREFIX}/{name}.jpg"
    coordinate_json = {'x': coord['x'], 'y': coord['y'], 'z': coord['z']}
    meta = json.dumps({'image_path': image_path, 'coordinate': coordinate_json})

    cursor.execute(
        """
        INSERT INTO object_data (image_path, coordinate, caption, embedding, meta)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (image_path) DO UPDATE
          SET coordinate  = EXCLUDED.coordinate,
              caption     = EXCLUDED.caption,
              embedding   = EXCLUDED.embedding,
              meta        = EXCLUDED.meta,
              created_at  = CURRENT_TIMESTAMP
        """,
        (image_path, json.dumps(coordinate_json), caption, embedding, meta)
    )
    print(f"✅ 資料庫更新/插入成功: {name}")

# ======== 生成 caption ========
def generate_caption(image_path, num_captions=1):
    image = Image.open(image_path).convert('RGB')
    inputs = processor(images=image, return_tensors='pt')
    pixel_values = inputs['pixel_values'].to(device)

    captions = []
    for _ in range(num_captions):
        out = model.generate(
            pixel_values=pixel_values,
            max_length=40,
            num_beams=5,
            temperature=1.0,
            do_sample=True,
            top_k=50,
            top_p=0.95
        )
        captions.append(processor.decode(out[0], skip_special_tokens=True))
    return captions

# ======== 生成 embedding（透過 Ollama SDK） ========
def embed_text(text):
    response = ollama.embeddings(model='nomic-embed-text:latest', prompt=text)
    return response.embedding

# ======== ROS 訂閱 callback ========
def handle_point(name):
    def callback(message):
        coords_dict[name] = message
        print(f"📍 座標更新: {name} -> {message}")
    return callback


def handle_image(name):
    def callback(message):
        try:
            image_bytes = base64.b64decode(message['data'])
            file_path = os.path.join(LOCAL_FOLDER, f"{name}.jpg")
            os.makedirs(LOCAL_FOLDER, exist_ok=True)
            with open(file_path, 'wb') as f:
                f.write(image_bytes)

            captions = generate_caption(file_path)
            merged = ' '.join(captions)

            coord = coords_dict.get(name)
            if coord is None:
                print(f"⚠️ 找不到座標：{name}")
                return

            embedding = embed_text(merged)
            save_to_db(name, coord, merged, embedding)
            print(f"📸 圖片處理完成: {name} -> {file_path}")

        except Exception as e:
            print(f"❌ 錯誤處理 {name}：{e}")
    return callback

# ======== 啟動 roslibpy 客戶端 ========
def start_roslibpy_client():
    client = roslibpy.Ros(host=IP, port=9090)
    client.run()
    time.sleep(1)

    topics = client.get_topics()
    print(f"🧵 所有 ROS topics: {topics}")

    object_names = set()
    for topic in topics:
        match = re.match(r"/object_(?:point|image)_(.+)", topic)
        if match:
            object_names.add(match.group(1))

    print(f"🔍 偵測到物件: {object_names}")
    if not object_names:
        print("⚠️ 目前沒有任何符合的 object topic，請確認 publisher 是否已啟動。")

    for name in object_names:
        point_topic = roslibpy.Topic(client, f"/object_point_{name}", 'geometry_msgs/Point')
        image_topic = roslibpy.Topic(client, f"/object_image_{name}", 'sensor_msgs/CompressedImage')
        point_topic.subscribe(handle_point(name))
        image_topic.subscribe(handle_image(name))
        print(f"📡 訂閱: {point_topic.name}, {image_topic.name}")

    try:
        while client.is_connected:
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("❌ 手動中斷")
        client.terminate()

if __name__ == '__main__':
    start_roslibpy_client()