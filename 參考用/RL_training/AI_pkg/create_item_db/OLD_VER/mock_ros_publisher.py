import os
import time
import random
import base64
import cv2
import roslibpy

# ======== ROS Bridge 設定 ========
ROSBRIDGE_HOST = 'localhost'
ROSBRIDGE_PORT = 9090

client = roslibpy.Ros(host=ROSBRIDGE_HOST, port=ROSBRIDGE_PORT)
client.run()
print("✅ 已連線到 rosbridge")

# ======== 圖片來源資料夾 ========
IMAGE_FOLDER = '/Users/chentingjie/Desktop/test_pub_pics'

# ======== 載入並編碼成 base64 的輔助函式 ========
def get_encoded_image(image_path: str) -> str:
    """
    讀取 image_path 指定的影像檔，轉成 JPEG 格式後 base64 編碼，
    回傳一個長度較小、適合放在 ROS sensor_msgs/CompressedImage 裡的 base64 字串。
    """
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"找不到圖片: {image_path}")
    # 把影像編碼成 JPEG 二進位
    success, buffer = cv2.imencode('.jpg', img)
    if not success:
        raise RuntimeError(f"無法將圖片轉成 JPEG: {image_path}")
    # 轉成 base64 字串
    encoded = base64.b64encode(buffer).decode('utf-8')
    return encoded

# ======== 將資料夾裡所有檔案列出來，過濾出常見影像副檔名 ========
def list_image_files(folder: str) -> list:
    """
    回傳 folder 底下所有副檔名符合常見影像檔（.jpg/.jpeg/.png）的小寫路徑清單。
    """
    files = []
    for fname in os.listdir(folder):
        lower = fname.lower()
        if lower.endswith('.jpg') or lower.endswith('.jpeg') or lower.endswith('.png'):
            files.append(os.path.join(folder, fname))
    return sorted(files)

# ======== 主要發布迴圈 ========
try:
    image_files = list_image_files(IMAGE_FOLDER)
    if not image_files:
        raise RuntimeError(f"在 {IMAGE_FOLDER} 找不到任何 .jpg/.jpeg/.png 影像檔")

    print(f"🔎 共找到 {len(image_files)} 張影像：")
    for p in image_files:
        print("   ", os.path.basename(p))

    # 進入無限循環，不斷輪播所有檔案
    while client.is_connected:
        for image_path in image_files:
            # 1) 取得不含副檔名的檔名，例如 "apple.jpg" → "apple"
            base = os.path.basename(image_path)
            name, _ext = os.path.splitext(base)

            # 2) 隨機整數座標（0~9）
            rand_point = {
                'x': random.randint(0, 9),
                'y': random.randint(0, 9),
                'z': random.randint(0, 9),
            }

            # 3) 將影像轉成 base64
            try:
                encoded = get_encoded_image(image_path)
            except Exception as e:
                print(f"❌ 讀取或編碼影像失敗：{image_path}，跳過此張。原因：{e}")
                continue

            # 4) 依據檔名動態產生兩個 Topic
            image_topic = roslibpy.Topic(client, f'/object_image_{name}', 'sensor_msgs/CompressedImage')
            point_topic = roslibpy.Topic(client, f'/object_point_{name}', 'geometry_msgs/Point')

            # 5) 在 publish 之前必須先 advertise
            image_topic.advertise()
            point_topic.advertise()

            # 6) publish 座標，格式對應 geometry_msgs/Point
            point_topic.publish(rand_point)
            print(f"📍 已發送座標至 /object_point_{name}：{rand_point}")

            # 7) publish 影像，格式對應 sensor_msgs/CompressedImage
            #    ROS 要求發送的是 {'format': 'jpeg', 'data': '<base64 字串>'}
            image_topic.publish({
                'format': 'jpeg',
                'data': encoded
            })
            print(f"📸 已發送影像至 /object_image_{name}：{base} （{len(encoded)} bytes）")

            # 8) 發送完後，停 2 秒再繼續下一張
            time.sleep(4)

            # 9) 發送結束後，unadvertise（釋放資源，下次迴圈會重新 advertise）
            image_topic.unadvertise()
            point_topic.unadvertise()

        # 10) 全部檔案輪播完一輪後，會回到 for-loop 一開始，繼續下一輪
        # 如果不想重複播放，就在這裡 break 即可；但題目要求一直重複，因此留空
        # break

except KeyboardInterrupt:
    print("❌ 手動中斷模擬程式（Ctrl+C）")

finally:
    # 確保退出前把 client 關閉
    if client.is_connected:
        client.terminate()
    print("💤 已終止 ROS 客戶端")