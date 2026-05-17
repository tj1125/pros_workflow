# View Agent — Unity 端

此目錄包含需附加至 Unity 機器人的感測器與物理環境 C# 腳本。

## 依賴套件
- **Unity Robotics Hub** (`com.unity.robotics.ros-tcp-connector`)
- **ROS TCP Endpoint** (執行於 ROS/ROS2 容器中)

## 腳本說明

1. `Sensors/ViewAgentSensorPublisher.cs`
   - 掛載在機器人本體上，設定 RGB / Depth 攝影機、IMU Transform 以及手臂關節。
   - 定期向 Rosbridge 發布影像、關節狀態、慣性數據。

2. `Sensors/OcclusionRaycastSensor.cs`
   - 利用網格射線計算相機到目標物 Bounding Box 間被障礙物遮擋的比例 (0~1)。

3. `Environment/PhysicsMetricPublisher.cs`
   - 負責計算並發布整合的物理指標至 `/view_agent/physics_metrics` (JSON格式)。

4. `Environment/EnvRandomizer.cs`
   - 訂閱 `/env_randomize` 主題。
   - 當收到隨機化 JSON 指令時，隨機重置目標物位置、障礙物分佈與光線，提供 RL 訓練多樣性。
