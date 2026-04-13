import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
import os

from arm_env import ArmEnv

def main():
    env = DummyVecEnv([lambda: Monitor(ArmEnv())])

    # 檢查是否已有儲存的模型
    model_path = "ppo_arm_model.zip"

    if os.path.exists(model_path):
        print("✅ 找到已存的模型，從上次繼續訓練...")
        model = PPO.load("ppo_arm_model", env=env)
    else:
        print("🆕 沒有找到舊模型，從頭開始訓練...")
        model = PPO("MlpPolicy", env, verbose=1)

    # 設定定期儲存的 CheckpointCallback
    checkpoint_callback = CheckpointCallback(
        save_freq=2000,  # 每 5000 steps 存一次
        save_path='./checkpoints/',
        name_prefix='ppo_arm_model'
    )

    # 開始訓練
    model.learn(
        total_timesteps=10000,
        callback=checkpoint_callback
    )

    # 最後結束後存一份完整模型
    model.save("ppo_arm_model")
    print("✅ 訓練完成並已儲存 ppo_arm_model.zip")

if __name__ == "__main__":
    main()