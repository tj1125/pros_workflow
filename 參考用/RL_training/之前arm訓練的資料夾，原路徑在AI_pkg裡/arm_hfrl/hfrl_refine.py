# hfrl_refine.py - 主程式入口
import argparse
import os
import time
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor

from arm_env import ArmEnv
from hf_data_collector import HumanFeedbackCollector
from buffer import OfflineBuffer
from ppo_trainer import CustomPPOTrainer
from td_updater import TDUpdater
from utils.model_loader import load_llm_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['collect', 'train'], required=True,
                       help='collect: 收集人類指令資料, train: 訓練模型')
    parser.add_argument('--base_model', default='ppo_arm_model.zip',
                       help='基礎 PPO 模型路徑')
    parser.add_argument('--buffer_size', type=int, default=10000,
                       help='Offline buffer 大小')
    parser.add_argument('--top_k', type=int, default=1000,
                       help='保留 top-k 高 reward 經驗')
    parser.add_argument('--offline_freq', type=int, default=100,
                       help='每 N 步使用一次 offline buffer')
    parser.add_argument('--total_timesteps', type=int, default=50000,
                       help='總訓練步數')
    
    args = parser.parse_args()
    
    if args.mode == 'collect':
        collect_human_feedback(args)
    elif args.mode == 'train':
        train_with_hfrl(args)

def collect_human_feedback(args):
    """收集人類指令並轉成經驗資料"""
    print("🔄 啟動人類指令收集模式...")
    
    # 初始化環境
    env = DummyVecEnv([lambda: Monitor(ArmEnv())])
    
    # 載入 Ollama LLM 模型
    llm_model = load_llm_model(
        llm_type="ollama",
        model="llama3.1:8b-instruct-q3_K_M"
    )
    
    # 初始化人類指令收集器
    collector = HumanFeedbackCollector(env, llm_model)
    
    # 初始化離線 buffer
    buffer = OfflineBuffer(max_size=args.buffer_size, top_k=args.top_k)
    
    print("📝 開始收集人類指令...")
    print("輸入 'quit' 結束收集")
    
    while True:
        try:
            # 獲取人類指令
            human_command = input("\n請輸入手臂動作指令 (例: '向左轉動第一關節30度'): ")
            
            if human_command.lower() == 'quit':
                break
            
            # 收集一個完整的軌跡
            trajectory = collector.collect_trajectory(human_command)
            
            # 加入到 buffer
            for transition in trajectory:
                buffer.add(transition)
            
            print(f"✅ 收集完成！當前 buffer 大小: {len(buffer)}")
            
        except KeyboardInterrupt:
            break
    
    # 保存 buffer
    buffer.save('data/offline_buffer.pkl')
    print(f"💾 已保存 {len(buffer)} 筆經驗到 offline_buffer.pkl")

def train_with_hfrl(args):
    """使用 Human Feedback RL 訓練"""
    print("🚀 啟動 Human Feedback RL 訓練...")
    
    # 初始化環境
    env = DummyVecEnv([lambda: Monitor(ArmEnv())])
    
    # 載入基礎模型
    if os.path.exists(args.base_model):
        print(f"📂 載入基礎模型: {args.base_model}")
        model = PPO.load(args.base_model, env=env)
    else:
        print("⚠️ 未找到基礎模型，從頭開始訓練")
        model = PPO("MlpPolicy", env, verbose=1)
    
    # 載入離線 buffer
    buffer = OfflineBuffer.load('data/offline_buffer.pkl')
    print(f"📚 載入離線 buffer，包含 {len(buffer)} 筆經驗")
    
    # 初始化自訂訓練器
    trainer = CustomPPOTrainer(
        model=model,
        env=env,
        offline_buffer=buffer,
        offline_freq=args.offline_freq
    )
    
    # 初始化 TD 更新器
    td_updater = TDUpdater(gamma=0.99)
    
    # 開始訓練
    print("🎯 開始 Online + Offline 混合訓練...")
    trainer.learn(
        total_timesteps=args.total_timesteps,
        td_updater=td_updater
    )
    
    # 保存最終模型
    final_model_path = "ppo_arm_hfrl.zip"
    trainer.model.save(final_model_path)
    print(f"✅ 訓練完成！模型已保存至 {final_model_path}")

if __name__ == "__main__":
    main()