# ppo_trainer.py - 自訂 PPO 訓練器
import numpy as np
import torch
import torch.nn.functional as F
from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import VecEnv

from buffer import OfflineBuffer, OnlineBuffer
from td_updater import TDUpdater

class OfflineDataCallback(BaseCallback):
    """自訂 Callback，整合離線資料訓練"""
    
    def __init__(self, offline_buffer: OfflineBuffer, offline_freq: int = 100, 
                 batch_size: int = 32, td_updater: Optional[TDUpdater] = None):
        super().__init__()
        self.offline_buffer = offline_buffer
        self.offline_freq = offline_freq
        self.batch_size = batch_size
        self.td_updater = td_updater
        self.step_count = 0
        
    def _on_step(self) -> bool:
        self.step_count += 1
        
        # 每 offline_freq 步進行一次離線資料更新
        if self.step_count % self.offline_freq == 0:
            self._update_with_offline_data()
        
        return True
    
    def _update_with_offline_data(self):
        """使用離線資料更新模型"""
        if len(self.offline_buffer) == 0:
            return
        
        print(f"🔄 Step {self.step_count}: 使用離線資料更新模型")
        
        # 從離線 buffer 中採樣高 reward 經驗
        transitions = self.offline_buffer.sample_high_reward(self.batch_size)
        
        if not transitions:
            return
        
        # 轉換為 tensors
        obs_batch = torch.FloatTensor([t.obs for t in transitions])
        action_batch = torch.LongTensor([t.action for t in transitions])
        reward_batch = torch.FloatTensor([t.reward for t in transitions])
        next_obs_batch = torch.FloatTensor([t.next_obs for t in transitions])
        done_batch = torch.BoolTensor([t.done for t in transitions])
        
        # 使用模型的 policy 和 value function
        with torch.no_grad():
            values = self.model.policy.predict_values(obs_batch)
            next_values = self.model.policy.predict_values(next_obs_batch)
            
        # TD-learning 更新 (如果有提供 td_updater)
        if self.td_updater:
            td_targets = self.td_updater.compute_td_targets(
                rewards=reward_batch.numpy(),
                values=values.squeeze().numpy(),
                next_values=next_values.squeeze().numpy(),
                dones=done_batch.numpy()
            )
            
            # 計算 TD error
            td_errors = td_targets - values.squeeze().numpy()
            avg_td_error = np.mean(np.abs(td_errors))
            
            self.logger.record("train/offline_td_error", avg_td_error)
            self.logger.record("train/offline_avg_reward", reward_batch.mean().item())
        
        # 執行額外的 policy 更新 (使用離線資料)
        self._update_policy_with_offline_data(obs_batch, action_batch, reward_batch, 
                                            next_obs_batch, done_batch)
    
    def _update_policy_with_offline_data(self, obs_batch, action_batch, reward_batch, 
                                       next_obs_batch, done_batch):
        """使用離線資料執行 policy 更新"""
        try:
            # 計算 advantage (簡化版)
            with torch.no_grad():
                values = self.model.policy.predict_values(obs_batch).squeeze()
                next_values = self.model.policy.predict_values(next_obs_batch).squeeze()
                
                # 計算 returns
                returns = reward_batch + self.model.gamma * next_values * ~done_batch
                advantages = returns - values
                
            # 獲取 action probabilities
            actions_pi, values_pi, log_prob = self.model.policy(obs_batch)
            
            # 計算當前 policy 下的 log probabilities
            log_prob_actions = log_prob.gather(1, action_batch.unsqueeze(1)).squeeze()
            
            # 簡化的 policy loss (不使用重要性採樣比率)
            policy_loss = -(log_prob_actions * advantages.detach()).mean()
            
            # Value loss
            value_loss = F.mse_loss(values_pi.squeeze(), returns)
            
            # Total loss
            total_loss = policy_loss + 0.5 * value_loss
            
            # 反向傳播
            self.model.policy.optimizer.zero_grad()
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.policy.parameters(), max_norm=0.5)
            
            self.model.policy.optimizer.step()
            
            # 記錄統計資訊
            self.logger.record("train/offline_policy_loss", policy_loss.item())
            self.logger.record("train/offline_value_loss", value_loss.item())
            self.logger.record("train/offline_total_loss", total_loss.item())
            
        except Exception as e:
            print(f"⚠️ 離線資料更新失敗: {e}")


class CustomPPOTrainer:
    """自訂 PPO 訓練器，整合線上和離線訓練 (macOS MPS 最佳化)"""
    
    def __init__(self, model: PPO, env: VecEnv, offline_buffer: OfflineBuffer, 
                 offline_freq: int = 100, batch_size: int = 32):
        self.model = model
        self.env = env
        self.offline_buffer = offline_buffer
        self.offline_freq = offline_freq
        self.batch_size = batch_size
        
        # 初始化在線緩衝區
        self.online_buffer = OnlineBuffer()
        
        # 🍎 macOS MPS 最佳化
        self._setup_mps_optimization()
        
    def _setup_mps_optimization(self):
        """設定 MPS 最佳化"""
        import torch
        if torch.backends.mps.is_available():
            print("🚀 啟用 Apple Silicon MPS 加速")
            self.device = torch.device('mps')
            try:
                # 嘗試將 policy 移到 MPS (可能不完全支援)
                self.model.policy.to(self.device)
                print("✅ PPO policy 已移到 MPS 設備")
            except Exception as e:
                print(f"⚠️ PPO policy 無法完全移到 MPS: {e}")
                print("📝 將使用混合模式 (部分 CPU + 部分 MPS)")
                self.device = torch.device('cpu')
        else:
            print("💻 使用 CPU 運算")
            self.device = torch.device('cpu')
        
    def learn(self, total_timesteps: int, td_updater: Optional[TDUpdater] = None):
        """執行混合訓練"""
        print("🎯 開始 Online + Offline 混合訓練...")
        
        # 創建離線資料 callback
        offline_callback = OfflineDataCallback(
            offline_buffer=self.offline_buffer,
            offline_freq=self.offline_freq,
            batch_size=self.batch_size,
            td_updater=td_updater
        )
        
        # 印出緩衝區統計資訊
        buffer_stats = self.offline_buffer.get_stats()
        print(f"📊 離線緩衝區統計:")
        print(f"  - 大小: {buffer_stats['size']}")
        print(f"  - 平均 reward: {buffer_stats['avg_reward']:.3f}")
        print(f"  - 最大 reward: {buffer_stats['max_reward']:.3f}")
        print(f"  - 最小 reward: {buffer_stats['min_reward']:.3f}")
        
        top_k_rewards = self.offline_buffer.get_top_k_rewards()
        if top_k_rewards:
            print(f"  - Top-5 rewards: {top_k_rewards[:5]}")
        
        # 執行 PPO 訓練，附加離線資料 callback
        self.model.learn(
            total_timesteps=total_timesteps,
            callback=offline_callback
        )
        
        print("✅ 混合訓練完成！")
    
    def evaluate_model(self, num_episodes: int = 5):
        """評估模型性能"""
        print(f"🔍 評估模型性能 ({num_episodes} 回合)...")
        
        episode_rewards = []
        episode_lengths = []
        
        for episode in range(num_episodes):
            obs = self.env.reset()
            episode_reward = 0
            episode_length = 0
            
            while True:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, done, info = self.env.step(action)
                
                episode_reward += reward[0]
                episode_length += 1
                
                if done[0]:
                    break
            
            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_length)
            
            print(f"  回合 {episode+1}: reward = {episode_reward:.3f}, length = {episode_length}")
        
        avg_reward = np.mean(episode_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"📈 評估結果:")
        print(f"  - 平均 reward: {avg_reward:.3f}")
        print(f"  - 平均回合長度: {avg_length:.1f}")
        
        return {
            "avg_reward": avg_reward,
            "avg_length": avg_length,
            "episode_rewards": episode_rewards,
            "episode_lengths": episode_lengths
        }