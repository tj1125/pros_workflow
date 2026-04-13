# buffer.py - 離線經驗緩衝區
import numpy as np
import pickle
import heapq
from typing import List, Tuple, Optional
from collections import deque
import random

from hf_data_collector import Transition

class OfflineBuffer:
    def __init__(self, max_size: int = 10000, top_k: int = 1000):
        """
        離線經驗緩衝區，專門儲存高 reward 的經驗
        
        Args:
            max_size: 緩衝區最大容量
            top_k: 保留的高 reward 經驗數量
        """
        self.max_size = max_size
        self.top_k = top_k
        self.buffer = deque(maxlen=max_size)
        self.reward_heap = []  # 用於維護 top-k 高 reward
        
    def add(self, transition: Transition):
        """添加一個轉移到緩衝區"""
        self.buffer.append(transition)
        
        # 維護 top-k reward heap
        if len(self.reward_heap) < self.top_k:
            heapq.heappush(self.reward_heap, (transition.reward, len(self.buffer)-1))
        else:
            # 如果當前 reward 比最小的高 reward 還大，替換
            min_reward, _ = self.reward_heap[0]
            if transition.reward > min_reward:
                heapq.heapreplace(self.reward_heap, (transition.reward, len(self.buffer)-1))
    
    def sample_high_reward(self, batch_size: int = 32) -> List[Transition]:
        """從高 reward 經驗中採樣"""
        if not self.reward_heap:
            return self.sample_random(batch_size)
        
        # 從 top-k 中隨機選擇
        available_indices = [idx for _, idx in self.reward_heap if idx < len(self.buffer)]
        
        if len(available_indices) < batch_size:
            # 如果高 reward 經驗不足，補充隨機採樣
            high_reward_samples = [self.buffer[idx] for idx in available_indices]
            random_samples = self.sample_random(batch_size - len(high_reward_samples))
            return high_reward_samples + random_samples
        
        sampled_indices = random.sample(available_indices, batch_size)
        return [self.buffer[idx] for idx in sampled_indices]
    
    def sample_random(self, batch_size: int = 32) -> List[Transition]:
        """隨機採樣"""
        if len(self.buffer) == 0:
            return []
        
        batch_size = min(batch_size, len(self.buffer))
        indices = random.sample(range(len(self.buffer)), batch_size)
        return [self.buffer[idx] for idx in indices]
    
    def sample_mixed(self, batch_size: int = 32, high_reward_ratio: float = 0.7) -> List[Transition]:
        """混合採樣：部分來自高 reward，部分隨機"""
        high_reward_size = int(batch_size * high_reward_ratio)
        random_size = batch_size - high_reward_size
        
        high_reward_samples = self.sample_high_reward(high_reward_size)
        random_samples = self.sample_random(random_size)
        
        # 混合並打亂順序
        mixed_samples = high_reward_samples + random_samples
        random.shuffle(mixed_samples)
        
        return mixed_samples
    
    def get_stats(self) -> dict:
        """獲取緩衝區統計信息"""
        if not self.buffer:
            return {"size": 0, "avg_reward": 0, "max_reward": 0, "min_reward": 0}
        
        rewards = [t.reward for t in self.buffer]
        return {
            "size": len(self.buffer),
            "avg_reward": np.mean(rewards),
            "max_reward": np.max(rewards),
            "min_reward": np.min(rewards),
            "std_reward": np.std(rewards)
        }
    
    def get_top_k_rewards(self) -> List[float]:
        """獲取 top-k 的 reward 值"""
        return sorted([reward for reward, _ in self.reward_heap], reverse=True)
    
    def save(self, filepath: str):
        """保存緩衝區到文件"""
        data = {
            "buffer": list(self.buffer),
            "reward_heap": self.reward_heap,
            "max_size": self.max_size,
            "top_k": self.top_k
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 緩衝區已保存到 {filepath}")
    
    @classmethod
    def load(cls, filepath: str) -> 'OfflineBuffer':
        """從文件載入緩衝區"""
        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)
            
            buffer = cls(data['max_size'], data['top_k'])
            buffer.buffer = deque(data['buffer'], maxlen=data['max_size'])
            buffer.reward_heap = data['reward_heap']
            
            print(f"📂 從 {filepath} 載入緩衝區，包含 {len(buffer.buffer)} 筆資料")
            return buffer
            
        except FileNotFoundError:
            print(f"⚠️ 找不到文件 {filepath}，創建新的緩衝區")
            return cls()
        except Exception as e:
            print(f"❌ 載入緩衝區失敗: {e}")
            return cls()
    
    def __len__(self):
        return len(self.buffer)


class OnlineBuffer:
    """在線緩衝區，用於暫存當前訓練的經驗"""
    
    def __init__(self, max_size: int = 2048):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
        
    def add(self, obs, action, reward, next_obs, done):
        """添加單個經驗"""
        self.buffer.append((obs, action, reward, next_obs, done))
    
    def get_all(self) -> Tuple[np.ndarray, ...]:
        """獲取所有經驗，返回分離的數組"""
        if not self.buffer:
            return tuple(np.array([]) for _ in range(5))
        
        obs_list, action_list, reward_list, next_obs_list, done_list = zip(*self.buffer)
        
        return (
            np.array(obs_list),
            np.array(action_list),
            np.array(reward_list),
            np.array(next_obs_list),
            np.array(done_list)
        )
    
    def clear(self):
        """清空緩衝區"""
        self.buffer.clear()
    
    def __len__(self):
        return len(self.buffer)