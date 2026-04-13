# td_updater.py - TD Learning 更新器
import numpy as np
from typing import Optional, Tuple

class TDUpdater:
    """TD-learning 更新器，實現各種 TD 方法"""
    
    def __init__(self, gamma: float = 0.99, td_lambda: float = 0.95):
        """
        Args:
            gamma: 折扣因子
            td_lambda: TD(λ) 的 λ 參數
        """
        self.gamma = gamma
        self.td_lambda = td_lambda
    
    def compute_td_targets(self, rewards: np.ndarray, values: np.ndarray, 
                          next_values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        計算 TD 目標值
        
        Args:
            rewards: 即時 rewards [batch_size]
            values: 當前狀態的 value [batch_size]
            next_values: 下一狀態的 value [batch_size]
            dones: 是否結束 [batch_size]
        
        Returns:
            TD targets [batch_size]
        """
        # TD(0) targets: r + γ * V(s') * (1 - done)
        td_targets = rewards + self.gamma * next_values * (1.0 - dones.astype(float))
        return td_targets
    
    def compute_td_errors(self, rewards: np.ndarray, values: np.ndarray, 
                         next_values: np.ndarray, dones: np.ndarray) -> np.ndarray:
        """
        計算 TD errors
        
        Returns:
            TD errors [batch_size]
        """
        td_targets = self.compute_td_targets(rewards, values, next_values, dones)
        td_errors = td_targets - values
        return td_errors
    
    def compute_gae_advantages(self, rewards: np.ndarray, values: np.ndarray, 
                              next_values: np.ndarray, dones: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        計算 Generalized Advantage Estimation (GAE)
        
        Returns:
            advantages, returns
        """
        batch_size = len(rewards)
        advantages = np.zeros_like(rewards)
        returns = np.zeros_like(rewards)
        
        # 計算 TD errors
        td_errors = self.compute_td_errors(rewards, values, next_values, dones)
        
        # 反向計算 GAE advantages
        gae = 0
        for t in reversed(range(batch_size)):
            if t == batch_size - 1:
                next_non_terminal = 1.0 - dones[t]
                next_advantage = 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_advantage = advantages[t + 1]
            
            # GAE calculation
            gae = td_errors[t] + self.gamma * self.td_lambda * next_non_terminal * next_advantage
            advantages[t] = gae
        
        # Returns = advantages + values
        returns = advantages + values
        
        return advantages, returns
    
    def compute_n_step_returns(self, rewards: np.ndarray, values: np.ndarray, 
                              dones: np.ndarray, n: int = 3) -> np.ndarray:
        """
        計算 n-step returns
        
        Args:
            n: n-step 的步數
        """
        batch_size = len(rewards)
        n_step_returns = np.zeros_like(rewards)
        
        for i in range(batch_size):
            n_step_return = 0
            discount = 1.0
            
            # 計算 n-step return
            for j in range(min(n, batch_size - i)):
                if i + j >= batch_size:
                    break
                
                n_step_return += discount * rewards[i + j]
                discount *= self.gamma
                
                # 如果遇到終止狀態，提前結束
                if dones[i + j]:
                    break
            
            # 如果沒有提前結束，加上 bootstrap value
            if i + n < batch_size and not any(dones[i:i+n]):
                n_step_return += discount * values[i + n]
            
            n_step_returns[i] = n_step_return
        
        return n_step_returns
    
    def update_value_function(self, values: np.ndarray, td_targets: np.ndarray, 
                            learning_rate: float = 0.01) -> np.ndarray:
        """
        簡單的 value function 更新 (僅作示範，實際會在 neural network 中進行)
        
        Returns:
            updated values
        """
        td_errors = td_targets - values
        updated_values = values + learning_rate * td_errors
        return updated_values
    
    def prioritized_experience_weights(self, td_errors: np.ndarray, 
                                     alpha: float = 0.6, beta: float = 0.4) -> np.ndarray:
        """
        計算優先經驗回放的權重
        
        Args:
            alpha: 優先級的指數
            beta: 重要性採樣的指數
        """
        # 計算優先級 (基於 TD error 的絕對值)
        priorities = np.abs(td_errors) + 1e-6  # 避免 0 優先級
        
        # 計算採樣概率
        probs = priorities ** alpha
        probs = probs / np.sum(probs)
        
        # 計算重要性採樣權重
        weights = (len(td_errors) * probs) ** (-beta)
        weights = weights / np.max(weights)  # 正規化
        
        return weights
    
    def get_stats(self, td_errors: np.ndarray) -> dict:
        """獲取 TD error 統計資訊"""
        return {
            "mean_td_error": np.mean(np.abs(td_errors)),
            "std_td_error": np.std(td_errors),
            "max_td_error": np.max(np.abs(td_errors)),
            "min_td_error": np.min(np.abs(td_errors))
        }