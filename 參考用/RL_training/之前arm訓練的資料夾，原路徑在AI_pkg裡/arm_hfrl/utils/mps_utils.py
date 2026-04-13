# mps_utils.py - Apple Silicon MPS 最佳化工具
import torch
import os
from typing import Optional

class MPSManager:
    """macOS MPS (Metal Performance Shaders) 管理器"""
    
    def __init__(self):
        self.device = self._get_optimal_device()
        self._setup_mps_optimizations()
    
    def _get_optimal_device(self) -> torch.device:
        """獲取最佳設備 (MPS > CPU)"""
        if torch.backends.mps.is_available():
            print("🚀 使用 Apple Silicon MPS 加速")
            return torch.device('mps')
        else:
            print("💻 使用 CPU (MPS 不可用)")
            return torch.device('cpu')
    
    def _setup_mps_optimizations(self):
        """設定 MPS 最佳化參數"""
        if self.is_mps():
            # 設定 MPS 相關環境變數
            os.environ['PYTORCH_MPS_HIGH_WATERMARK_RATIO'] = '0.0'  # 避免記憶體問題
            
            # 啟用 MPS fallback (某些操作 MPS 不支援時自動切換到 CPU)
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
            
            print("✅ MPS 最佳化設定已啟用")
    
    def is_mps(self) -> bool:
        """檢查是否使用 MPS"""
        return self.device.type == 'mps'
    
    def get_device(self) -> torch.device:
        """獲取設備"""
        return self.device
    
    def to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """將 tensor 移到設備上"""
        try:
            return tensor.to(self.device)
        except Exception as e:
            print(f"⚠️ 移動到 {self.device} 失敗: {e}")
            # Fallback to CPU
            return tensor.to('cpu')
    
    def get_memory_info(self) -> dict:
        """獲取記憶體資訊 (MPS 版本)"""
        if self.is_mps():
            # MPS 沒有直接的記憶體查詢 API，返回系統資訊
            import psutil
            memory = psutil.virtual_memory()
            return {
                "device": "MPS (Apple Silicon)",
                "total_memory": f"{memory.total / 1024**3:.1f} GB",
                "available_memory": f"{memory.available / 1024**3:.1f} GB",
                "memory_percent": f"{memory.percent}%"
            }
        else:
            return {
                "device": "CPU",
                "total_memory": "N/A",
                "available_memory": "N/A", 
                "memory_percent": "N/A"
            }

# 修改 PPO 訓練器以支援 MPS
class MPSOptimizedPPO:
    """MPS 最佳化的 PPO 包裝器"""
    
    def __init__(self, model, mps_manager: Optional[MPSManager] = None):
        self.model = model
        self.mps_manager = mps_manager or MPSManager()
        self._optimize_for_mps()
    
    def _optimize_for_mps(self):
        """針對 MPS 進行最佳化"""
        if self.mps_manager.is_mps():
            # 將模型移到 MPS 設備
            try:
                self.model.policy.to(self.mps_manager.get_device())
                print("✅ PPO 模型已移到 MPS 設備")
            except Exception as e:
                print(f"⚠️ 無法將模型移到 MPS: {e}")
                print("📝 某些 Stable-Baselines3 操作可能不支援 MPS")
    
    def predict(self, observation, **kwargs):
        """MPS 最佳化的預測"""
        try:
            # 嘗試使用原始模型預測
            return self.model.predict(observation, **kwargs)
        except Exception as e:
            print(f"⚠️ MPS 預測失敗，回退到 CPU: {e}")
            # 暫時移到 CPU 進行預測
            original_device = next(self.model.policy.parameters()).device
            self.model.policy.to('cpu')
            result = self.model.predict(observation, **kwargs)
            self.model.policy.to(original_device)
            return result
    
    def learn(self, total_timesteps, **kwargs):
        """MPS 最佳化的訓練"""
        print(f"🎯 開始 MPS 最佳化訓練 ({total_timesteps} steps)")
        
        # 顯示記憶體資訊
        memory_info = self.mps_manager.get_memory_info()
        print(f"💾 記憶體狀態: {memory_info}")
        
        try:
            return self.model.learn(total_timesteps, **kwargs)
        except RuntimeError as e:
            if "mps" in str(e).lower():
                print(f"⚠️ MPS 訓練錯誤: {e}")
                print("🔄 嘗試回退到 CPU 訓練...")
                
                # 移到 CPU
                self.model.policy.to('cpu')
                return self.model.learn(total_timesteps, **kwargs)
            else:
                raise e

# 全域 MPS 管理器實例
mps_manager = MPSManager()

def get_mps_device():
    """獲取 MPS 設備（全域函數）"""
    return mps_manager.get_device()

def optimize_tensor_for_mps(tensor: torch.Tensor) -> torch.Tensor:
    """將 tensor 最佳化到 MPS 設備"""
    return mps_manager.to_device(tensor)

# 使用範例
if __name__ == "__main__":
    # 測試 MPS 功能
    manager = MPSManager()
    
    print(f"設備: {manager.get_device()}")
    print(f"記憶體資訊: {manager.get_memory_info()}")
    
    # 測試 tensor 操作
    if manager.is_mps():
        x = torch.randn(1000, 1000)
        x_mps = manager.to_device(x)
        
        import time
        start = time.time()
        y = torch.mm(x_mps, x_mps.T)
        end = time.time()
        
        print(f"🚀 MPS 矩陣乘法耗時: {end-start:.4f} 秒")
    
    print("✅ MPS 測試完成！")