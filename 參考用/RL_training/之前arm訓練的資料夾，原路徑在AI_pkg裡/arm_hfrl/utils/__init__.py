# utils/__init__.py
"""
Utility modules for Human Feedback RL system

Contains helper functions and classes for:
- Joint limits and constraints
- LLM model loading and management  
- MPS (Apple Silicon) optimization
- Observation encoding and processing
"""

# 匯入所有 utils 模組
try:
    from .joint_limits import JOINT_LIMITS, is_valid_angle, clamp_angle
    from .model_loader import (
        LLMInterface, 
        OllamaLLM, 
        MockLLM, 
        load_llm_model,
        test_ollama_connection
    )
    from .mps_utils import MPSManager, MPSOptimizedPPO, get_mps_device
    from .observation_encoder import ObservationEncoder
    
    __all__ = [
        # Joint limits
        'JOINT_LIMITS',
        'is_valid_angle', 
        'clamp_angle',
        
        # LLM related
        'LLMInterface',
        'OllamaLLM',
        'MockLLM', 
        'load_llm_model',
        'test_ollama_connection',
        
        # MPS optimization
        'MPSManager',
        'MPSOptimizedPPO',
        'get_mps_device',
        
        # Observation processing
        'ObservationEncoder'
    ]
    
except ImportError as e:
    print(f"⚠️ 部分 utils 模組載入失敗: {e}")
    # 至少匯出可用的模組
    __all__ = []

# 便捷函數
def check_ollama_status():
    """快速檢查 Ollama 服務狀態"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=3)
        return response.status_code == 200
    except:
        return False

def get_available_models():
    """獲取可用的 Ollama 模型列表"""
    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            return [m['name'] for m in models]
        return []
    except:
        return []

def setup_mps_if_available():
    """如果可用，設定 MPS 最佳化"""
    try:
        import torch
        if torch.backends.mps.is_available():
            print("🚀 Apple Silicon MPS 加速已啟用")
            return torch.device('mps')
        else:
            print("💻 使用 CPU 運算")
            return torch.device('cpu')
    except ImportError:
        print("❌ PyTorch 未安裝")
        return None