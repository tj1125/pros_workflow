# __init__.py (專案根目錄)
"""
Human Feedback Reinforcement Learning for Robotic Arm Control

A complete system for training robotic arms using human feedback and reinforcement learning.
Integrates PPO, TD-learning, and Ollama LLM for natural language control.

Author: [Your Name]
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Human Feedback RL for Robotic Arm Control"

# 匯入主要模組，方便使用
try:
    from .hfrl_refine import main
    from .arm_env import ArmEnv
    from .buffer import OfflineBuffer, OnlineBuffer
    from .hf_data_collector import HumanFeedbackCollector
    from .ppo_trainer import CustomPPOTrainer
    from .td_updater import TDUpdater
    
    __all__ = [
        'main',
        'ArmEnv', 
        'OfflineBuffer',
        'OnlineBuffer',
        'HumanFeedbackCollector',
        'CustomPPOTrainer',
        'TDUpdater'
    ]
    
except ImportError:
    # 如果某些依賴未安裝，不影響套件載入
    pass

# 系統資訊
def get_system_info():
    """獲取系統和環境資訊"""
    import sys
    import platform
    
    info = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "architecture": platform.architecture()[0]
    }
    
    try:
        import torch
        info["pytorch_version"] = torch.__version__
        info["mps_available"] = torch.backends.mps.is_available()
    except ImportError:
        info["pytorch_version"] = "Not installed"
        info["mps_available"] = False
    
    try:
        import stable_baselines3
        info["sb3_version"] = stable_baselines3.__version__
    except ImportError:
        info["sb3_version"] = "Not installed"
    
    return info