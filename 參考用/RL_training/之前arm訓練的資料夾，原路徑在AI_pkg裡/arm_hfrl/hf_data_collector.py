# hf_data_collector.py - 人類指令收集器 (修復版)
import numpy as np
import json
import time
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# JSON 序列化工具函數
def safe_json_serialize(obj):
    """安全的 JSON 序列化，處理 numpy 類型"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {str(k): safe_json_serialize(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [safe_json_serialize(item) for item in obj]
    else:
        return obj

@dataclass
class Transition:
    """單個狀態轉移"""
    obs: np.ndarray
    action: int
    reward: float
    next_obs: np.ndarray
    done: bool
    human_command: str
    llm_response: str

class HumanFeedbackCollector:
    def __init__(self, env, llm_model, log_file="data/hf_logs.jsonl"):
        self.env = env
        self.llm_model = llm_model
        self.log_file = log_file
        
        # 確保 data 目錄存在
        import os
        os.makedirs("data", exist_ok=True)

    def collect_trajectory(self, human_command: str) -> List[Transition]:
        """收集一個完整的軌跡"""
        print(f"🤖 處理指令: {human_command}")
        
        # 1. 使用 LLM 將人類指令轉成動作序列
        action_sequence = self._command_to_actions(human_command)
        print(f"📋 LLM 解析結果: {action_sequence}")
        
        # 2. 執行動作序列並收集經驗
        trajectory = []
        obs_result = self.env.reset()
        
        # 處理不同版本的 reset 返回值
        if isinstance(obs_result, tuple):
            obs = obs_result[0]  # 新版本返回 (obs, info)
        else:
            obs = obs_result     # 舊版本直接返回 obs
            
        obs = obs[0] if isinstance(obs, (list, tuple)) and len(obs) > 0 else obs  # 處理向量環境
        
        for i, action_info in enumerate(action_sequence):
            action = action_info['action']
            
            # 執行動作 - 修復返回值解包問題
            step_result = self.env.step([action])
            
            # 處理不同版本的 gymnasium 返回值
            if len(step_result) == 4:
                next_obs, reward, terminated, info = step_result
                truncated = False  # 舊版本沒有 truncated
                done = terminated
            elif len(step_result) == 5:
                next_obs, reward, terminated, truncated, info = step_result
                done = terminated or truncated
            else:
                raise ValueError(f"Unexpected step() return length: {len(step_result)}")
            
            # 安全地提取向量環境的數據
            next_obs = next_obs[0] if isinstance(next_obs, (list, tuple)) and len(next_obs) > 0 else next_obs
            reward = reward[0] if isinstance(reward, (list, tuple)) and len(reward) > 0 else reward
            done = done[0] if isinstance(done, (list, tuple)) and len(done) > 0 else done
            
            # 創建轉移記錄 - 確保數據類型正確
            import numpy as np
            
            transition = Transition(
                obs=np.array(obs, dtype=np.float32),
                action=int(action),
                reward=float(reward),
                next_obs=np.array(next_obs, dtype=np.float32),
                done=bool(done),
                human_command=str(human_command),
                llm_response=json.dumps(action_sequence)
            )
            
            trajectory.append(transition)
            
            # 記錄到日誌
            self._log_interaction(human_command, action_info, reward, i+1)
            
            obs = next_obs
            
            if done:
                break
                
        # 3. 獲取人類反饋 (reward modification)
        final_reward = self._get_human_reward_feedback(trajectory)
        
        # 4. 修改軌跡的 reward
        for transition in trajectory:
            transition.reward += final_reward
            
        print(f"✅ 軌跡收集完成，包含 {len(trajectory)} 步")
        return trajectory

    def _command_to_actions(self, command: str) -> List[Dict]:
        """使用 Ollama LLM 將自然語言轉成角度變化"""
        print(f"🤖 使用 Ollama 解析指令: {command}")
        
        try:
            # 取得當前狀態 (從環境或預設值)
            current_angles = [90.0, 30.0, 160.0, 90.0, 10.0]  # 預設 T-pose 角度
            
            # 使用 LLM 生成角度回應
            response = self.llm_model.generate(command, current_angles)
            
            # 解析 JSON 回應並轉換為 action
            action = self.llm_model.get_action_from_angles(current_angles, response)
            
            # 轉換為原本期望的格式
            action_description = self._get_action_description(action)
            
            return [{"action": action, "description": action_description}]
            
        except Exception as e:
            print(f"❌ Ollama 解析失敗: {e}")
            # 回退到簡單的關鍵字匹配
            return self._fallback_command_parsing(command)
    
    def _get_action_description(self, action: int) -> str:
        """將 action ID 轉換為描述"""
        action_map = {
            0: "Base joint 正轉 (+15°)",
            1: "Base joint 反轉 (-15°)", 
            2: "Shoulder joint 正轉 (+15°)",
            3: "Shoulder joint 反轉 (-15°)",
            4: "Elbow joint 正轉 (+15°)",
            5: "Elbow joint 反轉 (-15°)",
            6: "Wrist joint 正轉 (+15°)",
            7: "Wrist joint 反轉 (-15°)",
            8: "無動作"
        }
        return action_map.get(action, "未知動作")

    def _fallback_command_parsing(self, command: str) -> List[Dict]:
        """簡單的關鍵字匹配作為 LLM 的回退方案"""
        actions = []
        command_lower = command.lower()
        
        # 簡單的關鍵字匹配
        if "joint0" in command_lower or "第一" in command_lower or "base" in command_lower:
            if "+" in command_lower or "正" in command_lower or "右" in command_lower:
                actions.append({"action": 0, "description": "joint0 正轉"})
            elif "-" in command_lower or "負" in command_lower or "左" in command_lower:
                actions.append({"action": 1, "description": "joint0 反轉"})
        
        if "joint1" in command_lower or "第二" in command_lower:
            if "+" in command_lower or "正" in command_lower or "上" in command_lower:
                actions.append({"action": 2, "description": "joint1 正轉"})
            elif "-" in command_lower or "負" in command_lower or "下" in command_lower:
                actions.append({"action": 3, "description": "joint1 反轉"})
        
        # 如果沒有匹配到任何動作，返回 no-op
        if not actions:
            actions = [{"action": 8, "description": "無動作"}]
            
        return actions

    def _get_human_reward_feedback(self, trajectory: List[Transition]) -> float:
        """獲取人類對軌跡的反饋 reward"""
        print("\n📊 軌跡執行完成！")
        print("請評價這次執行的品質:")
        print("1. 非常好 (+2.0)")
        print("2. 好 (+1.0)")  
        print("3. 普通 (0.0)")
        print("4. 差 (-1.0)")
        print("5. 非常差 (-2.0)")
        
        while True:
            try:
                choice = input("請選擇 (1-5): ").strip()
                reward_map = {
                    '1': 2.0, '2': 1.0, '3': 0.0, '4': -1.0, '5': -2.0
                }
                if choice in reward_map:
                    reward = reward_map[choice]
                    print(f"✅ 收到反饋: {reward}")
                    return reward
                else:
                    print("請輸入 1-5 之間的數字")
            except KeyboardInterrupt:
                return 0.0

    def _log_interaction(self, command: str, action_info: Dict, reward: float, step: int):
        """記錄互動到日誌檔案 (修復版)"""        
        log_entry = {
            "timestamp": time.time(),
            "human_command": str(command),
            "step": int(step),
            "action": safe_json_serialize(action_info['action']),
            "action_description": str(action_info['description']),
            "reward": safe_json_serialize(reward)
        }
        
        try:
            # 使用安全序列化
            serialized_entry = safe_json_serialize(log_entry)
            
            with open(self.log_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(serialized_entry, ensure_ascii=False) + '\n')
                
        except Exception as e:
            print(f"⚠️ 日誌記錄失敗: {e}")
            # 創建簡化版本的日誌條目
            try:
                simple_entry = {
                    "timestamp": time.time(),
                    "human_command": str(command)[:100],  # 截斷長字符串
                    "step": int(step),
                    "action": int(action_info.get('action', 0)),
                    "error": f"原始記錄失敗: {str(e)}"
                }
                with open(self.log_file, 'a', encoding='utf-8') as f:
                    f.write(json.dumps(simple_entry, ensure_ascii=False) + '\n')
            except:
                print("❌ 連簡化日誌都記錄失敗，跳過此次記錄")
