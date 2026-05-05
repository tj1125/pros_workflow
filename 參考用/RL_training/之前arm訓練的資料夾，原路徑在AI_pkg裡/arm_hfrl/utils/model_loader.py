# utils/model_loader.py
"""LLM 模型載入器 - 針對 Ollama llama3.1:8b-instruct-q3_K_M 最佳化"""

import json
import requests
from typing import Optional, Dict, Any
from abc import ABC, abstractmethod

class LLMInterface(ABC):
    """LLM 介面的抽象基類"""
    
    @abstractmethod
    def generate(self, prompt: str) -> str:
        """生成回應"""
        pass

class OllamaLLM(LLMInterface):
    """Ollama 本地 LLM 介面 - 針對機械手臂控制最佳化 (支援 Docker)"""
    
    def __init__(self, base_url: str = None, 
                 model: str = "llama3.1:8b-instruct-q3_K_M"):
        # 支援環境變數配置，方便 Docker 部署
        import os
        self.base_url = base_url or os.getenv("OLLAMA_URL", "http://localhost:11434")
        self.model = model
        self.system_prompt = self._get_arm_control_system_prompt()
        
        # 測試連接
        self._test_connection()
    
    def _get_arm_control_system_prompt(self) -> str:
        """獲取機械手臂控制的 system prompt"""
        return """Task: Determine One Joint Movement Based on Human Direction Command

Background:
• The current pose is a T-pose of a right arm lying sideways, palm facing downward.
• The arm has 5 joints:
  • Base (0): Shoulder flexion (arm moves left, +15°) / extension (arm moves right, −15°)
  • Shoulder (1): Shoulder adduction (arm moves down, toward body, +15°) / abduction (arm moves up, away from body, −15°)
  • Elbow (2): Elbow flexion (forearm moves down, +15°) / extension (forearm moves up, −15°)
  • Wrist (3): Clockwise (+15°) / Counter-clockwise (−15°)
  • Finger (4): Open (+15°) / Close (−15°)

Initial T-pose angles:
[Base, Shoulder, Elbow, Wrist, Finger] = [90.0, 30.0, 160.0, 90.0, 10.0]

Angle limits:
• Base: 0°–240°
• Shoulder: 0°–240°
• Elbow: 0°–150°
• Wrist: 50°–180°
• Finger: 10°–70°

Rules:
1. From natural language directional commands, infer which joint should move.
2. Modify only one joint per command.
3. Note that you can only change "one" joint at a time, and you can only change it by "+15°" or "-15°" or "not at all".
4. Ensure angle bounds are not violated.
5. Output a single updated angle set.

Please only output the final arm parameters in the following JSON format:

{
  "Base": <angle>,
  "Shoulder": <angle>,
  "Elbow": <angle>,
  "Wrist": <angle>,
  "Finger": <angle>
}

No explanation, no additional text. Just output the JSON."""
    
    def _test_connection(self):
        """測試 Ollama 連接 (Docker 支援)"""
        try:
            print(f"🐳 嘗試連接 Ollama: {self.base_url}")
            response = requests.get(f"{self.base_url}/api/tags", timeout=10)
            if response.status_code == 200:
                models = response.json().get('models', [])
                model_names = [m['name'] for m in models]
                if self.model in model_names:
                    print(f"✅ Docker Ollama 連接成功，模型 {self.model} 可用")
                else:
                    print(f"⚠️ 模型 {self.model} 未找到，可用模型: {model_names}")
                    print(f"請在 Docker 中執行: docker exec <container> ollama pull {self.model}")
            else:
                print(f"⚠️ Ollama 連接失敗，狀態碼: {response.status_code}")
                print(f"請檢查 Docker 容器是否運行，URL: {self.base_url}")
        except requests.exceptions.RequestException as e:
            print(f"❌ 無法連接到 Docker Ollama: {e}")
            print(f"請確認:")
            print(f"  1. Docker 容器正在運行")
            print(f"  2. 端口映射正確: -p 11434:11434") 
            print(f"  3. URL 正確: {self.base_url}")
            print(f"  4. 或設置環境變數: export OLLAMA_URL=http://your-docker-host:11434")
    
    def generate(self, prompt: str, current_angles: Optional[list] = None) -> str:
        """使用 Ollama 生成機械手臂控制指令"""
        try:
            # 構建完整 prompt
            if current_angles:
                current_state = f"\nCurrent angles: [Base: {current_angles[0]}°, Shoulder: {current_angles[1]}°, Elbow: {current_angles[2]}°, Wrist: {current_angles[3]}°, Finger: {current_angles[4]}°]"
            else:
                current_state = "\nCurrent angles: [Base: 90°, Shoulder: 30°, Elbow: 160°, Wrist: 90°, Finger: 10°]"
            
            full_prompt = f"{self.system_prompt}{current_state}\n\nHuman Command: {prompt}\n\nResponse:"
            
            payload = {
                "model": self.model,
                "prompt": full_prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,  # 低溫度確保一致性
                    "top_p": 0.9,
                    "top_k": 40,
                    "repeat_penalty": 1.1,
                    "num_predict": 200  # 限制輸出長度
                }
            }
            
            response = requests.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            
            response.raise_for_status()
            result = response.json()
            
            generated_text = result.get("response", "").strip()
            
            # 提取 JSON 內容
            json_response = self._extract_json_from_response(generated_text)
            
            if json_response:
                print(f"🤖 Ollama 回應: {json_response}")
                return json_response
            else:
                print(f"⚠️ 無法解析 JSON，原始回應: {generated_text}")
                return self._fallback_response(prompt)
                
        except Exception as e:
            print(f"Ollama 錯誤: {e}")
            return self._fallback_response(prompt)
    
    def _extract_json_from_response(self, response_text: str) -> Optional[str]:
        """從 LLM 回應中提取 JSON"""
        try:
            # 尋找 JSON 區塊
            start_idx = response_text.find('{')
            end_idx = response_text.rfind('}') + 1
            
            if start_idx != -1 and end_idx > start_idx:
                json_str = response_text[start_idx:end_idx]
                
                # 驗證 JSON 格式
                parsed_json = json.loads(json_str)
                
                # 驗證必要欄位
                required_fields = ["Base", "Shoulder", "Elbow", "Wrist", "Finger"]
                if all(field in parsed_json for field in required_fields):
                    return json_str
                else:
                    print(f"❌ JSON 缺少必要欄位: {required_fields}")
                    return None
            else:
                return None
                
        except json.JSONDecodeError as e:
            print(f"❌ JSON 解析錯誤: {e}")
            return None
    
    def _fallback_response(self, prompt: str) -> str:
        """LLM 失敗時的回退回應 - 保持當前角度不變"""
        return '{"Base": 90, "Shoulder": 30, "Elbow": 160, "Wrist": 90, "Finger": 10}'
    
    def get_action_from_angles(self, current_angles: list, new_angles_json: str) -> int:
        """將角度變化轉換為 action ID"""
        try:
            new_angles_dict = json.loads(new_angles_json)
            new_angles = [
                new_angles_dict["Base"],
                new_angles_dict["Shoulder"], 
                new_angles_dict["Elbow"],
                new_angles_dict["Wrist"],
                new_angles_dict["Finger"]
            ]
            
            # 找出變化的關節
            for i, (current, new) in enumerate(zip(current_angles, new_angles)):
                if current != new:
                    if new > current:
                        return i * 2  # 正向動作
                    else:
                        return i * 2 + 1  # 負向動作
            
            # 沒有變化
            return 8  # no-op
            
        except Exception as e:
            print(f"角度轉換錯誤: {e}")
            return 8

class MockLLM(LLMInterface):
    """模擬 LLM (用於測試)"""
    
    def generate(self, prompt: str, current_angles: Optional[list] = None) -> str:
        """返回預設的測試回應"""
        # 簡單的關鍵字匹配
        prompt_lower = prompt.lower()
        
        # 預設角度
        base_angles = {"Base": 90, "Shoulder": 30, "Elbow": 160, "Wrist": 90, "Finger": 10}
        
        if current_angles:
            base_angles = {
                "Base": current_angles[0],
                "Shoulder": current_angles[1], 
                "Elbow": current_angles[2],
                "Wrist": current_angles[3],
                "Finger": current_angles[4]
            }
        
        # 關鍵字匹配邏輯
        if "左" in prompt or "left" in prompt_lower:
            base_angles["Base"] = min(base_angles["Base"] + 15, 240)
        elif "右" in prompt or "right" in prompt_lower:
            base_angles["Base"] = max(base_angles["Base"] - 15, 0)
        elif "上" in prompt or "up" in prompt_lower:
            base_angles["Shoulder"] = max(base_angles["Shoulder"] - 15, 0)
        elif "下" in prompt or "down" in prompt_lower:
            base_angles["Shoulder"] = min(base_angles["Shoulder"] + 15, 240)
        elif "彎曲" in prompt or "bend" in prompt_lower:
            base_angles["Elbow"] = max(base_angles["Elbow"] - 15, 0)
        elif "伸直" in prompt or "extend" in prompt_lower:
            base_angles["Elbow"] = min(base_angles["Elbow"] + 15, 150)
        elif "轉" in prompt or "rotate" in prompt_lower:
            if "順時針" in prompt or "clockwise" in prompt_lower:
                base_angles["Wrist"] = min(base_angles["Wrist"] + 15, 180)
            else:
                base_angles["Wrist"] = max(base_angles["Wrist"] - 15, 50)
        elif "張開" in prompt or "open" in prompt_lower:
            base_angles["Finger"] = min(base_angles["Finger"] + 15, 70)
        elif "握拳" in prompt or "close" in prompt_lower:
            base_angles["Finger"] = max(base_angles["Finger"] - 15, 10)
        
        return json.dumps(base_angles)

def load_llm_model(llm_type: str = "ollama", **kwargs) -> LLMInterface:
    """
    載入 LLM 模型 (支援 Docker 部署)
    
    Args:
        llm_type: "ollama", "mock" 
        **kwargs: 模型特定參數
            base_url: Ollama 服務 URL (預設從環境變數 OLLAMA_URL 讀取)
            model: 模型名稱
    """
    if llm_type == "ollama":
        import os
        # 支援環境變數配置
        default_url = os.getenv("OLLAMA_URL", "http://localhost:11434")
        
        return OllamaLLM(
            kwargs.get("base_url", default_url),
            kwargs.get("model", "llama3.1:8b-instruct-q3_K_M")
        )
    elif llm_type == "mock":
        return MockLLM()
    else:
        print(f"⚠️ 未知的 LLM 類型: {llm_type}，使用 Mock LLM")
        return MockLLM()

# 測試函數
def test_ollama_connection():
    """測試 Ollama 連接和模型回應"""
    print("🧪 測試 Ollama 連接...")
    
    llm = load_llm_model("ollama")
    
    test_commands = [
        "向左移動手臂",
        "抬高手臂",
        "彎曲手肘", 
        "順時針轉動手腕",
        "張開手指"
    ]
    
    for command in test_commands:
        print(f"\n指令: {command}")
        response = llm.generate(command)
        print(f"回應: {response}")

if __name__ == "__main__":
    test_ollama_connection()
