from __future__ import annotations

import json
import os
from typing import Optional, Tuple

import numpy as np
import requests

def load_ollama_model(model: str = None, base_url: str = None):
    """
    載入 Ollama 模型與服務位址
    Args:
        model: 模型名稱
        base_url: Ollama API URL
    Returns:
        dict 包含 base_url 與 model
    """
    return {
        "base_url": base_url or os.getenv("OLLAMA_URL", "http://192.168.75.24:11434"),
        "model": model
    }

def get_action_parser_system_prompt() -> str:
    """
    提供人類指令 → action_id 的系統提示詞
    """
    return """You are a command parser for controlling a 4-DOF robotic arm.
Your task is to map natural language instructions to an **action ID** (integer 0–10).

Action ID Mapping:
0: Do nothing
1: Hand move forward (push the gripper forward)
2: Hand move backward (pull the gripper back)
3: Hand move right (shift gripper to the right)
4: Hand move left (shift gripper to the left)
5: Hand move upward (raise the gripper)
6: Hand move downward (lower the gripper)
7: Raise the elbow (shift elbow to the up)
8: Lower the elbow (shift elbow to the down)
9: Place elbow to the left  (shift elbow to the left)
10: Place elbow to the right (shift elbow to the right)

Return ONLY the action ID (just the number)."""


def normalize_human_feedback(value: Optional[float]) -> Tuple[Optional[int], float]:
    """Map raw human feedback to (level, scaled_reward)."""
    if value is None:
        return None, 0.0
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None, 0.0

    if 1.0 <= v <= 7.0:
        level = int(round(v))
        reward = 0.5 * (level - 4)
        return level, reward
    if v in (-1.0, 1.0):
        level = 7 if v > 0 else 1
        reward = 1.5 * v
        return level, reward
    if -3.0 <= v <= 3.0:
        reward = 0.5 * v
        level = int(round(v + 4))
        level = max(1, min(7, level))
        return level, reward
    return None, 0.0


class HFDataCollector:
    """
    Human feedback data collector with history-aware parsing.
    """

    ACTION_MAP = {
        0: "none",
        1: "forward",
        2: "backward",
        3: "right",
        4: "left",
        5: "up",
        6: "down",
        7: "elbow_backward",
        8: "elbow_forward",
        9: "elbow_left",
        10: "elbow_right",
    }

    def __init__(self, model="gemma3:4b", base_url=None):
        model_info = load_ollama_model(model, base_url)
        self.base_url = model_info["base_url"]
        self.model = model_info["model"]
        self.system_prompt = get_action_parser_system_prompt()
        self.history = []

    def parse_text_to_action(self, text: str) -> int:
        history_str = ""
        if self.history:
            history_str = "\nHistory:\n" + "\n".join(
                [f"Instruction: \"{h['text']}\" -> Action: {h['action']} ({h['action_id']})"
                 for h in self.history[-2:]]
            ) + "\n\nIf the new instruction is ambiguous, repeat the most recent action."
        
        full_prompt = f"{self.system_prompt}{history_str}\n\nNew instruction: \"{text}\""
        # print(f"📝 Full prompt sent to Ollama:\n{full_prompt}\n")

        payload = {
            "model": self.model,
            "prompt": full_prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 10}
        }

        try:
            r = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=30)
            r.raise_for_status()
            result = r.json().get("response", "").strip()
            action_id_str = ''.join(filter(str.isdigit, result))
            
            if not action_id_str:
                return 0  # No digits found in response

            try:
                return int(action_id_str)
            except ValueError:
                # This is a safeguard in case the model returns unexpected non-digit characters
                print(f"⚠️  Could not parse action ID from model response: '{result}'")
                return 0
        except Exception as e:
            print(f"❌ Error calling Ollama: {e}")
            return 0

    def record(
        self,
        obs: np.ndarray,
        action: int,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
        path: str,
        text: str = "",
        human_feedback: Optional[float] = None,
    ):
        """
        Save one transition to JSONL, and update history.
        """
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)

        # Update history
        self.history.append({
            "text": text,
            "action": self.ACTION_MAP.get(action, "unknown"),
            "action_id": int(action)
        })
        if len(self.history) > 3:
            self.history.pop(0)

        hf_level, hf_reward = normalize_human_feedback(human_feedback)

        entry = {
            "obs": obs.tolist(),
            "action": self.ACTION_MAP.get(action, "unknown"),
            "action_id": int(action),
            "reward": float(reward),
            "next_obs": next_obs.tolist(),
            "done": bool(done),
            "text": text,
            "human_level": hf_level,
            "human_reward": hf_reward if hf_level is not None else None,
        }
        with open(path, "a", encoding="utf-8", errors="replace") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
