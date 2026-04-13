from stable_baselines3 import PPO
import numpy as np

ACTION_LIST = [
    "joint0_plus", "joint0_minus",
    "joint1_plus", "joint1_minus",
    "joint2_plus", "joint2_minus",
    "joint3_plus", "joint3_minus",
]

def load_model_and_predict(model_path):
    model = PPO.load(model_path)
    
    def predict_fn(state):
        action, _ = model.predict(np.array(state), deterministic=True)
        return int(action)
    
    return predict_fn