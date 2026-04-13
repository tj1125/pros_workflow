# utils/observation_encoder.py
"""觀測狀態編碼器"""

import numpy as np
from typing import Tuple, Dict, Any

class ObservationEncoder:
    """將原始觀測編碼為適合 RL 的狀態表示"""
    
    def __init__(self, normalize: bool = True):
        self.normalize = normalize
        
        # 正規化參數 (根據你的環境調整)
        self.position_scale = 10.0  # 假設位置範圍 ±10
        self.distance_scale = 20.0  # 假設最大距離為 20
    
    def encode_position_difference(self, target_pos: np.ndarray, 
                                 current_pos: np.ndarray) -> np.ndarray:
        """編碼位置差異 (dx, dy, dz)"""
        diff = target_pos - current_pos
        
        if self.normalize:
            diff = diff / self.position_scale
            # 限制在 [-1, 1] 範圍
            diff = np.clip(diff, -1.0, 1.0)
        
        return diff.astype(np.float32)
    
    def encode_distance(self, target_pos: np.ndarray, 
                       current_pos: np.ndarray) -> float:
        """編碼歐幾里得距離"""
        distance = np.linalg.norm(target_pos - current_pos)
        
        if self.normalize:
            distance = distance / self.distance_scale
            distance = min(distance, 1.0)  # 限制最大值為 1
        
        return distance
    
    def encode_joint_angles(self, joint_angles: np.ndarray) -> np.ndarray:
        """編碼關節角度 (如果需要)"""
        if self.normalize:
            # 假設關節角度範圍為 [-180, 180]
            normalized_angles = joint_angles / 180.0
            normalized_angles = np.clip(normalized_angles, -1.0, 1.0)
            return normalized_angles.astype(np.float32)
        
        return joint_angles.astype(np.float32)
    
    def create_state_vector(self, cube_coords: np.ndarray) -> np.ndarray:
        """
        從 cube coordinates 創建狀態向量
        
        Args:
            cube_coords: [x1, y1, z1, x2, y2, z2] (current, target)
        
        Returns:
            state vector [dx, dy, dz]
        """
        current_pos = cube_coords[:3]
        target_pos = cube_coords[3:]
        
        return self.encode_position_difference(target_pos, current_pos)