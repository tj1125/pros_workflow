# utils/joint_limits.py
"""定義機械手臂關節限制"""

JOINT_LIMITS = {
    'joint0': {'min': -180, 'max': 180, 'step': 15},
    'joint1': {'min': -90, 'max': 90, 'step': 15},
    'joint2': {'min': -135, 'max': 135, 'step': 15},
    'joint3': {'min': -180, 'max': 180, 'step': 15}
}

def is_valid_angle(joint_id: int, angle: float) -> bool:
    """檢查關節角度是否在有效範圍內"""
    joint_name = f'joint{joint_id}'
    if joint_name not in JOINT_LIMITS:
        return False
    
    limits = JOINT_LIMITS[joint_name]
    return limits['min'] <= angle <= limits['max']

def clamp_angle(joint_id: int, angle: float) -> float:
    """將角度限制在有效範圍內"""
    joint_name = f'joint{joint_id}'
    if joint_name not in JOINT_LIMITS:
        return angle
    
    limits = JOINT_LIMITS[joint_name]
    return max(limits['min'], min(limits['max'], angle))