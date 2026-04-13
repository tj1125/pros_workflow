import math

def quantize_joint_angles(joints, step=10):
    return [round(math.degrees(j) / step) * step for j in joints]

def make_joint_key(qs):
    return f"q1_{qs[0]}_q2_{qs[1]}_q3_{qs[2]}_q4_{qs[3]}"

def within_limits(qs_deg, joint_limits):
    for i, q in enumerate(qs_deg):
        jlim = joint_limits[str(i)]
        if q < jlim["min_angle"] or q > jlim["max_angle"]:
            return False
    return True