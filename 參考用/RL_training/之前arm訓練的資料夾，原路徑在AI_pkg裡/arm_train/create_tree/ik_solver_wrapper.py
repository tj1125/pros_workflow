from robot_controller import PybulletRobotController, DummyAngleControlNode

class IKSolverWrapper:
    def __init__(self, config, reset_q):
        self.controller = PybulletRobotController(
            arm_params=config,
            arm_angle_control_node=DummyAngleControlNode(reset_q)
        )

    def get_target_pose(self, current_q, direction, distance):
        import math
        joint_rad = [math.radians(q) for q in current_q]
        self.controller.setJointPosition(joint_rad)
        return self.controller.calculate_ee_relative_target_positions(distance).get(direction, None)

    def solve(self, target_pose):
        return self.controller.solveInversePositionKinematics(target_pose)