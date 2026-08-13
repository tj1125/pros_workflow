"""In-memory joint-angle state and limit validation for the arm."""


class ArmAngleControl:
    def __init__(self, arm_params):
        self.arm_params = arm_params.get_arm_params()
        self.joint_positions = []
        self.arm_init()

    def get_arm_angles(self):
        return self.joint_positions

    def arm_init(self):
        joints_count = int(self.arm_params["global"]["joints_count"])
        self.joint_positions = []

        for i in range(joints_count):
            try:
                pos = float(self.arm_params["joints_reset"][i])
            except (KeyError, ValueError, TypeError):
                pos = 90.0
            self.joint_positions.append(pos)

    def arm_default_change(self):
        joints_reset = self.arm_params["joints_reset"]
        for index in range(len(self.joint_positions)):
            self.joint_positions[index] = float(joints_reset[index])
        return self.joint_positions

    def arm_index_change(self, index, angle):
        self.joint_positions[index] = angle
        self.joint_positions = self.validate_joint_limits(self.joint_positions)

    def arm_all_change(self, angles):
        self.joint_positions = self.validate_joint_limits(list(angles))

    def arm_increase_decrease(self, index, delta):
        joints_count = int(self.arm_params["global"]["joints_count"])
        if index < 0 or index >= joints_count:
            return self.joint_positions

        self.joint_positions[index] = self.joint_positions[index] + delta
        self.joint_positions = self.validate_joint_limits(self.joint_positions)
        return self.joint_positions

    def validate_joint_limits(self, positions):
        for joint_key in range(len(positions)):
            min_angle = float(
                self.arm_params["joints"][joint_key].get("min_angle", 0.0)
            )
            max_angle = float(
                self.arm_params["joints"][joint_key].get("max_angle", 180.0)
            )
            positions[joint_key] = max(min(positions[joint_key], max_angle), min_angle)
        return positions
