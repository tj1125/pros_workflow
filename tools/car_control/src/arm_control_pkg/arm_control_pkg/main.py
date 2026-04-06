import rclpy
from rclpy.executors import MultiThreadedExecutor

from arm_control_pkg.arm_action_server import ArmActionServer
from arm_control_pkg.arm_angle_control import ArmAngleControl
from arm_control_pkg.arm_auto_controller import ArmAutoController
from arm_control_pkg.arm_commute_node import ArmCummuteNode
from arm_control_pkg.arm_manual import ManualControlNode
from arm_control_pkg.load_params import LoadParams
from arm_control_pkg.pybullet_ik_gripper import PybulletRobotController


def main(args=None):
    rclpy.init(args=args)

    load_params = LoadParams("arm_control_pkg")
    arm_agnle_control = ArmAngleControl(arm_params=load_params)
    arm_commute_node = ArmCummuteNode(
        arm_params=load_params, arm_angle_control=arm_agnle_control
    )
    pybullet_robot_controller = PybulletRobotController(
        arm_params=load_params,
        arm_angle_control_node=arm_agnle_control,
    )
    arm_auto_controller = ArmAutoController(
        arm_params=load_params,
        arm_commute_node=arm_commute_node,
        pybulletRobotController=pybullet_robot_controller,
        arm_agnle_control=arm_agnle_control,
    )
    arm_action_server = ArmActionServer(
        arm_commute_node=arm_commute_node,
        arm_auto_controller=arm_auto_controller,
    )
    arm_manual_node = ManualControlNode(
        arm_commute_node=arm_commute_node,
        arm_angle_control_node=arm_agnle_control,
        arm_params=load_params,
    )

    executor = MultiThreadedExecutor()
    executor.add_node(arm_commute_node)
    executor.add_node(arm_manual_node)
    executor.add_node(arm_action_server)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        arm_action_server.destroy_node()
        arm_manual_node.destroy_node()
        arm_commute_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
