from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    arm_control_node = Node(
        package="arm_control_pkg",
        executable="arm_control_node",
        name="arm_control_node",
        output="screen",
        parameters=[{"use_sim_time": True}],
    )

    return LaunchDescription([arm_control_node])
