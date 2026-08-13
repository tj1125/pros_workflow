from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nav_params_file = (
        get_package_share_directory("nav_goal_bridge_pkg") + "/config/mapper_params.yaml"
    )
    car_control_node = Node(
        package="car_control_pkg",
        executable="car_control_node",
        name="car_control_node",
        output="screen",
        parameters=[nav_params_file],
    )

    return LaunchDescription([car_control_node])
