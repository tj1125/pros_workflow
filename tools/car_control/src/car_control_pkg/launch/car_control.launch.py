from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    car_control_node = Node(
        package="car_control_pkg",
        executable="car_control_node",
        name="car_control_node",
        output="screen",
    )

    return LaunchDescription([car_control_node])
