import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    vlm_rl_nav_dir = get_package_share_directory("vlm_rl_nav")
    localization_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(vlm_rl_nav_dir, "pros_demo", "localization_unity.xml")
        )
    )
    navigation_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(vlm_rl_nav_dir, "pros_demo", "navigation_unity.xml")
        )
    )
    rplidar_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(vlm_rl_nav_dir, "pros_demo", "rplidar_unity.xml")
        )
    )

    car_control_node = Node(
        package="car_control_pkg",
        executable="car_control_node",
        name="car_control_node",
        output="screen",
    )

    nav_goal_bridge_node = Node(
        package="nav_goal_bridge_pkg",
        executable="nav_goal_bridge_node",
        name="nav_goal_bridge_node",
        output="screen",
    )

    return LaunchDescription(
        [
            rplidar_launch,
            localization_launch,
            navigation_launch,
            car_control_node,
            nav_goal_bridge_node,
        ]
    )
