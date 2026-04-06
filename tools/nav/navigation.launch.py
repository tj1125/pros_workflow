import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    vlm_rl_nav_dir = get_package_share_directory("vlm_rl_nav")
    params_file = os.path.join(vlm_rl_nav_dir, "config", "mapper_params.yaml")
    localization_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(vlm_rl_nav_dir, "config", "localization_unity.xml")
        )
    )
    # NOTE: localization_unity.xml already includes rplidar_unity.xml — do NOT launch it again!
    navigation_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(vlm_rl_nav_dir, "config", "navigation_unity.xml")
        )
    )

    nav_goal_bridge_node = Node(
        package="nav_goal_bridge_pkg",
        executable="nav_goal_bridge_node",
        name="nav_goal_bridge_node",
        output="screen",
        parameters=[params_file],
    )

    return LaunchDescription(
        [
            localization_launch,
            navigation_launch,
            nav_goal_bridge_node,
        ]
    )
