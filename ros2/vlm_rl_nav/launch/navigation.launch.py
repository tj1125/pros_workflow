"""
VLM-RL Nav2 Navigation Launch File
====================================
Launches the full Nav2 stack using:
- Our own config:  share/vlm_rl_nav/config/nav2_params.yaml
- Our own map:     share/vlm_rl_nav/map/map01.yaml

Usage:
    ros2 launch vlm_rl_nav navigation.launch.py
    ros2 launch vlm_rl_nav navigation.launch.py map_file:=/path/to/map.yaml
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration


def generate_launch_description() -> LaunchDescription:
    # --- Package directories ---
    vlm_rl_nav_dir = get_package_share_directory("vlm_rl_nav")
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    # --- Default paths (can be overridden via CLI arguments) ---
    default_map = os.path.join(vlm_rl_nav_dir, "map", "map01.yaml")
    default_params = os.path.join(vlm_rl_nav_dir, "config", "nav2_params.yaml")

    # --- Declare launch arguments ---
    declare_map = DeclareLaunchArgument(
        "map_file",
        default_value=default_map,
        description="Full path to the ROS 2 map .yaml file",
    )
    declare_params = DeclareLaunchArgument(
        "params_file",
        default_value=default_params,
        description="Full path to the Nav2 params .yaml file",
    )
    declare_use_sim_time = DeclareLaunchArgument(
        "use_sim_time",
        default_value="false",
        description="Use simulation clock if true",
    )

    # --- Include Nav2 bringup's navigation_launch.py ---
    # This starts: amcl, map_server, bt_navigator, controller_server,
    #              planner_server, behavior_server, velocity_smoother,
    #              waypoint_follower, lifecycle_manager
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, "launch", "navigation_launch.py")
        ),
        launch_arguments={
            "map": LaunchConfiguration("map_file"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "params_file": LaunchConfiguration("params_file"),
            "autostart": "true",
        }.items(),
    )

    return LaunchDescription(
        [
            declare_map,
            declare_params,
            declare_use_sim_time,
            nav2_launch,
        ]
    )
