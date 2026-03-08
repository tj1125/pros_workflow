"""
VLM-RL Nav2 Navigation Launch File
=================================

Bring up the Unity lidar / scan matcher pipeline first, then localization.
After AMCL and the TF chain are stable, start planner_server only.
Path execution is handled in the application layer by a pros-style discrete
follower over /received_global_plan, so bt_navigator/controller_server are not
launched here.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    vlm_rl_nav_dir = get_package_share_directory("vlm_rl_nav")
    nav2_bringup_dir = get_package_share_directory("nav2_bringup")

    default_map = os.path.join(vlm_rl_nav_dir, "map01.yaml")
    default_params = os.path.join(vlm_rl_nav_dir, "nav2_params.yaml")

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

    base_to_laser_tf_cmd = Node(
        package="tf2_ros",
        executable="static_transform_publisher",
        name="base_to_laser_tf",
        arguments=["0.0", "0.0", "0.15", "3.14", "0.0", "0.0", "base_footprint", "laser"],
    )

    scan_relayer_cmd = Node(
        package="vlm_rl_nav",
        executable="scan_relayer",
        name="scan_relayer",
        output="screen",
    )

    scan_matcher_cmd = Node(
        package="ros2_laser_scan_matcher",
        executable="laser_scan_matcher",
        name="scan_matcher",
        output="screen",
        parameters=[
            {
                "base_frame": "base_footprint",
                "publish_tf": True,
                "publish_odom": "odom",
            }
        ],
    )

    localization_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_bringup_dir, "launch", "localization_launch.py")
        ),
        launch_arguments={
            "slam": "False",
            "map": LaunchConfiguration("map_file"),
            "params_file": LaunchConfiguration("params_file"),
            "use_sim_time": LaunchConfiguration("use_sim_time"),
            "autostart": "true",
        }.items(),
    )

    nav_start_gate_cmd = Node(
        package="vlm_rl_nav",
        executable="nav_start_gate",
        name="nav_start_gate",
        output="screen",
    )

    planner_server_cmd = Node(
        package="nav2_planner",
        executable="planner_server",
        name="planner_server",
        output="screen",
        parameters=[
            LaunchConfiguration("params_file"),
            {"use_sim_time": LaunchConfiguration("use_sim_time")},
        ],
    )

    planner_lifecycle_manager_cmd = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_navigation",
        output="screen",
        parameters=[
            {
                "use_sim_time": LaunchConfiguration("use_sim_time"),
                "autostart": True,
                "node_names": ["planner_server"],
            }
        ],
    )

    start_navigation_after_gate = RegisterEventHandler(
        OnProcessExit(
            target_action=nav_start_gate_cmd,
            on_exit=[
                planner_server_cmd,
                planner_lifecycle_manager_cmd,
            ],
        )
    )

    return LaunchDescription(
        [
            declare_map,
            declare_params,
            declare_use_sim_time,
            base_to_laser_tf_cmd,
            scan_relayer_cmd,
            scan_matcher_cmd,
            localization_launch,
            nav_start_gate_cmd,
            start_navigation_after_gate,
        ]
    )
