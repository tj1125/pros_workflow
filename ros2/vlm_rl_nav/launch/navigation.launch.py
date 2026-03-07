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
from launch_ros.actions import Node


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

    # -------------------------------------------------------------
    # 建立 Unity Lidar 所需的 TF 樹與 Odometry
    # -------------------------------------------------------------
    # 1. 靜態 TF: base_footprint -> laser (Z 軸 +0.15, Yaw 反轉 3.14)
    base_to_laser_tf_cmd = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='base_to_laser_tf',
        arguments=['0.0', '0.0', '0.15', '3.14', '0.0', '0.0', 'base_footprint', 'laser']
    )

    # 1.5 Scan Relayer: 解決 Unity 與 Docker 容器間的時鐘誤差 (Clock Skew)
    # Unity 發送的雷達資料帶有過去的時間戳，導致 TF Extrapolation Error
    # 此節點將 /scan_tmp 重新打包為 /scan 並賦予當下 (now) 的系統時間
    scan_relayer_cmd = Node(
        package='vlm_rl_nav',
        executable='scan_relayer',
        name='scan_relayer',
        output='screen'
    )

    # 2. Laser Scan Matcher: 利用 Lidar 掃描資料推算 Odom
    # 這會補足 Unity 沒有直接發布 /odom topic 的問題，
    # 負責發布 transform: odom -> base_footprint
    scan_matcher_cmd = Node(
        package='ros2_laser_scan_matcher',
        executable='laser_scan_matcher',
        name='scan_matcher',
        output='screen',
        parameters=[{
            'base_frame': 'base_footprint',
            'publish_tf': True,
            'publish_odom': 'odom'
        }]
    )

    # --- Include Nav2 bringup's bringup_launch.py ---
    # This starts EVERYTHING: amcl, map_server, bt_navigator, controller_server,
    #                         planner_server, behavior_server, velocity_smoother,
    #                         and lifecycle_manager.
    nav2_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory("nav2_bringup"),
                "launch",
                "bringup_launch.py",
            )
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
            base_to_laser_tf_cmd,
            scan_relayer_cmd,
            scan_matcher_cmd,
            nav2_launch,
        ]
    )
