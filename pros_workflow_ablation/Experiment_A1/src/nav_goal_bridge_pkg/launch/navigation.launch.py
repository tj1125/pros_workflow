import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    nav_goal_bridge_dir = get_package_share_directory("nav_goal_bridge_pkg")
    params_file = os.path.join(nav_goal_bridge_dir, "config", "mapper_params.yaml")
    keepout_map_file = os.path.join(nav_goal_bridge_dir, "config", "keepout_map.yaml")
    scan_throttle_node = Node(
        package="nav_goal_bridge_pkg",
        executable="scan_throttle_node",
        name="scan_throttle_node",
        output="screen",
        parameters=[
            {
                # Must match the Nav2 stack's clock: scan_throttle re-stamps /scan
                # with get_clock().now(), and AMCL stamps map->odom from /scan. If
                # this node runs on wall time while AMCL/controller use sim time,
                # map->odom ends up on the wrong clock -> "Transform data too old".
                "use_sim_time": False,
                "input_topic": "/scan_tmp",
                "output_topic": "/scan",
                "target_rate_hz": 20.0,
            }
        ],
    )
    localization_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(nav_goal_bridge_dir, "launch", "localization_unity.xml")
        )
    )
    # NOTE: localization_unity.xml already includes rplidar_unity.xml — do NOT launch it again!
    navigation_launch = IncludeLaunchDescription(
        AnyLaunchDescriptionSource(
            os.path.join(nav_goal_bridge_dir, "launch", "navigation_unity.xml")
        )
    )

    keepout_filter_mask_server = Node(
        package="nav2_map_server",
        executable="map_server",
        name="keepout_filter_mask_server",
        output="screen",
        parameters=[
            params_file,
            {"yaml_filename": keepout_map_file},
        ],
    )

    keepout_costmap_filter_info_server = Node(
        package="nav2_map_server",
        executable="costmap_filter_info_server",
        name="keepout_costmap_filter_info_server",
        output="screen",
        parameters=[params_file],
    )

    keepout_lifecycle_manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_keepout_filter",
        output="screen",
        parameters=[
            {"use_sim_time": False},
            {"autostart": True},
            {
                "node_names": [
                    "keepout_filter_mask_server",
                    "keepout_costmap_filter_info_server",
                ]
            },
        ],
    )

    nav_goal_bridge_node = Node(
        package="nav_goal_bridge_pkg",
        executable="nav_goal_bridge_node",
        name="nav_goal_bridge_node",
        output="screen",
        parameters=[
            params_file,
            {"use_sim_time": False},
        ],
    )

    return LaunchDescription(
        [
            scan_throttle_node,
            localization_launch,
            keepout_filter_mask_server,
            keepout_costmap_filter_info_server,
            keepout_lifecycle_manager,
            navigation_launch,
            nav_goal_bridge_node,
        ]
    )
