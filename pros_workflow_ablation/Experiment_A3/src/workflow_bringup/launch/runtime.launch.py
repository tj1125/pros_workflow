import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import AnyLaunchDescriptionSource


def _include(package_name: str, relative_launch_path: str) -> IncludeLaunchDescription:
    package_dir = get_package_share_directory(package_name)
    return IncludeLaunchDescription(
        AnyLaunchDescriptionSource(os.path.join(package_dir, relative_launch_path))
    )


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            _include("nav_goal_bridge_pkg", "launch/navigation.launch.py"),
            _include("car_control_pkg", "launch/car_control.launch.py"),
            _include("arm_control_pkg", "launch/arm_control.launch.py"),
            _include("rosbridge_server", "launch/rosbridge_websocket_launch.xml"),
        ]
    )
