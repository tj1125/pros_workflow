#!/usr/bin/env bash

set -eo pipefail

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-1}"

source /opt/ros/humble/setup.bash

if [ -f /opt/ros/vendor_laser_scan_matcher/setup.bash ]; then
    source /opt/ros/vendor_laser_scan_matcher/setup.bash
fi

if [ ! -f /workspaces/install/setup.bash ]; then
    echo "Missing /workspaces/install/setup.bash. Build the ROS workspace with: r" >&2
    exit 1
fi

source /workspaces/install/setup.bash

exec ros2 launch workflow_bringup runtime.launch.py
