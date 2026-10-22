#!/usr/bin/env bash
# Arguments pass through to the launch file:
#     ./scripts/run_demo.sh ego_use_v2x:=false
#     ./scripts/run_demo.sh headless:=true rviz:=false
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ ! -f install/setup.bash ]]; then
    echo "Workspace is not built. Run ./scripts/build.sh first." >&2
    exit 1
fi

# shellcheck disable=SC1091
source /opt/ros/jazzy/setup.bash
# shellcheck disable=SC1091
source install/setup.bash

# A second workspace on the same LAN would otherwise join the same "track".
export ROS_LOCALHOST_ONLY=${ROS_LOCALHOST_ONLY:-1}

exec ros2 launch bfmc_v2x_ros demo.launch.py "$@"
