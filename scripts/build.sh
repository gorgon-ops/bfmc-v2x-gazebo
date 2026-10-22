#!/usr/bin/env bash
# Build the workspace and run the tests that do not need a simulator.
set -euo pipefail

cd "$(dirname "$0")/.."

if [[ -z "${ROS_DISTRO:-}" ]]; then
    if [[ -f /opt/ros/jazzy/setup.bash ]]; then
        # shellcheck disable=SC1091
        source /opt/ros/jazzy/setup.bash
    else
        echo "No ROS 2 environment found. Run ./scripts/bootstrap.sh first," >&2
        echo "or source your own installation before building." >&2
        exit 1
    fi
fi

echo "==> Regenerating the world from the track definition"
python3 src/bfmc_sim/scripts/generate_world.py

echo "==> colcon build"
colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

echo "==> Core tests (no simulator required)"
python3 -m pytest src/bfmc_v2x/test src/bfmc_sim/test -q

echo
echo "Built. Source the overlay and launch:"
echo "    source install/setup.bash"
echo "    ros2 launch bfmc_v2x_ros demo.launch.py"
