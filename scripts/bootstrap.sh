#!/usr/bin/env bash
# Install ROS 2 Jazzy and Gazebo Harmonic on Ubuntu 24.04. Idempotent; about
# 10-20 minutes and 3 GB on a clean machine.
set -euo pipefail

if [[ "$(. /etc/os-release && echo "$VERSION_ID")" != "24.04" ]]; then
    echo "This script targets Ubuntu 24.04 (ROS 2 Jazzy + Gazebo Harmonic)." >&2
    echo "On another release, install the matching ROS 2 / Gazebo pair by hand." >&2
    exit 1
fi

echo "==> Prerequisites"
sudo apt-get update
sudo apt-get install -y software-properties-common curl gnupg lsb-release
sudo add-apt-repository -y universe

echo "==> ROS 2 apt repository"
if [[ ! -f /etc/apt/sources.list.d/ros2.list ]]; then
    sudo curl -sSL https://raw.githubusercontent.com/ros/rosdistro/master/ros.key \
        -o /usr/share/keyrings/ros-archive-keyring.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] \
http://packages.ros.org/ros2/ubuntu $(. /etc/os-release && echo "$UBUNTU_CODENAME") main" \
        | sudo tee /etc/apt/sources.list.d/ros2.list > /dev/null
fi

sudo apt-get update

echo "==> ROS 2 Jazzy, Gazebo Harmonic and the ros_gz bridge"
sudo apt-get install -y \
    ros-jazzy-desktop \
    ros-jazzy-ros-gz \
    ros-jazzy-xacro \
    ros-jazzy-robot-state-publisher \
    ros-jazzy-tf2-ros \
    python3-colcon-common-extensions \
    python3-rosdep \
    python3-numpy \
    python3-yaml \
    python3-pytest \
    python3-matplotlib

echo "==> rosdep"
sudo rosdep init 2>/dev/null || true
rosdep update

echo
echo "Done. Next:"
echo "    source /opt/ros/jazzy/setup.bash"
echo "    ./scripts/build.sh"
echo "    ./scripts/run_demo.sh"
