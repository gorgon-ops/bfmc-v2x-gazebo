# ROS 2 Jazzy + Gazebo Harmonic, with the workspace pre-built.
#
# Exists so the demo can be reproduced without touching the host: the ROS
# and Gazebo versions are pinned by the base image, and the build is the
# same one CI runs.
FROM ros:jazzy-ros-base-noble

SHELL ["/bin/bash", "-c"]

RUN apt-get update && apt-get install -y --no-install-recommends \
        ros-jazzy-ros-gz \
        ros-jazzy-xacro \
        ros-jazzy-robot-state-publisher \
        ros-jazzy-tf2-ros \
        ros-jazzy-rviz2 \
        python3-colcon-common-extensions \
        python3-numpy \
        python3-yaml \
        python3-pytest \
        python3-matplotlib \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /ws

# Copy sources last so dependency installation stays cached across edits.
COPY src/ /ws/src/
COPY scripts/ /ws/scripts/

RUN source /opt/ros/jazzy/setup.bash \
    && python3 src/bfmc_sim/scripts/generate_world.py \
    && colcon build --symlink-install --cmake-args -DCMAKE_BUILD_TYPE=Release

COPY docker/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

ENV ROS_LOCALHOST_ONLY=1
ENTRYPOINT ["/entrypoint.sh"]
CMD ["ros2", "launch", "bfmc_v2x_ros", "demo.launch.py", "headless:=true", "rviz:=false"]
