"""The whole demonstration: world, infrastructure, two vehicles, RViz.

    ros2 launch bfmc_v2x_ros demo.launch.py
    ros2 launch bfmc_v2x_ros demo.launch.py ego_use_v2x:=false        # odometry only
    ros2 launch bfmc_v2x_ros demo.launch.py silent_lights:=[1]        # stale signal
    ros2 launch bfmc_v2x_ros demo.launch.py fix_latency:=0.6 fix_dropout:=0.35
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _setup(context, *args, **kwargs):
    ros_share = get_package_share_directory("bfmc_v2x_ros")
    sim_share = get_package_share_directory("bfmc_sim")

    world = os.path.join(sim_share, "worlds", "mini_city.sdf")
    track = os.path.join(sim_share, "track", "mini_city.yaml")
    rviz_config = os.path.join(ros_share, "config", "demo.rviz")

    headless = LaunchConfiguration("headless").perform(context).lower() in ("true", "1")
    use_rviz = LaunchConfiguration("rviz").perform(context).lower() in ("true", "1")
    silent = LaunchConfiguration("silent_lights").perform(context)

    gz_args = "-r -v 2 {}".format(world) if not headless else "-r -s -v 2 {}".format(world)
    common = {"use_sim_time": True}

    actions = [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")
            ),
            launch_arguments={"gz_args": gz_args}.items(),
        ),
        # Without the clock bridge nodes run on wall time and every message looks
        # hours old.
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="clock_bridge",
            output="screen",
            arguments=["/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock"],
        ),
        # -------------------------------------------------- infrastructure
        Node(
            package="bfmc_v2x_ros",
            executable="localization_server",
            name="localization_server",
            output="screen",
            parameters=[
                dict(
                    common,
                    track=track,
                    vehicles=["ego", "peer"],
                    vehicle_ids=[1, 2],
                    rate_hz=float(LaunchConfiguration("fix_rate").perform(context)),
                    latency_s=float(LaunchConfiguration("fix_latency").perform(context)),
                    dropout_probability=float(
                        LaunchConfiguration("fix_dropout").perform(context)
                    ),
                )
            ],
        ),
        Node(
            package="bfmc_v2x_ros",
            executable="semaphores",
            name="semaphores",
            output="screen",
            parameters=[
                dict(common, track=track, silent_lights=_parse_int_list(silent))
            ],
        ),
        Node(
            package="bfmc_v2x_ros",
            executable="environment_server",
            name="environment_server",
            output="screen",
            parameters=[dict(common, track=track)],
        ),
        Node(
            package="bfmc_v2x_ros",
            executable="metrics",
            name="metrics",
            output="screen",
            parameters=[
                dict(
                    common,
                    vehicles=["ego", "peer"],
                    output_dir=LaunchConfiguration("output_dir").perform(context),
                )
            ],
        ),
    ]

    # ------------------------------------------------------------ vehicles
    vehicle_launch = PythonLaunchDescriptionSource(
        os.path.join(ros_share, "launch", "vehicle.launch.py")
    )
    for name, vehicle_id, route, colour, bias, scale in (
        ("ego", "1", "ego", "0.15 0.35 0.75 1", "0.020", "1.03"),
        # Different bias and scale error, so the two drift differently.
        ("peer", "2", "peer", "0.85 0.35 0.10 1", "-0.014", "0.978"),
    ):
        arguments = {
            "name": name,
            "vehicle_id": vehicle_id,
            "route": route,
            "colour": colour,
            "gyro_bias": bias,
            "wheel_scale_error": scale,
            "track": track,
            "cruise_speed": LaunchConfiguration("cruise_speed").perform(context),
            "use_v2x": LaunchConfiguration("{}_use_v2x".format(name)).perform(context),
        }
        actions.append(
            IncludeLaunchDescription(vehicle_launch, launch_arguments=arguments.items())
        )

    if use_rviz and not headless:
        actions.append(
            Node(
                package="rviz2",
                executable="rviz2",
                name="rviz2",
                output="log",
                arguments=["-d", rviz_config],
                parameters=[common],
            )
        )

    return actions


def _parse_int_list(raw: str):
    """Accept ``[1,2]``, ``1,2`` or an empty string."""
    cleaned = raw.strip().strip("[]")
    if not cleaned:
        return []
    return [int(part) for part in cleaned.split(",") if part.strip()]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("headless", default_value="false"),
            DeclareLaunchArgument("rviz", default_value="true"),
            DeclareLaunchArgument("cruise_speed", default_value="0.35"),
            DeclareLaunchArgument(
                "ego_use_v2x",
                default_value="true",
                description="false drives the ego car on wheel odometry alone.",
            ),
            DeclareLaunchArgument("peer_use_v2x", default_value="true"),
            DeclareLaunchArgument(
                "fix_rate", default_value="4.0", description="Trackside camera rate, Hz."
            ),
            DeclareLaunchArgument(
                "fix_latency",
                default_value="0.25",
                description="Seconds between exposure and delivery of a pose fix.",
            ),
            DeclareLaunchArgument(
                "fix_dropout",
                default_value="0.08",
                description="Fraction of camera frames that miss a vehicle.",
            ),
            DeclareLaunchArgument(
                "silent_lights",
                default_value="[]",
                description="Light ids to take off the air, e.g. [1].",
            ),
            DeclareLaunchArgument(
                "output_dir",
                default_value="",
                description="Where metrics CSVs are written. Defaults to ~/bfmc_runs.",
            ),
            OpaqueFunction(function=_setup),
        ]
    )
