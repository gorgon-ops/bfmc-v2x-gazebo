"""One vehicle's stack, namespaced by name. The spawn pose comes from the track definition,
so moving a road never starts a car in a wall.
"""

from __future__ import annotations

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def _spawn_pose(track_path: str, route_name: str):
    from bfmc_v2x.track import Track

    route = Track.from_yaml(track_path).build_route(route_name)
    point, heading = route.start_pose()
    return point.x, point.y, heading


def _setup(context, *args, **kwargs):
    name = LaunchConfiguration("name").perform(context)
    vehicle_id = int(LaunchConfiguration("vehicle_id").perform(context))
    route = LaunchConfiguration("route").perform(context)
    colour = LaunchConfiguration("colour").perform(context)
    gyro_bias = LaunchConfiguration("gyro_bias").perform(context)
    scale_error = float(LaunchConfiguration("wheel_scale_error").perform(context))
    use_v2x = LaunchConfiguration("use_v2x").perform(context).lower() in ("true", "1")
    cruise_speed = float(LaunchConfiguration("cruise_speed").perform(context))

    sim_share = get_package_share_directory("bfmc_sim")
    track_path = LaunchConfiguration("track").perform(context) or os.path.join(
        sim_share, "track", "mini_city.yaml"
    )
    xacro_path = os.path.join(sim_share, "urdf", "vehicle.urdf.xacro")

    x, y, yaw = _spawn_pose(track_path, route)

    import xacro

    description = xacro.process_file(
        xacro_path,
        mappings={"name": name, "colour": colour, "gyro_bias": gyro_bias},
    ).toxml()

    common = {"use_sim_time": True}
    vehicle_params = dict(common, vehicle=name, vehicle_id=vehicle_id, track=track_path)

    return [
        # ------------------------------------------------ description + spawn
        Node(
            package="robot_state_publisher",
            executable="robot_state_publisher",
            name="robot_state_publisher",
            namespace=name,
            output="screen",
            parameters=[
                dict(common, robot_description=description, frame_prefix=name + "/")
            ],
        ),
        Node(
            package="ros_gz_sim",
            executable="create",
            name="spawn_" + name,
            output="screen",
            arguments=[
                "-string", description,
                "-name", name,
                "-x", "{:.4f}".format(x),
                "-y", "{:.4f}".format(y),
                "-z", "0.02",
                "-Y", "{:.5f}".format(yaw),
            ],
        ),
        # ------------------------------------------------------------- bridge
        Node(
            package="ros_gz_bridge",
            executable="parameter_bridge",
            name="gz_bridge_" + name,
            output="screen",
            parameters=[common],
            arguments=[
                # Ground truth: only the sensor simulators and scoring nodes may subscribe.
                "/model/{}/odometry@nav_msgs/msg/Odometry[gz.msgs.Odometry".format(name),
                "/model/{}/imu@sensor_msgs/msg/Imu[gz.msgs.IMU".format(name),
                "/model/{}/joint_state@sensor_msgs/msg/JointState[gz.msgs.Model".format(name),
                "/model/{}/cmd_vel@geometry_msgs/msg/Twist]gz.msgs.Twist".format(name),
            ],
            remappings=[
                ("/model/{}/odometry".format(name), "/{}/odom_truth".format(name)),
                ("/model/{}/imu".format(name), "/{}/imu".format(name)),
                ("/model/{}/joint_state".format(name), "/{}/joint_states".format(name)),
                ("/model/{}/cmd_vel".format(name), "/{}/cmd_vel".format(name)),
            ],
        ),
        # ------------------------------------------------ simulated sensors
        Node(
            package="bfmc_v2x_ros",
            executable="wheel_encoder",
            name="wheel_encoder",
            namespace=name,
            output="screen",
            parameters=[dict(common, vehicle=name, scale_error=scale_error)],
        ),
        # ------------------------------------------------------ the vehicle
        Node(
            package="bfmc_v2x_ros",
            executable="v2x_bridge",
            name="v2x_bridge",
            namespace=name,
            output="screen",
            parameters=[dict(common, vehicle=name, vehicle_id=vehicle_id)],
        ),
        Node(
            package="bfmc_v2x_ros",
            executable="localization",
            name="localization",
            namespace=name,
            output="screen",
            parameters=[dict(vehicle_params, route=route, use_v2x_fixes=use_v2x)],
        ),
        Node(
            package="bfmc_v2x_ros",
            executable="driver",
            name="driver",
            namespace=name,
            output="screen",
            parameters=[dict(vehicle_params, route=route, cruise_speed=cruise_speed)],
        ),
        Node(
            package="bfmc_v2x_ros",
            executable="detector",
            name="detector",
            namespace=name,
            output="screen",
            parameters=[dict(vehicle_params, seed=1000 + vehicle_id)],
        ),
    ]


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument("name", default_value="ego"),
            DeclareLaunchArgument("vehicle_id", default_value="1"),
            DeclareLaunchArgument("route", default_value="ego"),
            DeclareLaunchArgument("colour", default_value="0.15 0.35 0.75 1"),
            DeclareLaunchArgument("track", default_value=""),
            DeclareLaunchArgument(
                "gyro_bias",
                default_value="0.02",
                description="rad/s of gyro bias. This is what makes dead reckoning drift.",
            ),
            DeclareLaunchArgument(
                "wheel_scale_error",
                default_value="1.03",
                description="Effective tyre-radius error, the other half of the drift.",
            ),
            DeclareLaunchArgument(
                "use_v2x",
                default_value="true",
                description="Set false to drive on dead reckoning alone.",
            ),
            DeclareLaunchArgument("cruise_speed", default_value="0.35"),
            OpaqueFunction(function=_setup),
        ]
    )
