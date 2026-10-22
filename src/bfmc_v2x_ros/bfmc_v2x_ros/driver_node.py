"""The decision loop. It cannot see ground truth: localisation failure shows up as a car
stopping in the wrong place. Output is a twist for Gazebo's Ackermann system.
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, Twist
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

import bfmc_v2x_msgs.msg as bfmc_msgs
from bfmc_v2x.driver import Driver, DriverConfig
from bfmc_v2x.geometry import Pose2D
from bfmc_v2x.semaphore import SemaphoreTracker, StopLinePolicy
from bfmc_v2x.v2v import PeerRegistry

from .common import LATCHED_QOS, STATE_QOS, V2X_QOS, line_strip, load_track, marker
from .conversions import msg_to_peer_state, msg_to_semaphore, yaw_from_quaternion


class DriverNode(Node):
    def __init__(self) -> None:
        super().__init__("driver")

        self.declare_parameter("vehicle", "ego")
        self.declare_parameter("vehicle_id", 1)
        self.declare_parameter("route", "ego")
        self.declare_parameter("track", "")
        self.declare_parameter("control_hz", 50.0)
        self.declare_parameter("cruise_speed", 0.35)
        self.declare_parameter("wheelbase", 0.26)
        self.declare_parameter("vehicle_length", 0.36)
        self.declare_parameter("max_steer_deg", 25.0)
        self.declare_parameter("lookahead_min_m", 0.32)
        self.declare_parameter("lookahead_gain_s", 0.9)
        self.declare_parameter("signal_stale_after_s", 1.0)
        self.declare_parameter("peer_expire_after_s", 1.0)
        #: Seconds without a pose estimate before the vehicle stops itself.
        self.declare_parameter("pose_timeout_s", 0.5)

        self.vehicle = str(self.get_parameter("vehicle").value)
        self.vehicle_id = int(self.get_parameter("vehicle_id").value)
        self.pose_timeout = float(self.get_parameter("pose_timeout_s").value)

        track = load_track(self.get_parameter("track").value or None)
        self.route = track.build_route(str(self.get_parameter("route").value))

        config = DriverConfig(
            wheelbase=float(self.get_parameter("wheelbase").value),
            vehicle_length=float(self.get_parameter("vehicle_length").value),
            cruise_speed=float(self.get_parameter("cruise_speed").value),
            max_steer_rad=math.radians(float(self.get_parameter("max_steer_deg").value)),
            lookahead_min_m=float(self.get_parameter("lookahead_min_m").value),
            lookahead_gain_s=float(self.get_parameter("lookahead_gain_s").value),
            vehicle_id=self.vehicle_id,
        )
        self.config = config
        self._driver = Driver(
            self.route, config, StopLinePolicy(cruise_speed=config.cruise_speed)
        )

        self._lights = SemaphoreTracker(
            stale_after_s=float(self.get_parameter("signal_stale_after_s").value)
        )
        self._peers = PeerRegistry(
            expire_after_s=float(self.get_parameter("peer_expire_after_s").value)
        )

        self._pose: Optional[Pose2D] = None
        self._speed = 0.0
        self._yaw_rate = 0.0
        self._pose_stamp = 0.0

        ns = "/{}".format(self.vehicle)
        self.create_subscription(
            PoseWithCovarianceStamped, ns + "/localization/pose", self._on_pose, STATE_QOS
        )
        self.create_subscription(
            bfmc_msgs.PeerState, ns + "/localization/peer_state", self._on_own_state, STATE_QOS
        )
        self.create_subscription(
            bfmc_msgs.SemaphoreState, ns + "/v2x/semaphore", self._on_semaphore, V2X_QOS
        )
        self.create_subscription(bfmc_msgs.PeerState, ns + "/v2x/peer", self._on_peer, V2X_QOS)

        self._cmd_pub = self.create_publisher(Twist, ns + "/cmd_vel", STATE_QOS)
        self._status_pub = self.create_publisher(
            bfmc_msgs.DriveStatus, ns + "/drive_status", STATE_QOS
        )
        self._route_pub = self.create_publisher(
            MarkerArray, ns + "/route_markers", LATCHED_QOS
        )

        self.create_timer(1.0 / float(self.get_parameter("control_hz").value), self._control)
        self.create_timer(2.0, self._publish_route)
        self._publish_route()

        self.get_logger().info(
            "driver up for {} on route '{}' ({:.1f} m, {} stop line(s))".format(
                self.vehicle, self.route.name, self.route.length, len(self.route.stop_lines)
            )
        )

    # -------------------------------------------------------------- inputs

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _on_pose(self, message: PoseWithCovarianceStamped) -> None:
        self._pose = Pose2D(
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw_from_quaternion(message.pose.pose.orientation),
        )
        self._pose_stamp = self._now()

    def _on_own_state(self, message: bfmc_msgs.PeerState) -> None:
        # The filter is the authority on our speed: the encoder is biased and the
        # command is only what we asked for.
        self._speed = message.speed
        self._yaw_rate = message.yaw_rate

    def _on_semaphore(self, message: bfmc_msgs.SemaphoreState) -> None:
        payload, ttl = msg_to_semaphore(message)
        self._lights.ingest(payload, rx_time=self._now(), link_age_s=0.0)

    def _on_peer(self, message: bfmc_msgs.PeerState) -> None:
        payload, age = msg_to_peer_state(message)
        self._peers.ingest(payload, rx_time=self._now(), link_age_s=age)

    # ------------------------------------------------------------- control

    def _control(self) -> None:
        now = self._now()

        if self._pose is None or (now - self._pose_stamp) > self.pose_timeout:
            # No idea where we are: driving on a stale pose is the failure this project
            # is about.
            self._cmd_pub.publish(Twist())
            if self._pose is not None:
                self.get_logger().warn(
                    "no pose for {:.2f} s, holding".format(now - self._pose_stamp),
                    throttle_duration_sec=2.0,
                )
            return

        arc_s, _ = self.route.locate(self._pose.point)
        stop_line, _ = self.route.next_stop_line(arc_s)
        light = None
        if stop_line is not None and stop_line.light_id is not None:
            light = self._lights.observe(stop_line.light_id, now)

        command = self._driver.step(
            pose=self._pose,
            speed=self._speed,
            yaw_rate=self._yaw_rate,
            light=light,
            peers=self._peers.active(now),
        )

        twist = Twist()
        twist.linear.x = float(command.speed)
        # Gazebo's Ackermann system takes a yaw rate, so convert through the bicycle
        # model.
        twist.angular.z = float(
            command.speed * math.tan(command.steer) / self.config.wheelbase
        )
        self._cmd_pub.publish(twist)

        self._publish_status(command)

    def _publish_status(self, command) -> None:
        message = bfmc_msgs.DriveStatus()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "map"
        message.commanded_speed = float(command.speed)
        message.steering_angle = float(command.steer)
        message.limiting_factor = command.limiting_factor
        message.reason = command.reason
        message.arc_s = float(command.arc_s)
        message.lateral_error = float(command.lateral_error_m)
        message.distance_to_line = float(
            command.distance_to_line_m if math.isfinite(command.distance_to_line_m) else -1.0
        )
        message.approach_decision = (
            command.approach.decision.value if command.approach else ""
        )
        message.right_of_way = command.conflict.verdict.value if command.conflict else ""
        message.conflicting_peer = (
            int(command.conflict.peer_id)
            if command.conflict and command.conflict.peer_id is not None
            else -1
        )
        self._status_pub.publish(message)

    def _publish_route(self) -> None:
        markers = MarkerArray()
        markers.markers.append(
            line_strip(
                self.route.lane.points,
                marker_id=0,
                namespace="{}_route".format(self.vehicle),
                colour=(0.25, 0.55, 0.95) if self.vehicle_id == 1 else (0.95, 0.45, 0.20),
                width=0.015,
                alpha=0.75,
            )
        )
        for index, line in enumerate(self.route.stop_lines):
            point = self.route.lane.point_at(line.arc_s)
            item = marker(
                10 + index,
                "{}_stop_lines".format(self.vehicle),
                Marker.CUBE,
                scale=(0.03, 0.34, 0.01),
                colour=(1.0, 1.0, 1.0),
            )
            item.pose.position.x = point.x
            item.pose.position.y = point.y
            item.pose.position.z = 0.005
            markers.markers.append(item)
        self._route_pub.publish(markers)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DriverNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
