"""Simulated detector; what is real is ``detection (body) + pose estimate -> report (track)``.

Ground truth decides visibility (the camera is on the real car); the fused
estimate decides placement. Mixing them up would quietly invalidate the result.
"""

from __future__ import annotations

import math
import random
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

import bfmc_v2x_msgs.msg as bfmc_msgs
from bfmc_v2x.environment import ObstacleDetector
from bfmc_v2x.geometry import Pose2D
from bfmc_v2x.protocol import ObstacleReport

from .common import SENSOR_QOS, STATE_QOS, load_track
from .conversions import obstacle_report_to_msg, yaw_from_quaternion


class DetectorNode(Node):
    def __init__(self) -> None:
        super().__init__("detector")

        self.declare_parameter("vehicle", "ego")
        self.declare_parameter("vehicle_id", 1)
        self.declare_parameter("track", "")
        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("max_range_m", 1.6)
        self.declare_parameter("half_fov_deg", 35.0)
        self.declare_parameter("range_noise_fraction", 0.06)
        self.declare_parameter("bearing_noise_deg", 2.0)
        self.declare_parameter("min_confidence", 0.45)
        self.declare_parameter("seed", 8675309)
        #: Report using the dead-reckoned pose instead, to show the contrast.
        self.declare_parameter("use_fused_pose", True)

        self.vehicle = str(self.get_parameter("vehicle").value)
        self.vehicle_id = int(self.get_parameter("vehicle_id").value)
        self.min_confidence = float(self.get_parameter("min_confidence").value)
        self.use_fused = bool(self.get_parameter("use_fused_pose").value)

        track = load_track(self.get_parameter("track").value or None)
        self._detector = ObstacleDetector(
            obstacles=track.obstacles,
            max_range_m=float(self.get_parameter("max_range_m").value),
            half_fov_rad=math.radians(float(self.get_parameter("half_fov_deg").value)),
            range_noise_frac=float(self.get_parameter("range_noise_fraction").value),
            bearing_noise_rad=math.radians(float(self.get_parameter("bearing_noise_deg").value)),
            rng=random.Random(int(self.get_parameter("seed").value)),
        )

        self._true_pose: Optional[Pose2D] = None
        self._estimated_pose: Optional[Pose2D] = None
        self._seq = 0
        self._sent = 0

        ns = "/{}".format(self.vehicle)
        self.create_subscription(Odometry, ns + "/odom_truth", self._on_truth, SENSOR_QOS)
        source = "/localization/pose" if self.use_fused else "/localization/dead_reckoned"
        self.create_subscription(
            PoseWithCovarianceStamped, ns + source, self._on_estimate, STATE_QOS
        )

        self._report_pub = self.create_publisher(
            bfmc_msgs.ObstacleReport, ns + "/obstacle_report", STATE_QOS
        )
        self.create_subscription(
            bfmc_msgs.ObstacleAck, ns + "/v2x/obstacle_ack", self._on_ack, STATE_QOS
        )

        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._scan)

        self.get_logger().info(
            "detector up for {}: {:.1f} m range, {:.0f} deg field of view, "
            "reporting from the {} pose".format(
                self.vehicle,
                float(self.get_parameter("max_range_m").value),
                2.0 * float(self.get_parameter("half_fov_deg").value),
                "fused" if self.use_fused else "DEAD-RECKONED",
            )
        )

    def _on_truth(self, message: Odometry) -> None:
        self._true_pose = Pose2D(
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw_from_quaternion(message.pose.pose.orientation),
        )

    def _on_estimate(self, message: PoseWithCovarianceStamped) -> None:
        self._estimated_pose = Pose2D(
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw_from_quaternion(message.pose.pose.orientation),
        )

    def _scan(self) -> None:
        if self._true_pose is None or self._estimated_pose is None:
            return

        for detection in self._detector.detect(self._true_pose):
            if detection.confidence < self.min_confidence:
                continue

            placed = detection.to_track_frame(self._estimated_pose)

            self._seq += 1
            payload = ObstacleReport(
                kind=int(detection.kind),
                x=placed.x,
                y=placed.y,
                confidence=detection.confidence,
            )
            message = obstacle_report_to_msg(payload, self.vehicle_id, self._seq)
            message.header.stamp = self.get_clock().now().to_msg()
            message.header.frame_id = "map"
            self._report_pub.publish(message)
            self._sent += 1

    def _on_ack(self, message: bfmc_msgs.ObstacleAck) -> None:
        if message.accepted:
            self.get_logger().info(
                "obstacle report {} accepted, {:.3f} m from truth".format(
                    message.report_seq, message.error
                )
            )


def main(args=None) -> None:
    rclpy.init(args=args)
    node = DetectorNode()
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
