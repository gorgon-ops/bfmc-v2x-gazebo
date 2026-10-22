"""ROS adapter over :class:`bfmc_v2x.ekf.DelayedFusionEKF`.

Fixes go in with their exposure time, never arrival time. A dead-reckoning
shadow runs on identical inputs, both as the baseline and as a warm fallback.
"""

from __future__ import annotations

import math
from typing import Optional

import rclpy
from geometry_msgs.msg import PoseWithCovarianceStamped, TransformStamped, Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node
from sensor_msgs.msg import Imu
from tf2_ros import TransformBroadcaster

import bfmc_v2x_msgs.msg as bfmc_msgs
from bfmc_v2x.ekf import (
    IDX_X,
    IDX_Y,
    IDX_YAW,
    DeadReckoning,
    DelayedFusionEKF,
    ProcessNoise,
)
from bfmc_v2x.geometry import Pose2D, wrap_angle
from bfmc_v2x.protocol import Intent

from .common import SENSOR_QOS, STATE_QOS, V2X_QOS, load_track
from .conversions import (
    msg_to_pose_fix,
    quaternion_from_yaw,
    stamp_to_seconds,
    yaw_from_quaternion,
)


class LocalizationNode(Node):
    def __init__(self) -> None:
        super().__init__("localization")

        self.declare_parameter("vehicle", "ego")
        self.declare_parameter("vehicle_id", 1)
        self.declare_parameter("route", "ego")
        self.declare_parameter("track", "")
        self.declare_parameter("publish_hz", 50.0)
        self.declare_parameter("wheel_speed_sigma", 0.05)
        self.declare_parameter("buffer_horizon_s", 1.5)
        #: Set false to watch the vehicle drive on dead reckoning alone.
        self.declare_parameter("use_v2x_fixes", True)
        self.declare_parameter("publish_tf", True)

        # The only knobs that meaningfully change filter behaviour.
        self.declare_parameter("accel_noise", 0.6)
        self.declare_parameter("gyro_noise", 0.02)
        self.declare_parameter("gyro_bias_noise", 0.001)
        self.declare_parameter("slip_noise", 0.02)

        self.vehicle = str(self.get_parameter("vehicle").value)
        self.vehicle_id = int(self.get_parameter("vehicle_id").value)
        self.use_v2x = bool(self.get_parameter("use_v2x_fixes").value)
        self.wheel_sigma = float(self.get_parameter("wheel_speed_sigma").value)

        track = load_track(self.get_parameter("track").value or None)
        route = track.build_route(str(self.get_parameter("route").value))
        start, heading = route.start_pose()
        initial = Pose2D(start.x, start.y, heading)

        noise = ProcessNoise(
            accel=float(self.get_parameter("accel_noise").value),
            gyro=float(self.get_parameter("gyro_noise").value),
            gyro_bias=float(self.get_parameter("gyro_bias_noise").value),
            slip=float(self.get_parameter("slip_noise").value),
        )

        now = self._now()
        self._ekf = DelayedFusionEKF(
            initial_pose=initial,
            initial_stamp=now,
            process_noise=noise,
            buffer_horizon_s=float(self.get_parameter("buffer_horizon_s").value),
        )
        self._reckoner = DeadReckoning(initial, now)

        self._truth: Optional[Pose2D] = None
        self._last_fix_age = 0.0
        self._yaw_rate = 0.0
        self._started = False

        ns = "/{}".format(self.vehicle)
        self.create_subscription(Imu, ns + "/imu", self._on_imu, SENSOR_QOS)
        self.create_subscription(
            Vector3Stamped, ns + "/wheel_speed", self._on_wheel_speed, SENSOR_QOS
        )
        self.create_subscription(
            bfmc_msgs.PoseFix, ns + "/v2x/pose_fix", self._on_pose_fix, V2X_QOS
        )
        # Ground truth is for scoring only; nothing downstream sees it.
        self.create_subscription(Odometry, ns + "/odom_truth", self._on_truth, SENSOR_QOS)

        self._pose_pub = self.create_publisher(
            PoseWithCovarianceStamped, ns + "/localization/pose", STATE_QOS
        )
        self._reckoned_pub = self.create_publisher(
            PoseWithCovarianceStamped, ns + "/localization/dead_reckoned", STATE_QOS
        )
        self._peer_pub = self.create_publisher(
            bfmc_msgs.PeerState, ns + "/localization/peer_state", STATE_QOS
        )
        self._diag_pub = self.create_publisher(
            bfmc_msgs.LocalizationDiagnostics, ns + "/localization/diagnostics", STATE_QOS
        )

        self._tf = TransformBroadcaster(self) if self.get_parameter("publish_tf").value else None
        self.create_timer(1.0 / float(self.get_parameter("publish_hz").value), self._publish)

        self.get_logger().info(
            "localisation up for {} at ({:.2f}, {:.2f}), V2X fixes {}".format(
                self.vehicle, initial.x, initial.y, "on" if self.use_v2x else "OFF"
            )
        )

    def _now(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    # ------------------------------------------------------------- sensors

    def _on_imu(self, message: Imu) -> None:
        stamp = stamp_to_seconds(message.header.stamp)
        rate = message.angular_velocity.z
        self._yaw_rate = rate
        self._ekf.predict(stamp, rate)
        # Raw gyro, bias and all: correcting it would flatter the baseline.
        self._reckoner.predict(stamp, rate)
        self._started = True

    def _on_wheel_speed(self, message: Vector3Stamped) -> None:
        stamp = stamp_to_seconds(message.header.stamp)
        speed = message.vector.x
        self._ekf.update_wheel_speed(stamp, speed, self.wheel_sigma)
        self._reckoner.update_wheel_speed(stamp, speed)

    def _on_pose_fix(self, message: bfmc_msgs.PoseFix) -> None:
        if not self.use_v2x:
            return
        payload, measured_at = msg_to_pose_fix(message)
        self._last_fix_age = max(0.0, self._now() - measured_at)

        # measured_at, never header.stamp: the delayed fusion rests on it.
        self._ekf.update_pose_fix(
            measured_at,
            payload.x,
            payload.y,
            payload.yaw,
            max(1e-3, payload.sigma_pos),
            max(1e-3, payload.sigma_yaw),
        )

    def _on_truth(self, message: Odometry) -> None:
        self._truth = Pose2D(
            message.pose.pose.position.x,
            message.pose.pose.position.y,
            yaw_from_quaternion(message.pose.pose.orientation),
        )

    # ------------------------------------------------------------ outbound

    def _publish(self) -> None:
        if not self._started:
            return

        estimate = self._ekf.estimate()
        stamp = self.get_clock().now().to_msg()

        self._pose_pub.publish(self._pose_message(estimate.pose, estimate.covariance, stamp))
        self._reckoned_pub.publish(self._pose_message(self._reckoner.pose, None, stamp))
        self._publish_peer_state(estimate, stamp)
        self._publish_diagnostics(estimate, stamp)

        if self._tf is not None:
            transform = TransformStamped()
            transform.header.stamp = stamp
            transform.header.frame_id = "map"
            transform.child_frame_id = "{}/base_link".format(self.vehicle)
            transform.transform.translation.x = estimate.pose.x
            transform.transform.translation.y = estimate.pose.y
            transform.transform.rotation = quaternion_from_yaw(estimate.pose.yaw)
            self._tf.sendTransform(transform)

    def _pose_message(self, pose: Pose2D, covariance, stamp) -> PoseWithCovarianceStamped:
        message = PoseWithCovarianceStamped()
        message.header.stamp = stamp
        message.header.frame_id = "map"
        message.pose.pose.position.x = float(pose.x)
        message.pose.pose.position.y = float(pose.y)
        message.pose.pose.orientation = quaternion_from_yaw(pose.yaw)

        if covariance is not None:
            # ROS wants a 6x6 in (x, y, z, roll, pitch, yaw); only the planar entries are
            # real.
            flat = [0.0] * 36
            flat[0] = float(covariance[IDX_X, IDX_X])
            flat[1] = float(covariance[IDX_X, IDX_Y])
            flat[5] = float(covariance[IDX_X, IDX_YAW])
            flat[6] = float(covariance[IDX_Y, IDX_X])
            flat[7] = float(covariance[IDX_Y, IDX_Y])
            flat[11] = float(covariance[IDX_Y, IDX_YAW])
            flat[30] = float(covariance[IDX_YAW, IDX_X])
            flat[31] = float(covariance[IDX_YAW, IDX_Y])
            flat[35] = float(covariance[IDX_YAW, IDX_YAW])
            message.pose.covariance = flat
        return message

    def _publish_peer_state(self, estimate, stamp) -> None:
        message = bfmc_msgs.PeerState()
        message.header.stamp = stamp
        message.header.frame_id = "map"
        message.vehicle_id = self.vehicle_id
        message.x = float(estimate.pose.x)
        message.y = float(estimate.pose.y)
        message.yaw = float(estimate.pose.yaw)
        message.speed = float(estimate.speed)
        message.yaw_rate = float(self._yaw_rate - estimate.gyro_bias)
        message.intent = int(Intent.STRAIGHT)
        message.age = 0.0
        self._peer_pub.publish(message)

    def _publish_diagnostics(self, estimate, stamp) -> None:
        message = bfmc_msgs.LocalizationDiagnostics()
        message.header.stamp = stamp
        message.header.frame_id = "map"

        if self._truth is not None:
            message.position_error = float(
                math.hypot(estimate.pose.x - self._truth.x, estimate.pose.y - self._truth.y)
            )
            message.heading_error = float(wrap_angle(estimate.pose.yaw - self._truth.yaw))
            message.dead_reckoning_error = float(
                math.hypot(
                    self._reckoner.pose.x - self._truth.x,
                    self._reckoner.pose.y - self._truth.y,
                )
            )
        else:
            message.position_error = -1.0
            message.heading_error = 0.0
            message.dead_reckoning_error = -1.0

        message.position_sigma = float(estimate.position_sigma)
        message.yaw_sigma = float(estimate.yaw_sigma)
        message.gyro_bias = float(estimate.gyro_bias)
        message.last_nis = float(self._ekf.last_nis)
        message.accepted_fixes = int(self._ekf.accepted_fixes)
        message.rejected_fixes = int(self._ekf.rejected_fixes)
        message.stale_fixes = int(self._ekf.stale_fixes)
        message.replayed_events = int(self._ekf.replayed_events)
        message.fix_age = float(self._last_fix_age)
        self._diag_pub.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalizationNode()
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
