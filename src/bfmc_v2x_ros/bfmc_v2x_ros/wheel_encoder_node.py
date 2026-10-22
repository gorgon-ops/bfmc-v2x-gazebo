"""Wheel encoder from Gazebo ground truth, with the error that matters: a fixed scale error
from the real tyre radius, which integrates straight into position. Perfect odometry
would leave the V2X fixes nothing to correct.
"""

from __future__ import annotations

import math
import random

import rclpy
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
from rclpy.node import Node

from .common import SENSOR_QOS


class WheelEncoderNode(Node):
    def __init__(self) -> None:
        super().__init__("wheel_encoder")

        self.declare_parameter("vehicle", "ego")
        self.declare_parameter("rate_hz", 50.0)
        #: 1.03: the car believes it went 3 % further than it did.
        self.declare_parameter("scale_error", 1.03)
        self.declare_parameter("noise_std", 0.010)
        #: Sets the quantisation floor.
        self.declare_parameter("counts_per_revolution", 40.0)
        self.declare_parameter("wheel_radius", 0.033)
        self.declare_parameter("seed", 424242)

        self.vehicle = str(self.get_parameter("vehicle").value)
        self.scale_error = float(self.get_parameter("scale_error").value)
        self.noise_std = float(self.get_parameter("noise_std").value)
        self.wheel_radius = float(self.get_parameter("wheel_radius").value)
        counts = float(self.get_parameter("counts_per_revolution").value)
        rate = float(self.get_parameter("rate_hz").value)

        # One count per sample interval is the smallest expressible speed change.
        circumference = 2.0 * math.pi * self.wheel_radius
        self._quantum = (circumference / counts) * rate

        self._rng = random.Random(int(self.get_parameter("seed").value))
        self._speed = 0.0
        self._have_truth = False

        self.create_subscription(
            Odometry, "/{}/odom_truth".format(self.vehicle), self._on_truth, SENSOR_QOS
        )
        self._publisher = self.create_publisher(
            Vector3Stamped, "/{}/wheel_speed".format(self.vehicle), SENSOR_QOS
        )
        self.create_timer(1.0 / rate, self._publish)

        self.get_logger().info(
            "wheel encoder for {}: scale error {:+.1f} %, quantum {:.3f} m/s".format(
                self.vehicle, (self.scale_error - 1.0) * 100.0, self._quantum
            )
        )

    def _on_truth(self, message: Odometry) -> None:
        # Body-frame twist, so forward is x; projected anyway in case the source is
        # reconfigured to the world frame.
        vx = message.twist.twist.linear.x
        vy = message.twist.twist.linear.y
        self._speed = math.hypot(vx, vy) * (1.0 if vx >= 0.0 else -1.0)
        self._have_truth = True

    def _publish(self) -> None:
        if not self._have_truth:
            return

        measured = self._speed * self.scale_error + self._rng.gauss(0.0, self.noise_std)
        if self._quantum > 0.0:
            measured = round(measured / self._quantum) * self._quantum

        message = Vector3Stamped()
        message.header.stamp = self.get_clock().now().to_msg()
        message.header.frame_id = "{}/base_link".format(self.vehicle)
        message.vector.x = float(measured)
        self._publisher.publish(message)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = WheelEncoderNode()
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
