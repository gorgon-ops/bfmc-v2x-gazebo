"""The overhead-camera localisation system, simulated from ground truth with its defects:
slow (one fix per 250 ms), late (stamped at exposure), and less accurate towards
the frame edge. It only broadcasts on V2X, never on a ROS topic.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

import rclpy
from nav_msgs.msg import Odometry
from rclpy.node import Node

from bfmc_v2x.protocol import PoseFix
from bfmc_v2x.transport import DEFAULT_GROUP, DEFAULT_PORT, V2XEndpoint

from .common import SENSOR_QOS, load_track
from .conversions import yaw_from_quaternion

#: So vehicles can tell the trackside system from a car claiming to see them.
LOCALISATION_SENDER_ID = 1000


@dataclass
class _Pending:
    release_at: float
    measured_at: float
    payload: PoseFix


class LocalizationServerNode(Node):
    def __init__(self) -> None:
        super().__init__("localization_server")

        self.declare_parameter("vehicles", ["ego", "peer"])
        self.declare_parameter("vehicle_ids", [1, 2])
        self.declare_parameter("rate_hz", 4.0)
        self.declare_parameter("latency_s", 0.25)
        self.declare_parameter("latency_jitter_s", 0.04)
        self.declare_parameter("sigma_pos_centre", 0.025)
        self.declare_parameter("sigma_pos_edge", 0.090)
        self.declare_parameter("sigma_yaw", 0.030)
        self.declare_parameter("dropout_probability", 0.08)
        self.declare_parameter("seed", 20240601)
        self.declare_parameter("port", DEFAULT_PORT)
        self.declare_parameter("group", DEFAULT_GROUP)
        self.declare_parameter("track", "")

        names = list(self.get_parameter("vehicles").value)
        ids = list(self.get_parameter("vehicle_ids").value)
        if len(names) != len(ids):
            raise ValueError("vehicles and vehicle_ids must be the same length")
        self._ids: Dict[str, int] = dict(zip(names, ids))

        self.rate_hz = float(self.get_parameter("rate_hz").value)
        self.latency_s = float(self.get_parameter("latency_s").value)
        self.jitter_s = float(self.get_parameter("latency_jitter_s").value)
        self.sigma_centre = float(self.get_parameter("sigma_pos_centre").value)
        self.sigma_edge = float(self.get_parameter("sigma_pos_edge").value)
        self.sigma_yaw = float(self.get_parameter("sigma_yaw").value)
        self.dropout = float(self.get_parameter("dropout_probability").value)

        track = load_track(self.get_parameter("track").value or None)
        min_x, min_y, max_x, max_y = track.extent
        # The camera looks straight down at the middle of the track.
        self._camera = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        self._max_range = max(max_x - self._camera[0], max_y - self._camera[1])

        self._rng = random.Random(int(self.get_parameter("seed").value))
        self._truth: Dict[str, Tuple[float, float, float]] = {}
        self._pending: List[_Pending] = []

        self._link = V2XEndpoint(
            sender_id=LOCALISATION_SENDER_ID,
            port=int(self.get_parameter("port").value),
            group=str(self.get_parameter("group").value),
            receive=False,
        )

        for name in names:
            self.create_subscription(
                Odometry,
                "/{}/odom_truth".format(name),
                self._make_truth_callback(name),
                SENSOR_QOS,
            )

        self.create_timer(1.0 / self.rate_hz, self._expose)
        # Faster than the exposure rate, so latency is set by the configured delay.
        self.create_timer(0.01, self._transmit)

        self.get_logger().info(
            "trackside localisation up: {} vehicles at {:.1f} Hz, "
            "{:.0f} ms latency, sigma {:.0f}-{:.0f} mm, {:.0f}% dropout".format(
                len(names), self.rate_hz, self.latency_s * 1000.0,
                self.sigma_centre * 1000.0, self.sigma_edge * 1000.0, self.dropout * 100.0,
            )
        )

    def _make_truth_callback(self, name: str):
        def callback(message: Odometry) -> None:
            self._truth[name] = (
                message.pose.pose.position.x,
                message.pose.pose.position.y,
                yaw_from_quaternion(message.pose.pose.orientation),
            )

        return callback

    def _sigma_at(self, x: float, y: float) -> float:
        """Quoted accuracy, interpolated by distance from the camera axis."""
        if self._max_range <= 0.0:
            return self.sigma_centre
        offset = ((x - self._camera[0]) ** 2 + (y - self._camera[1]) ** 2) ** 0.5
        t = min(1.0, offset / self._max_range)
        return self.sigma_centre + (self.sigma_edge - self.sigma_centre) * t

    def _expose(self) -> None:
        measured_at = self.get_clock().now().nanoseconds * 1e-9

        for name, vehicle_id in self._ids.items():
            truth = self._truth.get(name)
            if truth is None:
                continue
            if self._rng.random() < self.dropout:
                # Missed this frame: occlusion, blur, or the tracker ran late.
                continue

            x, y, yaw = truth
            sigma = self._sigma_at(x, y)
            payload = PoseFix(
                vehicle_id=vehicle_id,
                x=x + self._rng.gauss(0.0, sigma),
                y=y + self._rng.gauss(0.0, sigma),
                yaw=yaw + self._rng.gauss(0.0, self.sigma_yaw),
                # Honest sigma: overstating accuracy makes a well-tuned filter overconfident.
                sigma_pos=sigma,
                sigma_yaw=self.sigma_yaw,
            )
            delay = max(0.0, self.latency_s + self._rng.gauss(0.0, self.jitter_s))
            self._pending.append(
                _Pending(
                    release_at=measured_at + delay,
                    measured_at=measured_at,
                    payload=payload,
                )
            )

    def _transmit(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        due = [p for p in self._pending if p.release_at <= now]
        if not due:
            return
        self._pending = [p for p in self._pending if p.release_at > now]

        for item in due:
            # The exposure time, not now: what makes delayed fusion possible.
            self._link.send(item.payload, tx_time_us=int(item.measured_at * 1e6))

    def destroy_node(self) -> bool:
        self._link.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = LocalizationServerNode()
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
