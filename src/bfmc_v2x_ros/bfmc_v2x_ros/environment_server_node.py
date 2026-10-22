"""The organisers' environment server: sees only what vehicles broadcast, grades it against
ground truth, and prints the scoreboard at shutdown.
"""

from __future__ import annotations

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from bfmc_v2x.environment import EnvironmentServer
from bfmc_v2x.protocol import ObstacleReport
from bfmc_v2x.transport import DEFAULT_GROUP, DEFAULT_PORT, V2XEndpoint

from .common import LATCHED_QOS, STATE_QOS, load_track, marker

ENVIRONMENT_SENDER_ID = 1200

#: By outcome.
_ACCEPTED = (0.20, 0.80, 0.30)
_REJECTED = (0.90, 0.25, 0.25)
_TRUTH = (0.55, 0.55, 0.60)


class EnvironmentServerNode(Node):
    def __init__(self) -> None:
        super().__init__("environment_server")

        self.declare_parameter("track", "")
        self.declare_parameter("tolerance_m", 0.35)
        self.declare_parameter("min_confidence", 0.4)
        self.declare_parameter("poll_hz", 50.0)
        self.declare_parameter("report_every_s", 15.0)
        self.declare_parameter("port", DEFAULT_PORT)
        self.declare_parameter("group", DEFAULT_GROUP)

        self.track = load_track(self.get_parameter("track").value or None)
        self._server = EnvironmentServer(
            obstacles=self.track.obstacles,
            tolerance_m=float(self.get_parameter("tolerance_m").value),
            min_confidence=float(self.get_parameter("min_confidence").value),
        )

        self._link = V2XEndpoint(
            sender_id=ENVIRONMENT_SENDER_ID,
            port=int(self.get_parameter("port").value),
            group=str(self.get_parameter("group").value),
        )

        self._markers = self.create_publisher(
            MarkerArray, "/environment/markers", STATE_QOS
        )
        self._truth_markers = self.create_publisher(
            MarkerArray, "/environment/ground_truth", LATCHED_QOS
        )

        self.create_timer(1.0 / float(self.get_parameter("poll_hz").value), self._poll)
        self.create_timer(float(self.get_parameter("report_every_s").value), self._log_scoreboard)
        self.create_timer(2.0, self._publish_truth)
        self._publish_truth()

        self._marker_seq = 0
        self.get_logger().info(
            "environment server up: {} obstacles, {:.2f} m tolerance".format(
                len(self.track.obstacles), self._server.tolerance_m
            )
        )

    def _poll(self) -> None:
        for received in self._link.poll():
            if not isinstance(received.payload, ObstacleReport):
                continue

            ack, scored = self._server.submit(
                received.payload,
                vehicle_id=received.header.sender_id,
                seq=received.header.seq,
            )
            self._link.send(
                ack, tx_time_us=int(self.get_clock().now().nanoseconds * 1e-3)
            )
            self._publish_report_marker(scored)

            if scored.accepted:
                self.get_logger().info(
                    "vehicle {}: {}".format(scored.vehicle_id, scored.note)
                )

    def _publish_report_marker(self, scored) -> None:
        self._marker_seq = (self._marker_seq + 1) % 500
        item = marker(
            self._marker_seq,
            "reports",
            Marker.SPHERE,
            scale=(0.06, 0.06, 0.06),
            colour=_ACCEPTED if scored.accepted else _REJECTED,
            alpha=0.9 if scored.accepted else 0.35,
            # Rejected guesses fade; accepted ones stay as the set actually found.
            lifetime_s=0.0 if scored.accepted else 3.0,
        )
        item.header.stamp = self.get_clock().now().to_msg()
        item.pose.position.x = float(scored.reported.x)
        item.pose.position.y = float(scored.reported.y)
        item.pose.position.z = 0.06

        array = MarkerArray()
        array.markers.append(item)
        self._markers.publish(array)

    def _publish_truth(self) -> None:
        array = MarkerArray()
        for obstacle in self.track.obstacles:
            item = marker(
                obstacle.obstacle_id,
                "ground_truth",
                Marker.CYLINDER,
                scale=(2 * self._server.tolerance_m, 2 * self._server.tolerance_m, 0.002),
                colour=_TRUTH,
                alpha=0.18,
            )
            item.pose.position.x = obstacle.x
            item.pose.position.y = obstacle.y
            item.pose.position.z = 0.002
            array.markers.append(item)

            label = marker(
                100 + obstacle.obstacle_id,
                "ground_truth_labels",
                Marker.TEXT_VIEW_FACING,
                scale=(0.0, 0.0, 0.06),
                colour=(0.85, 0.85, 0.9),
            )
            label.pose.position.x = obstacle.x
            label.pose.position.y = obstacle.y
            label.pose.position.z = 0.18
            label.text = obstacle.kind.name
            array.markers.append(label)
        self._truth_markers.publish(array)

    def _log_scoreboard(self) -> None:
        self.get_logger().info(self._server.summary())

    def destroy_node(self) -> bool:
        self.get_logger().info("final score: {}".format(self._server.summary()))
        for obstacle in self.track.obstacles:
            error = self._server.scoreboard.best_error_m.get(obstacle.obstacle_id)
            self.get_logger().info(
                "  {:<12} {}".format(
                    obstacle.kind.name,
                    "located to {:.3f} m".format(error) if error is not None else "MISSED",
                )
            )
        self._link.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = EnvironmentServerNode()
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
