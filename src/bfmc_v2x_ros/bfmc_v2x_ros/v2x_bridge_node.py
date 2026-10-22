"""The vehicle's radio: the only node that opens a socket, so none can read ground truth and
call it perception. Datagrams are stamped from the ROS clock, which under
``use_sim_time`` differs from wall time.
"""

from __future__ import annotations

from typing import Optional

import rclpy
from rclpy.node import Node

import bfmc_v2x_msgs.msg as bfmc_msgs
from bfmc_v2x.protocol import (
    MessageType,
    ObstacleAck,
    ObstacleReport,
    PeerState,
    PoseFix,
    SemaphoreBroadcast,
)
from bfmc_v2x.transport import DEFAULT_GROUP, DEFAULT_PORT, V2XEndpoint

from .common import STATE_QOS, V2X_QOS
from .conversions import (
    msg_to_obstacle_report,
    peer_state_to_msg,
    pose_fix_to_msg,
    semaphore_to_msg,
)


class V2XBridgeNode(Node):
    def __init__(self) -> None:
        super().__init__("v2x_bridge")

        self.declare_parameter("vehicle", "ego")
        self.declare_parameter("vehicle_id", 1)
        self.declare_parameter("poll_hz", 100.0)
        self.declare_parameter("broadcast_hz", 10.0)
        self.declare_parameter("port", DEFAULT_PORT)
        self.declare_parameter("group", DEFAULT_GROUP)

        self.vehicle = str(self.get_parameter("vehicle").value)
        self.vehicle_id = int(self.get_parameter("vehicle_id").value)

        self._link = V2XEndpoint(
            sender_id=self.vehicle_id,
            port=int(self.get_parameter("port").value),
            group=str(self.get_parameter("group").value),
        )

        ns = "/{}".format(self.vehicle)
        self._fix_pub = self.create_publisher(bfmc_msgs.PoseFix, ns + "/v2x/pose_fix", V2X_QOS)
        self._light_pub = self.create_publisher(
            bfmc_msgs.SemaphoreState, ns + "/v2x/semaphore", V2X_QOS
        )
        self._peer_pub = self.create_publisher(bfmc_msgs.PeerState, ns + "/v2x/peer", V2X_QOS)
        self._ack_pub = self.create_publisher(
            bfmc_msgs.ObstacleAck, ns + "/v2x/obstacle_ack", STATE_QOS
        )

        self._own_state: Optional[bfmc_msgs.PeerState] = None
        self.create_subscription(
            bfmc_msgs.PeerState, ns + "/localization/peer_state", self._on_own_state, STATE_QOS
        )
        self.create_subscription(
            bfmc_msgs.ObstacleReport, ns + "/obstacle_report", self._on_own_report, STATE_QOS
        )

        self.create_timer(1.0 / float(self.get_parameter("poll_hz").value), self._poll)
        self.create_timer(1.0 / float(self.get_parameter("broadcast_hz").value), self._broadcast)

        self._rx = {kind: 0 for kind in MessageType}
        self._not_for_us = 0

        self.get_logger().info(
            "radio up for {} (id {}) on port {}".format(
                self.vehicle, self.vehicle_id, self.get_parameter("port").value
            )
        )

    # ------------------------------------------------------------- inbound

    def _now_seconds(self) -> float:
        return self.get_clock().now().nanoseconds * 1e-9

    def _poll(self) -> None:
        now = self._now_seconds()
        now_us = int(now * 1e6)

        for received in self._link.poll():
            header, payload = received.header, received.payload
            self._rx[header.msg_type] += 1
            # A sender clock slightly ahead would give a negative age and a growing
            # countdown.
            age = max(0.0, header.age_s(now_us))

            if isinstance(payload, PoseFix):
                if payload.vehicle_id != self.vehicle_id:
                    self._not_for_us += 1
                    continue
                message = pose_fix_to_msg(payload, measured_at=header.tx_time_us * 1e-6)
                message.header.stamp = self.get_clock().now().to_msg()
                self._fix_pub.publish(message)

            elif isinstance(payload, SemaphoreBroadcast):
                # Age the countdown by the flight time.
                message = semaphore_to_msg(payload, ttl_s=max(0.0, payload.ttl_s - age))
                message.header.stamp = self.get_clock().now().to_msg()
                self._light_pub.publish(message)

            elif isinstance(payload, PeerState):
                if payload.vehicle_id == self.vehicle_id:
                    continue
                message = peer_state_to_msg(payload, age_s=age)
                message.header.stamp = self.get_clock().now().to_msg()
                self._peer_pub.publish(message)

            elif isinstance(payload, ObstacleAck):
                message = bfmc_msgs.ObstacleAck()
                message.header.stamp = self.get_clock().now().to_msg()
                message.vehicle_id = self.vehicle_id
                message.report_seq = payload.report_seq
                message.accepted = payload.was_accepted
                message.error = payload.error_m
                self._ack_pub.publish(message)

    # ------------------------------------------------------------ outbound

    def _on_own_state(self, message: bfmc_msgs.PeerState) -> None:
        self._own_state = message

    def _broadcast(self) -> None:
        if self._own_state is None:
            return
        state = self._own_state
        self._link.send(
            PeerState(
                vehicle_id=self.vehicle_id,
                x=state.x,
                y=state.y,
                yaw=state.yaw,
                speed=state.speed,
                yaw_rate=state.yaw_rate,
                intent=int(state.intent),
            ),
            tx_time_us=int(self._now_seconds() * 1e6),
        )

    def _on_own_report(self, message: bfmc_msgs.ObstacleReport) -> None:
        payload, _, _ = msg_to_obstacle_report(message)
        self._link.send(payload, tx_time_us=int(self._now_seconds() * 1e6))

    def destroy_node(self) -> bool:
        counts = ", ".join(
            "{}={}".format(kind.name.lower(), count)
            for kind, count in self._rx.items()
            if count
        )
        self.get_logger().info(
            "radio down for {}: received {} (dropped {} addressed elsewhere, "
            "{} malformed)".format(
                self.vehicle, counts or "nothing", self._not_for_us, self._link.malformed
            )
        )
        self._link.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = V2XBridgeNode()
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
