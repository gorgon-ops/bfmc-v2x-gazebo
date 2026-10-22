"""All trackside lights in one node. Each colour is a pure function of the clock and its
offset, so one process or twelve gives identical output; safety comes from the offsets.
"""

from __future__ import annotations

from typing import Dict, Tuple

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray

from bfmc_v2x.protocol import SemaphoreState
from bfmc_v2x.semaphore import Phase, SemaphoreCycle
from bfmc_v2x.transport import DEFAULT_GROUP, DEFAULT_PORT, V2XEndpoint

from .common import SEMAPHORE_COLOURS, STATE_QOS, load_track, marker

#: Well clear of vehicle ids.
SEMAPHORE_SENDER_BASE = 1100


class SemaphoreNode(Node):
    def __init__(self) -> None:
        super().__init__("semaphores")

        self.declare_parameter("rate_hz", 5.0)
        self.declare_parameter("green_s", 7.0)
        self.declare_parameter("yellow_s", 2.0)
        self.declare_parameter("red_s", 9.0)
        self.declare_parameter("port", DEFAULT_PORT)
        self.declare_parameter("group", DEFAULT_GROUP)
        self.declare_parameter("track", "")
        #: Simulate a light off the network, to exercise the stale-signal fallback.
        self.declare_parameter("silent_lights", [])

        self.track = load_track(self.get_parameter("track").value or None)
        self._silent = set(int(v) for v in self.get_parameter("silent_lights").value)

        phases = (
            Phase(SemaphoreState.GREEN, float(self.get_parameter("green_s").value)),
            Phase(SemaphoreState.YELLOW, float(self.get_parameter("yellow_s").value)),
            Phase(SemaphoreState.RED, float(self.get_parameter("red_s").value)),
        )
        self._check_phase_plan(phases)

        self._cycles: Dict[int, SemaphoreCycle] = {}
        self._poses: Dict[int, Tuple[float, float, float]] = {}
        for placement in self.track.light_placements():
            self._cycles[placement.light_id] = SemaphoreCycle(
                light_id=placement.light_id, phases=phases, offset_s=placement.offset_s
            )
            position, facing = self.track.light_pose(placement)
            self._poses[placement.light_id] = (position.x, position.y, facing)

        # One sender id per light, as on the real track.
        self._links: Dict[int, V2XEndpoint] = {
            light_id: V2XEndpoint(
                sender_id=SEMAPHORE_SENDER_BASE + light_id,
                port=int(self.get_parameter("port").value),
                group=str(self.get_parameter("group").value),
                receive=False,
            )
            for light_id in self._cycles
        }

        self._markers = self.create_publisher(MarkerArray, "/v2x/semaphore_markers", STATE_QOS)
        self.create_timer(1.0 / float(self.get_parameter("rate_hz").value), self._tick)

        silent = " (silent: {})".format(sorted(self._silent)) if self._silent else ""
        self.get_logger().info(
            "{} lights up, {:.0f} s cycle{}".format(
                len(self._cycles), sum(p.duration_s for p in phases), silent
            )
        )

    def _check_phase_plan(self, phases) -> None:
        """Refuse offsets and durations that would show green to two crossing approaches."""
        offsets = sorted({p.offset_s for p in self.track.light_placements()})
        if len(offsets) < 2:
            return

        period = sum(p.duration_s for p in phases)
        permissive = {SemaphoreState.GREEN, SemaphoreState.YELLOW}
        cycles = [SemaphoreCycle(0, phases, offset_s=o) for o in offsets]

        for step in range(int(period * 20)):
            t = step / 20.0
            live = [c for c in cycles if c.state_at(t)[0] in permissive]
            if len(live) > 1:
                raise ValueError(
                    "unsafe phase plan: offsets {} are both permissive at t={:.2f} s. "
                    "Red must last at least green + yellow.".format(offsets, t)
                )

    def _tick(self) -> None:
        now = self.get_clock().now().nanoseconds * 1e-9
        markers = MarkerArray()

        for light_id, cycle in self._cycles.items():
            state, remaining = cycle.state_at(now)

            if light_id not in self._silent:
                # ROS clock, not wall clock: under use_sim_time they are unrelated, and every
                # message would look ancient.
                self._links[light_id].send(
                    cycle.broadcast_at(now), tx_time_us=int(now * 1e6)
                )

            markers.markers.append(self._marker_for(light_id, state, remaining))
            markers.markers.append(self._label_for(light_id, state, remaining))

        self._markers.publish(markers)

    def _marker_for(self, light_id: int, state: SemaphoreState, remaining: float) -> Marker:
        x, y, _ = self._poses[light_id]
        colour = SEMAPHORE_COLOURS[state]
        # Hollow, so "deliberately off the air" is not mistaken for broken.
        alpha = 0.35 if light_id in self._silent else 1.0

        item = marker(
            light_id,
            "semaphores",
            Marker.SPHERE,
            scale=(0.09, 0.09, 0.09),
            colour=colour,
            alpha=alpha,
        )
        item.header.stamp = self.get_clock().now().to_msg()
        item.pose.position.x = x
        item.pose.position.y = y
        item.pose.position.z = 0.22
        return item

    def _label_for(self, light_id: int, state: SemaphoreState, remaining: float) -> Marker:
        x, y, _ = self._poses[light_id]
        item = marker(
            1000 + light_id,
            "semaphore_labels",
            Marker.TEXT_VIEW_FACING,
            scale=(0.0, 0.0, 0.07),
            colour=(1.0, 1.0, 1.0),
        )
        item.header.stamp = self.get_clock().now().to_msg()
        item.pose.position.x = x
        item.pose.position.y = y
        item.pose.position.z = 0.34
        if light_id in self._silent:
            item.text = "{}: OFF AIR".format(light_id)
        else:
            item.text = "{}: {} {:.1f}s".format(light_id, state.name, remaining)
        return item

    def destroy_node(self) -> bool:
        for link in self._links.values():
            link.close()
        return super().destroy_node()


def main(args=None) -> None:
    rclpy.init(args=args)
    node = SemaphoreNode()
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
