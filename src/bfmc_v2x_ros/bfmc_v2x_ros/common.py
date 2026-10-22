from __future__ import annotations

import os
from typing import Iterable, Optional, Tuple

from ament_index_python.packages import get_package_share_directory
from builtin_interfaces.msg import Duration as DurationMsg
from geometry_msgs.msg import Point as PointMsg
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker

from bfmc_v2x.geometry import Point
from bfmc_v2x.protocol import SemaphoreState
from bfmc_v2x.track import Track

__all__ = [
    "V2X_QOS",
    "SENSOR_QOS",
    "STATE_QOS",
    "LATCHED_QOS",
    "default_track_path",
    "load_track",
    "marker",
    "line_strip",
    "SEMAPHORE_COLOURS",
]

#: Best effort, shallow: a reliable queue would deliver a stale pose fix rather
#: than drop it.
V2X_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

#: The filter tolerates gaps far better than latency.
SENSOR_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

#: Reliable: dropping a stop command is not an option.
STATE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    history=HistoryPolicy.KEEP_LAST,
    depth=10,
)

#: Latched so a late RViz still gets the track.
LATCHED_QOS = QoSProfile(
    reliability=ReliabilityPolicy.RELIABLE,
    durability=DurabilityPolicy.TRANSIENT_LOCAL,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

SEMAPHORE_COLOURS = {
    SemaphoreState.RED: (0.90, 0.15, 0.15),
    SemaphoreState.YELLOW: (0.95, 0.75, 0.10),
    SemaphoreState.GREEN: (0.20, 0.80, 0.30),
    SemaphoreState.FLASHING_YELLOW: (0.95, 0.75, 0.10),
}


def default_track_path() -> str:
    """Falls back to the source tree so nodes run straight from a clone."""
    try:
        share = get_package_share_directory("bfmc_sim")
        candidate = os.path.join(share, "track", "mini_city.yaml")
        if os.path.exists(candidate):
            return candidate
    except Exception:  # pragma: no cover - package not built yet
        pass

    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.normpath(
        os.path.join(here, "..", "..", "bfmc_sim", "track", "mini_city.yaml")
    )


def load_track(path: Optional[str] = None) -> Track:
    return Track.from_yaml(path or default_track_path())


def marker(
    marker_id: int,
    namespace: str,
    marker_type: int,
    frame_id: str = "map",
    scale: Tuple[float, float, float] = (0.05, 0.05, 0.05),
    colour: Tuple[float, float, float] = (1.0, 1.0, 1.0),
    alpha: float = 1.0,
    lifetime_s: float = 0.0,
) -> Marker:
    item = Marker()
    item.header.frame_id = frame_id
    item.ns = namespace
    item.id = marker_id
    item.type = marker_type
    item.action = Marker.ADD
    item.scale.x, item.scale.y, item.scale.z = scale
    item.color = ColorRGBA(r=colour[0], g=colour[1], b=colour[2], a=alpha)
    item.pose.orientation.w = 1.0
    if lifetime_s > 0.0:
        item.lifetime = DurationMsg(
            sec=int(lifetime_s), nanosec=int((lifetime_s % 1.0) * 1e9)
        )
    return item


def line_strip(
    points: Iterable[Point],
    marker_id: int,
    namespace: str,
    colour: Tuple[float, float, float],
    width: float = 0.02,
    z: float = 0.01,
    alpha: float = 1.0,
) -> Marker:
    item = marker(
        marker_id,
        namespace,
        Marker.LINE_STRIP,
        scale=(width, 0.0, 0.0),
        colour=colour,
        alpha=alpha,
    )
    item.points = [PointMsg(x=float(p.x), y=float(p.y), z=float(z)) for p in points]
    return item
