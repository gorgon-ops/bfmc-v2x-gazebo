"""Cooperative awareness. The central junction is blind on all four corners.

Two checks, because they fail differently: geometric closest approach catches a
peer somewhere it should not be; occupancy windows at the conflict point are far
less twitchy than time-to-collision for cars passing in sequence. Peer states are
always aged forward, and dropped once too old to extrapolate.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .geometry import Point, Pose2D, distance, wrap_angle
from .protocol import Intent, PeerState

__all__ = [
    "PeerTrack",
    "PeerRegistry",
    "Trajectory",
    "predict_trajectory",
    "closest_approach",
    "OccupancyWindow",
    "JunctionAssessment",
    "RightOfWay",
    "assess_junction",
]


@dataclass(frozen=True)
class PeerTrack:
    vehicle_id: int
    pose: Pose2D
    speed: float
    yaw_rate: float
    intent: Intent
    #: Seconds since the peer stamped the message.
    age_s: float

    def extrapolated(self, dt: float) -> "PeerTrack":
        """Constant turn rate: fine for a few hundred ms; past ~1 s peers start driving in circles.
        """
        yaw = wrap_angle(self.pose.yaw + self.yaw_rate * dt)
        # Mid-interval heading, so a turning peer is not placed outside its own arc.
        mid = self.pose.yaw + 0.5 * self.yaw_rate * dt
        return PeerTrack(
            vehicle_id=self.vehicle_id,
            pose=Pose2D(
                self.pose.x + self.speed * math.cos(mid) * dt,
                self.pose.y + self.speed * math.sin(mid) * dt,
                yaw,
            ),
            speed=self.speed,
            yaw_rate=self.yaw_rate,
            intent=self.intent,
            age_s=self.age_s + dt,
        )


class PeerRegistry:
    def __init__(self, expire_after_s: float = 1.0) -> None:
        self.expire_after_s = expire_after_s
        self._tracks: Dict[int, Tuple[PeerTrack, float]] = {}
        #: Peers dropped for going quiet.
        self.expired: int = 0

    def ingest(self, state: PeerState, rx_time: float, link_age_s: float = 0.0) -> None:
        track = PeerTrack(
            vehicle_id=state.vehicle_id,
            pose=Pose2D(state.x, state.y, state.yaw),
            speed=state.speed,
            yaw_rate=state.yaw_rate,
            intent=state.manoeuvre,
            age_s=max(0.0, link_age_s),
        )
        self._tracks[state.vehicle_id] = (track, rx_time)

    def active(self, now: float) -> List[PeerTrack]:
        out: List[PeerTrack] = []
        stale: List[int] = []
        for vehicle_id, (track, rx_time) in self._tracks.items():
            elapsed = now - rx_time
            if elapsed > self.expire_after_s:
                stale.append(vehicle_id)
                continue
            out.append(track.extrapolated(max(0.0, elapsed)))
        for vehicle_id in stale:
            del self._tracks[vehicle_id]
            self.expired += 1
        return sorted(out, key=lambda t: t.vehicle_id)

    def get(self, vehicle_id: int, now: float) -> Optional[PeerTrack]:
        for track in self.active(now):
            if track.vehicle_id == vehicle_id:
                return track
        return None

    def __len__(self) -> int:
        return len(self._tracks)


#: ``(time_offset, position)`` samples.
Trajectory = Sequence[Tuple[float, Point]]


def predict_trajectory(
    pose: Pose2D,
    speed: float,
    yaw_rate: float,
    horizon_s: float = 3.0,
    step_s: float = 0.1,
) -> List[Tuple[float, Point]]:
    """Constant-turn-rate prediction over ``horizon_s``."""
    samples: List[Tuple[float, Point]] = []
    steps = max(1, int(round(horizon_s / step_s)))
    x, y, yaw = pose.x, pose.y, pose.yaw
    for index in range(steps + 1):
        samples.append((index * step_s, Point(x, y)))
        mid = yaw + 0.5 * yaw_rate * step_s
        x += speed * math.cos(mid) * step_s
        y += speed * math.sin(mid) * step_s
        yaw = wrap_angle(yaw + yaw_rate * step_s)
    return samples


def closest_approach(ego: Trajectory, peer: Trajectory) -> Tuple[float, float]:
    """Both trajectories must share a time base and sample rate."""
    best_time = 0.0
    best_gap = math.inf
    for (t_ego, p_ego), (t_peer, p_peer) in zip(ego, peer):
        gap = distance(p_ego, p_peer)
        if gap < best_gap:
            best_gap = gap
            best_time = 0.5 * (t_ego + t_peer)
    return best_time, best_gap


@dataclass(frozen=True)
class OccupancyWindow:
    enter_s: float
    exit_s: float
    #: ``False`` when too slow to ever arrive, or leaving.
    will_enter: bool

    def overlaps(self, other: "OccupancyWindow", margin_s: float) -> bool:
        if not (self.will_enter and other.will_enter):
            return False
        return (
            self.enter_s - margin_s < other.exit_s and other.enter_s - margin_s < self.exit_s
        )


class RightOfWay(Enum):
    PROCEED = "PROCEED"
    #: Conflict predicted. Give up the junction and stop at the line.
    YIELD = "YIELD"
    #: Too close to resolve by yielding at the line.
    EMERGENCY_STOP = "EMERGENCY_STOP"


@dataclass(frozen=True)
class JunctionAssessment:
    verdict: RightOfWay
    ego_window: OccupancyWindow
    peer_window: Optional[OccupancyWindow]
    #: Metres.
    min_separation_m: float
    #: Seconds from now.
    time_to_closest_s: float
    peer_id: Optional[int]
    reason: str


def _occupancy(
    position: Point,
    heading: float,
    speed: float,
    conflict_point: Point,
    zone_radius_m: float,
    vehicle_length_m: float,
) -> OccupancyWindow:
    """Along the heading, so a car past the junction gets a negative range."""
    to_point = Point(conflict_point.x - position.x, conflict_point.y - position.y)
    along = to_point.x * math.cos(heading) + to_point.y * math.sin(heading)
    reach = along - zone_radius_m
    clear = along + zone_radius_m + vehicle_length_m

    if speed < 1e-3:
        # Stationary: occupies the zone only if already in it, indefinitely.
        if reach <= 0.0 <= clear:
            return OccupancyWindow(0.0, math.inf, True)
        return OccupancyWindow(math.inf, math.inf, False)

    if clear <= 0.0:
        return OccupancyWindow(math.inf, math.inf, False)

    return OccupancyWindow(max(0.0, reach / speed), clear / speed, True)


def assess_junction(
    ego_pose: Pose2D,
    ego_speed: float,
    ego_yaw_rate: float,
    peers: Iterable[PeerTrack],
    conflict_point: Point,
    distance_to_line: float,
    zone_radius_m: float = 0.55,
    vehicle_length_m: float = 0.36,
    headway_margin_s: float = 1.2,
    min_separation_m: float = 0.45,
    ego_id: int = 0,
) -> JunctionAssessment:
    """Ties go to the lower vehicle id: arbitrary but symmetric, so exactly one car yields.
    "Yield to the right" would deadlock a four-way.
    """
    ego_window = _occupancy(
        ego_pose.point, ego_pose.yaw, ego_speed, conflict_point, zone_radius_m, vehicle_length_m
    )
    ego_path = predict_trajectory(ego_pose, ego_speed, ego_yaw_rate)

    worst: Optional[JunctionAssessment] = None
    for peer in peers:
        if peer.intent is Intent.HOLD:
            # A peer announcing HOLD will not take the junction.
            continue

        peer_window = _occupancy(
            peer.pose.point,
            peer.pose.yaw,
            peer.speed,
            conflict_point,
            zone_radius_m,
            vehicle_length_m,
        )
        peer_path = predict_trajectory(peer.pose, peer.speed, peer.yaw_rate)
        t_closest, separation = closest_approach(ego_path, peer_path)

        conflicting = ego_window.overlaps(peer_window, headway_margin_s)
        too_close = separation < min_separation_m

        if not conflicting and not too_close:
            continue

        # First to reach the zone keeps it; ties to the lower id.
        if ego_window.enter_s < peer_window.enter_s - 1e-6:
            ours = True
        elif peer_window.enter_s < ego_window.enter_s - 1e-6:
            ours = False
        else:
            ours = ego_id < peer.vehicle_id

        if ours and not too_close:
            continue

        braking_room = distance_to_line - ego_speed * 0.3
        if too_close and separation < min_separation_m * 0.6:
            verdict = RightOfWay.EMERGENCY_STOP
            reason = "peer {} within {:.2f} m in {:.1f} s".format(
                peer.vehicle_id, separation, t_closest
            )
        elif braking_room <= 0.0 and ego_window.enter_s < headway_margin_s:
            verdict = RightOfWay.EMERGENCY_STOP
            reason = "peer {} conflicts and the stop line is behind us".format(peer.vehicle_id)
        else:
            verdict = RightOfWay.YIELD
            reason = "peer {} enters in {:.1f} s, we would enter in {:.1f} s".format(
                peer.vehicle_id, peer_window.enter_s, ego_window.enter_s
            )

        candidate = JunctionAssessment(
            verdict=verdict,
            ego_window=ego_window,
            peer_window=peer_window,
            min_separation_m=separation,
            time_to_closest_s=t_closest,
            peer_id=peer.vehicle_id,
            reason=reason,
        )
        if worst is None or _severity(candidate.verdict) > _severity(worst.verdict):
            worst = candidate

    if worst is not None:
        return worst

    return JunctionAssessment(
        verdict=RightOfWay.PROCEED,
        ego_window=ego_window,
        peer_window=None,
        min_separation_m=math.inf,
        time_to_closest_s=math.inf,
        peer_id=None,
        reason="no conflicting peer",
    )


def _severity(verdict: RightOfWay) -> int:
    return {RightOfWay.PROCEED: 0, RightOfWay.YIELD: 1, RightOfWay.EMERGENCY_STOP: 2}[verdict]
