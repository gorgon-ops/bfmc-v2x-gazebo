"""Behaviour arbitration: the slowest speed request wins, and nothing may raise one.

Steering is never arbitrated: the car steers along its route even while stopping.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

from .geometry import Pose2D, pure_pursuit_steer
from .semaphore import Approach, ApproachDecision, ObservedLight, StopLinePolicy
from .track import Route, StopLine
from .v2v import JunctionAssessment, PeerTrack, RightOfWay, assess_junction

__all__ = ["DriveCommand", "DriverConfig", "Driver"]


@dataclass(frozen=True)
class DriveCommand:
    speed: float
    steer: float
    #: Which constraint is binding.
    limiting_factor: str
    reason: str
    arc_s: float
    lateral_error_m: float
    stop_line: Optional[StopLine] = None
    distance_to_line_m: float = math.inf
    approach: Optional[Approach] = None
    conflict: Optional[JunctionAssessment] = None


@dataclass(frozen=True)
class DriverConfig:
    wheelbase: float = 0.26
    vehicle_length: float = 0.36
    cruise_speed: float = 0.35
    max_steer_rad: float = math.radians(25.0)
    #: Grows with speed so the controller stays stable at pace.
    lookahead_min_m: float = 0.32
    lookahead_gain_s: float = 0.9
    curvature_speed_gain: float = 0.55
    vehicle_id: int = 1


class Driver:
    """Holds no ROS types and no clock: :meth:`step` is a pure function of its inputs."""

    def __init__(
        self,
        route: Route,
        config: DriverConfig = DriverConfig(),
        policy: Optional[StopLinePolicy] = None,
    ) -> None:
        self.route = route
        self.config = config
        self.policy = policy or StopLinePolicy(cruise_speed=config.cruise_speed)

        # The only state, and it earns its place: a car commanded to zero creeps, and
        # a couple of centimetres past the line a stateless policy sees itself in the
        # junction, switches to CLEAR and drives through the red. Once stopped at a
        # line, stay stopped until the line releases you.
        self._held_line_s: Optional[float] = None

    #: Below this speed, and this close, the vehicle counts as stopped.
    HOLD_SPEED = 0.06
    HOLD_DISTANCE = 0.30

    def step(
        self,
        pose: Pose2D,
        speed: float,
        yaw_rate: float,
        light: Optional[ObservedLight] = None,
        peers: Sequence[PeerTrack] = (),
    ) -> DriveCommand:
        """``pose`` must be the fused pose: the stop-line distance is computed from it."""
        arc_s, lateral = self.route.locate(pose.point)

        lookahead = self.config.lookahead_min_m + self.config.lookahead_gain_s * max(0.0, speed)
        target = self.route.lookahead_point(arc_s, lookahead)
        steer = pure_pursuit_steer(
            pose, target, self.config.wheelbase, self.config.max_steer_rad
        )

        # Lowest wins; the label says which constraint is holding the car back.
        candidates: List[Tuple[float, str, str]] = [
            (self.config.cruise_speed, "cruise", "route clear")
        ]

        corner_limit = self.config.cruise_speed * (
            1.0 - self.config.curvature_speed_gain * abs(steer) / self.config.max_steer_rad
        )
        candidates.append((max(0.12, corner_limit), "corner", "steering {:.0f} deg".format(math.degrees(steer))))

        stop_line, gap = self.route.next_stop_line(arc_s)
        approach: Optional[Approach] = None
        conflict: Optional[JunctionAssessment] = None

        if stop_line is not None and gap < 4.0:
            held = (
                self._held_line_s is not None
                and abs(self._held_line_s - stop_line.arc_s) < 1e-6
            )
            # Creep past the line while held is an artefact, not a decision to enter.
            effective_gap = max(gap, 0.0) if held else gap

            approach = self.policy.evaluate(effective_gap, speed, light)
            candidates.append((approach.target_speed, "signal", approach.reason))

            conflict = assess_junction(
                ego_pose=pose,
                ego_speed=speed,
                ego_yaw_rate=yaw_rate,
                peers=peers,
                conflict_point=stop_line.conflict_point,
                distance_to_line=effective_gap,
                zone_radius_m=stop_line.zone_radius,
                vehicle_length_m=self.config.vehicle_length,
                ego_id=self.config.vehicle_id,
            )
            if conflict.verdict is RightOfWay.EMERGENCY_STOP:
                candidates.append((0.0, "v2v", conflict.reason))
            elif conflict.verdict is RightOfWay.YIELD:
                # Yield at the line, with the same braking profile the light would use.
                yield_speed = self.policy.evaluate(effective_gap, speed, None).target_speed
                candidates.append((yield_speed, "v2v", conflict.reason))

            self._update_hold(stop_line, effective_gap, speed, approach, conflict, held)
        else:
            self._held_line_s = None

        speed_cmd, factor, reason = min(candidates, key=lambda c: c[0])
        if self._held_line_s is not None:
            speed_cmd = 0.0

        return DriveCommand(
            speed=max(0.0, speed_cmd),
            steer=steer,
            limiting_factor=factor,
            reason=reason,
            arc_s=arc_s,
            lateral_error_m=lateral,
            stop_line=stop_line,
            distance_to_line_m=gap,
            approach=approach,
            conflict=conflict,
        )

    def _update_hold(
        self,
        stop_line: StopLine,
        gap: float,
        speed: float,
        approach: Approach,
        conflict: Optional[JunctionAssessment],
        held: bool,
    ) -> None:
        """Release needs both constraints clear at once: a green with a peer crossing is not a go.
        """
        signal_clear = approach.decision in (ApproachDecision.PROCEED, ApproachDecision.CLEAR)
        peer_clear = conflict is None or conflict.verdict is RightOfWay.PROCEED

        if held:
            if signal_clear and peer_clear:
                self._held_line_s = None
            return

        blocked = not signal_clear or not peer_clear
        if blocked and speed < self.HOLD_SPEED and gap < self.HOLD_DISTANCE:
            self._held_line_s = stop_line.arc_s
