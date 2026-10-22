"""Traffic lights: the fixed-time cycle inside a light, and the tracker and policy inside a car.

Knowing the red expires in 700 ms lets a car roll instead of stopping; without a
countdown the policy degrades to the conservative full stop.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple

from .protocol import SemaphoreBroadcast, SemaphoreState

__all__ = [
    "Phase",
    "SemaphoreCycle",
    "ObservedLight",
    "SemaphoreTracker",
    "Approach",
    "ApproachDecision",
    "StopLinePolicy",
    "DEFAULT_PHASES",
]

#: Red lasts the opposing green plus yellow, so a two-axis junction runs this plan
#: on both axes nine seconds apart. Any shorter red shows green both ways.
DEFAULT_PHASES: Tuple["Phase", ...] = ()


@dataclass(frozen=True)
class Phase:
    state: SemaphoreState
    duration_s: float

    def __post_init__(self) -> None:
        if self.duration_s <= 0.0:
            raise ValueError("a phase must last a positive time")


DEFAULT_PHASES = (
    Phase(SemaphoreState.GREEN, 7.0),
    Phase(SemaphoreState.YELLOW, 2.0),
    Phase(SemaphoreState.RED, 9.0),
)


class SemaphoreCycle:
    """The colour is a pure function of the clock, so a restarted light comes back in phase."""

    def __init__(
        self,
        light_id: int,
        phases: Sequence[Phase] = DEFAULT_PHASES,
        offset_s: float = 0.0,
    ) -> None:
        if not phases:
            raise ValueError("a cycle needs at least one phase")
        self.light_id = light_id
        self.phases: Tuple[Phase, ...] = tuple(phases)
        self.offset_s = offset_s
        self.period_s = sum(p.duration_s for p in self.phases)

    def state_at(self, now: float) -> Tuple[SemaphoreState, float]:
        elapsed = (now - self.offset_s) % self.period_s
        for phase in self.phases:
            if elapsed < phase.duration_s:
                return phase.state, phase.duration_s - elapsed
            elapsed -= phase.duration_s
        # Only reachable through floating-point slop at the period boundary.
        last = self.phases[-1]
        return last.state, 0.0

    def broadcast_at(self, now: float) -> SemaphoreBroadcast:
        state, remaining = self.state_at(now)
        return SemaphoreBroadcast(
            light_id=self.light_id,
            state=int(state),
            ttl_ms=int(max(0.0, min(65.535, remaining)) * 1000.0),
        )


@dataclass(frozen=True)
class ObservedLight:
    light_id: int
    state: SemaphoreState
    #: Seconds, or ``None`` if not trustworthy.
    ttl_s: Optional[float]
    #: Seconds since the light stamped the broadcast.
    age_s: float

    @property
    def is_stale(self) -> bool:
        return self.ttl_s is None


class SemaphoreTracker:
    """A light that stops broadcasting goes stale: last colour, no countdown."""

    def __init__(self, stale_after_s: float = 1.0) -> None:
        self.stale_after_s = stale_after_s
        self._latest: Dict[int, Tuple[SemaphoreState, float, float]] = {}

    def ingest(self, broadcast: SemaphoreBroadcast, rx_time: float, link_age_s: float = 0.0) -> None:
        """Flight time is subtracted: a 700 ms countdown that took 200 ms to arrive has 500 ms left.
        """
        remaining = max(0.0, broadcast.ttl_s - max(0.0, link_age_s))
        self._latest[broadcast.light_id] = (broadcast.colour, remaining, rx_time)

    def observe(self, light_id: int, now: float) -> Optional[ObservedLight]:
        entry = self._latest.get(light_id)
        if entry is None:
            return None
        state, remaining, rx_time = entry
        age = max(0.0, now - rx_time)
        if age > self.stale_after_s:
            return ObservedLight(light_id, state, None, age)
        countdown = remaining - age
        if countdown <= 0.0:
            # Should have changed and we have not been told to what: withdraw the
            # countdown.
            return ObservedLight(light_id, state, None, age)
        return ObservedLight(light_id, state, countdown, age)

    def known_lights(self) -> List[int]:
        return sorted(self._latest)


class ApproachDecision(Enum):
    PROCEED = "PROCEED"
    #: Shed speed now so a stop at the line is comfortable if needed.
    PREPARE = "PREPARE"
    STOP = "STOP"
    #: Past the point of no return on a changing light: clear the junction.
    CLEAR = "CLEAR"


@dataclass(frozen=True)
class Approach:
    decision: ApproachDecision
    #: m/s.
    target_speed: float
    reason: str


class StopLinePolicy:
    """Parameterised by braking capability rather than tuned distances, so it holds at any speed."""

    def __init__(
        self,
        cruise_speed: float = 0.35,
        comfortable_decel: float = 0.6,
        max_decel: float = 1.2,
        reaction_time_s: float = 0.15,
        stop_margin_m: float = 0.08,
        junction_clearance_m: float = 0.9,
    ) -> None:
        self.cruise_speed = cruise_speed
        self.comfortable_decel = comfortable_decel
        self.max_decel = max_decel
        self.reaction_time_s = reaction_time_s
        self.stop_margin_m = stop_margin_m
        self.junction_clearance_m = junction_clearance_m

    def braking_distance(self, speed: float, decel: float) -> float:
        if decel <= 0.0:
            return math.inf
        return speed * self.reaction_time_s + (speed * speed) / (2.0 * decel)

    def evaluate(
        self,
        distance_to_line: float,
        speed: float,
        light: Optional[ObservedLight],
    ) -> Approach:
        """``distance_to_line`` is negative past the line, where the only answer is to keep going.
        """
        if distance_to_line < 0.0:
            return Approach(ApproachDecision.CLEAR, self.cruise_speed, "past the line")

        if light is None:
            # A give-way, or a light never heard from: treat it as a stop sign.
            return self._stop_profile(distance_to_line, "no signal: treating as give-way")

        comfortable = self.braking_distance(speed, self.comfortable_decel)
        latest = self.braking_distance(speed, self.max_decel)

        # Not the current speed: a stopped car's time-to-line would be infinite and
        # it would sit at a green forever - a deadlock this project actually hit.
        reference_speed = max(speed, 0.5 * self.cruise_speed)
        time_to_line = distance_to_line / reference_speed

        if light.state is SemaphoreState.FLASHING_YELLOW:
            return self._stop_profile(distance_to_line, "flashing yellow: give way")

        if light.state is SemaphoreState.GREEN:
            if light.ttl_s is None:
                # Green for an unknown time: arrive slow enough to stop on a late change.
                return Approach(
                    ApproachDecision.PREPARE,
                    min(self.cruise_speed, self._speed_for(distance_to_line)),
                    "green, countdown unknown",
                )
            crossing_time = time_to_line + self.junction_clearance_m / reference_speed
            if crossing_time <= light.ttl_s:
                return Approach(
                    ApproachDecision.PROCEED,
                    self.cruise_speed,
                    "green for {:.1f} s, clears in {:.1f} s".format(light.ttl_s, crossing_time),
                )
            # Changes before we are through: plan the stop, do not sprint the amber.
            return self._stop_profile(
                distance_to_line, "green expires in {:.1f} s, too late to clear".format(light.ttl_s)
            )

        if light.state is SemaphoreState.YELLOW:
            if distance_to_line < latest:
                # Dilemma zone: stopping needs more braking than the car has.
                return Approach(
                    ApproachDecision.CLEAR, self.cruise_speed, "yellow inside dilemma zone"
                )
            return self._stop_profile(distance_to_line, "yellow, stoppable")

        if light.ttl_s is not None and time_to_line > light.ttl_s + self.reaction_time_s:
            # Green before we get there: no stop, no pull-away.
            return Approach(
                ApproachDecision.PROCEED,
                self.cruise_speed,
                "red clears in {:.1f} s, arriving in {:.1f} s".format(light.ttl_s, time_to_line),
            )
        if light.ttl_s is not None and light.ttl_s < 1.5 and distance_to_line > comfortable:
            # Close to clearing: creep rather than stopping dead.
            return Approach(
                ApproachDecision.PREPARE,
                min(self.cruise_speed, max(0.08, self._speed_for(distance_to_line))),
                "red clears in {:.1f} s, creeping".format(light.ttl_s),
            )
        return self._stop_profile(distance_to_line, "red")

    def _stop_profile(self, distance_to_line: float, reason: str) -> Approach:
        usable = distance_to_line - self.stop_margin_m
        if usable <= 0.0:
            return Approach(ApproachDecision.STOP, 0.0, reason)
        target = self._speed_for(distance_to_line)
        if target <= 0.05:
            return Approach(ApproachDecision.STOP, 0.0, reason)
        decision = (
            ApproachDecision.PROCEED if target >= self.cruise_speed else ApproachDecision.PREPARE
        )
        return Approach(decision, min(self.cruise_speed, target), reason)

    def _speed_for(self, distance_to_line: float) -> float:
        """Exact inverse of :meth:`braking_distance`, reaction time included, so a speed it
        returns can always stop in time and the car does not creep over the line.
        """
        usable = max(0.0, distance_to_line - self.stop_margin_m)
        lag = self.comfortable_decel * self.reaction_time_s
        return math.sqrt(lag * lag + 2.0 * self.comfortable_decel * usable) - lag
