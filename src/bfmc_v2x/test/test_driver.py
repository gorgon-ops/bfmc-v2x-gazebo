"""A kinematic bicycle stands in for Gazebo: it tests the decisions, not the physics, fast
enough to assert "never crosses the stop line on a red" at every step.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List, Optional

import pytest

from bfmc_v2x.driver import Driver, DriverConfig
from bfmc_v2x.geometry import Pose2D, wrap_angle
from bfmc_v2x.protocol import Intent, SemaphoreState
from bfmc_v2x.semaphore import ObservedLight
from bfmc_v2x.v2v import PeerTrack, RightOfWay

DT = 0.02  # 50 Hz control loop


class Bicycle:
    def __init__(self, pose: Pose2D, wheelbase: float = 0.26, speed_tau: float = 0.25) -> None:
        self.pose = pose
        self.speed = 0.0
        self.yaw_rate = 0.0
        self.wheelbase = wheelbase
        self.speed_tau = speed_tau

    def step(self, target_speed: float, steer: float, dt: float = DT) -> None:
        self.speed += (target_speed - self.speed) * min(1.0, dt / self.speed_tau)
        self.yaw_rate = self.speed * math.tan(steer) / self.wheelbase
        mid = self.pose.yaw + 0.5 * self.yaw_rate * dt
        self.pose = Pose2D(
            self.pose.x + self.speed * math.cos(mid) * dt,
            self.pose.y + self.speed * math.sin(mid) * dt,
            wrap_angle(self.pose.yaw + self.yaw_rate * dt),
        )


def green(ttl: float = 30.0) -> ObservedLight:
    return ObservedLight(1, SemaphoreState.GREEN, ttl, 0.02)


def red(ttl: Optional[float] = 30.0) -> ObservedLight:
    return ObservedLight(1, SemaphoreState.RED, ttl, 0.02)


@dataclass
class Run:
    lateral: List[float]
    distance_to_line: List[float]
    speed: List[float]
    arc: List[float]
    factors: List[str]
    final: Pose2D
    peers: List[List[PeerTrack]] = field(default_factory=list)


def drive(
    route,
    seconds: float,
    light=None,
    peers=(),
    config: DriverConfig = DriverConfig(),
    start_arc: Optional[float] = None,
) -> Run:
    """``light`` and ``peers`` may be values or zero-argument callables."""
    start, heading = route.start_pose(start_arc)
    vehicle = Bicycle(Pose2D(start.x, start.y, heading), wheelbase=config.wheelbase)
    driver = Driver(route, config)

    run = Run([], [], [], [], [], vehicle.pose)
    for _ in range(int(seconds / DT)):
        observed = light() if callable(light) else light
        current_peers = peers() if callable(peers) else peers
        command = driver.step(
            vehicle.pose, vehicle.speed, vehicle.yaw_rate, observed, current_peers
        )
        vehicle.step(command.speed, command.steer)

        run.lateral.append(command.lateral_error_m)
        run.distance_to_line.append(command.distance_to_line_m)
        run.speed.append(vehicle.speed)
        run.arc.append(command.arc_s)
        run.factors.append(command.limiting_factor)
        run.peers.append(list(current_peers))
    run.final = vehicle.pose
    return run


# ---- path following ------------------------------------------------------


def test_the_vehicle_completes_a_lap(ego_route):
    run = drive(ego_route, seconds=70.0, light=green())

    travelled = sum(
        (b - a) % ego_route.length for a, b in zip(run.arc, run.arc[1:])
    )
    assert travelled > ego_route.length, "only covered {:.1f} m".format(travelled)


def test_the_vehicle_stays_in_its_lane(ego_route):
    run = drive(ego_route, seconds=70.0, light=green())

    settled = run.lateral[50:]
    worst = max(abs(v) for v in settled)
    mean = sum(abs(v) for v in settled) / len(settled)

    # Half a lane width touches the centre line.
    assert worst < 0.12, "peak lateral error {:.3f} m".format(worst)
    assert mean < 0.04, "mean lateral error {:.3f} m".format(mean)


def test_the_vehicle_slows_for_corners(ego_route):
    run = drive(ego_route, seconds=70.0, light=green())

    assert "corner" in run.factors, "the corner limit never bound"
    cornering = [s for s, f in zip(run.speed, run.factors) if f == "corner"]
    cruising = [s for s, f in zip(run.speed, run.factors) if f == "cruise"]

    assert max(cornering) < max(cruising)


# ---- the light -----------------------------------------------------------


def test_the_vehicle_stops_at_a_red_light(ego_route):
    run = drive(ego_route, seconds=30.0, light=red())

    assert run.speed[-1] == pytest.approx(0.0, abs=0.02), "still moving at the line"
    assert 0.0 <= run.distance_to_line[-1] < 0.15


def test_the_vehicle_never_crosses_a_red_line(ego_route):
    run = drive(ego_route, seconds=30.0, light=red())

    stop_line_arc = ego_route.stop_lines[0].arc_s
    assert max(run.arc) <= stop_line_arc + 1e-6


def test_a_green_light_does_not_stop_the_vehicle(ego_route):
    run = drive(ego_route, seconds=30.0, light=green())

    stop_line_arc = ego_route.stop_lines[0].arc_s
    assert max(run.arc) > stop_line_arc + 1.0, "did not clear the junction"
    assert min(run.speed[50:]) > 0.05, "came to a stop on a green"


def test_a_light_that_turns_green_releases_the_vehicle(ego_route):
    clock = {"t": 0.0}

    def changing():
        clock["t"] += DT
        return red(30.0) if clock["t"] < 18.0 else green(30.0)

    run = drive(ego_route, seconds=40.0, light=changing)

    stopped = min(run.speed[600:880])
    assert stopped == pytest.approx(0.0, abs=0.02), "never stopped for the red"
    assert max(run.arc) > ego_route.stop_lines[0].arc_s + 0.5, "never pulled away"


def test_an_unsignalled_line_is_treated_as_a_give_way(ego_route):
    run = drive(ego_route, seconds=30.0, light=None)

    assert run.speed[-1] == pytest.approx(0.0, abs=0.02)
    assert run.factors[-1] == "signal"


# ---- the peer ------------------------------------------------------------


def _pose_short_of_the_line(route, gap: float) -> Pose2D:
    """Derived from the map, so moving a road cannot silently relocate it past the line."""
    arc = route.stop_lines[0].arc_s - gap
    return Pose2D(*route.lane.point_at(arc).as_tuple(), math.pi / 2.0)


def _crossing_peer(speed: float = 0.35) -> PeerTrack:
    return PeerTrack(
        vehicle_id=2,
        pose=Pose2D(3.5, 3.0, math.pi),
        speed=speed,
        yaw_rate=0.0,
        intent=Intent.STRAIGHT,
        age_s=0.05,
    )


class MovingPeer:
    """A frozen peer claiming 0.35 m/s is unfair: the ego correctly decides it arrives first."""

    def __init__(self, x: float, speed: float = 0.35, vehicle_id: int = 2) -> None:
        self.x = x
        self.speed = speed
        self.vehicle_id = vehicle_id

    def __call__(self):
        self.x -= self.speed * DT
        return [
            PeerTrack(
                vehicle_id=self.vehicle_id,
                pose=Pose2D(self.x, 3.0, math.pi),
                speed=self.speed,
                yaw_rate=0.0,
                intent=Intent.STRAIGHT,
                age_s=0.05,
            )
        ]

    def occupies_junction(self, zone: float = 0.55, length: float = 0.36) -> bool:
        return 3.0 - zone - length < self.x < 3.0 + zone


def test_the_vehicle_yields_to_a_conflicting_peer(ego_route):
    peer = MovingPeer(x=5.45)
    run = drive(
        ego_route, seconds=14.0, light=green(), peers=peer,
        config=DriverConfig(vehicle_id=5), start_arc=3.2,
    )

    assert "v2v" in run.factors, "the peer never constrained the vehicle"

    stopped = [i for i, v in enumerate(run.speed) if v < 0.05]
    assert stopped, "never yielded"
    first_stop = stopped[len(stopped) // 2]
    assert run.distance_to_line[first_stop] > -0.05, "stopped inside the junction"

    assert max(run.arc) > ego_route.stop_lines[0].arc_s, "never resumed"


def test_the_two_vehicles_are_never_in_the_junction_together(ego_route):
    peer = MovingPeer(x=5.45)
    run = drive(
        ego_route, seconds=14.0, light=green(), peers=peer,
        config=DriverConfig(vehicle_id=5), start_arc=3.2,
    )

    line = ego_route.stop_lines[0]
    zone, length = line.zone_radius, 0.36

    for gap, tracked in zip(run.distance_to_line, run.peers):
        # The ego's nose reaches the zone past the line by the setback; its tail clears
        # a vehicle length later.
        past_line = -gap
        ego_inside = past_line > (line.conflict_s - line.arc_s) - zone

        nose = tracked[0].pose.x
        tail = nose + length
        peer_inside = nose < 3.0 + zone and tail > 3.0 - zone

        assert not (ego_inside and peer_inside), (
            "both in the junction: ego {:.3f} m past the line, peer nose at x={:.2f}".format(
                past_line, nose
            )
        )


def test_the_vehicle_takes_the_junction_when_it_has_priority(ego_route):
    peer = MovingPeer(x=6.2, vehicle_id=9)
    run = drive(
        ego_route, seconds=16.0, light=green(), peers=peer,
        config=DriverConfig(vehicle_id=1), start_arc=3.9,
    )

    assert max(run.arc) > ego_route.stop_lines[0].arc_s, "yielded when it had priority"


def test_a_peer_holding_does_not_block_us(ego_route):
    parked = PeerTrack(
        vehicle_id=2, pose=Pose2D(4.6, 3.0, math.pi), speed=0.0,
        yaw_rate=0.0, intent=Intent.HOLD, age_s=0.05,
    )

    run = drive(ego_route, seconds=16.0, light=green(), peers=[parked],
                config=DriverConfig(vehicle_id=5))

    assert max(run.arc) > ego_route.stop_lines[0].arc_s


# ---- arbitration ---------------------------------------------------------


def test_the_slowest_request_wins(ego_route):
    driver = Driver(ego_route, DriverConfig(vehicle_id=5))
    # Right at the line: further back a red is comfortably stoppable and the signal
    # would not be the binding constraint.
    pose = _pose_short_of_the_line(ego_route, 0.04)

    unconstrained = driver.step(pose, 0.35, 0.0, green())
    with_red = driver.step(pose, 0.35, 0.0, red())
    with_peer = driver.step(pose, 0.35, 0.0, green(), [_crossing_peer()])

    assert with_red.speed <= unconstrained.speed
    assert with_peer.speed <= unconstrained.speed
    assert with_red.limiting_factor == "signal"
    assert with_peer.limiting_factor == "v2v"


def test_the_command_explains_itself(ego_route):
    driver = Driver(ego_route, DriverConfig(vehicle_id=5))
    pose = _pose_short_of_the_line(ego_route, 0.04)

    command = driver.step(pose, 0.35, 0.0, green(), [_crossing_peer()])

    assert command.reason
    assert command.conflict is not None
    assert command.conflict.verdict is not RightOfWay.PROCEED
    assert command.stop_line is not None


def test_steering_is_never_abandoned_while_stopping(ego_route):
    run = drive(ego_route, seconds=30.0, light=red())

    assert abs(run.lateral[-1]) < 0.08
