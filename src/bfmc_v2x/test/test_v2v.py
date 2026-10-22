"""Symmetry: two cars on the same shared state must reach opposite conclusions. Both
yielding deadlocks; neither yielding collides.
"""

from __future__ import annotations

import math

import pytest

from bfmc_v2x.geometry import Point, Pose2D
from bfmc_v2x.protocol import Intent, PeerState
from bfmc_v2x.v2v import (
    PeerRegistry,
    PeerTrack,
    RightOfWay,
    assess_junction,
    closest_approach,
    predict_trajectory,
)

JUNCTION = Point(3.0, 3.0)

NORTH = math.pi / 2.0
SOUTH = -math.pi / 2.0
EAST = 0.0
WEST = math.pi


def track(vehicle_id, x, y, yaw, speed=0.35, yaw_rate=0.0, intent=Intent.STRAIGHT, age=0.05):
    return PeerTrack(
        vehicle_id=vehicle_id,
        pose=Pose2D(x, y, yaw),
        speed=speed,
        yaw_rate=yaw_rate,
        intent=intent,
        age_s=age,
    )


# ---- registry ------------------------------------------------------------


def test_registry_ages_a_peer_forward():
    registry = PeerRegistry()
    registry.ingest(
        PeerState(vehicle_id=2, x=0.0, y=0.0, yaw=0.0, speed=0.5, yaw_rate=0.0,
                  intent=int(Intent.STRAIGHT)),
        rx_time=100.0,
    )

    (peer,) = registry.active(now=100.4)

    assert peer.pose.x == pytest.approx(0.2, abs=1e-6)
    assert peer.age_s == pytest.approx(0.4, abs=1e-6)


def test_registry_accounts_for_time_in_flight():
    registry = PeerRegistry()
    registry.ingest(
        PeerState(vehicle_id=2, x=0.0, y=0.0, yaw=0.0, speed=0.5, yaw_rate=0.0,
                  intent=int(Intent.STRAIGHT)),
        rx_time=100.0,
        link_age_s=0.2,
    )

    (peer,) = registry.active(now=100.0)

    assert peer.age_s == pytest.approx(0.2)


def test_a_peer_that_goes_quiet_is_dropped():
    registry = PeerRegistry(expire_after_s=1.0)
    registry.ingest(
        PeerState(vehicle_id=2, x=0.0, y=0.0, yaw=0.0, speed=0.0, yaw_rate=0.0,
                  intent=int(Intent.HOLD)),
        rx_time=100.0,
    )

    assert len(registry.active(now=100.5)) == 1
    assert registry.active(now=102.0) == []
    assert registry.expired == 1
    assert len(registry) == 0


def test_turning_peer_is_extrapolated_around_its_arc():
    turning = track(2, 0.0, 0.0, 0.0, speed=0.5, yaw_rate=1.0)
    moved = turning.extrapolated(0.5)

    assert moved.pose.yaw == pytest.approx(0.5)
    # Mid-interval heading: off the +x axis, short of the naive 0.25 m.
    assert 0.0 < moved.pose.y < 0.1
    assert moved.pose.x < 0.25


# ---- trajectories --------------------------------------------------------


def test_closest_approach_finds_the_crossing():
    ego = predict_trajectory(Pose2D(3.0, 1.0, NORTH), 0.5, 0.0, horizon_s=6.0)
    peer = predict_trajectory(Pose2D(5.0, 3.0, WEST), 0.5, 0.0, horizon_s=6.0)

    time_s, gap = closest_approach(ego, peer)

    assert time_s == pytest.approx(4.0, abs=0.2)
    assert gap == pytest.approx(0.0, abs=0.05)


def test_parallel_traffic_never_conflicts():
    ego = predict_trajectory(Pose2D(3.0, 1.0, NORTH), 0.4, 0.0)
    peer = predict_trajectory(Pose2D(3.6, 1.0, NORTH), 0.4, 0.0)

    _, gap = closest_approach(ego, peer)

    assert gap == pytest.approx(0.6, abs=1e-6)


# ---- right of way --------------------------------------------------------


def test_exactly_one_of_two_symmetric_vehicles_yields():
    ego = Pose2D(3.0, 1.5, NORTH)
    peer = Pose2D(4.5, 3.0, WEST)

    from_ego = assess_junction(
        ego_pose=ego, ego_speed=0.35, ego_yaw_rate=0.0,
        peers=[track(2, peer.x, peer.y, peer.yaw)],
        conflict_point=JUNCTION, distance_to_line=0.95, ego_id=1,
    )
    from_peer = assess_junction(
        ego_pose=peer, ego_speed=0.35, ego_yaw_rate=0.0,
        peers=[track(1, ego.x, ego.y, ego.yaw)],
        conflict_point=JUNCTION, distance_to_line=0.95, ego_id=2,
    )

    yielding = [a.verdict is not RightOfWay.PROCEED for a in (from_ego, from_peer)]
    assert sum(yielding) == 1, "{} vs {}".format(from_ego.verdict, from_peer.verdict)
    # The lower id wins the tie, so vehicle 1 goes.
    assert from_ego.verdict is RightOfWay.PROCEED
    assert from_peer.verdict is RightOfWay.YIELD


def test_the_vehicle_that_arrives_first_keeps_the_junction():
    # Ego 0.6 m out, peer 2.0 m out, same speed.
    assessment = assess_junction(
        ego_pose=Pose2D(3.0, 2.4, NORTH), ego_speed=0.35, ego_yaw_rate=0.0,
        peers=[track(1, 5.0, 3.0, WEST)],       # lower id, but further away
        conflict_point=JUNCTION, distance_to_line=0.05, ego_id=9,
    )

    assert assessment.verdict is RightOfWay.PROCEED


def test_a_peer_that_has_already_crossed_is_not_a_conflict():
    assessment = assess_junction(
        ego_pose=Pose2D(3.0, 1.5, NORTH), ego_speed=0.35, ego_yaw_rate=0.0,
        peers=[track(2, 1.5, 3.0, WEST)],       # west of the junction, leaving
        conflict_point=JUNCTION, distance_to_line=0.95, ego_id=1,
    )

    assert assessment.verdict is RightOfWay.PROCEED
    assert assessment.peer_id is None


def test_a_peer_holding_at_the_line_does_not_make_us_yield():
    assessment = assess_junction(
        ego_pose=Pose2D(3.0, 1.5, NORTH), ego_speed=0.35, ego_yaw_rate=0.0,
        peers=[track(0, 4.5, 3.0, WEST, speed=0.0, intent=Intent.HOLD)],
        conflict_point=JUNCTION, distance_to_line=0.95, ego_id=1,
    )

    assert assessment.verdict is RightOfWay.PROCEED


def test_a_stationary_peer_blocking_the_junction_is_a_conflict():
    assessment = assess_junction(
        ego_pose=Pose2D(3.0, 2.2, NORTH), ego_speed=0.35, ego_yaw_rate=0.0,
        peers=[track(2, 3.05, 3.0, WEST, speed=0.0, intent=Intent.STRAIGHT)],
        conflict_point=JUNCTION, distance_to_line=0.25, ego_id=1,
    )

    assert assessment.verdict is not RightOfWay.PROCEED


def test_an_imminent_collision_is_an_emergency_stop():
    assessment = assess_junction(
        ego_pose=Pose2D(3.0, 2.85, NORTH), ego_speed=0.35, ego_yaw_rate=0.0,
        peers=[track(2, 3.2, 3.0, WEST, speed=0.35)],
        conflict_point=JUNCTION, distance_to_line=-0.1, ego_id=1,
    )

    assert assessment.verdict is RightOfWay.EMERGENCY_STOP
    assert assessment.min_separation_m < 0.45


def test_the_worst_of_several_peers_wins():
    assessment = assess_junction(
        ego_pose=Pose2D(3.0, 2.85, NORTH), ego_speed=0.35, ego_yaw_rate=0.0,
        peers=[
            track(5, 1.0, 3.0, WEST),            # harmless, already past
            track(2, 3.2, 3.0, WEST, speed=0.35),  # about to hit us
        ],
        conflict_point=JUNCTION, distance_to_line=-0.1, ego_id=1,
    )

    assert assessment.verdict is RightOfWay.EMERGENCY_STOP
    assert assessment.peer_id == 2


def test_an_empty_road_always_proceeds():
    assessment = assess_junction(
        ego_pose=Pose2D(3.0, 1.5, NORTH), ego_speed=0.35, ego_yaw_rate=0.0,
        peers=[], conflict_point=JUNCTION, distance_to_line=0.95, ego_id=1,
    )

    assert assessment.verdict is RightOfWay.PROCEED
    assert assessment.reason == "no conflicting peer"
    assert math.isinf(assessment.min_separation_m)


def test_assessment_reports_the_windows_it_compared():
    assessment = assess_junction(
        ego_pose=Pose2D(3.0, 1.5, NORTH), ego_speed=0.35, ego_yaw_rate=0.0,
        peers=[track(2, 4.5, 3.0, WEST)],
        conflict_point=JUNCTION, distance_to_line=0.95, ego_id=3,
    )

    assert assessment.peer_window is not None
    assert assessment.ego_window.will_enter
    assert assessment.ego_window.enter_s < assessment.ego_window.exit_s
    assert str(assessment.peer_id) in assessment.reason
