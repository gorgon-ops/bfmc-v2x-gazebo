"""The last test ties the project together: same detector, scored with a fused pose and
with a drifted one.
"""

from __future__ import annotations

import math
import random

import pytest

from bfmc_v2x.environment import (
    Detection,
    EnvironmentServer,
    GroundTruthObstacle,
    ObstacleDetector,
)
from bfmc_v2x.geometry import Point, Pose2D
from bfmc_v2x.protocol import ObstacleKind, ObstacleReport

OBSTACLES = [
    GroundTruthObstacle(1, ObstacleKind.STATIC_CAR, 2.10, 0.12),
    GroundTruthObstacle(2, ObstacleKind.PEDESTRIAN, 3.35, 4.30),
    GroundTruthObstacle(3, ObstacleKind.ROADBLOCK, 1.55, 5.75),
]


def report(kind, x, y, confidence=0.9):
    return ObstacleReport(kind=int(kind), x=x, y=y, confidence=confidence)


# ---- the server ----------------------------------------------------------


def test_a_good_report_is_accepted():
    server = EnvironmentServer(OBSTACLES)

    ack, scored = server.submit(report(ObstacleKind.STATIC_CAR, 2.15, 0.14), vehicle_id=1, seq=5)

    assert scored.accepted and ack.was_accepted
    assert ack.report_seq == 5
    assert scored.matched_id == 1
    assert ack.error_m == pytest.approx(math.hypot(0.05, 0.02), abs=1e-4)
    assert server.scoreboard.found == 1


def test_a_report_in_the_wrong_place_is_a_false_positive():
    server = EnvironmentServer(OBSTACLES, tolerance_m=0.35)

    ack, scored = server.submit(report(ObstacleKind.STATIC_CAR, 4.0, 4.0), vehicle_id=1, seq=1)

    assert not scored.accepted and not ack.was_accepted
    assert server.scoreboard.false_positives == 1
    assert server.scoreboard.found == 0
    assert "no STATIC_CAR within" in scored.note


def test_right_place_wrong_label_is_reported_separately():
    server = EnvironmentServer(OBSTACLES)

    _, scored = server.submit(report(ObstacleKind.STATIC_CAR, 3.35, 4.30), vehicle_id=1, seq=1)

    assert not scored.accepted
    assert server.scoreboard.misclassified == 1
    assert server.scoreboard.false_positives == 0
    assert "PEDESTRIAN" in scored.note


def test_reporting_the_same_obstacle_twice_scores_once():
    server = EnvironmentServer(OBSTACLES)

    server.submit(report(ObstacleKind.STATIC_CAR, 2.15, 0.14), vehicle_id=1, seq=1)
    _, second = server.submit(report(ObstacleKind.STATIC_CAR, 2.11, 0.13), vehicle_id=1, seq=2)

    assert not second.accepted
    assert server.scoreboard.accepted == 1
    assert server.scoreboard.duplicates == 1
    assert server.scoreboard.found == 1


def test_a_closer_second_look_improves_the_recorded_error():
    server = EnvironmentServer(OBSTACLES)

    server.submit(report(ObstacleKind.STATIC_CAR, 2.40, 0.12), vehicle_id=1, seq=1)
    coarse = server.scoreboard.best_error_m[1]
    server.submit(report(ObstacleKind.STATIC_CAR, 2.11, 0.12), vehicle_id=1, seq=2)

    assert server.scoreboard.best_error_m[1] < coarse


def test_two_vehicles_are_scored_independently():
    server = EnvironmentServer(OBSTACLES)

    server.submit(report(ObstacleKind.STATIC_CAR, 2.15, 0.14), vehicle_id=1, seq=1)
    _, other = server.submit(report(ObstacleKind.STATIC_CAR, 2.12, 0.11), vehicle_id=2, seq=1)

    assert other.accepted
    assert server.scoreboard.accepted == 2


def test_a_low_confidence_guess_is_not_scored():
    server = EnvironmentServer(OBSTACLES, min_confidence=0.4)

    _, scored = server.submit(
        report(ObstacleKind.STATIC_CAR, 2.10, 0.12, confidence=0.1), vehicle_id=1, seq=1
    )

    assert not scored.accepted
    assert "below threshold" in scored.note
    assert server.scoreboard.found == 0


def test_summary_counts_everything():
    server = EnvironmentServer(OBSTACLES)
    server.submit(report(ObstacleKind.STATIC_CAR, 2.10, 0.12), 1, 1)
    server.submit(report(ObstacleKind.ROADBLOCK, 0.0, 0.0), 1, 2)

    summary = server.summary()

    assert "1/3 obstacles located" in summary
    assert "1 false positive" in summary


# ---- the detector --------------------------------------------------------


def test_the_detector_sees_what_is_in_front_of_it():
    detector = ObstacleDetector(OBSTACLES, rng=random.Random(1))
    found = detector.detect(Pose2D(2.10, -0.88, math.pi / 2.0))

    assert len(found) == 1
    assert found[0].kind is ObstacleKind.STATIC_CAR
    assert found[0].forward == pytest.approx(1.0, abs=0.2)


def test_the_detector_ignores_what_is_behind_it():
    detector = ObstacleDetector(OBSTACLES, rng=random.Random(1))
    facing_away = detector.detect(Pose2D(2.10, -0.88, -math.pi / 2.0))

    assert facing_away == []


def test_the_detector_ignores_what_is_out_of_range():
    detector = ObstacleDetector(OBSTACLES, max_range_m=1.6, rng=random.Random(1))

    assert detector.detect(Pose2D(2.10, -3.0, math.pi / 2.0)) == []


def test_confidence_falls_off_with_distance():
    detector = ObstacleDetector(OBSTACLES, rng=random.Random(1))

    (near,) = detector.detect(Pose2D(2.10, -0.42, math.pi / 2.0))
    (far,) = detector.detect(Pose2D(2.10, -1.28, math.pi / 2.0))

    assert near.confidence > far.confidence


def test_a_detection_is_placed_using_the_pose_it_is_given():
    detection = Detection(ObstacleKind.PEDESTRIAN, forward=1.0, left=0.5, confidence=0.9)

    placed = detection.to_track_frame(Pose2D(0.0, 0.0, math.pi / 2.0))

    # Facing north: 1 m ahead is +y, 0.5 m left is -x.
    assert placed.x == pytest.approx(-0.5)
    assert placed.y == pytest.approx(1.0)


# ---- the point of the exercise -------------------------------------------


def test_localisation_error_is_what_decides_the_score():
    """Drifted half a metre, the car maps nothing and litters confident false positives."""
    detector = ObstacleDetector(OBSTACLES, rng=random.Random(4))
    drift = Point(0.55, -0.35)

    fused_server = EnvironmentServer(OBSTACLES, tolerance_m=0.35)
    drifted_server = EnvironmentServer(OBSTACLES, tolerance_m=0.35)

    seq = 0
    for step in range(40):
        true_pose = Pose2D(2.10, -1.20 + step * 0.03, math.pi / 2.0)
        drifted_pose = Pose2D(true_pose.x + drift.x, true_pose.y + drift.y, true_pose.yaw)

        for detection in detector.detect(true_pose):
            seq += 1
            good = detection.to_track_frame(true_pose)
            bad = detection.to_track_frame(drifted_pose)
            fused_server.submit(report(detection.kind, good.x, good.y), 1, seq)
            drifted_server.submit(report(detection.kind, bad.x, bad.y), 1, seq)

    assert seq > 0, "the detector never saw anything"
    assert fused_server.scoreboard.found == 1
    assert fused_server.scoreboard.mean_error_m < 0.15

    assert drifted_server.scoreboard.found == 0
    assert drifted_server.scoreboard.false_positives > 0
