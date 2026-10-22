"""Environment server: vehicles report obstacles in the track frame, the server scores them.

A report is placed using the car's own pose, so a car 40 cm lost reports every
obstacle 40 cm wrong - the most honest end-to-end metric of the localisation.
Only the first match per obstacle scores; a match to nothing is a false positive.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from .geometry import Point, Pose2D, distance
from .protocol import ObstacleAck, ObstacleKind, ObstacleReport

__all__ = [
    "GroundTruthObstacle",
    "ScoredReport",
    "Scoreboard",
    "EnvironmentServer",
    "Detection",
    "ObstacleDetector",
]


@dataclass(frozen=True)
class GroundTruthObstacle:
    obstacle_id: int
    kind: ObstacleKind
    x: float
    y: float

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


@dataclass(frozen=True)
class ScoredReport:
    vehicle_id: int
    seq: int
    kind: ObstacleKind
    reported: Point
    matched_id: Optional[int]
    error_m: float
    accepted: bool
    note: str


@dataclass
class Scoreboard:
    accepted: int = 0
    duplicates: int = 0
    false_positives: int = 0
    misclassified: int = 0
    #: Smallest error achieved per obstacle id.
    best_error_m: Dict[int, float] = field(default_factory=dict)

    @property
    def found(self) -> int:
        return len(self.best_error_m)

    @property
    def mean_error_m(self) -> float:
        if not self.best_error_m:
            return math.nan
        return sum(self.best_error_m.values()) / len(self.best_error_m)

    def summary(self, total_obstacles: int) -> str:
        return (
            "{}/{} obstacles located, mean error {:.3f} m "
            "({} accepted, {} duplicate, {} false positive, {} misclassified)".format(
                self.found,
                total_obstacles,
                self.mean_error_m,
                self.accepted,
                self.duplicates,
                self.false_positives,
                self.misclassified,
            )
        )


class EnvironmentServer:
    def __init__(
        self,
        obstacles: Iterable[GroundTruthObstacle],
        tolerance_m: float = 0.35,
        min_confidence: float = 0.4,
    ) -> None:
        self.obstacles: Tuple[GroundTruthObstacle, ...] = tuple(obstacles)
        self.tolerance_m = tolerance_m
        self.min_confidence = min_confidence
        self.scoreboard = Scoreboard()
        self.log: List[ScoredReport] = []
        self._credited: Dict[int, set] = {}

    def submit(self, report: ObstacleReport, vehicle_id: int, seq: int) -> Tuple[ObstacleAck, ScoredReport]:
        reported = Point(report.x, report.y)

        if report.confidence < self.min_confidence:
            scored = ScoredReport(
                vehicle_id, seq, report.obstacle, reported, None, math.inf, False,
                "confidence {:.2f} below threshold".format(report.confidence),
            )
            return self._finish(scored)

        nearest, nearest_gap = self._nearest(reported, kind=report.obstacle)
        if nearest is None or nearest_gap > self.tolerance_m:
            # "Wrong place" is a localisation problem, "wrong label" a detector problem;
            # keep them apart.
            any_obstacle, any_gap = self._nearest(reported, kind=None)
            if any_obstacle is not None and any_gap <= self.tolerance_m:
                self.scoreboard.misclassified += 1
                scored = ScoredReport(
                    vehicle_id, seq, report.obstacle, reported, None, any_gap, False,
                    "matched a {} at {:.2f} m, not a {}".format(
                        any_obstacle.kind.name, any_gap, report.obstacle.name
                    ),
                )
            else:
                self.scoreboard.false_positives += 1
                scored = ScoredReport(
                    vehicle_id, seq, report.obstacle, reported, None, nearest_gap, False,
                    "no {} within {:.2f} m".format(report.obstacle.name, self.tolerance_m),
                )
            return self._finish(scored)

        credited = self._credited.setdefault(vehicle_id, set())
        if nearest.obstacle_id in credited:
            self.scoreboard.duplicates += 1
            # A second, closer pass can still improve the recorded error.
            previous = self.scoreboard.best_error_m.get(nearest.obstacle_id, math.inf)
            if nearest_gap < previous:
                self.scoreboard.best_error_m[nearest.obstacle_id] = nearest_gap
            scored = ScoredReport(
                vehicle_id, seq, report.obstacle, reported, nearest.obstacle_id, nearest_gap,
                False, "already credited for obstacle {}".format(nearest.obstacle_id),
            )
            return self._finish(scored)

        credited.add(nearest.obstacle_id)
        self.scoreboard.accepted += 1
        previous = self.scoreboard.best_error_m.get(nearest.obstacle_id, math.inf)
        self.scoreboard.best_error_m[nearest.obstacle_id] = min(previous, nearest_gap)
        scored = ScoredReport(
            vehicle_id, seq, report.obstacle, reported, nearest.obstacle_id, nearest_gap,
            True, "located obstacle {} to {:.3f} m".format(nearest.obstacle_id, nearest_gap),
        )
        return self._finish(scored)

    def _finish(self, scored: ScoredReport) -> Tuple[ObstacleAck, ScoredReport]:
        self.log.append(scored)
        ack = ObstacleAck(
            report_seq=scored.seq & 0xFFFFFFFF,
            accepted=int(scored.accepted),
            error_m=float(scored.error_m if math.isfinite(scored.error_m) else -1.0),
        )
        return ack, scored

    def _nearest(
        self, point: Point, kind: Optional[ObstacleKind]
    ) -> Tuple[Optional[GroundTruthObstacle], float]:
        best: Optional[GroundTruthObstacle] = None
        best_gap = math.inf
        for obstacle in self.obstacles:
            if kind is not None and obstacle.kind is not kind:
                continue
            gap = distance(point, obstacle.point)
            if gap < best_gap:
                best, best_gap = obstacle, gap
        return best, best_gap

    def summary(self) -> str:
        return self.scoreboard.summary(len(self.obstacles))


@dataclass(frozen=True)
class Detection:
    """Body frame, as a camera gives it; the pose error enters when it is placed on the map."""

    kind: ObstacleKind
    #: Metres ahead of the rear axle.
    forward: float
    #: Metres left of the centreline.
    left: float
    confidence: float

    def to_track_frame(self, pose: Pose2D) -> Point:
        """Fused pose: lands near the obstacle. Dead-reckoned: lands wherever the drift went."""
        placed = pose.translated(forward=self.forward, left=self.left)
        return Point(placed.x, placed.y)


class ObstacleDetector:
    """Simulated from ground truth: the project is about localisation, not vision. A real
    detector replaces this class and nothing else.
    """

    def __init__(
        self,
        obstacles: Iterable[GroundTruthObstacle],
        max_range_m: float = 1.6,
        half_fov_rad: float = math.radians(35.0),
        range_noise_frac: float = 0.06,
        bearing_noise_rad: float = math.radians(2.0),
        rng: Optional[random.Random] = None,
    ) -> None:
        self.obstacles = tuple(obstacles)
        self.max_range_m = max_range_m
        self.half_fov_rad = half_fov_rad
        self.range_noise_frac = range_noise_frac
        self.bearing_noise_rad = bearing_noise_rad
        self._rng = rng if rng is not None else random.Random(20240601)

    def detect(self, true_pose: Pose2D) -> List[Detection]:
        """From the true pose: what the camera sees depends on where the car really is."""
        found: List[Detection] = []
        for obstacle in self.obstacles:
            local = true_pose.to_body(obstacle.point)
            range_m = math.hypot(local.x, local.y)
            if range_m > self.max_range_m or range_m < 1e-3:
                continue
            bearing = math.atan2(local.y, local.x)
            if abs(bearing) > self.half_fov_rad:
                continue

            noisy_range = range_m * (1.0 + self._rng.gauss(0.0, self.range_noise_frac))
            noisy_bearing = bearing + self._rng.gauss(0.0, self.bearing_noise_rad)
            confidence = max(0.0, min(1.0, 1.0 - (range_m / self.max_range_m) ** 2))
            found.append(
                Detection(
                    kind=obstacle.kind,
                    forward=noisy_range * math.cos(noisy_bearing),
                    left=noisy_range * math.sin(noisy_bearing),
                    confidence=confidence,
                )
            )
        return found
