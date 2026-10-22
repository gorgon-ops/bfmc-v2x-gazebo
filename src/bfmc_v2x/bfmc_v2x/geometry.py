"""Track-frame geometry: metres, x east, y north, yaw counter-clockwise from +x."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

__all__ = [
    "wrap_angle",
    "Point",
    "Pose2D",
    "distance",
    "project_on_segment",
    "Polyline",
    "pure_pursuit_steer",
]


def wrap_angle(angle: float) -> float:
    """Fold into ``(-pi, pi]``; getting this wrong diverges the EKF on the westbound straight."""
    return math.atan2(math.sin(angle), math.cos(angle))


@dataclass(frozen=True)
class Point:
    x: float
    y: float

    def __add__(self, other: "Point") -> "Point":
        return Point(self.x + other.x, self.y + other.y)

    def __sub__(self, other: "Point") -> "Point":
        return Point(self.x - other.x, self.y - other.y)

    def scaled(self, factor: float) -> "Point":
        return Point(self.x * factor, self.y * factor)

    def norm(self) -> float:
        return math.hypot(self.x, self.y)

    def as_tuple(self) -> Tuple[float, float]:
        return (self.x, self.y)


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    yaw: float

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)

    def translated(self, forward: float, left: float = 0.0) -> "Pose2D":
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        return Pose2D(
            self.x + forward * cos_y - left * sin_y,
            self.y + forward * sin_y + left * cos_y,
            self.yaw,
        )

    def to_body(self, target: Point) -> Point:
        dx, dy = target.x - self.x, target.y - self.y
        cos_y, sin_y = math.cos(self.yaw), math.sin(self.yaw)
        return Point(dx * cos_y + dy * sin_y, -dx * sin_y + dy * cos_y)


def distance(a: Point, b: Point) -> float:
    return math.hypot(a.x - b.x, a.y - b.y)


def project_on_segment(point: Point, start: Point, end: Point) -> Tuple[Point, float, float]:
    """Closest point, arc length from ``start``, perpendicular distance; clamped to the segment."""
    seg = end - start
    length_sq = seg.x * seg.x + seg.y * seg.y
    if length_sq < 1e-12:
        return start, 0.0, distance(point, start)
    rel = point - start
    t = (rel.x * seg.x + rel.y * seg.y) / length_sq
    t = min(1.0, max(0.0, t))
    closest = Point(start.x + seg.x * t, start.y + seg.y * t)
    return closest, t * math.sqrt(length_sq), distance(point, closest)


class Polyline:
    """Cumulative arc length is computed once, for "how far along" and "what is L metres ahead"."""

    def __init__(self, points: Iterable[Point], closed: bool = False) -> None:
        pts = list(points)
        if len(pts) < 2:
            raise ValueError("a polyline needs at least two points")
        if closed and distance(pts[0], pts[-1]) > 1e-9:
            pts.append(pts[0])
        self._points: List[Point] = pts
        self.closed = closed
        self._cumulative: List[float] = [0.0]
        for a, b in zip(pts, pts[1:]):
            self._cumulative.append(self._cumulative[-1] + distance(a, b))

    @property
    def points(self) -> Sequence[Point]:
        return tuple(self._points)

    @property
    def length(self) -> float:
        return self._cumulative[-1]

    def __len__(self) -> int:
        return len(self._points)

    def locate(self, query: Point) -> Tuple[float, float]:
        """Lateral error is positive left of the path direction."""
        best_s = 0.0
        best_lateral = math.inf
        best_abs = math.inf
        for index, (a, b) in enumerate(zip(self._points, self._points[1:])):
            closest, along, perp = project_on_segment(query, a, b)
            if perp < best_abs:
                seg = b - a
                rel = query - closest
                cross = seg.x * rel.y - seg.y * rel.x
                sign = 1.0 if cross >= 0.0 else -1.0
                best_abs = perp
                best_lateral = perp * sign
                best_s = self._cumulative[index] + along
        return best_s, best_lateral

    def point_at(self, arc_length: float) -> Point:
        """Wraps for a closed path; clamps to the ends for an open one."""
        total = self.length
        if self.closed and total > 0.0:
            arc_length %= total
        arc_length = min(total, max(0.0, arc_length))
        for index, (start_s, end_s) in enumerate(zip(self._cumulative, self._cumulative[1:])):
            if arc_length <= end_s:
                span = end_s - start_s
                t = 0.0 if span < 1e-12 else (arc_length - start_s) / span
                a, b = self._points[index], self._points[index + 1]
                return Point(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t)
        return self._points[-1]

    def heading_at(self, arc_length: float) -> float:
        ahead = self.point_at(arc_length + 1e-3)
        behind = self.point_at(max(0.0, arc_length - 1e-3))
        return math.atan2(ahead.y - behind.y, ahead.x - behind.x)

    def offset(self, distance_m: float) -> "Polyline":
        """Shifted left of travel. Each vertex moves along its averaged normal: exact on
        straights, fine for this track's shallow corners, not a general polygon offset.
        """
        count = len(self._points)
        shifted: List[Point] = []
        for index, point in enumerate(self._points):
            normals: List[Point] = []
            if index > 0:
                normals.append(self._normal(self._points[index - 1], point))
            if index < count - 1:
                normals.append(self._normal(point, self._points[index + 1]))
            if self.closed and index in (0, count - 1):
                normals = [
                    self._normal(self._points[-2], self._points[0]),
                    self._normal(self._points[0], self._points[1]),
                ]
            nx = sum(n.x for n in normals) / len(normals)
            ny = sum(n.y for n in normals) / len(normals)
            scale = math.hypot(nx, ny)
            if scale > 1e-9:
                nx, ny = nx / scale, ny / scale
            shifted.append(Point(point.x + nx * distance_m, point.y + ny * distance_m))
        return Polyline(shifted, closed=self.closed)

    def rounded(self, radius: float, samples_per_corner: int = 8) -> "Polyline":
        """Fillet sharp vertices: handed a right angle, pure pursuit cuts it and reports a geometry
        error as tracking error. Clamped to half of each adjacent segment.
        """
        if radius <= 0.0:
            return self

        base = self._points[:-1] if self.closed else self._points
        count = len(base)
        out: List[Point] = []

        for index, vertex in enumerate(base):
            has_prev = self.closed or index > 0
            has_next = self.closed or index < count - 1
            if not (has_prev and has_next):
                out.append(vertex)
                continue

            previous = base[(index - 1) % count]
            following = base[(index + 1) % count]
            incoming = self._unit(previous, vertex)
            outgoing = self._unit(vertex, following)

            cross = incoming.x * outgoing.y - incoming.y * outgoing.x
            dot = incoming.x * outgoing.x + incoming.y * outgoing.y
            turn = math.atan2(cross, dot)
            if abs(turn) < 1e-6 or abs(abs(turn) - math.pi) < 1e-6:
                # Straight through, or a reversal a fillet cannot help.
                out.append(vertex)
                continue

            half = abs(turn) / 2.0
            tangent = min(
                radius * math.tan(half),
                0.5 * distance(previous, vertex),
                0.5 * distance(vertex, following),
            )
            effective = tangent / math.tan(half)

            start = Point(vertex.x - incoming.x * tangent, vertex.y - incoming.y * tangent)
            sign = 1.0 if turn > 0.0 else -1.0
            normal = Point(-incoming.y * sign, incoming.x * sign)
            centre = Point(start.x + normal.x * effective, start.y + normal.y * effective)

            begin = math.atan2(start.y - centre.y, start.x - centre.x)
            for step in range(samples_per_corner + 1):
                angle = begin + turn * (step / samples_per_corner)
                out.append(
                    Point(
                        centre.x + effective * math.cos(angle),
                        centre.y + effective * math.sin(angle),
                    )
                )

        return Polyline(out, closed=self.closed)

    @staticmethod
    def _unit(a: Point, b: Point) -> Point:
        dx, dy = b.x - a.x, b.y - a.y
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return Point(0.0, 0.0)
        return Point(dx / length, dy / length)

    @staticmethod
    def _normal(a: Point, b: Point) -> Point:
        dx, dy = b.x - a.x, b.y - a.y
        length = math.hypot(dx, dy)
        if length < 1e-9:
            return Point(0.0, 0.0)
        return Point(-dy / length, dx / length)


def pure_pursuit_steer(
    pose: Pose2D,
    target: Point,
    wheelbase: float,
    max_steer: float = math.radians(25.0),
) -> float:
    """Pure pursuit. Zero when the target collapses onto the car (misconfigured look-ahead)."""
    local = pose.to_body(target)
    range_sq = local.x * local.x + local.y * local.y
    if range_sq < 1e-9:
        return 0.0
    curvature = 2.0 * local.y / range_sq
    steer = math.atan(wheelbase * curvature)
    return min(max_steer, max(-max_steer, steer))
