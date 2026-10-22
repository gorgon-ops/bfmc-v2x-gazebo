"""Track graph and routes. One YAML map generates the Gazebo world, the routes, stop lines,
lights and ground truth, so the simulation and the car's idea of it cannot drift apart.
Vehicles drive on the right of the centreline.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from .environment import GroundTruthObstacle
from .geometry import Point, Polyline
from .protocol import ObstacleKind

__all__ = [
    "TrackNode",
    "TrackEdge",
    "LightPlacement",
    "Junction",
    "StopLine",
    "Route",
    "Track",
]


@dataclass(frozen=True)
class TrackNode:
    name: str
    x: float
    y: float

    @property
    def point(self) -> Point:
        return Point(self.x, self.y)


@dataclass(frozen=True)
class TrackEdge:
    start: str
    end: str

    def key(self) -> Tuple[str, str]:
        """Order-independent, so A-B and B-A are the same road."""
        return (self.start, self.end) if self.start <= self.end else (self.end, self.start)


@dataclass(frozen=True)
class LightPlacement:
    light_id: int
    junction: str
    #: Node a vehicle is coming from when this light applies.
    approach_from: str
    #: Seconds into the shared phase plan.
    offset_s: float = 0.0


@dataclass(frozen=True)
class Junction:
    node: str
    radius: float
    lights: Tuple[LightPlacement, ...] = ()

    def light_for(self, approach_from: str) -> Optional[LightPlacement]:
        for light in self.lights:
            if light.approach_from == approach_from:
                return light
        return None


@dataclass(frozen=True)
class StopLine:
    #: Arc length along the lane where the vehicle must stop.
    arc_s: float
    junction: str
    #: Arc length of the junction centre.
    conflict_s: float
    conflict_point: Point
    light_id: Optional[int]
    zone_radius: float


class Route:
    def __init__(
        self,
        name: str,
        centreline: Polyline,
        lane: Polyline,
        stop_lines: Sequence[StopLine],
        spawn_s: float = 0.0,
    ) -> None:
        self.name = name
        self.centreline = centreline
        self.lane = lane
        self.stop_lines: Tuple[StopLine, ...] = tuple(sorted(stop_lines, key=lambda s: s.arc_s))
        self.spawn_s = spawn_s

    @property
    def length(self) -> float:
        return self.lane.length

    def start_pose(self, arc_s: Optional[float] = None) -> Tuple[Point, float]:
        """Defaults to ``spawn_s``: arc zero is a filleted corner, and spawning mid-turn starts
        every log with a steering transient.
        """
        arc = self.spawn_s if arc_s is None else arc_s
        return self.lane.point_at(arc), self.lane.heading_at(arc)

    def locate(self, position: Point) -> Tuple[float, float]:
        return self.lane.locate(position)

    def lookahead_point(self, arc_s: float, distance_m: float) -> Point:
        return self.lane.point_at(arc_s + distance_m)

    def next_stop_line(self, arc_s: float) -> Tuple[Optional[StopLine], float]:
        """Wraps around the loop: a car just before the finish still stops for the junction after
        it.
        """
        if not self.stop_lines:
            return None, math.inf
        best: Optional[StopLine] = None
        best_gap = math.inf
        for line in self.stop_lines:
            gap = line.arc_s - arc_s
            if gap < -0.5:
                # Behind us: the next lap's.
                gap += self.length
            if 0.0 <= gap < best_gap or (gap < 0.0 and gap > -0.5 and best is None):
                best, best_gap = line, gap
        return best, best_gap


class Track:
    def __init__(
        self,
        name: str,
        nodes: Dict[str, TrackNode],
        edges: Sequence[TrackEdge],
        junctions: Dict[str, Junction],
        obstacles: Sequence[GroundTruthObstacle],
        route_specs: Dict[str, Sequence[str]],
        road_width: float = 0.74,
        lane_width: float = 0.37,
        bounds_margin: float = 0.6,
        corner_radius: float = 0.6,
        stop_line_setback: float = 0.12,
        spawn_offsets: Optional[Dict[str, float]] = None,
    ) -> None:
        self.name = name
        self.nodes = dict(nodes)
        self.edges: Tuple[TrackEdge, ...] = tuple(edges)
        self.junctions = dict(junctions)
        self.obstacles: Tuple[GroundTruthObstacle, ...] = tuple(obstacles)
        self.route_specs = {k: tuple(v) for k, v in route_specs.items()}
        self.road_width = road_width
        self.lane_width = lane_width
        self.bounds_margin = bounds_margin
        self.corner_radius = corner_radius
        self.stop_line_setback = stop_line_setback
        self.spawn_offsets = dict(spawn_offsets or {})
        self._validate()

    @classmethod
    def from_dict(cls, data: Dict[str, object]) -> "Track":
        raw_nodes = data.get("nodes") or {}
        nodes = {
            name: TrackNode(name, float(xy[0]), float(xy[1]))
            for name, xy in raw_nodes.items()  # type: ignore[union-attr]
        }

        edges = [TrackEdge(str(a), str(b)) for a, b in data.get("edges", [])]  # type: ignore[misc]

        junctions: Dict[str, Junction] = {}
        for node_name, spec in (data.get("junctions") or {}).items():  # type: ignore[union-attr]
            lights = tuple(
                LightPlacement(
                    light_id=int(light["id"]),
                    junction=str(node_name),
                    approach_from=str(light["approach_from"]),
                    offset_s=float(light.get("offset_s", 0.0)),
                )
                for light in spec.get("lights", [])
            )
            junctions[str(node_name)] = Junction(
                node=str(node_name),
                radius=float(spec.get("radius", 0.55)),
                lights=lights,
            )

        obstacles = [
            GroundTruthObstacle(
                obstacle_id=int(item["id"]),
                kind=ObstacleKind[str(item["kind"]).upper()],
                x=float(item["x"]),
                y=float(item["y"]),
            )
            for item in data.get("obstacles", [])  # type: ignore[union-attr]
        ]

        # A bare node list, or a mapping with a spawn offset.
        routes: Dict[str, Sequence[str]] = {}
        spawns: Dict[str, float] = {}
        for name, spec in (data.get("routes") or {}).items():  # type: ignore[union-attr]
            if isinstance(spec, dict):
                routes[str(name)] = [str(n) for n in spec["nodes"]]
                spawns[str(name)] = float(spec.get("spawn_s", 0.0))
            else:
                routes[str(name)] = [str(n) for n in spec]
                spawns[str(name)] = 0.0

        return cls(
            name=str(data.get("name", "track")),
            nodes=nodes,
            edges=edges,
            junctions=junctions,
            obstacles=obstacles,
            route_specs=routes,
            road_width=float(data.get("road_width", 0.74)),
            lane_width=float(data.get("lane_width", 0.37)),
            bounds_margin=float(data.get("bounds_margin", 0.6)),
            corner_radius=float(data.get("corner_radius", 0.6)),
            stop_line_setback=float(data.get("stop_line_setback", 0.12)),
            spawn_offsets=spawns,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "Track":
        import yaml

        with open(path, "r", encoding="utf-8") as handle:
            return cls.from_dict(yaml.safe_load(handle))

    def _validate(self) -> None:
        for edge in self.edges:
            for end in (edge.start, edge.end):
                if end not in self.nodes:
                    raise ValueError("edge {}-{} names unknown node {}".format(edge.start, edge.end, end))
        for name in self.junctions:
            if name not in self.nodes:
                raise ValueError("junction {} is not a node".format(name))
        known = {edge.key() for edge in self.edges}
        for route_name, spec in self.route_specs.items():
            if len(spec) < 2:
                raise ValueError("route {} needs at least two nodes".format(route_name))
            for a, b in zip(spec, spec[1:]):
                key = (a, b) if a <= b else (b, a)
                if key not in known:
                    raise ValueError(
                        "route {} uses road {}-{}, which does not exist".format(route_name, a, b)
                    )

    @property
    def extent(self) -> Tuple[float, float, float, float]:
        xs = [n.x for n in self.nodes.values()]
        ys = [n.y for n in self.nodes.values()]
        margin = self.road_width * 0.5 + self.bounds_margin
        return (min(xs) - margin, min(ys) - margin, max(xs) + margin, max(ys) + margin)

    def light_placements(self) -> List[LightPlacement]:
        out: List[LightPlacement] = []
        for junction in self.junctions.values():
            out.extend(junction.lights)
        return sorted(out, key=lambda p: p.light_id)

    def light_pose(self, placement: LightPlacement) -> Tuple[Point, float]:
        """At the approach's stop line, offset to the right-hand kerb, facing oncoming traffic."""
        junction = self.junctions[placement.junction]
        centre = self.nodes[junction.node].point
        approach = self.nodes[placement.approach_from].point
        dx, dy = centre.x - approach.x, centre.y - approach.y
        heading = math.atan2(dy, dx)
        span = math.hypot(dx, dy)
        if span < 1e-9:
            return centre, heading
        back = junction.radius + 0.10
        kerb = self.road_width * 0.5 + 0.10
        base = Point(centre.x - dx / span * back, centre.y - dy / span * back)
        # Right of travel is the heading rotated by -90 degrees.
        right = Point(math.sin(heading), -math.cos(heading))
        return Point(base.x + right.x * kerb, base.y + right.y * kerb), heading + math.pi

    def build_route(self, name: str) -> Route:
        spec = self.route_specs.get(name)
        if spec is None:
            raise KeyError("no route named {}".format(name))

        closed = spec[0] == spec[-1]
        points = [self.nodes[n].point for n in spec]
        centreline = Polyline(points, closed=closed)
        # Negative offset is right of travel. Filleted after offsetting, so the fillets
        # sit on the lane actually driven.
        lane = centreline.offset(-self.lane_width * 0.5).rounded(self.corner_radius)

        stop_lines: List[StopLine] = []
        walk = spec[:-1] if closed else spec
        for index, node_name in enumerate(walk):
            junction = self.junctions.get(node_name)
            if junction is None:
                continue
            approach_from = spec[index - 1] if index > 0 else spec[-2]
            conflict_s, _ = lane.locate(self.nodes[node_name].point)
            # Set back from the zone: a stopped car creeps a few millimetres, and without a
            # setback "stopped at the line" would already be "in the junction".
            arc_s = conflict_s - junction.radius - self.stop_line_setback
            if closed:
                arc_s %= lane.length
            light = junction.light_for(approach_from)
            stop_lines.append(
                StopLine(
                    arc_s=arc_s,
                    junction=node_name,
                    conflict_s=conflict_s,
                    conflict_point=self.nodes[node_name].point,
                    light_id=light.light_id if light else None,
                    zone_radius=junction.radius,
                )
            )
        return Route(
            name,
            centreline,
            lane,
            stop_lines,
            spawn_s=self.spawn_offsets.get(name, 0.0) % lane.length,
        )

    def routes(self) -> Dict[str, Route]:
        return {name: self.build_route(name) for name in self.route_specs}

    def road_segments(self) -> List[Tuple[Point, Point]]:
        return [(self.nodes[e.start].point, self.nodes[e.end].point) for e in self.edges]

    def junction_points(self) -> List[Tuple[str, Point, float]]:
        return [
            (name, self.nodes[name].point, junction.radius)
            for name, junction in sorted(self.junctions.items())
        ]
