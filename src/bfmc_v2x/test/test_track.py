"""The YAML is the only description of the world; everything derived from it must agree."""

from __future__ import annotations

import math

import pytest

from bfmc_v2x.geometry import Point, distance
from bfmc_v2x.protocol import ObstacleKind
from bfmc_v2x.track import Track


def test_track_loads(track):
    assert track.name == "mini_city"
    assert len(track.nodes) == 9
    assert len(track.edges) == 12
    assert set(track.route_specs) == {"ego", "peer"}


def test_every_road_joins_two_real_nodes(track):
    for edge in track.edges:
        assert edge.start in track.nodes
        assert edge.end in track.nodes


def test_routes_are_closed_tours(track):
    for name in track.route_specs:
        spec = track.route_specs[name]
        assert spec[0] == spec[-1], "route {} does not return to its start".format(name)


def test_both_routes_pass_through_the_junction(track):
    for name in track.route_specs:
        assert "X" in track.route_specs[name]


def test_a_route_that_uses_a_missing_road_is_rejected():
    with pytest.raises(ValueError, match="does not exist"):
        Track.from_dict(
            {
                "nodes": {"A": [0, 0], "B": [1, 0], "C": [1, 1]},
                "edges": [["A", "B"]],
                "routes": {"bad": ["A", "C"]},
            }
        )


def test_a_junction_must_be_a_node():
    with pytest.raises(ValueError, match="is not a node"):
        Track.from_dict(
            {
                "nodes": {"A": [0, 0], "B": [1, 0]},
                "edges": [["A", "B"]],
                "junctions": {"Z": {"radius": 0.5}},
            }
        )


# ---- lanes ---------------------------------------------------------------


def test_vehicles_spawn_on_a_straight(track):
    """Arc zero is a filleted corner."""
    for name in track.route_specs:
        route = track.build_route(name)
        _, heading = route.start_pose()
        # Every road on this map runs due north or due east.
        assert min(
            abs(math.remainder(heading - axis, 2 * math.pi))
            for axis in (0.0, math.pi / 2, math.pi, -math.pi / 2)
        ) < 1e-6, "{} spawns at {:.1f} deg".format(name, math.degrees(heading))


def test_both_vehicles_start_the_same_distance_from_the_junction(track):
    gaps = [
        track.build_route(name).next_stop_line(track.build_route(name).spawn_s)[1]
        for name in ("ego", "peer")
    ]

    assert gaps[0] == pytest.approx(gaps[1], abs=0.01)
    assert gaps[0] > 2.0, "too close to the junction to show the approach"


@pytest.mark.parametrize("route_name", ["ego", "peer"])
def test_the_lane_is_always_to_the_right_of_travel(track, route_name):
    """Right-hand traffic."""
    route = track.build_route(route_name)
    roads = track.road_segments()
    offset = track.lane_width / 2.0

    checked = 0
    step = 0.1
    for index in range(int(route.length / step)):
        arc = index * step
        point = route.lane.point_at(arc)
        heading = route.lane.heading_at(arc)

        # Corner fillets are legitimately far from any one centreline.
        centre, gap = _nearest_road_point(point, roads)
        if abs(gap - offset) > 0.02:
            continue

        # Driving on the right puts the road centre to the left.
        to_centre = Point(centre.x - point.x, centre.y - point.y)
        cross = math.cos(heading) * to_centre.y - math.sin(heading) * to_centre.x
        assert cross > 0.0, (
            "{} drives on the wrong side at ({:.2f}, {:.2f})".format(
                route_name, point.x, point.y
            )
        )
        checked += 1

    # Fillets and the run along the crossing road's centreline are excluded;
    # several metres of straight remain on every road.
    assert checked > 25, "only {} straight-run samples checked".format(checked)


def _nearest_road_point(point, roads):
    from bfmc_v2x.geometry import project_on_segment

    best, best_gap = None, math.inf
    for start, end in roads:
        closest, _, gap = project_on_segment(point, start, end)
        if gap < best_gap:
            best, best_gap = closest, gap
    return best, best_gap


def test_the_lane_stays_on_the_carriageway(track, ego_route):
    half_road = track.road_width / 2.0
    roads = track.road_segments()

    steps = int(ego_route.length / 0.05)
    for step in range(steps):
        point = ego_route.lane.point_at(step * 0.05)
        clearance = min(
            _distance_to_segment(point, start, end) for start, end in roads
        )
        assert clearance <= half_road + 1e-6, (
            "lane leaves the road at {:.2f}, {:.2f}".format(point.x, point.y)
        )


def _distance_to_segment(point, start, end):
    from bfmc_v2x.geometry import project_on_segment

    return project_on_segment(point, start, end)[2]


def test_corner_fillets_are_drivable(track, ego_route):
    # wheelbase 0.26 m at 25 deg of steer
    min_radius = 0.26 / math.tan(math.radians(25.0))

    # A 0.25 m baseline: shorter ones report the sampling chords' kinks.
    worst = math.inf
    step = 0.25
    for index in range(int(ego_route.length / step) - 2):
        a = ego_route.lane.point_at(index * step)
        b = ego_route.lane.point_at((index + 1) * step)
        c = ego_route.lane.point_at((index + 2) * step)
        worst = min(worst, _circumradius(a, b, c))

    assert worst >= min_radius * 0.95, "tightest radius {:.3f} m".format(worst)


def _circumradius(a, b, c):
    ab, bc, ca = distance(a, b), distance(b, c), distance(c, a)
    area = abs((b.x - a.x) * (c.y - a.y) - (c.x - a.x) * (b.y - a.y)) / 2.0
    if area < 1e-12:
        return math.inf
    return (ab * bc * ca) / (4.0 * area)


# ---- stop lines ----------------------------------------------------------


def test_the_stop_line_sits_before_the_junction(track, ego_route):
    (line,) = ego_route.stop_lines

    assert line.junction == "X"
    assert line.arc_s < line.conflict_s
    assert line.conflict_s - line.arc_s == pytest.approx(
        track.junctions["X"].radius + track.stop_line_setback, abs=1e-6
    )


def test_the_stop_line_is_set_back_from_the_conflict_zone(track, ego_route):
    (line,) = ego_route.stop_lines

    assert track.stop_line_setback > 0.0
    assert line.conflict_s - line.arc_s > line.zone_radius


def test_each_route_gets_the_light_for_its_own_approach(track):
    (ego_line,) = track.build_route("ego").stop_lines
    (peer_line,) = track.build_route("peer").stop_lines

    assert ego_line.light_id == 1     # approach_from S
    assert peer_line.light_id == 4    # approach_from E


def test_the_next_stop_line_wraps_around_the_lap(ego_route):
    (line,) = ego_route.stop_lines

    just_before = ego_route.next_stop_line(line.arc_s - 0.4)
    assert just_before[0] is line
    assert just_before[1] == pytest.approx(0.4, abs=1e-6)

    just_after = ego_route.next_stop_line(line.arc_s + 1.0)
    assert just_after[0] is line
    assert just_after[1] == pytest.approx(ego_route.length - 1.0, abs=1e-6)


def test_a_route_with_no_junction_has_no_stop_line():
    track = Track.from_dict(
        {
            "nodes": {"A": [0, 0], "B": [2, 0], "C": [2, 2], "D": [0, 2]},
            "edges": [["A", "B"], ["B", "C"], ["C", "D"], ["D", "A"]],
            "routes": {"loop": ["A", "B", "C", "D", "A"]},
        }
    )
    route = track.build_route("loop")

    assert route.stop_lines == ()
    assert route.next_stop_line(0.0) == (None, math.inf)


# ---- lights and extras ---------------------------------------------------


def test_the_junction_has_one_light_per_approach(track):
    placements = track.light_placements()

    assert [p.light_id for p in placements] == [1, 2, 3, 4]
    assert {p.approach_from for p in placements} == {"S", "N", "W", "E"}


def test_opposing_approaches_share_a_phase_offset(track):
    offsets = {p.approach_from: p.offset_s for p in track.light_placements()}

    assert offsets["S"] == offsets["N"]
    assert offsets["W"] == offsets["E"]
    assert offsets["S"] != offsets["W"]


def test_lights_stand_beside_the_road_not_in_it(track):
    half_road = track.road_width / 2.0

    for placement in track.light_placements():
        position, _ = track.light_pose(placement)
        clearance = min(
            _distance_to_segment(position, start, end)
            for start, end in track.road_segments()
        )
        assert clearance > half_road, "light {} stands in the road".format(placement.light_id)


def test_a_light_faces_the_traffic_it_controls(track):
    placement = next(p for p in track.light_placements() if p.approach_from == "S")
    _, facing = track.light_pose(placement)

    # Traffic from the south drives north, so the light looks south.
    assert math.cos(facing) == pytest.approx(0.0, abs=1e-6)
    assert math.sin(facing) < 0.0


def test_obstacles_load_with_their_classes(track):
    kinds = {o.obstacle_id: o.kind for o in track.obstacles}

    assert len(track.obstacles) == 5
    assert kinds[1] is ObstacleKind.STATIC_CAR
    assert kinds[2] is ObstacleKind.PEDESTRIAN
    assert kinds[4] is ObstacleKind.STOP_SIGN


def test_obstacles_are_reachable_from_a_route(track, ego_route):
    for obstacle in track.obstacles:
        _, lateral = ego_route.locate(obstacle.point)
        assert abs(lateral) < 1.6, (
            "obstacle {} is {:.2f} m off the ego route".format(obstacle.obstacle_id, lateral)
        )


def test_extent_covers_every_node_with_margin(track):
    min_x, min_y, max_x, max_y = track.extent

    for node in track.nodes.values():
        assert min_x < node.x < max_x
        assert min_y < node.y < max_y
