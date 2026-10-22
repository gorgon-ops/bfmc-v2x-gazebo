"""The generated world must stay loadable (a literal ``--`` in an XML comment already broke
it once) and must match the map, or a correct car looks broken.
"""

from __future__ import annotations

import importlib.util
import math
import pathlib
import sys
import xml.etree.ElementTree as ET

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
REPO_ROOT = PACKAGE_ROOT.parents[1]

sys.path.insert(0, str(REPO_ROOT / "src" / "bfmc_v2x"))

from bfmc_v2x.track import Track  # noqa: E402

TRACK_YAML = PACKAGE_ROOT / "track" / "mini_city.yaml"
WORLD_SDF = PACKAGE_ROOT / "worlds" / "mini_city.sdf"
GENERATOR = PACKAGE_ROOT / "scripts" / "generate_world.py"


def _load_generator():
    spec = importlib.util.spec_from_file_location("generate_world", GENERATOR)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def track() -> Track:
    return Track.from_yaml(str(TRACK_YAML))


@pytest.fixture(scope="module")
def world_xml() -> ET.ElementTree:
    if not WORLD_SDF.exists():
        pytest.skip("world not generated")
    return ET.parse(str(WORLD_SDF))


def _models(world: ET.ElementTree):
    return {m.get("name"): m for m in world.getroot().iter("model")}


def _pose(model) -> tuple:
    return tuple(float(v) for v in model.find("pose").text.split())


# ---- validity ------------------------------------------------------------


def test_the_world_is_well_formed_xml(world_xml):
    assert world_xml.getroot().tag == "sdf"


def test_no_xml_comment_contains_a_double_hyphen():
    """``--`` inside a comment is illegal XML."""
    text = WORLD_SDF.read_text(encoding="utf-8")
    depth = 0
    index = 0
    while True:
        start = text.find("<!--", index)
        if start < 0:
            break
        end = text.find("-->", start + 4)
        assert end > 0, "unterminated comment"
        body = text[start + 4 : end]
        assert "--" not in body, "illegal '--' in comment: {!r}".format(body[:60])
        index = end + 3
        depth += 1
    assert depth > 0, "expected at least the generated-file banner"


def test_the_world_is_up_to_date():
    """Regenerating must be a no-op."""
    module = _load_generator()
    expected = module.build_world(Track.from_yaml(str(TRACK_YAML)))

    assert WORLD_SDF.read_text(encoding="utf-8") == expected, (
        "worlds/mini_city.sdf is stale; run scripts/generate_world.py"
    )


def test_required_systems_are_present(world_xml):
    plugins = {p.get("name") for p in world_xml.getroot().iter("plugin")}

    assert "gz::sim::systems::Physics" in plugins
    assert "gz::sim::systems::Imu" in plugins, "IMU system missing: no gyro, no filter"
    assert "gz::sim::systems::SceneBroadcaster" in plugins


# ---- the world matches the map -------------------------------------------


def test_every_road_is_built(track, world_xml):
    models = _models(world_xml)
    for index in range(len(track.edges)):
        assert "road_{}".format(index) in models


def test_stop_lines_are_painted_where_the_vehicle_brakes(track, world_xml):
    models = _models(world_xml)

    for name, route in track.routes().items():
        for index, line in enumerate(route.stop_lines):
            model = models["stop_line_{}_{}".format(name, index)]
            x, y = _pose(model)[:2]
            expected = route.lane.point_at(line.arc_s)

            assert math.hypot(x - expected.x, y - expected.y) < 1e-3, (
                "{} stop line painted at ({:.3f}, {:.3f}) but braked for at "
                "({:.3f}, {:.3f})".format(name, x, y, expected.x, expected.y)
            )


def test_traffic_lights_are_built_where_the_map_puts_them(track, world_xml):
    models = _models(world_xml)

    for placement in track.light_placements():
        model = models["light_post_{}".format(placement.light_id)]
        x, y = _pose(model)[:2]
        expected, _ = track.light_pose(placement)

        assert math.hypot(x - expected.x, y - expected.y) < 1e-3


def test_every_obstacle_exists_in_the_world(track, world_xml):
    models = _models(world_xml)

    for obstacle in track.obstacles:
        name = "obstacle_{}_{}".format(obstacle.obstacle_id, obstacle.kind.name.lower())
        assert name in models, "obstacle {} is scored but not built".format(name)

        x, y = _pose(models[name])[:2]
        assert math.hypot(x - obstacle.x, y - obstacle.y) < 1e-3


def test_the_walls_enclose_the_track(track, world_xml):
    models = _models(world_xml)
    min_x, min_y, max_x, max_y = track.extent

    walls = [models["wall_{}".format(i)] for i in range(4)]
    xs = [_pose(w)[0] for w in walls]
    ys = [_pose(w)[1] for w in walls]

    assert min(xs) == pytest.approx(min_x, abs=1e-3)
    assert max(xs) == pytest.approx(max_x, abs=1e-3)
    assert min(ys) == pytest.approx(min_y, abs=1e-3)
    assert max(ys) == pytest.approx(max_y, abs=1e-3)


def test_nothing_obstructs_the_driving_lanes(track, world_xml):
    """An obstacle in the carriageway would stop the demo and look like a controller bug."""
    models = _models(world_xml)

    for obstacle in track.obstacles:
        name = "obstacle_{}_{}".format(obstacle.obstacle_id, obstacle.kind.name.lower())
        has_collision = models[name].find(".//collision") is not None
        if not has_collision:
            continue

        for route in track.routes().values():
            _, lateral = route.locate(obstacle.point)
            assert abs(lateral) > track.lane_width * 0.5, (
                "solid obstacle {} sits {:.3f} m from the {} lane centre".format(
                    obstacle.obstacle_id, abs(lateral), route.name
                )
            )
