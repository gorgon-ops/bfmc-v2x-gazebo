#!/usr/bin/env python3
"""Build the Gazebo world from the track definition.

The car plans against the YAML and drives in the SDF; deriving one from the other
makes them unable to disagree.

    python3 generate_world.py                    # regenerate in place
    python3 generate_world.py --check            # fail if stale (used by CI)
    python3 generate_world.py --track other.yaml --output other.sdf
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import List, Optional, Sequence, Tuple

HERE = os.path.dirname(os.path.abspath(__file__))
PACKAGE = os.path.dirname(HERE)
# The map reader, without the core package installed.
sys.path.insert(0, os.path.join(os.path.dirname(PACKAGE), "bfmc_v2x"))

from bfmc_v2x.geometry import Point, distance  # noqa: E402
from bfmc_v2x.protocol import ObstacleKind, SemaphoreState  # noqa: E402
from bfmc_v2x.track import Track  # noqa: E402

DEFAULT_TRACK = os.path.join(PACKAGE, "track", "mini_city.yaml")
DEFAULT_OUTPUT = os.path.join(PACKAGE, "worlds", "mini_city.sdf")

#: Thin boxes rather than textures: no assets, and every renderer draws them.
ROAD_THICKNESS = 0.004
PAINT_THICKNESS = 0.002
PAINT_WIDTH = 0.020
WALL_HEIGHT = 0.25
WALL_THICKNESS = 0.04

ASPHALT = (0.22, 0.22, 0.24)
PAINT = (0.92, 0.92, 0.92)
JUNCTION = (0.26, 0.26, 0.28)
WALL = (0.45, 0.45, 0.50)

OBSTACLE_STYLE = {
    ObstacleKind.STATIC_CAR: ((0.30, 0.16, 0.12), (0.75, 0.20, 0.20)),
    ObstacleKind.PEDESTRIAN: ((0.12, 0.12, 0.34), (0.90, 0.70, 0.45)),
    ObstacleKind.ROADBLOCK: ((0.16, 0.16, 0.22), (0.95, 0.55, 0.10)),
    ObstacleKind.STOP_SIGN: ((0.04, 0.12, 0.26), (0.85, 0.12, 0.12)),
    ObstacleKind.CROSSWALK: ((0.36, 0.30, 0.004), (0.92, 0.92, 0.92)),
    ObstacleKind.PARKING_SPOT: ((0.40, 0.24, 0.004), (0.70, 0.70, 0.95)),
}


def _material(colour: Tuple[float, float, float], alpha: float = 1.0) -> str:
    r, g, b = colour
    return (
        "<material>"
        "<ambient>{r:.3f} {g:.3f} {b:.3f} {a:.2f}</ambient>"
        "<diffuse>{r:.3f} {g:.3f} {b:.3f} {a:.2f}</diffuse>"
        "<specular>0.1 0.1 0.1 1</specular>"
        "</material>".format(r=r, g=g, b=b, a=alpha)
    )


def _box(
    name: str,
    centre: Tuple[float, float, float],
    size: Tuple[float, float, float],
    yaw: float,
    colour: Tuple[float, float, float],
    alpha: float = 1.0,
    collide: bool = False,
) -> str:
    """Visual-only unless ``collide``: paint has no physics."""
    geometry = "<geometry><box><size>{:.4f} {:.4f} {:.4f}</size></box></geometry>".format(*size)
    collision = (
        "<collision name='collision'>{}</collision>".format(geometry) if collide else ""
    )
    return (
        "    <model name='{name}'>\n"
        "      <static>true</static>\n"
        "      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 {yaw:.5f}</pose>\n"
        "      <link name='link'>\n"
        "        {collision}\n"
        "        <visual name='visual'>{geometry}{material}</visual>\n"
        "      </link>\n"
        "    </model>\n".format(
            name=name,
            x=centre[0],
            y=centre[1],
            z=centre[2],
            yaw=yaw,
            geometry=geometry,
            collision=collision,
            material=_material(colour, alpha),
        )
    )


def _cylinder(
    name: str,
    centre: Tuple[float, float, float],
    radius: float,
    length: float,
    colour: Tuple[float, float, float],
    alpha: float = 1.0,
) -> str:
    geometry = (
        "<geometry><cylinder><radius>{:.4f}</radius>"
        "<length>{:.4f}</length></cylinder></geometry>".format(radius, length)
    )
    return (
        "    <model name='{name}'>\n"
        "      <static>true</static>\n"
        "      <pose>{x:.4f} {y:.4f} {z:.4f} 0 0 0</pose>\n"
        "      <link name='link'>\n"
        "        <visual name='visual'>{geometry}{material}</visual>\n"
        "      </link>\n"
        "    </model>\n".format(
            name=name,
            x=centre[0],
            y=centre[1],
            z=centre[2],
            geometry=geometry,
            material=_material(colour, alpha),
        )
    )


def _segment_box(
    name: str,
    start: Point,
    end: Point,
    width: float,
    thickness: float,
    z: float,
    colour: Tuple[float, float, float],
    alpha: float = 1.0,
    extend: float = 0.0,
) -> str:
    dx, dy = end.x - start.x, end.y - start.y
    length = math.hypot(dx, dy) + extend
    yaw = math.atan2(dy, dx)
    centre = ((start.x + end.x) / 2.0, (start.y + end.y) / 2.0, z)
    return _box(name, centre, (length, width, thickness), yaw, colour, alpha)


def _dashed_centre_line(start: Point, end: Point, index: int) -> List[str]:
    """Cosmetic, but shows at a glance which side of the centre line each car is on."""
    span = distance(start, end)
    dash, gap = 0.16, 0.12
    period = dash + gap
    # Leave the junction ends clear.
    margin = 0.55
    usable = span - 2 * margin
    if usable <= dash:
        return []

    heading = math.atan2(end.y - start.y, end.x - start.x)
    ux, uy = math.cos(heading), math.sin(heading)
    out: List[str] = []
    count = int(usable // period)
    offset = margin + (usable - count * period + gap) / 2.0

    for n in range(count):
        centre_s = offset + n * period + dash / 2.0
        cx = start.x + ux * centre_s
        cy = start.y + uy * centre_s
        out.append(
            _box(
                "lane_dash_{}_{}".format(index, n),
                (cx, cy, ROAD_THICKNESS + PAINT_THICKNESS / 2.0),
                (dash, PAINT_WIDTH, PAINT_THICKNESS),
                heading,
                PAINT,
            )
        )
    return out


def build_world(track: Track) -> str:
    parts: List[str] = []
    min_x, min_y, max_x, max_y = track.extent
    width, height = max_x - min_x, max_y - min_y
    centre_x, centre_y = (min_x + max_x) / 2.0, (min_y + max_y) / 2.0

    # ---- carriageway ---------------------------------------------------------
    for index, (start, end) in enumerate(track.road_segments()):
        parts.append(
            _segment_box(
                "road_{}".format(index),
                start,
                end,
                track.road_width,
                ROAD_THICKNESS,
                ROAD_THICKNESS / 2.0,
                ASPHALT,
                # Overhang by a road width so corners and tees join without a notch.
                extend=track.road_width,
            )
        )

    for index, (start, end) in enumerate(track.road_segments()):
        parts.extend(_dashed_centre_line(start, end, index))

    # ---- junctions -----------------------------------------------------------
    for name, point, radius in track.junction_points():
        parts.append(
            _box(
                "junction_{}".format(name),
                (point.x, point.y, ROAD_THICKNESS + 0.0005),
                (2 * radius, 2 * radius, PAINT_THICKNESS),
                0.0,
                JUNCTION,
            )
        )

    # ---- stop lines ----------------------------------------------------------
    # From the routes, so the paint is where the car actually brakes.
    for route_name, route in sorted(track.routes().items()):
        for index, line in enumerate(route.stop_lines):
            point = route.lane.point_at(line.arc_s)
            heading = route.lane.heading_at(line.arc_s)
            parts.append(
                _box(
                    "stop_line_{}_{}".format(route_name, index),
                    (point.x, point.y, ROAD_THICKNESS + PAINT_THICKNESS / 2.0),
                    (0.03, track.lane_width, PAINT_THICKNESS),
                    heading,
                    PAINT,
                )
            )

    # ---- traffic lights ------------------------------------------------------
    for placement in track.light_placements():
        position, facing = track.light_pose(placement)
        parts.append(
            _box(
                "light_post_{}".format(placement.light_id),
                (position.x, position.y, 0.11),
                (0.03, 0.03, 0.22),
                facing,
                (0.20, 0.20, 0.22),
                collide=True,
            )
        )
        parts.append(
            _box(
                "light_head_{}".format(placement.light_id),
                (position.x, position.y, 0.27),
                (0.05, 0.07, 0.15),
                facing,
                (0.12, 0.12, 0.14),
            )
        )
        # Dim, fixed lamps: Gazebo cannot recolour at run time without a plugin, so
        # the live state is the RViz marker.
        for lamp, (offset, colour) in enumerate(
            ((0.045, SemaphoreState.RED), (0.0, SemaphoreState.YELLOW), (-0.045, SemaphoreState.GREEN))
        ):
            rgb = {
                SemaphoreState.RED: (0.55, 0.12, 0.12),
                SemaphoreState.YELLOW: (0.55, 0.45, 0.10),
                SemaphoreState.GREEN: (0.12, 0.50, 0.20),
            }[colour]
            parts.append(
                _cylinder(
                    "light_lamp_{}_{}".format(placement.light_id, lamp),
                    (position.x, position.y, 0.27 + offset),
                    0.016,
                    0.01,
                    rgb,
                )
            )

    # ---- obstacles -----------------------------------------------------------
    for obstacle in track.obstacles:
        size, colour = OBSTACLE_STYLE[obstacle.kind]
        flat = obstacle.kind in (ObstacleKind.CROSSWALK, ObstacleKind.PARKING_SPOT)
        parts.append(
            _box(
                "obstacle_{}_{}".format(obstacle.obstacle_id, obstacle.kind.name.lower()),
                (obstacle.x, obstacle.y, ROAD_THICKNESS + size[2] / 2.0),
                size,
                0.0,
                colour,
                collide=not flat,
            )
        )

    # ---- walls ---------------------------------------------------------------
    for index, (cx, cy, sx, sy) in enumerate(
        (
            (centre_x, min_y, width, WALL_THICKNESS),
            (centre_x, max_y, width, WALL_THICKNESS),
            (min_x, centre_y, WALL_THICKNESS, height),
            (max_x, centre_y, WALL_THICKNESS, height),
        )
    ):
        parts.append(
            _box(
                "wall_{}".format(index),
                (cx, cy, WALL_HEIGHT / 2.0),
                (sx, sy, WALL_HEIGHT),
                0.0,
                WALL,
                collide=True,
            )
        )

    return _WORLD_TEMPLATE.format(
        name=track.name,
        camera_x=centre_x,
        camera_y=centre_y - height * 0.75,
        camera_z=max(width, height) * 0.85,
        models="".join(parts),
    )


_WORLD_TEMPLATE = """<?xml version="1.0" ?>
<!--
  GENERATED FILE. Do not edit.

  Produced by bfmc_sim/scripts/generate_world.py from
  bfmc_sim/track/mini_city.yaml. Change the track definition and re-run the
  generator; edits made here are lost the next time anyone does.
-->
<sdf version="1.9">
  <world name="{name}">

    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>

    <plugin filename="gz-sim-physics-system"
            name="gz::sim::systems::Physics"/>
    <plugin filename="gz-sim-user-commands-system"
            name="gz::sim::systems::UserCommands"/>
    <plugin filename="gz-sim-scene-broadcaster-system"
            name="gz::sim::systems::SceneBroadcaster"/>
    <plugin filename="gz-sim-contact-system"
            name="gz::sim::systems::Contact"/>
    <!-- Required for the IMU on each vehicle to produce readings. -->
    <plugin filename="gz-sim-imu-system"
            name="gz::sim::systems::Imu"/>
    <plugin filename="gz-sim-sensors-system"
            name="gz::sim::systems::Sensors">
      <render_engine>ogre2</render_engine>
    </plugin>

    <gravity>0 0 -9.8</gravity>

    <scene>
      <ambient>0.55 0.55 0.58 1</ambient>
      <background>0.75 0.80 0.86 1</background>
      <grid>false</grid>
      <shadows>true</shadows>
    </scene>

    <light type="directional" name="sun">
      <cast_shadows>true</cast_shadows>
      <pose>0 0 10 0 0 0</pose>
      <diffuse>0.9 0.9 0.9 1</diffuse>
      <specular>0.25 0.25 0.25 1</specular>
      <direction>-0.4 0.3 -0.9</direction>
    </light>

    <gui fullscreen="0">
      <plugin filename="MinimalScene" name="3D View">
        <gz-gui>
          <property type="string" key="state">docked</property>
        </gz-gui>
        <engine>ogre2</engine>
        <scene>scene</scene>
        <ambient_light>0.55 0.55 0.58</ambient_light>
        <background_color>0.75 0.80 0.86</background_color>
        <camera_pose>{camera_x:.2f} {camera_y:.2f} {camera_z:.2f} 0 0.95 1.5708</camera_pose>
      </plugin>
      <plugin filename="GzSceneManager" name="Scene Manager"/>
      <plugin filename="InteractiveViewControl" name="Interactive view control"/>
      <plugin filename="CameraTracking" name="Camera Tracking"/>
      <plugin filename="WorldControl" name="World control">
        <gz-gui>
          <property type="bool" key="resizable">false</property>
          <property type="double" key="height">72</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View">
            <line own="left" target="left"/>
            <line own="bottom" target="bottom"/>
          </anchors>
        </gz-gui>
        <play_pause>true</play_pause>
        <step>true</step>
        <use_event>true</use_event>
      </plugin>
      <plugin filename="WorldStats" name="World stats">
        <gz-gui>
          <property type="bool" key="resizable">false</property>
          <property type="string" key="state">floating</property>
          <anchors target="3D View">
            <line own="right" target="right"/>
            <line own="bottom" target="bottom"/>
          </anchors>
        </gz-gui>
        <sim_time>true</sim_time>
        <real_time_factor>true</real_time_factor>
      </plugin>
    </gui>

    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
        <visual name="visual">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
          <material>
            <ambient>0.35 0.42 0.33 1</ambient>
            <diffuse>0.35 0.42 0.33 1</diffuse>
          </material>
        </visual>
      </link>
    </model>

{models}
  </world>
</sdf>
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--track", default=DEFAULT_TRACK)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--check",
        action="store_true",
        help="exit non-zero if the world on disk is not what this would write",
    )
    args = parser.parse_args(argv)

    track = Track.from_yaml(args.track)
    world = build_world(track)

    if args.check:
        if not os.path.exists(args.output):
            print("missing: {}".format(args.output), file=sys.stderr)
            return 1
        with open(args.output, "r", encoding="utf-8") as handle:
            if handle.read() != world:
                print(
                    "{} is stale -- re-run generate_world.py".format(args.output),
                    file=sys.stderr,
                )
                return 1
        print("{} is up to date".format(os.path.basename(args.output)))
        return 0

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8", newline="\n") as handle:
        handle.write(world)

    models = world.count("<model name=")
    print(
        "wrote {} ({} models) from {}".format(
            os.path.relpath(args.output), models, os.path.basename(args.track)
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
