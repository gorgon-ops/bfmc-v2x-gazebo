"""The path insertion lets pytest run straight from a clone; harmless under colcon."""

from __future__ import annotations

import pathlib
import sys

import pytest

PACKAGE_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

REPO_ROOT = PACKAGE_ROOT.parents[1]
TRACK_YAML = REPO_ROOT / "src" / "bfmc_sim" / "track" / "mini_city.yaml"


@pytest.fixture(scope="session")
def track_yaml_path() -> pathlib.Path:
    if not TRACK_YAML.exists():  # pragma: no cover - only if the repo is split up
        pytest.skip("track definition not found at {}".format(TRACK_YAML))
    return TRACK_YAML


@pytest.fixture(scope="session")
def track(track_yaml_path):
    from bfmc_v2x.track import Track

    return Track.from_yaml(str(track_yaml_path))


@pytest.fixture()
def ego_route(track):
    return track.build_route("ego")
