# Models

Deliberately empty.

Every object in the world -- roads, paint, junction boxes, traffic lights,
obstacles, walls -- is generated as inline SDF by
`scripts/generate_world.py` from `track/mini_city.yaml`. There are no mesh
files and no external model dependencies, so the world loads on a fresh
machine with no asset download and no `GZ_SIM_RESOURCE_PATH` surprises.

This directory exists so that a model added later (a textured building, a
proper vehicle mesh) has an obvious home, and so the resource-path hook has
something to point at.
