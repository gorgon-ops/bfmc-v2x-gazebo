# BFMC V2X — cooperative localisation in Gazebo

A Gazebo simulation of the Bosch Future Mobility Challenge track, built to
demonstrate one specific thing: **what a vehicle gains from the V2X link, and
what it costs to use it properly.**

Two 1:10 Ackermann cars drive a generated mini-city. Neither can see where it
is. Each has wheel encoders with a tyre-radius error, a MEMS gyro with a
bias, and a radio. Over that radio arrive absolute pose fixes from a
trackside camera, traffic-light states with countdowns, and the other car's
position. Everything the cars do — staying in lane, stopping at the line,
deciding who takes the junction, mapping obstacles for the organisers —
follows from fusing those three things well.

The interesting part is that fusing them *naively* makes things worse in a
way that is easy to miss, and most of this project is about that.

---

## The result

A pose fix describes where the vehicle was when the camera shutter opened,
not where it is when the datagram lands. At 4 Hz with 300 ms of latency, a
car doing 0.5 m/s has moved 15 cm by the time the fix arrives. Fuse it as
though it were current and you install a constant 15 cm bias that the filter
can never learn away — it looks like it is working, and it is quietly
dragging the estimate backwards along the direction of travel.

The fix is to rewind. Every fix carries its exposure timestamp; the filter
rewinds to the buffered state at that instant, applies the correction
*there*, and replays the odometry recorded since.

| Position error (RMS) | | |
|---|---|---|
| Wheel odometry alone | **0.613 m** | drifts without bound |
| V2X fused at arrival time | **0.166 m** | ≈ speed × latency, as predicted |
| V2X fused at exposure time | **0.008 m** | 20× better than naive |

<sub>Five seeds, 30 s runs, 0.5 m/s, 300 ms latency, 4 Hz fixes, 2 cm fix
noise, 0.02 rad/s gyro bias. These come from the deterministic Python
harness in `src/bfmc_v2x/test/test_ekf.py`, not from a Gazebo run — the
simulator adds real contact physics and its own IMU noise model, so live
figures differ. Reproduce them with `pytest src/bfmc_v2x/test/test_ekf.py`;
the live equivalents are printed by the metrics node at the end of every
run.</sub>

The stronger claim is a test rather than a number:
`test_replay_is_exactly_equivalent_to_never_having_lagged` asserts that the
replayed filter lands within 1e-9 of a filter that received the same fixes
instantly. Delayed fusion is not an approximation here — it is the same
arithmetic, reordered.

---

## What is in the demo

**Localisation.** A 5-state EKF (`x, y, θ, v, gyro bias`) fusing wheel speed,
gyro and V2X fixes, with out-of-sequence measurement replay, chi-square
gating for when the tracker locks onto the wrong car, and Joseph-form
updates. A dead-reckoning shadow runs on identical inputs all run long, so
every log answers "what would this have looked like without V2X?".

**Traffic lights.** Fixed-time phase plans broadcast with a countdown, not
just a colour. A car 3 m from a line that knows the red expires in 700 ms
keeps rolling instead of braking to a stop and pulling away again. A light
that goes off the air degrades to colour-only, and then to give-way — the
node refuses at startup to run a phase plan that would show green to two
crossing approaches.

**V2V.** Both routes cross the same signalled junction from perpendicular
approaches. Each car reduces itself and its peer to the time window each
will occupy the conflict zone and compares them. The tie-break is by vehicle
id — arbitrary, but *symmetric*, so exactly one of the two yields. Both
yielding is a deadlock; neither is a collision; there is no useful middle.

**Environment server.** Cars report the obstacles they drive past, placed on
the map using their own pose estimate. This closes the loop: a car that is
40 cm lost reports every obstacle 40 cm wrong and scores nothing, however
good its detector is. The scoreboard is the most honest end-to-end metric
here — it counts real things successfully put on a map.

---

## Running it

Ubuntu 24.04 → ROS 2 Jazzy + Gazebo Harmonic.

```bash
./scripts/bootstrap.sh   # ROS 2 Jazzy + Gazebo Harmonic (~3 GB, 10-20 min)
./scripts/build.sh       # generate world, colcon build, run the tests
./scripts/run_demo.sh    # Gazebo + RViz + both vehicles
```

Or without touching the host:

```bash
docker compose -f docker/compose.yaml run --rm test
docker compose -f docker/compose.yaml up sim
```

The core logic has no ROS or Gazebo dependency, so the interesting half is
testable on any machine with Python:

```bash
pip install numpy PyYAML pytest && pytest src/bfmc_v2x/test -q
```

### Variations worth trying

```bash
# Drive the ego car blind, on wheel odometry alone. Watch it leave the road.
./scripts/run_demo.sh ego_use_v2x:=false

# Take a traffic light off the air; the car falls back to give-way.
./scripts/run_demo.sh silent_lights:=[1]

# Make the radio worse and watch the gating counters climb.
./scripts/run_demo.sh fix_latency:=0.6 fix_dropout:=0.35
```

In RViz, the blue arrow is what the ego car believes after fusing V2X; the
red arrow is the same car on wheel odometry alone. The gap between them is
the whole argument.

Afterwards:

```bash
python3 scripts/plot_results.py            # reads ~/bfmc_runs/*.csv
```

---

## Layout

```
src/bfmc_v2x/       Pure Python. EKF, wire protocol, right-of-way, map,
                    behaviour arbitration. No ROS, no sockets except one
                    module, no simulator. 143 tests, runs in 6 seconds.
src/bfmc_v2x_msgs/  ROS message definitions.
src/bfmc_v2x_ros/   Thin ROS 2 adapters: nine nodes, launch files, RViz.
src/bfmc_sim/       Track definition, vehicle URDF, generated Gazebo world.
```

Two boundaries are enforced rather than merely intended:

**Logic is ROS-free.** Anything decidable without a clock, a socket or a
simulator lives in `bfmc_v2x`, where it can be exercised ten thousand times
a second instead of by watching a running system. `conversions.py` is the
only file allowed to know about both worlds.

**The map exists once.** `src/bfmc_sim/track/mini_city.yaml` is the single
source of truth. The Gazebo world, the driving routes, the stop-line
positions, the traffic-light placements and phase offsets, and the
environment server's ground truth are all generated from it. A test asserts
the painted stop line sits within 1 mm of the arc length the vehicle
actually brakes for — because a car that is provably correct and visibly
wrong is the worst thing to debug.

---

## Notes on what this is and is not

The detector is **simulated from ground truth** — range, bearing and
confidence with noise, not a vision stack. This project is about the V2X and
localisation layers, and a half-hearted classifier would only add noise to
the results. The seam is one class (`ObstacleDetector`); replacing it with
real perception means changing nothing else. Crucially, it takes ground
truth for *visibility* and the fused estimate for *placement*, which is what
makes obstacle scoring a genuine localisation benchmark.

Traffic-light colour in **Gazebo is static**; the live state is the RViz
marker. Gazebo cannot recolour a visual at runtime without a custom C++
system plugin, and adding one would put a compiled dependency in the way of
a workspace that otherwise builds anywhere. The housings are there so the
junction reads as a junction.

The **kinematic bicycle** used in the unit tests is not a physics engine —
no tyre model, no mass transfer. It is the same model the steering
controller assumes, so it tests whether the decisions are right; whether the
vehicle can execute them is what Gazebo is for.

Deeper detail: [architecture](docs/architecture.md)
