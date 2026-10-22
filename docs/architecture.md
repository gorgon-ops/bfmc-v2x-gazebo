# Architecture

## The shape of it

```
                        ┌──────────────────────────────────────┐
     GAZEBO             │  mini_city.sdf  (generated)          │
                        │  physics · IMU · Ackermann · truth    │
                        └───────┬──────────────────────┬────────┘
                    ground truth│                      │cmd_vel
                                ▼                      ▲
     ┌──────────────────────────────────────────┐      │
     │  TRACKSIDE (no ROS topic a car can read) │      │
     │  ┌────────────────────┐ ┌──────────────┐ │      │
     │  │ localization_server│ │  semaphores  │ │      │
     │  │ noise·latency·drop │ │  phase plans │ │      │
     │  └─────────┬──────────┘ └──────┬───────┘ │      │
     │  ┌─────────┴───────────────────┴──────┐  │      │
     │  │        environment_server          │  │      │
     │  └────────────────┬───────────────────┘  │      │
     └───────────────────┼──────────────────────┘      │
                         │                             │
              ═══════════▼═════════════════════════════│═══════
                   UDP BROADCAST · port 50009          │
              ═══════════╤═════════════════════════════│═══════
                         │                             │
     ┌───────────────────┼─────────────────────────────┼──────┐
     │  VEHICLE          ▼                             │      │
     │            ┌─────────────┐                      │      │
     │            │ v2x_bridge  │  the only socket     │      │
     │            └──┬───┬───┬──┘                      │      │
     │      pose_fix │   │   │ peer / semaphore        │      │
     │               ▼   │   ▼                         │      │
     │  ┌─────────────────┐ │  ┌──────────────────┐    │      │
     │  │  localization   │ │  │     driver       ├────┘      │
     │  │  EKF + replay   ├─┼─►│  arbitration     │           │
     │  │  + DR shadow    │ │  └──────────────────┘           │
     │  └────────┬────────┘ │                                 │
     │           │          │  ┌──────────────────┐           │
     │           └──────────┴─►│    detector      │           │
     │                         └──────────────────┘           │
     └─────────────────────────────────────────────────────────┘
```

## Two boundaries that are enforced, not just intended

### 1. The logic knows nothing about ROS

`src/bfmc_v2x/` has no `rclpy` import anywhere. The EKF, the wire format,
the right-of-way rules, the stop-line policy, the map and the behaviour
arbiter are all plain Python operating on plain dataclasses.

This is not tidiness for its own sake. A filter that can only be exercised
inside a simulator gets tested by watching it, once, and the properties that
matter — *does the covariance stay honest over ten minutes, does the replay
double-count an interval, does exactly one of two symmetric cars yield* —
are not things you can watch. Running the whole suite takes six seconds,
which is the difference between checking those properties on every change
and checking them never.

`bfmc_v2x_ros/conversions.py` is the only file permitted to know about both
sides. If a ROS type ever appears inside `bfmc_v2x`, the boundary has been
lost.

### 2. The map exists exactly once

`src/bfmc_sim/track/mini_city.yaml` describes nodes, roads, junctions,
traffic lights, obstacles and routes. Derived from it, at build time or at
node startup:

| Consumer | What it takes |
|---|---|
| `generate_world.py` | carriageways, paint, junctions, light housings, obstacles, walls |
| `Track.build_route` | drivable lane (right-hand offset, filleted corners), stop lines |
| `semaphore_node` | which lights exist, where they stand, their phase offsets |
| `environment_server_node` | ground truth for scoring |
| `vehicle.launch.py` | spawn poses |

The failure this prevents is specific and nasty: the car plans against the
YAML and drives in the SDF, so a disagreement between them produces a
vehicle that is *provably correct and visibly wrong* — braking for a stop
line painted somewhere else, or clipping a wall that moved. Both look like
controller bugs and neither is. `test_world_generation.py` asserts the
painted stop line lands within 1 mm of the arc length the vehicle brakes
for.

## The nodes

### Trackside — what the organisers run

| Node | Does |
|---|---|
| `localization_server` | Reads ground truth, adds noise / latency / dropout, broadcasts pose fixes stamped with **exposure** time |
| `semaphores` | Runs every light's phase plan, broadcasts colour + countdown, refuses an unsafe plan at startup |
| `environment_server` | Receives obstacle reports off the radio, scores them against ground truth |

None of these publishes a ROS topic a vehicle subscribes to. On the real
track nothing on the car can see these processes; whatever the car knows, it
heard on the radio. Keeping that true in simulation is what stops the demo
from quietly cheating.

### Vehicle — what runs on the car

| Node | Does |
|---|---|
| `wheel_encoder` | Simulates encoders: tyre-radius **scale error**, quantisation, noise |
| `v2x_bridge` | The only socket on the vehicle. Filters inbound, broadcasts own state |
| `localization` | EKF with delayed-fix replay, plus the dead-reckoning shadow |
| `driver` | Route following, stop-line policy, right-of-way, speed arbitration |
| `detector` | Detections from ground truth, **placed with the fused estimate** |

`driver` cannot see ground truth. It steers, stops and yields entirely on
the filter's output, so a localisation failure shows up as a car stopping in
the wrong place rather than as a number in a log — which is the difference
between measuring localisation and demonstrating why it matters.

### Scoring

`metrics` records per-sample CSV and prints the summary table at shutdown:
fused error, dead-reckoning error, the ratio, gyro bias estimate, fix
latency, gating counts, and how many events the replay path re-integrated.

## Why the sensors are deliberately spoiled

Gazebo will hand you the vehicle's exact velocity, and using it makes the
localisation problem disappear. The two errors reintroduced are the two that
matter on real hardware, and they are chosen because they are *systematic*
rather than random — averaging does not remove them, and they integrate
straight into position:

* **Tyre-radius scale error** (`wheel_encoder`, default +3 %): every measured
  distance is out by a fixed percentage.
* **Gyro bias** (declared in the URDF's IMU sensor, default 0.02 rad/s):
  heading ramps away linearly.

The second is why the filter carries the bias in its state vector. It is
observable once absolute headings start arriving, and estimating it is what
stops the heading drifting between fixes.

## Arbitration

Three subsystems want to set the speed and there is one throttle. The rule
is deliberately boring: **the slowest request wins**, and nothing may raise
a speed another module asked to lower. A new constraint is one more
candidate in a list, not an edit to a priority table.

Steering is never arbitrated. The vehicle always steers along its route,
even while stopping — a car that abandons its lane because a light went red
is a worse outcome than one that stops in it.

The driver holds exactly one piece of state: which stop line it is held at.
It earns its place — a vehicle commanded to zero does not stop dead, it
decays towards zero and creeps. A few millimetres of creep puts the nose
past the line, and a stateless policy then sees itself as "in the junction",
switches to *clear the junction*, and drives away through a red light. That
bug was real; the latch and the stop-line setback are the two fixes.
