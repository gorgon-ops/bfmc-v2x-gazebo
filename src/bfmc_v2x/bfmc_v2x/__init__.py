"""V2X localisation and cooperative driving. Plain Python - no ROS, no Gazebo, sockets only
in :mod:`bfmc_v2x.transport` - so it can be tested at many times real time.
"""

from __future__ import annotations

from .driver import DriveCommand, Driver, DriverConfig
from .ekf import DeadReckoning, DelayedFusionEKF, FilterEstimate, ProcessNoise
from .environment import (
    Detection,
    EnvironmentServer,
    GroundTruthObstacle,
    ObstacleDetector,
    Scoreboard,
)
from .geometry import Point, Polyline, Pose2D, pure_pursuit_steer, wrap_angle
from .protocol import (
    Header,
    Intent,
    MessageType,
    ObstacleAck,
    ObstacleKind,
    ObstacleReport,
    PeerState,
    PoseFix,
    ProtocolError,
    SemaphoreBroadcast,
    SemaphoreState,
    decode,
    encode,
)
from .semaphore import (
    Approach,
    ApproachDecision,
    ObservedLight,
    Phase,
    SemaphoreCycle,
    SemaphoreTracker,
    StopLinePolicy,
)
from .track import Junction, Route, StopLine, Track, TrackEdge, TrackNode
from .transport import LoopbackLink, Received, V2XEndpoint
from .v2v import (
    JunctionAssessment,
    PeerRegistry,
    PeerTrack,
    RightOfWay,
    assess_junction,
    closest_approach,
    predict_trajectory,
)

__version__ = "1.0.0"

__all__ = [
    "__version__",
    # protocol
    "Header",
    "Intent",
    "MessageType",
    "ObstacleAck",
    "ObstacleKind",
    "ObstacleReport",
    "PeerState",
    "PoseFix",
    "ProtocolError",
    "SemaphoreBroadcast",
    "SemaphoreState",
    "decode",
    "encode",
    # transport
    "LoopbackLink",
    "Received",
    "V2XEndpoint",
    # geometry
    "Point",
    "Polyline",
    "Pose2D",
    "pure_pursuit_steer",
    "wrap_angle",
    # driving
    "DriveCommand",
    "Driver",
    "DriverConfig",
    # localisation
    "DeadReckoning",
    "DelayedFusionEKF",
    "FilterEstimate",
    "ProcessNoise",
    # signals
    "Approach",
    "ApproachDecision",
    "ObservedLight",
    "Phase",
    "SemaphoreCycle",
    "SemaphoreTracker",
    "StopLinePolicy",
    # cooperation
    "JunctionAssessment",
    "PeerRegistry",
    "PeerTrack",
    "RightOfWay",
    "assess_junction",
    "closest_approach",
    "predict_trajectory",
    # environment
    "Detection",
    "EnvironmentServer",
    "GroundTruthObstacle",
    "ObstacleDetector",
    "Scoreboard",
    # map
    "Junction",
    "Route",
    "StopLine",
    "Track",
    "TrackEdge",
    "TrackNode",
]
