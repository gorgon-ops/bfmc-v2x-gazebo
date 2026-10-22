"""The one file allowed to know both ROS types and the plain-Python core."""

from __future__ import annotations

import math
from typing import Optional, Tuple

from builtin_interfaces.msg import Time as TimeMsg
from geometry_msgs.msg import Quaternion

from bfmc_v2x.geometry import Pose2D, wrap_angle
from bfmc_v2x.protocol import (
    Intent,
    ObstacleKind,
    ObstacleReport,
    PeerState,
    PoseFix,
    SemaphoreBroadcast,
    SemaphoreState,
)

import bfmc_v2x_msgs.msg as bfmc_msgs

__all__ = [
    "yaw_from_quaternion",
    "quaternion_from_yaw",
    "stamp_to_seconds",
    "seconds_to_stamp",
    "pose_fix_to_msg",
    "msg_to_pose_fix",
    "semaphore_to_msg",
    "msg_to_semaphore",
    "peer_state_to_msg",
    "msg_to_peer_state",
    "obstacle_report_to_msg",
    "msg_to_obstacle_report",
]


def yaw_from_quaternion(q: Quaternion) -> float:
    """Yaw-only extraction, exact for any rotation about z."""
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def quaternion_from_yaw(yaw: float) -> Quaternion:
    half = wrap_angle(yaw) / 2.0
    return Quaternion(x=0.0, y=0.0, z=math.sin(half), w=math.cos(half))


def stamp_to_seconds(stamp: TimeMsg) -> float:
    return stamp.sec + stamp.nanosec * 1e-9


def seconds_to_stamp(seconds: float) -> TimeMsg:
    if seconds < 0.0:
        seconds = 0.0
    sec = int(seconds)
    return TimeMsg(sec=sec, nanosec=int(round((seconds - sec) * 1e9)))


# ----------------------------------------------------------------- pose fix


def pose_fix_to_msg(
    payload: PoseFix, measured_at: float, frame_id: str = "map"
) -> bfmc_msgs.PoseFix:
    message = bfmc_msgs.PoseFix()
    message.header.frame_id = frame_id
    message.measured_at = seconds_to_stamp(measured_at)
    message.vehicle_id = int(payload.vehicle_id)
    message.x = float(payload.x)
    message.y = float(payload.y)
    message.yaw = float(payload.yaw)
    message.sigma_pos = float(payload.sigma_pos)
    message.sigma_yaw = float(payload.sigma_yaw)
    return message


def msg_to_pose_fix(message: bfmc_msgs.PoseFix) -> Tuple[PoseFix, float]:
    payload = PoseFix(
        vehicle_id=message.vehicle_id,
        x=message.x,
        y=message.y,
        yaw=message.yaw,
        sigma_pos=message.sigma_pos,
        sigma_yaw=message.sigma_yaw,
    )
    return payload, stamp_to_seconds(message.measured_at)


# ---------------------------------------------------------------- semaphore


def semaphore_to_msg(payload: SemaphoreBroadcast, ttl_s: Optional[float]) -> bfmc_msgs.SemaphoreState:
    message = bfmc_msgs.SemaphoreState()
    message.light_id = int(payload.light_id)
    message.state = int(payload.state)
    # Negative: colour only.
    message.ttl = float(ttl_s) if ttl_s is not None else -1.0
    return message


def msg_to_semaphore(
    message: bfmc_msgs.SemaphoreState,
) -> Tuple[SemaphoreBroadcast, Optional[float]]:
    ttl = None if message.ttl < 0.0 else float(message.ttl)
    payload = SemaphoreBroadcast(
        light_id=message.light_id,
        state=int(message.state),
        ttl_ms=int(max(0.0, ttl or 0.0) * 1000.0),
    )
    return payload, ttl


def semaphore_state_name(state: int) -> str:
    return SemaphoreState(state).name


# -------------------------------------------------------------- peer state


def peer_state_to_msg(payload: PeerState, age_s: float) -> bfmc_msgs.PeerState:
    message = bfmc_msgs.PeerState()
    message.vehicle_id = int(payload.vehicle_id)
    message.x = float(payload.x)
    message.y = float(payload.y)
    message.yaw = float(payload.yaw)
    message.speed = float(payload.speed)
    message.yaw_rate = float(payload.yaw_rate)
    message.intent = int(payload.intent)
    message.age = float(age_s)
    return message


def msg_to_peer_state(message: bfmc_msgs.PeerState) -> Tuple[PeerState, float]:
    payload = PeerState(
        vehicle_id=message.vehicle_id,
        x=message.x,
        y=message.y,
        yaw=message.yaw,
        speed=message.speed,
        yaw_rate=message.yaw_rate,
        intent=int(message.intent),
    )
    return payload, float(message.age)


def pose2d_from_peer(message: bfmc_msgs.PeerState) -> Pose2D:
    return Pose2D(message.x, message.y, message.yaw)


# ---------------------------------------------------------------- obstacles


def obstacle_report_to_msg(
    payload: ObstacleReport, vehicle_id: int, seq: int
) -> bfmc_msgs.ObstacleReport:
    message = bfmc_msgs.ObstacleReport()
    message.vehicle_id = int(vehicle_id)
    message.seq = int(seq) & 0xFFFFFFFF
    message.kind = int(payload.kind)
    message.x = float(payload.x)
    message.y = float(payload.y)
    message.confidence = float(payload.confidence)
    return message


def msg_to_obstacle_report(
    message: bfmc_msgs.ObstacleReport,
) -> Tuple[ObstacleReport, int, int]:
    payload = ObstacleReport(
        kind=int(message.kind),
        x=message.x,
        y=message.y,
        confidence=message.confidence,
    )
    return payload, int(message.vehicle_id), int(message.seq)


def obstacle_kind_name(kind: int) -> str:
    return ObstacleKind(kind).name


def intent_name(intent: int) -> str:
    return Intent(intent).name
