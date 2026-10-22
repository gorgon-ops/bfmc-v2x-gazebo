"""Wire format for the V2X link: one self-contained UDP datagram per message, no retransmission.

Datagram layout (all fields big-endian)::

    0        2       3       4          6         10             18      18+n
    +--------+-------+-------+----------+---------+--------------+-----+-------+
    | magic  |  ver  | type  | sender   |   seq   |   tx_time    | pay | crc16 |
    | 2 B    |  1 B  | 1 B   |   2 B    |   4 B   |     8 B      | n B |  2 B  |
    +--------+-------+-------+----------+---------+--------------+-----+-------+

``tx_time`` is epoch microseconds, stamped as late as possible: the EKF uses it.
``crc16`` (CCITT-FALSE) rejects corruption only; the protocol has no security.
"""

from __future__ import annotations

import struct
import time
from dataclasses import dataclass
from enum import IntEnum
from typing import ClassVar, Dict, Optional, Tuple, Type

__all__ = [
    "MAGIC",
    "VERSION",
    "MAX_DATAGRAM",
    "MessageType",
    "SemaphoreState",
    "Intent",
    "ObstacleKind",
    "ProtocolError",
    "Header",
    "Payload",
    "PoseFix",
    "SemaphoreBroadcast",
    "PeerState",
    "ObstacleReport",
    "ObstacleAck",
    "encode",
    "decode",
    "crc16_ccitt",
    "now_us",
]

MAGIC = b"BF"

VERSION = 1

_HEADER = struct.Struct("!2sBBHIQ")
_CRC = struct.Struct("!H")

#: Comfortably inside one Ethernet MTU.
MAX_DATAGRAM = 256


class MessageType(IntEnum):
    #: Infrastructure to vehicle: absolute pose from the overhead tracker.
    POSE_FIX = 0x01
    SEMAPHORE = 0x02
    PEER_STATE = 0x03
    OBSTACLE_REPORT = 0x04
    OBSTACLE_ACK = 0x05


class SemaphoreState(IntEnum):
    RED = 0
    YELLOW = 1
    GREEN = 2
    #: Night mode or fault fallback.
    FLASHING_YELLOW = 3


class Intent(IntEnum):
    STRAIGHT = 0
    LEFT = 1
    RIGHT = 2
    #: Parked, or yielding indefinitely.
    HOLD = 3


class ObstacleKind(IntEnum):
    STATIC_CAR = 0
    PEDESTRIAN = 1
    ROADBLOCK = 2
    STOP_SIGN = 3
    CROSSWALK = 4
    PARKING_SPOT = 5


class ProtocolError(ValueError):
    pass


def crc16_ccitt(data: bytes, seed: int = 0xFFFF) -> int:
    """Bit-at-a-time: the same routine runs on the light firmware, where a table is real money."""
    crc = seed
    for byte in data:
        crc ^= byte << 8
        for _ in range(8):
            if crc & 0x8000:
                crc = ((crc << 1) ^ 0x1021) & 0xFFFF
            else:
                crc = (crc << 1) & 0xFFFF
    return crc


def now_us() -> int:
    return time.time_ns() // 1000


@dataclass(frozen=True)
class Header:
    msg_type: MessageType
    sender_id: int
    seq: int
    tx_time_us: int
    version: int = VERSION

    def age_s(self, reference_us: Optional[int] = None) -> float:
        """Can be slightly negative with unsynced clocks; clamp rather than trust the sign."""
        reference = now_us() if reference_us is None else reference_us
        return (reference - self.tx_time_us) / 1e6


class Payload:
    """Subclasses declare a :class:`struct.Struct` and a :class:`MessageType`."""

    TYPE: ClassVar[MessageType]
    STRUCT: ClassVar[struct.Struct]

    def pack(self) -> bytes:
        return self.STRUCT.pack(*self._fields())

    def _fields(self) -> Tuple[object, ...]:
        raise NotImplementedError

    @classmethod
    def unpack(cls, raw: bytes) -> "Payload":
        if len(raw) != cls.STRUCT.size:
            raise ProtocolError(
                "{} expects {} payload bytes, got {}".format(
                    cls.__name__, cls.STRUCT.size, len(raw)
                )
            )
        return cls(*cls.STRUCT.unpack(raw))  # type: ignore[call-arg]


@dataclass(frozen=True)
class PoseFix(Payload):
    """Slow, late and noisy, but drift-free. The sigmas grow towards the edge of the camera's view
    and feed straight into the EKF's measurement covariance.
    """

    TYPE = MessageType.POSE_FIX
    STRUCT = struct.Struct("!Hfffff")

    vehicle_id: int
    x: float
    y: float
    yaw: float
    sigma_pos: float
    sigma_yaw: float

    def _fields(self) -> Tuple[object, ...]:
        return (self.vehicle_id, self.x, self.y, self.yaw, self.sigma_pos, self.sigma_yaw)


@dataclass(frozen=True)
class SemaphoreBroadcast(Payload):
    """``ttl_ms`` lets a car 3 m out that knows the light turns green in 800 ms keep rolling."""

    TYPE = MessageType.SEMAPHORE
    STRUCT = struct.Struct("!HBH")

    light_id: int
    state: int
    ttl_ms: int

    def _fields(self) -> Tuple[object, ...]:
        return (self.light_id, int(self.state), self.ttl_ms)

    @property
    def colour(self) -> SemaphoreState:
        return SemaphoreState(self.state)

    @property
    def ttl_s(self) -> float:
        return self.ttl_ms / 1000.0


@dataclass(frozen=True)
class PeerState(Payload):
    """Kinematic, not a planned trajectory, so receivers need not agree on a path format."""

    TYPE = MessageType.PEER_STATE
    STRUCT = struct.Struct("!HfffffB")

    vehicle_id: int
    x: float
    y: float
    yaw: float
    speed: float
    yaw_rate: float
    intent: int

    def _fields(self) -> Tuple[object, ...]:
        return (
            self.vehicle_id,
            self.x,
            self.y,
            self.yaw,
            self.speed,
            self.yaw_rate,
            int(self.intent),
        )

    @property
    def manoeuvre(self) -> Intent:
        return Intent(self.intent)


@dataclass(frozen=True)
class ObstacleReport(Payload):
    TYPE = MessageType.OBSTACLE_REPORT
    STRUCT = struct.Struct("!Bfff")

    kind: int
    x: float
    y: float
    confidence: float

    def _fields(self) -> Tuple[object, ...]:
        return (int(self.kind), self.x, self.y, self.confidence)

    @property
    def obstacle(self) -> ObstacleKind:
        return ObstacleKind(self.kind)


@dataclass(frozen=True)
class ObstacleAck(Payload):
    TYPE = MessageType.OBSTACLE_ACK
    STRUCT = struct.Struct("!IBf")

    report_seq: int
    accepted: int
    error_m: float

    def _fields(self) -> Tuple[object, ...]:
        return (self.report_seq, int(bool(self.accepted)), self.error_m)

    @property
    def was_accepted(self) -> bool:
        return bool(self.accepted)


_PAYLOADS: Dict[MessageType, Type[Payload]] = {
    cls.TYPE: cls
    for cls in (PoseFix, SemaphoreBroadcast, PeerState, ObstacleReport, ObstacleAck)
}


def encode(
    payload: Payload,
    sender_id: int,
    seq: int,
    tx_time_us: Optional[int] = None,
) -> bytes:
    """``seq`` wraps at 32 bits; treat it as modular."""
    if not 0 <= sender_id <= 0xFFFF:
        raise ProtocolError("sender_id {} out of range".format(sender_id))
    stamp = now_us() if tx_time_us is None else tx_time_us
    head = _HEADER.pack(
        MAGIC,
        VERSION,
        int(payload.TYPE),
        sender_id,
        seq & 0xFFFFFFFF,
        stamp & 0xFFFFFFFFFFFFFFFF,
    )
    frame = head + payload.pack()
    datagram = frame + _CRC.pack(crc16_ccitt(frame))
    if len(datagram) > MAX_DATAGRAM:
        raise ProtocolError(
            "datagram of {} B exceeds {} B budget".format(len(datagram), MAX_DATAGRAM)
        )
    return datagram


def decode(datagram: bytes) -> Tuple[Header, Payload]:
    """Every failure is a rejection, never a partial result."""
    if len(datagram) < _HEADER.size + _CRC.size:
        raise ProtocolError("runt datagram of {} B".format(len(datagram)))

    magic, version, raw_type, sender_id, seq, tx_time_us = _HEADER.unpack_from(datagram)
    if magic != MAGIC:
        raise ProtocolError("bad magic {!r}".format(magic))
    if version != VERSION:
        raise ProtocolError("unsupported protocol version {}".format(version))

    body = datagram[: -_CRC.size]
    (received_crc,) = _CRC.unpack_from(datagram, len(datagram) - _CRC.size)
    expected_crc = crc16_ccitt(body)
    if received_crc != expected_crc:
        raise ProtocolError(
            "crc mismatch: got 0x{:04x}, want 0x{:04x}".format(received_crc, expected_crc)
        )

    try:
        msg_type = MessageType(raw_type)
    except ValueError as exc:
        raise ProtocolError("unknown message type 0x{:02x}".format(raw_type)) from exc

    payload = _PAYLOADS[msg_type].unpack(body[_HEADER.size :])
    header = Header(
        msg_type=msg_type,
        sender_id=sender_id,
        seq=seq,
        tx_time_us=tx_time_us,
        version=version,
    )
    return header, payload
