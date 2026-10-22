"""UDP broadcast transport. Everyone hears everyone, hence a sender id in every frame.

The receive buffer is small on purpose: a node that stalls should come back to
recent messages, not a backlog.
"""

from __future__ import annotations

import errno
import logging
import socket
from dataclasses import dataclass
from typing import List, Optional, Tuple

from .protocol import Header, Payload, ProtocolError, decode, encode

__all__ = ["V2XEndpoint", "Received", "LoopbackLink", "DEFAULT_PORT", "DEFAULT_GROUP"]

log = logging.getLogger(__name__)

#: Outside the ephemeral range.
DEFAULT_PORT = 50009

#: Stays inside the local segment and needs no multicast setup, on a competition
#: LAN nobody may reconfigure.
DEFAULT_GROUP = "255.255.255.255"

#: Small on purpose.
_RECV_BUFFER_BYTES = 8192


@dataclass(frozen=True)
class Received:
    header: Header
    payload: Payload
    source: Tuple[str, int]


class V2XEndpoint:
    """Not thread-safe: each node owns one and polls it from its own timer."""

    def __init__(
        self,
        sender_id: int,
        port: int = DEFAULT_PORT,
        group: str = DEFAULT_GROUP,
        bind_host: str = "",
        receive: bool = True,
    ) -> None:
        self.sender_id = sender_id
        self.port = port
        self.group = group
        self._seq = 0
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        if hasattr(socket, "SO_REUSEPORT"):
            # Several nodes on one machine share the port; without it the second fails
            # to start on Linux.
            try:
                self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
            except OSError:  # pragma: no cover - platform dependent
                log.debug("SO_REUSEPORT unavailable, continuing without it")
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, _RECV_BUFFER_BYTES)
        if receive:
            self._sock.bind((bind_host, port))
        self._sock.setblocking(False)

        #: A proxy for link health.
        self.malformed: int = 0
        self.self_echoes: int = 0

    def send(self, payload: Payload, tx_time_us: Optional[int] = None) -> int:
        """``tx_time_us`` overrides the stamp for reports about another moment, such as when a
        camera's shutter opened: receivers use it to decide when the measurement applies.
        """
        seq = self._seq
        datagram = encode(payload, sender_id=self.sender_id, seq=seq, tx_time_us=tx_time_us)
        self._seq = (self._seq + 1) & 0xFFFFFFFF
        try:
            self._sock.sendto(datagram, (self.group, self.port))
        except OSError as exc:  # pragma: no cover - transient network faults
            if exc.errno not in (errno.ENETUNREACH, errno.ENETDOWN, errno.EPERM):
                raise
            log.warning("v2x send failed: %s", exc)
        return seq

    def poll(self, limit: int = 64, drop_self: bool = True) -> List[Received]:
        """Our own broadcasts loop back and are dropped by default."""
        out: List[Received] = []
        for _ in range(limit):
            try:
                datagram, source = self._sock.recvfrom(self.max_datagram)
            except BlockingIOError:
                break
            except OSError as exc:  # pragma: no cover - transient
                log.warning("v2x recv failed: %s", exc)
                break
            try:
                header, payload = decode(datagram)
            except ProtocolError as exc:
                self.malformed += 1
                log.debug("dropping malformed datagram from %s: %s", source, exc)
                continue
            if drop_self and header.sender_id == self.sender_id:
                self.self_echoes += 1
                continue
            out.append(Received(header=header, payload=payload, source=source))
        return out

    @property
    def max_datagram(self) -> int:
        from .protocol import MAX_DATAGRAM

        return MAX_DATAGRAM

    def fileno(self) -> int:
        return self._sock.fileno()

    def close(self) -> None:
        self._sock.close()

    def __enter__(self) -> "V2XEndpoint":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()


class LoopbackLink:
    """Holds encoded bytes, so tests exercise the real wire format."""

    def __init__(self) -> None:
        self._queue: List[Tuple[int, bytes]] = []
        self._seq = 0

    def send(self, payload: Payload, sender_id: int, tx_time_us: Optional[int] = None) -> int:
        seq = self._seq
        self._queue.append(
            (sender_id, encode(payload, sender_id=sender_id, seq=seq, tx_time_us=tx_time_us))
        )
        self._seq += 1
        return seq

    def deliver(self, receiver_id: Optional[int] = None) -> List[Received]:
        out: List[Received] = []
        for sender_id, datagram in self._queue:
            if receiver_id is not None and sender_id == receiver_id:
                continue
            header, payload = decode(datagram)
            out.append(Received(header=header, payload=payload, source=("loopback", 0)))
        self._queue.clear()
        return out

    def corrupt_last(self, byte_index: int = -3) -> None:
        """Flip a bit in the most recent datagram, to exercise the CRC path."""
        sender_id, datagram = self._queue[-1]
        mutable = bytearray(datagram)
        mutable[byte_index] ^= 0x01
        self._queue[-1] = (sender_id, bytes(mutable))
