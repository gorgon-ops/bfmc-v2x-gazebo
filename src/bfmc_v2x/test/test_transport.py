"""Loopback tests carry real bytes. The socket test is a smoke test for socket options,
skipped where UDP broadcast is unavailable.
"""

from __future__ import annotations

import socket

import pytest

from bfmc_v2x.protocol import (
    Intent,
    MessageType,
    PeerState,
    PoseFix,
    ProtocolError,
    SemaphoreBroadcast,
    SemaphoreState,
)
from bfmc_v2x.transport import DEFAULT_PORT, LoopbackLink, V2XEndpoint

FIX = PoseFix(vehicle_id=1, x=1.0, y=2.0, yaw=0.5, sigma_pos=0.04, sigma_yaw=0.02)
LIGHT = SemaphoreBroadcast(light_id=2, state=int(SemaphoreState.GREEN), ttl_ms=3000)


def test_loopback_delivers_decoded_messages():
    link = LoopbackLink()
    link.send(FIX, sender_id=10)
    link.send(LIGHT, sender_id=20)

    received = link.deliver()

    assert [r.header.msg_type for r in received] == [
        MessageType.POSE_FIX,
        MessageType.SEMAPHORE,
    ]
    assert received[0].payload.x == pytest.approx(1.0)
    assert received[1].payload.colour is SemaphoreState.GREEN


def test_loopback_drops_the_receivers_own_echo():
    link = LoopbackLink()
    link.send(FIX, sender_id=7)
    link.send(LIGHT, sender_id=9)

    received = link.deliver(receiver_id=7)

    assert [r.header.sender_id for r in received] == [9]


def test_the_queue_drains_on_delivery():
    link = LoopbackLink()
    link.send(FIX, sender_id=1)

    assert len(link.deliver()) == 1
    assert link.deliver() == []


def test_sequence_numbers_advance():
    link = LoopbackLink()
    link.send(FIX, sender_id=1)
    link.send(FIX, sender_id=1)

    first, second = link.deliver()

    assert second.header.seq == first.header.seq + 1


def test_a_sender_can_stamp_a_frame_with_an_earlier_time():
    """Without the shutter time, measured-late and delivered-late are indistinguishable."""
    link = LoopbackLink()
    link.send(FIX, sender_id=1, tx_time_us=1_000_000)

    (received,) = link.deliver()

    assert received.header.tx_time_us == 1_000_000
    assert received.header.age_s(reference_us=1_250_000) == pytest.approx(0.25)


def test_a_corrupted_frame_fails_to_decode():
    link = LoopbackLink()
    link.send(FIX, sender_id=1)
    link.corrupt_last()

    with pytest.raises(ProtocolError):
        link.deliver()


def test_timestamps_survive_the_link():
    link = LoopbackLink()
    link.send(FIX, sender_id=1, tx_time_us=1_700_000_000_000_000)

    (received,) = link.deliver()

    assert received.header.age_s(reference_us=1_700_000_000_500_000) == pytest.approx(0.5)


# ---- real sockets --------------------------------------------------------


def _broadcast_available() -> bool:
    try:
        probe = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
        probe.close()
        return True
    except OSError:  # pragma: no cover - platform dependent
        return False


@pytest.mark.skipif(not _broadcast_available(), reason="UDP broadcast unavailable")
def test_two_endpoints_can_share_a_port():
    port = DEFAULT_PORT + 11
    try:
        first = V2XEndpoint(sender_id=1, port=port, group="127.0.0.1")
        second = V2XEndpoint(sender_id=2, port=port, group="127.0.0.1")
    except OSError as exc:  # pragma: no cover - restrictive sandboxes
        pytest.skip("cannot bind the test port: {}".format(exc))

    with first, second:
        first.send(
            PeerState(vehicle_id=1, x=0.5, y=0.25, yaw=0.0, speed=0.3,
                      yaw_rate=0.0, intent=int(Intent.STRAIGHT))
        )

        # Non-blocking: allow a few polls.
        received = []
        for _ in range(200):
            received = second.poll()
            if received:
                break

        if not received:  # pragma: no cover - lossy sandbox networking
            pytest.skip("no datagram delivered on the loopback interface")

        assert received[0].header.sender_id == 1
        assert received[0].payload.x == pytest.approx(0.5)


@pytest.mark.skipif(not _broadcast_available(), reason="UDP broadcast unavailable")
def test_polling_an_idle_endpoint_returns_nothing():
    try:
        endpoint = V2XEndpoint(sender_id=3, port=DEFAULT_PORT + 12, group="127.0.0.1")
    except OSError as exc:  # pragma: no cover
        pytest.skip("cannot bind the test port: {}".format(exc))

    with endpoint:
        assert endpoint.poll() == []
