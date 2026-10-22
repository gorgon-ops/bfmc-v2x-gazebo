"""Everything goes through real bytes: format mismatches and sign extension only show up
after a round trip.
"""

from __future__ import annotations

import math

import pytest

from bfmc_v2x.protocol import (
    MAGIC,
    MAX_DATAGRAM,
    VERSION,
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
    crc16_ccitt,
    decode,
    encode,
)

PAYLOADS = [
    PoseFix(vehicle_id=7, x=1.25, y=-3.5, yaw=1.5, sigma_pos=0.05, sigma_yaw=0.02),
    SemaphoreBroadcast(light_id=3, state=int(SemaphoreState.YELLOW), ttl_ms=1750),
    PeerState(
        vehicle_id=2, x=4.0, y=0.5, yaw=-2.0, speed=0.33, yaw_rate=-0.1,
        intent=int(Intent.LEFT),
    ),
    ObstacleReport(kind=int(ObstacleKind.PEDESTRIAN), x=3.35, y=4.3, confidence=0.82),
    ObstacleAck(report_seq=4_000_000_000, accepted=1, error_m=0.12),
]


@pytest.mark.parametrize("payload", PAYLOADS, ids=lambda p: type(p).__name__)
def test_round_trip_preserves_payload(payload):
    datagram = encode(payload, sender_id=11, seq=42, tx_time_us=1_700_000_000_000_000)
    header, decoded = decode(datagram)

    assert header.msg_type is payload.TYPE
    assert header.sender_id == 11
    assert header.seq == 42
    assert header.tx_time_us == 1_700_000_000_000_000
    assert header.version == VERSION
    assert type(decoded) is type(payload)

    for field, original in vars(payload).items():
        recovered = getattr(decoded, field)
        if isinstance(original, float):
            # f32 on the wire.
            assert recovered == pytest.approx(original, rel=1e-6, abs=1e-6)
        else:
            assert recovered == original


@pytest.mark.parametrize("payload", PAYLOADS, ids=lambda p: type(p).__name__)
def test_every_datagram_fits_the_budget(payload):
    assert len(encode(payload, sender_id=1, seq=0)) <= MAX_DATAGRAM


def test_crc16_matches_the_reference_vector():
    # The check value every CRC-16/CCITT-FALSE implementation agrees on.
    assert crc16_ccitt(b"123456789") == 0x29B1


def test_corrupt_payload_is_rejected():
    datagram = bytearray(encode(PAYLOADS[0], sender_id=1, seq=1))
    datagram[20] ^= 0x01

    with pytest.raises(ProtocolError, match="crc mismatch"):
        decode(bytes(datagram))


def test_corrupt_header_is_rejected():
    datagram = bytearray(encode(PAYLOADS[1], sender_id=1, seq=1))
    datagram[6] ^= 0xFF

    with pytest.raises(ProtocolError, match="crc mismatch"):
        decode(bytes(datagram))


def test_foreign_traffic_on_the_port_is_rejected():
    with pytest.raises(ProtocolError, match="bad magic"):
        decode(b"GET / HTTP/1.1\r\n\r\npadding-to-length")


def test_future_version_is_rejected():
    datagram = bytearray(encode(PAYLOADS[0], sender_id=1, seq=1))
    datagram[2] = VERSION + 1
    # Re-CRC so the version check is what trips, not the checksum.
    body = bytes(datagram[:-2])
    datagram[-2:] = crc16_ccitt(body).to_bytes(2, "big")

    with pytest.raises(ProtocolError, match="unsupported protocol version"):
        decode(bytes(datagram))


def test_unknown_message_type_is_rejected():
    datagram = bytearray(encode(PAYLOADS[0], sender_id=1, seq=1))
    datagram[3] = 0x7F
    body = bytes(datagram[:-2])
    datagram[-2:] = crc16_ccitt(body).to_bytes(2, "big")

    with pytest.raises(ProtocolError, match="unknown message type"):
        decode(bytes(datagram))


def test_truncated_datagram_is_rejected():
    datagram = encode(PAYLOADS[0], sender_id=1, seq=1)

    with pytest.raises(ProtocolError):
        decode(datagram[:10])


def test_payload_with_wrong_length_is_rejected():
    datagram = bytearray(encode(PAYLOADS[0], sender_id=1, seq=1))
    del datagram[-6:-2]
    body = bytes(datagram[:-2])
    datagram[-2:] = crc16_ccitt(body).to_bytes(2, "big")

    with pytest.raises(ProtocolError, match="payload bytes"):
        decode(bytes(datagram))


def test_sender_id_is_range_checked():
    with pytest.raises(ProtocolError, match="sender_id"):
        encode(PAYLOADS[0], sender_id=70000, seq=0)


def test_sequence_number_wraps_rather_than_overflowing():
    datagram = encode(PAYLOADS[0], sender_id=1, seq=0x1_0000_0001)
    header, _ = decode(datagram)

    assert header.seq == 1


def test_header_reports_message_age():
    datagram = encode(PAYLOADS[0], sender_id=1, seq=0, tx_time_us=1_000_000)
    header, _ = decode(datagram)

    assert header.age_s(reference_us=1_250_000) == pytest.approx(0.25)


def test_semaphore_countdown_converts_to_seconds():
    _, decoded = decode(encode(PAYLOADS[1], sender_id=1, seq=0))

    assert decoded.colour is SemaphoreState.YELLOW
    assert decoded.ttl_s == pytest.approx(1.75)


def test_magic_is_what_the_firmware_expects():
    # Hard-coded in the light firmware.
    assert MAGIC == b"BF"


def test_negative_coordinates_survive_the_wire():
    payload = PoseFix(vehicle_id=1, x=-2.5, y=-0.125, yaw=-math.pi + 0.01,
                      sigma_pos=0.03, sigma_yaw=0.01)
    _, decoded = decode(encode(payload, sender_id=1, seq=0))

    assert decoded.x == pytest.approx(-2.5)
    assert decoded.y == pytest.approx(-0.125)
    assert decoded.yaw == pytest.approx(-math.pi + 0.01, abs=1e-6)


def test_message_types_are_stable():
    # On the wire and in firmware: reordering silently reinterprets every datagram.
    assert MessageType.POSE_FIX == 0x01
    assert MessageType.SEMAPHORE == 0x02
    assert MessageType.PEER_STATE == 0x03
    assert MessageType.OBSTACLE_REPORT == 0x04
    assert MessageType.OBSTACLE_ACK == 0x05
