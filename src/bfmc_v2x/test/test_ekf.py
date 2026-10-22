"""The comparison that matters is three-way: dead reckoning, fixes fused at the wrong
timestamp, and fixes fused at the right one.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pytest

from bfmc_v2x.ekf import DeadReckoning, DelayedFusionEKF, ProcessNoise
from bfmc_v2x.geometry import Pose2D, wrap_angle

#: Discretisation error well below the sensor error being measured.
SIM_HZ = 100.0
SIM_DT = 1.0 / SIM_HZ

TRUE_SPEED = 0.5
TRUE_GYRO_BIAS = 0.02
FIX_PERIOD_STEPS = 25      # 4 Hz
SPEED_PERIOD_STEPS = 2     # 50 Hz


@dataclass
class Sample:
    t: float
    truth: Pose2D
    gyro: float
    wheel_speed: Optional[float]
    fix: Optional[Tuple[float, float, float]]


def simulate(
    duration_s: float = 20.0,
    seed: int = 7,
    fix_sigma_pos: float = 0.02,
    fix_sigma_yaw: float = 0.01,
    gyro_noise: float = 0.01,
    speed_noise: float = 0.01,
) -> List[Sample]:
    """The turn turns an unlearned gyro bias into position error; a straight would hide it."""
    rng = random.Random(seed)
    samples: List[Sample] = []
    x = y = yaw = 0.0
    steps = int(duration_s * SIM_HZ)

    for index in range(steps):
        t = index * SIM_DT
        yaw_rate = 0.0 if t < 4.0 else 0.35

        samples.append(
            Sample(
                t=t,
                truth=Pose2D(x, y, yaw),
                gyro=yaw_rate + TRUE_GYRO_BIAS + rng.gauss(0.0, gyro_noise),
                wheel_speed=(
                    TRUE_SPEED + rng.gauss(0.0, speed_noise)
                    if index % SPEED_PERIOD_STEPS == 0
                    else None
                ),
                fix=(
                    (
                        x + rng.gauss(0.0, fix_sigma_pos),
                        y + rng.gauss(0.0, fix_sigma_pos),
                        wrap_angle(yaw + rng.gauss(0.0, fix_sigma_yaw)),
                    )
                    if index % FIX_PERIOD_STEPS == 0
                    else None
                ),
            )
        )

        mid_yaw = yaw + 0.5 * yaw_rate * SIM_DT
        x += TRUE_SPEED * math.cos(mid_yaw) * SIM_DT
        y += TRUE_SPEED * math.sin(mid_yaw) * SIM_DT
        yaw = wrap_angle(yaw + yaw_rate * SIM_DT)

    return samples


def run_filter(
    samples: List[Sample],
    latency_s: float,
    honest_timestamps: bool,
    fix_sigma_pos: float = 0.02,
    fix_sigma_yaw: float = 0.01,
) -> Tuple[DelayedFusionEKF, List[float]]:
    """Without ``honest_timestamps`` fixes are fused as if just measured."""
    ekf = DelayedFusionEKF(
        initial_pose=samples[0].truth,
        initial_stamp=samples[0].t,
        process_noise=ProcessNoise(),
    )
    pending: List[Tuple[float, float, Tuple[float, float, float]]] = []
    errors: List[float] = []

    for sample in samples:
        ekf.predict(sample.t, sample.gyro)
        if sample.wheel_speed is not None:
            ekf.update_wheel_speed(sample.t, sample.wheel_speed)
        if sample.fix is not None:
            pending.append((sample.t + latency_s, sample.t, sample.fix))

        while pending and pending[0][0] <= sample.t:
            _, measured_at, (fx, fy, fyaw) = pending.pop(0)
            stamp = measured_at if honest_timestamps else sample.t
            ekf.update_pose_fix(stamp, fx, fy, fyaw, fix_sigma_pos, fix_sigma_yaw)

        errors.append(math.hypot(ekf.pose.x - sample.truth.x, ekf.pose.y - sample.truth.y))

    return ekf, errors


def rms_tail(values: List[float], tail_fraction: float = 0.6) -> float:
    start = int(len(values) * (1.0 - tail_fraction))
    tail = values[start:]
    return math.sqrt(sum(v * v for v in tail) / len(tail))


# ---- the claims ----------------------------------------------------------


def test_fusion_beats_dead_reckoning():
    samples = simulate()

    _, fused_errors = run_filter(samples, latency_s=0.25, honest_timestamps=True)

    reckoner = DeadReckoning(samples[0].truth, samples[0].t)
    dr_errors: List[float] = []
    for sample in samples:
        reckoner.predict(sample.t, sample.gyro)
        if sample.wheel_speed is not None:
            reckoner.update_wheel_speed(sample.t, sample.wheel_speed)
        dr_errors.append(
            math.hypot(reckoner.pose.x - sample.truth.x, reckoner.pose.y - sample.truth.y)
        )

    fused = rms_tail(fused_errors)
    reckoned = rms_tail(dr_errors)

    assert reckoned > 0.20, "dead reckoning should drift; got {:.3f} m".format(reckoned)
    assert fused < 0.06, "fused estimate should stay tight; got {:.3f} m".format(fused)
    assert fused < reckoned / 4.0


def test_honest_timestamps_beat_naive_ones():
    """When a fix is fused matters as much as whether."""
    samples = simulate()

    _, honest = run_filter(samples, latency_s=0.3, honest_timestamps=True)
    _, naive = run_filter(samples, latency_s=0.3, honest_timestamps=False)

    honest_rms = rms_tail(honest)
    naive_rms = rms_tail(naive)

    # Late fixes fused as current drag the estimate back by about speed x latency.
    assert naive_rms > 0.5 * TRUE_SPEED * 0.3
    assert honest_rms < naive_rms / 2.0, (
        "delayed fusion {:.3f} m vs naive {:.3f} m".format(honest_rms, naive_rms)
    )


def test_replay_is_exactly_equivalent_to_never_having_lagged():
    """The rewind must reproduce the zero-latency filter exactly; any drift means an interval
    is double-counted or dropped.
    """
    samples = simulate(duration_s=12.0)

    instant, _ = run_filter(samples, latency_s=0.0, honest_timestamps=True)
    delayed, _ = run_filter(samples, latency_s=0.22, honest_timestamps=True)

    assert delayed.accepted_fixes == instant.accepted_fixes
    assert delayed.replayed_events > 0, "the replay path was never exercised"

    np.testing.assert_allclose(
        [delayed.pose.x, delayed.pose.y, delayed.pose.yaw],
        [instant.pose.x, instant.pose.y, instant.pose.yaw],
        atol=1e-9,
    )


def test_gyro_bias_is_estimated():
    samples = simulate(duration_s=25.0)
    ekf, _ = run_filter(samples, latency_s=0.2, honest_timestamps=True)

    assert ekf.gyro_bias == pytest.approx(TRUE_GYRO_BIAS, abs=0.008)


def test_mis_associated_fix_is_gated_out():
    """The tracker occasionally locks onto the wrong car."""
    samples = simulate(duration_s=10.0)
    ekf = DelayedFusionEKF(samples[0].truth, samples[0].t)

    for sample in samples[:400]:
        ekf.predict(sample.t, sample.gyro)
        if sample.wheel_speed is not None:
            ekf.update_wheel_speed(sample.t, sample.wheel_speed)
        if sample.fix is not None:
            ekf.update_pose_fix(sample.t, *sample.fix, 0.02, 0.01)

    before = ekf.pose
    # A 95 % gate rejects a few honest fixes by construction: count the delta.
    rejected_before = ekf.rejected_fixes
    last = samples[399]

    accepted = ekf.update_pose_fix(last.t, last.truth.x + 2.5, last.truth.y - 1.8, 0.0, 0.02, 0.01)

    assert accepted is False
    assert ekf.rejected_fixes == rejected_before + 1
    assert ekf.pose.x == pytest.approx(before.x)
    assert ekf.pose.y == pytest.approx(before.y)


@pytest.mark.parametrize("seed", [7, 11, 23])
def test_filter_is_statistically_consistent(seed):
    """Mean NIS near 3 with ~5 % gated: the covariance is honest, not just small. An
    overconfident filter gates good fixes away and drifts while insisting it knows.
    """
    samples = simulate(duration_s=30.0, seed=seed)
    ekf = DelayedFusionEKF(samples[0].truth, samples[0].t)

    scores: List[float] = []
    for sample in samples:
        ekf.predict(sample.t, sample.gyro)
        if sample.wheel_speed is not None:
            ekf.update_wheel_speed(sample.t, sample.wheel_speed)
        if sample.fix is not None:
            ekf.update_pose_fix(sample.t, *sample.fix, 0.02, 0.01)
            scores.append(ekf.last_nis)

    mean_nis = sum(scores) / len(scores)
    gate_rate = ekf.rejected_fixes / len(scores)

    assert mean_nis == pytest.approx(3.0, abs=0.75), "mean NIS {:.2f}".format(mean_nis)
    assert gate_rate < 0.08, "gated {:.1%} of honest fixes".format(gate_rate)


def test_a_plausible_fix_is_still_accepted_after_a_rejection():
    """Gating must not latch."""
    samples = simulate(duration_s=10.0)
    ekf, _ = run_filter(samples[:300], latency_s=0.1, honest_timestamps=True)

    ekf.update_pose_fix(ekf.stamp, 9.0, 9.0, 0.0, 0.02, 0.01)
    assert ekf.rejected_fixes == 1

    truth = samples[299].truth
    assert ekf.update_pose_fix(ekf.stamp, truth.x, truth.y, truth.yaw, 0.05, 0.02) is True


def test_covariance_stays_symmetric_and_positive_definite():
    samples = simulate(duration_s=20.0)
    ekf, _ = run_filter(samples, latency_s=0.25, honest_timestamps=True)

    covariance = ekf.estimate().covariance

    np.testing.assert_allclose(covariance, covariance.T, atol=1e-12)
    assert np.all(np.linalg.eigvalsh(covariance) > 0.0)


def test_uncertainty_grows_without_fixes_and_shrinks_with_one():
    samples = simulate(duration_s=6.0)
    ekf = DelayedFusionEKF(samples[0].truth, samples[0].t)

    for sample in samples[:300]:
        ekf.predict(sample.t, sample.gyro)
        if sample.wheel_speed is not None:
            ekf.update_wheel_speed(sample.t, sample.wheel_speed)

    drifted = ekf.estimate().position_sigma
    truth = samples[299].truth
    ekf.update_pose_fix(ekf.stamp, truth.x, truth.y, truth.yaw, 0.02, 0.01)
    corrected = ekf.estimate().position_sigma

    assert drifted > corrected * 2.0


def test_non_monotonic_predict_is_ignored():
    ekf = DelayedFusionEKF(Pose2D(0.0, 0.0, 0.0), 0.0)
    ekf.predict(0.1, 0.0)
    ekf.update_wheel_speed(0.1, 0.5)
    before = ekf.pose

    ekf.predict(0.05, 0.0)
    ekf.predict(0.1, 0.0)

    assert ekf.stamp == pytest.approx(0.1)
    assert ekf.pose == before


def test_fix_older_than_the_buffer_is_counted_not_dropped():
    samples = simulate(duration_s=6.0)
    ekf, _ = run_filter(samples[:400], latency_s=0.1, honest_timestamps=True)

    stale_stamp = ekf.stamp - 5.0  # far outside the 1.5 s horizon
    truth = samples[100].truth
    ekf.update_pose_fix(stale_stamp, truth.x, truth.y, truth.yaw, 0.05, 0.02)

    assert ekf.stale_fixes == 1


def test_buffer_is_bounded():
    samples = simulate(duration_s=20.0)
    ekf, _ = run_filter(samples, latency_s=0.2, honest_timestamps=True)

    # 1.5 s of 100 Hz predicts plus 50 Hz speed updates, with slack.
    assert ekf.buffered_events < 260


def test_heading_wrap_does_not_destabilise_the_filter():
    """Due west sits on the +/-pi discontinuity."""
    ekf = DelayedFusionEKF(Pose2D(0.0, 0.0, math.pi - 0.02), 0.0)
    t = 0.0
    for _ in range(600):
        t += SIM_DT
        ekf.predict(t, 0.05)
        ekf.update_wheel_speed(t, 0.4)

    # A fix on the other side of the branch cut.
    accepted = ekf.update_pose_fix(ekf.stamp, ekf.pose.x, ekf.pose.y, -math.pi + 0.01, 0.05, 0.05)

    assert accepted is True
    assert -math.pi <= ekf.pose.yaw <= math.pi
    assert abs(wrap_angle(ekf.pose.yaw)) > 3.0
