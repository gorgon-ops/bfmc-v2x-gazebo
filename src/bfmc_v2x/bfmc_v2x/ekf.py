"""EKF fusing wheel odometry, a gyro and late V2X pose fixes.

A fix describes where the car was when the camera shutter opened, 10-20 cm ago.
Applied as if current, it drags the estimate back by latency x speed - a constant
bias the filter cannot learn away. So the filter rewinds to the buffered state at
the fix's stamp, updates there, and replays the inputs since.

State ``[x, y, theta, v, b_omega]``: the gyro bias becomes observable once absolute
headings arrive, and estimating it keeps the heading from ramping between fixes.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from .geometry import Pose2D, wrap_angle

__all__ = [
    "ProcessNoise",
    "FilterEstimate",
    "DelayedFusionEKF",
    "DeadReckoning",
    "STATE_DIM",
    "IDX_X",
    "IDX_Y",
    "IDX_YAW",
    "IDX_V",
    "IDX_GYRO_BIAS",
]

STATE_DIM = 5
IDX_X, IDX_Y, IDX_YAW, IDX_V, IDX_GYRO_BIAS = range(STATE_DIM)

#: 95 % chi-square gate for a 3-DoF (x, y, yaw) measurement.
_CHI2_3DOF_95 = 7.815


@dataclass(frozen=True)
class ProcessNoise:
    """Continuous-time process noise, SI units, sized for the 1:10 car and a consumer MEMS gyro."""

    #: m/s^2 per sqrt(s).
    accel: float = 0.6
    #: rad/s per sqrt(s).
    gyro: float = 0.02
    #: rad/s^2 per sqrt(s).
    gyro_bias: float = 0.001
    #: Lateral slip the kinematic model cannot represent, m/s per sqrt(s).
    slip: float = 0.02


@dataclass(frozen=True)
class FilterEstimate:
    stamp: float
    pose: Pose2D
    speed: float
    gyro_bias: float
    covariance: np.ndarray

    @property
    def position_sigma(self) -> float:
        """One-sigma position uncertainty as a single radius, metres."""
        return math.sqrt(max(0.0, self.covariance[IDX_X, IDX_X] + self.covariance[IDX_Y, IDX_Y]))

    @property
    def yaw_sigma(self) -> float:
        return math.sqrt(max(0.0, self.covariance[IDX_YAW, IDX_YAW]))


@dataclass
class _Event:
    """Storing the posterior with each event makes the rewind a binary search, not a re-integration.
    """

    stamp: float
    kind: str
    dt: float = 0.0
    yaw_rate: float = 0.0
    speed: float = 0.0
    speed_sigma: float = 0.0
    state: np.ndarray = field(default_factory=lambda: np.zeros(STATE_DIM))
    covariance: np.ndarray = field(default_factory=lambda: np.zeros((STATE_DIM, STATE_DIM)))


class DelayedFusionEKF:
    def __init__(
        self,
        initial_pose: Pose2D = Pose2D(0.0, 0.0, 0.0),
        initial_stamp: float = 0.0,
        process_noise: ProcessNoise = ProcessNoise(),
        buffer_horizon_s: float = 1.5,
        gate_chi2: float = _CHI2_3DOF_95,
    ) -> None:
        self.noise = process_noise
        self.buffer_horizon_s = buffer_horizon_s
        self.gate_chi2 = gate_chi2

        self._state = np.array(
            [initial_pose.x, initial_pose.y, initial_pose.yaw, 0.0, 0.0], dtype=float
        )
        # Unsure of position and heading, sure it is stationary: nothing launches mid-lap.
        self._covariance = np.diag([0.25, 0.25, 0.25, 0.01, 0.01])
        self._stamp = initial_stamp
        self._history: List[_Event] = []

        #: Normalised innovation squared of the last fix.
        self.last_nis: float = 0.0
        self.rejected_fixes: int = 0
        self.stale_fixes: int = 0
        #: Total events replayed: what latency costs.
        self.replayed_events: int = 0
        self.accepted_fixes: int = 0

    @property
    def stamp(self) -> float:
        return self._stamp

    @property
    def pose(self) -> Pose2D:
        return Pose2D(self._state[IDX_X], self._state[IDX_Y], self._state[IDX_YAW])

    @property
    def speed(self) -> float:
        return float(self._state[IDX_V])

    @property
    def gyro_bias(self) -> float:
        return float(self._state[IDX_GYRO_BIAS])

    def estimate(self) -> FilterEstimate:
        return FilterEstimate(
            stamp=self._stamp,
            pose=self.pose,
            speed=self.speed,
            gyro_bias=self.gyro_bias,
            covariance=self._covariance.copy(),
        )

    def predict(self, stamp: float, yaw_rate: float) -> None:
        """The gyro is an input, not a measurement: a small state that follows fast rotations."""
        dt = stamp - self._stamp
        if dt <= 0.0:
            # A duplicate IMU sample or a clock step.
            return
        self._state, self._covariance = self._propagate(
            self._state, self._covariance, dt, yaw_rate
        )
        self._stamp = stamp
        self._record(_Event(stamp=stamp, kind="predict", dt=dt, yaw_rate=yaw_rate))

    def _propagate(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        dt: float,
        yaw_rate: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        x, y, yaw, v, bias = state
        omega = yaw_rate - bias

        nxt = np.array(
            [
                x + v * math.cos(yaw) * dt,
                y + v * math.sin(yaw) * dt,
                wrap_angle(yaw + omega * dt),
                v,
                bias,
            ]
        )

        jac = np.eye(STATE_DIM)
        jac[IDX_X, IDX_YAW] = -v * math.sin(yaw) * dt
        jac[IDX_X, IDX_V] = math.cos(yaw) * dt
        jac[IDX_Y, IDX_YAW] = v * math.cos(yaw) * dt
        jac[IDX_Y, IDX_V] = math.sin(yaw) * dt
        jac[IDX_YAW, IDX_GYRO_BIAS] = -dt

        return nxt, jac @ covariance @ jac.T + self._process_covariance(dt, yaw)

    def _process_covariance(self, dt: float, yaw: float) -> np.ndarray:
        """Slip is injected perpendicular to the heading: the one direction the model forbids."""
        noise = self.noise
        pos_var = (0.5 * noise.accel * dt * dt) ** 2
        slip_var = (noise.slip * dt) ** 2
        sin_y, cos_y = math.sin(yaw), math.cos(yaw)

        # (longitudinal, lateral) variance pair into the track frame.
        lateral = np.array([[-sin_y], [cos_y]])
        longitudinal = np.array([[cos_y], [sin_y]])
        block = pos_var * (longitudinal @ longitudinal.T) + slip_var * (lateral @ lateral.T)

        cov = np.zeros((STATE_DIM, STATE_DIM))
        cov[0:2, 0:2] = block
        cov[IDX_YAW, IDX_YAW] = (noise.gyro * dt) ** 2
        cov[IDX_V, IDX_V] = (noise.accel * dt) ** 2
        cov[IDX_GYRO_BIAS, IDX_GYRO_BIAS] = (noise.gyro_bias * dt) ** 2
        return cov

    def update_wheel_speed(self, stamp: float, speed: float, sigma: float = 0.05) -> None:
        if stamp > self._stamp:
            # Encoder ahead of the filter: a zero-turn propagation (passing the bias as the
            # rate cancels it) so the update lands at its own stamp.
            self.predict(stamp, float(self._state[IDX_GYRO_BIAS]))

        obs = np.zeros((1, STATE_DIM))
        obs[0, IDX_V] = 1.0
        residual = np.array([speed - self._state[IDX_V]])
        self._state, self._covariance = self._apply_update(
            self._state, self._covariance, obs, residual, np.array([[sigma**2]])
        )
        self._record(
            _Event(
                stamp=self._stamp,
                kind="speed",
                speed=speed,
                speed_sigma=sigma,
            )
        )

    def update_pose_fix(
        self,
        stamp: float,
        x: float,
        y: float,
        yaw: float,
        sigma_pos: float,
        sigma_yaw: float,
    ) -> bool:
        """The chi-square gate stops a mis-association (the tracker locking onto the other car)."""
        if not self._history or stamp >= self._stamp:
            return self._fuse_pose_here(x, y, yaw, sigma_pos, sigma_yaw)

        oldest = self._history[0].stamp
        if stamp < oldest:
            # Older than the buffer: applied at the edge, wrong by however much older it is,
            # but better than drifting longer.
            self.stale_fixes += 1
            stamp = oldest

        index = self._index_at(stamp)
        anchor = self._history[index]
        state = anchor.state.copy()
        covariance = anchor.covariance.copy()

        gap = stamp - anchor.stamp
        if gap > 1e-6:
            state, covariance = self._propagate(state, covariance, gap, state[IDX_GYRO_BIAS])

        fused = self._fuse_pose(state, covariance, x, y, yaw, sigma_pos, sigma_yaw)
        if fused is None:
            self.rejected_fixes += 1
            return False
        state, covariance = fused
        self.accepted_fixes += 1

        # From the fix's stamp, not the anchor's: the gap was already propagated, and
        # replaying the next event's full dt would integrate it twice.
        replay = self._history[index + 1 :]
        self.replayed_events += len(replay)
        cursor = stamp
        for event in replay:
            if event.kind == "predict":
                dt = event.stamp - cursor
                cursor = event.stamp
                if dt > 1e-9:
                    state, covariance = self._propagate(
                        state, covariance, dt, event.yaw_rate
                    )
            elif event.kind == "speed":
                obs = np.zeros((1, STATE_DIM))
                obs[0, IDX_V] = 1.0
                residual = np.array([event.speed - state[IDX_V]])
                state, covariance = self._apply_update(
                    state,
                    covariance,
                    obs,
                    residual,
                    np.array([[event.speed_sigma**2]]),
                )
            event.state = state.copy()
            event.covariance = covariance.copy()

        self._state = state
        self._covariance = covariance
        return True

    def _fuse_pose_here(
        self, x: float, y: float, yaw: float, sigma_pos: float, sigma_yaw: float
    ) -> bool:
        fused = self._fuse_pose(
            self._state, self._covariance, x, y, yaw, sigma_pos, sigma_yaw
        )
        if fused is None:
            self.rejected_fixes += 1
            return False
        self._state, self._covariance = fused
        self.accepted_fixes += 1
        return True

    def _fuse_pose(
        self,
        state: np.ndarray,
        covariance: np.ndarray,
        x: float,
        y: float,
        yaw: float,
        sigma_pos: float,
        sigma_yaw: float,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        obs = np.zeros((3, STATE_DIM))
        obs[0, IDX_X] = 1.0
        obs[1, IDX_Y] = 1.0
        obs[2, IDX_YAW] = 1.0

        residual = np.array(
            [
                x - state[IDX_X],
                y - state[IDX_Y],
                wrap_angle(yaw - state[IDX_YAW]),
            ]
        )
        meas_cov = np.diag([sigma_pos**2, sigma_pos**2, sigma_yaw**2])
        innovation_cov = obs @ covariance @ obs.T + meas_cov

        try:
            nis = float(residual @ np.linalg.solve(innovation_cov, residual))
        except np.linalg.LinAlgError:
            return None
        self.last_nis = nis
        if nis > self.gate_chi2:
            return None

        return self._apply_update(state, covariance, obs, residual, meas_cov)

    @staticmethod
    def _apply_update(
        state: np.ndarray,
        covariance: np.ndarray,
        obs: np.ndarray,
        residual: np.ndarray,
        meas_cov: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Joseph form: keeps the covariance positive-definite over tens of minutes at 100 Hz."""
        innovation_cov = obs @ covariance @ obs.T + meas_cov
        gain = covariance @ obs.T @ np.linalg.inv(innovation_cov)
        state = state + gain @ residual
        state[IDX_YAW] = wrap_angle(state[IDX_YAW])

        ident = np.eye(STATE_DIM)
        factor = ident - gain @ obs
        covariance = factor @ covariance @ factor.T + gain @ meas_cov @ gain.T
        return state, 0.5 * (covariance + covariance.T)

    def _record(self, event: _Event) -> None:
        event.state = self._state.copy()
        event.covariance = self._covariance.copy()
        self._history.append(event)
        self._prune()

    def _prune(self) -> None:
        cutoff = self._stamp - self.buffer_horizon_s
        keep = bisect.bisect_left([e.stamp for e in self._history], cutoff)
        # One event before the cutoff, so a fix exactly at the horizon has an anchor.
        if keep > 1:
            del self._history[: keep - 1]

    def _index_at(self, stamp: float) -> int:
        stamps = [e.stamp for e in self._history]
        index = bisect.bisect_right(stamps, stamp) - 1
        return max(0, index)

    @property
    def buffered_events(self) -> int:
        return len(self._history)


class DeadReckoning:
    """Odometry-only baseline: the control in the experiment, and the fallback when V2X goes quiet.
    """

    def __init__(self, initial_pose: Pose2D = Pose2D(0.0, 0.0, 0.0), stamp: float = 0.0) -> None:
        self.pose = initial_pose
        self.speed = 0.0
        self._stamp = stamp

    def predict(self, stamp: float, yaw_rate: float) -> None:
        dt = stamp - self._stamp
        if dt <= 0.0:
            return
        yaw = wrap_angle(self.pose.yaw + yaw_rate * dt)
        self.pose = Pose2D(
            self.pose.x + self.speed * math.cos(self.pose.yaw) * dt,
            self.pose.y + self.speed * math.sin(self.pose.yaw) * dt,
            yaw,
        )
        self._stamp = stamp

    def update_wheel_speed(self, stamp: float, speed: float) -> None:
        if stamp > self._stamp:
            self.predict(stamp, 0.0)
        self.speed = speed
