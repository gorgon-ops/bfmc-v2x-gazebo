"""One CSV row per sample, and a summary at shutdown. Percentiles, because a mean hides the
moment the filter lost the plot in a corner.
"""

from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Dict, List

import rclpy
from rclpy.node import Node

import bfmc_v2x_msgs.msg as bfmc_msgs

from .common import STATE_QOS


@dataclass
class _VehicleLog:
    name: str
    fused_error: List[float] = field(default_factory=list)
    reckoned_error: List[float] = field(default_factory=list)
    heading_error: List[float] = field(default_factory=list)
    sigma: List[float] = field(default_factory=list)
    fix_age: List[float] = field(default_factory=list)
    gyro_bias: List[float] = field(default_factory=list)
    accepted: int = 0
    rejected: int = 0
    stale: int = 0
    replayed: int = 0
    stops: int = 0
    yields: int = 0
    emergency_stops: int = 0
    samples: int = 0


def _percentile(values: List[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def _mean(values: List[float]) -> float:
    return sum(values) / len(values) if values else math.nan


class MetricsNode(Node):
    def __init__(self) -> None:
        super().__init__("metrics")

        self.declare_parameter("vehicles", ["ego", "peer"])
        self.declare_parameter("output_dir", "")
        self.declare_parameter("write_csv", True)

        self.vehicles = list(self.get_parameter("vehicles").value)
        self._logs: Dict[str, _VehicleLog] = {v: _VehicleLog(v) for v in self.vehicles}

        self._writers: Dict[str, object] = {}
        self._files: Dict[str, object] = {}
        if bool(self.get_parameter("write_csv").value):
            self._open_files(str(self.get_parameter("output_dir").value))

        for vehicle in self.vehicles:
            self.create_subscription(
                bfmc_msgs.LocalizationDiagnostics,
                "/{}/localization/diagnostics".format(vehicle),
                self._make_diagnostics_callback(vehicle),
                STATE_QOS,
            )
            self.create_subscription(
                bfmc_msgs.DriveStatus,
                "/{}/drive_status".format(vehicle),
                self._make_status_callback(vehicle),
                STATE_QOS,
            )

        self.create_timer(20.0, self._log_progress)
        self.get_logger().info("metrics recording {}".format(", ".join(self.vehicles)))

    def _open_files(self, output_dir: str) -> None:
        directory = output_dir or os.path.join(os.path.expanduser("~"), "bfmc_runs")
        try:
            os.makedirs(directory, exist_ok=True)
        except OSError as exc:  # pragma: no cover - read-only home
            self.get_logger().warn("cannot write metrics to {}: {}".format(directory, exc))
            return

        for vehicle in self.vehicles:
            path = os.path.join(directory, "{}_localization.csv".format(vehicle))
            handle = open(path, "w", newline="", encoding="utf-8")
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "t",
                    "fused_error_m",
                    "dead_reckoning_error_m",
                    "heading_error_rad",
                    "position_sigma_m",
                    "fix_age_s",
                    "gyro_bias_rad_s",
                    "last_nis",
                    "accepted_fixes",
                    "rejected_fixes",
                    "replayed_events",
                ]
            )
            self._files[vehicle] = handle
            self._writers[vehicle] = writer
            self.get_logger().info("writing {}".format(path))

    def _make_diagnostics_callback(self, vehicle: str):
        def callback(message: bfmc_msgs.LocalizationDiagnostics) -> None:
            log = self._logs[vehicle]
            log.samples += 1
            log.accepted = message.accepted_fixes
            log.rejected = message.rejected_fixes
            log.stale = message.stale_fixes
            log.replayed = message.replayed_events
            log.sigma.append(message.position_sigma)
            log.fix_age.append(message.fix_age)
            log.gyro_bias.append(message.gyro_bias)

            # -1: no ground truth (real hardware). Not an error.
            if message.position_error >= 0.0:
                log.fused_error.append(message.position_error)
                log.heading_error.append(abs(message.heading_error))
            if message.dead_reckoning_error >= 0.0:
                log.reckoned_error.append(message.dead_reckoning_error)

            writer = self._writers.get(vehicle)
            if writer is not None:
                writer.writerow(
                    [
                        "{:.3f}".format(self.get_clock().now().nanoseconds * 1e-9),
                        "{:.5f}".format(message.position_error),
                        "{:.5f}".format(message.dead_reckoning_error),
                        "{:.5f}".format(message.heading_error),
                        "{:.5f}".format(message.position_sigma),
                        "{:.4f}".format(message.fix_age),
                        "{:.6f}".format(message.gyro_bias),
                        "{:.4f}".format(message.last_nis),
                        message.accepted_fixes,
                        message.rejected_fixes,
                        message.replayed_events,
                    ]
                )

        return callback

    def _make_status_callback(self, vehicle: str):
        previous = {"factor": ""}

        def callback(message: bfmc_msgs.DriveStatus) -> None:
            log = self._logs[vehicle]
            # Transitions, not samples: four seconds stopped is one stop.
            if message.limiting_factor != previous["factor"]:
                if message.limiting_factor == "signal" and message.commanded_speed <= 0.01:
                    log.stops += 1
                elif message.limiting_factor == "v2v":
                    if message.right_of_way == "EMERGENCY_STOP":
                        log.emergency_stops += 1
                    elif message.right_of_way == "YIELD":
                        log.yields += 1
                previous["factor"] = message.limiting_factor

        return callback

    def _log_progress(self) -> None:
        for log in self._logs.values():
            if not log.fused_error:
                continue
            self.get_logger().info(
                "{}: fused {:.3f} m mean / {:.3f} m p95, dead reckoning {:.3f} m, "
                "{} fixes ({} gated)".format(
                    log.name,
                    _mean(log.fused_error),
                    _percentile(log.fused_error, 0.95),
                    _mean(log.reckoned_error),
                    log.accepted,
                    log.rejected,
                )
            )

    def destroy_node(self) -> bool:
        self._print_summary()
        for handle in self._files.values():
            try:
                handle.close()
            except Exception:  # pragma: no cover
                pass
        return super().destroy_node()

    def _print_summary(self) -> None:
        logger = self.get_logger()
        logger.info("=" * 72)
        logger.info("RUN SUMMARY")
        logger.info("=" * 72)

        for log in self._logs.values():
            if log.samples == 0:
                logger.info("{}: no data".format(log.name))
                continue

            fused_mean = _mean(log.fused_error)
            reckoned_mean = _mean(log.reckoned_error)
            improvement = (
                reckoned_mean / fused_mean
                if fused_mean and fused_mean > 1e-9 and not math.isnan(reckoned_mean)
                else math.nan
            )

            logger.info("")
            logger.info("{} -- {} samples".format(log.name, log.samples))
            logger.info("  position error, fused     mean {:.3f} m   p95 {:.3f} m   max {:.3f} m"
                        .format(fused_mean, _percentile(log.fused_error, 0.95),
                                max(log.fused_error) if log.fused_error else math.nan))
            logger.info("  position error, odometry  mean {:.3f} m   p95 {:.3f} m   max {:.3f} m"
                        .format(reckoned_mean, _percentile(log.reckoned_error, 0.95),
                                max(log.reckoned_error) if log.reckoned_error else math.nan))
            logger.info("  improvement               {:.1f}x".format(improvement))
            logger.info("  heading error             mean {:.2f} deg"
                        .format(math.degrees(_mean(log.heading_error))))
            logger.info("  filter's own sigma        mean {:.3f} m".format(_mean(log.sigma)))
            logger.info("  gyro bias estimate        {:.4f} rad/s".format(
                log.gyro_bias[-1] if log.gyro_bias else math.nan))
            logger.info("  V2X fixes                 {} accepted, {} gated, {} stale"
                        .format(log.accepted, log.rejected, log.stale))
            logger.info("  fix latency               mean {:.0f} ms"
                        .format(_mean(log.fix_age) * 1000.0))
            logger.info("  events replayed           {}".format(log.replayed))
            logger.info("  behaviour                 {} signal stops, {} yields, {} emergency"
                        .format(log.stops, log.yields, log.emergency_stops))
        logger.info("=" * 72)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = MetricsNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
