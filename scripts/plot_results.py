#!/usr/bin/env python3
"""Plot the localisation error recorded by a run.

    python3 scripts/plot_results.py ~/bfmc_runs/ego_localization.csv

Fused error and the dead-reckoning shadow come from the same run, so the gap is the
V2X link's contribution. Without matplotlib it prints the summary instead.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import sys
from typing import Dict, List, Optional, Sequence


def read_csv(path: str) -> Dict[str, List[float]]:
    columns: Dict[str, List[float]] = {}
    with open(path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for field in reader.fieldnames or []:
            columns[field] = []
        for row in reader:
            for field, raw in row.items():
                try:
                    columns[field].append(float(raw))
                except (TypeError, ValueError):
                    columns[field].append(math.nan)
    return columns


def _clean(values: Sequence[float]) -> List[float]:
    """Drop the -1 "no ground truth" sentinel."""
    return [v for v in values if v is not None and not math.isnan(v) and v >= 0.0]


def _percentile(values: Sequence[float], fraction: float) -> float:
    if not values:
        return math.nan
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(round(fraction * (len(ordered) - 1)))))
    return ordered[index]


def summarise(name: str, columns: Dict[str, List[float]]) -> None:
    fused = _clean(columns.get("fused_error_m", []))
    reckoned = _clean(columns.get("dead_reckoning_error_m", []))
    if not fused:
        print("{}: no ground-truth samples, nothing to compare".format(name))
        return

    fused_mean = sum(fused) / len(fused)
    reckoned_mean = sum(reckoned) / len(reckoned) if reckoned else math.nan

    print("")
    print(name)
    print("-" * len(name))
    print("  samples                  {}".format(len(fused)))
    print("  fused error      mean {:.3f} m   p95 {:.3f} m   max {:.3f} m".format(
        fused_mean, _percentile(fused, 0.95), max(fused)))
    if reckoned:
        print("  dead reckoning   mean {:.3f} m   p95 {:.3f} m   max {:.3f} m".format(
            reckoned_mean, _percentile(reckoned, 0.95), max(reckoned)))
        if fused_mean > 1e-9:
            print("  improvement      {:.1f}x".format(reckoned_mean / fused_mean))

    accepted = columns.get("accepted_fixes") or []
    rejected = columns.get("rejected_fixes") or []
    replayed = columns.get("replayed_events") or []
    if accepted:
        print("  V2X fixes        {:.0f} accepted, {:.0f} gated".format(
            accepted[-1], rejected[-1] if rejected else 0))
    if replayed:
        print("  events replayed  {:.0f}".format(replayed[-1]))

    ages = _clean(columns.get("fix_age_s", []))
    if ages:
        print("  fix latency      mean {:.0f} ms   max {:.0f} ms".format(
            1000.0 * sum(ages) / len(ages), 1000.0 * max(ages)))


def plot(paths: Sequence[str], output: Optional[str]) -> int:
    try:
        import matplotlib
        if output:
            matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib is not installed, so no figure was drawn.", file=sys.stderr)
        print("Install it with: sudo apt-get install python3-matplotlib", file=sys.stderr)
        return 0

    figure, axes = plt.subplots(
        len(paths), 1, figsize=(10, 4 * len(paths)), squeeze=False, sharex=True
    )

    for index, path in enumerate(paths):
        columns = read_csv(path)
        axis = axes[index][0]
        name = os.path.basename(path).replace("_localization.csv", "")

        t = columns.get("t", [])
        if t:
            start = t[0]
            t = [v - start for v in t]

        fused = columns.get("fused_error_m", [])
        reckoned = columns.get("dead_reckoning_error_m", [])
        # Mask rather than drop, so both series stay aligned with time.
        fused = [v if v >= 0.0 else math.nan for v in fused]
        reckoned = [v if v >= 0.0 else math.nan for v in reckoned]

        axis.plot(t, reckoned, label="wheel odometry only", color="#e05046", linewidth=1.4)
        axis.plot(t, fused, label="fused with V2X", color="#3c8cf0", linewidth=1.6)

        sigma = columns.get("position_sigma_m", [])
        if sigma:
            # If the 1-sigma tracks the error the covariance is honest; well below, the
            # filter is overconfident.
            axis.plot(t, sigma, label="filter 1-sigma", color="#9aa0a6",
                      linewidth=1.0, linestyle="--")

        axis.set_title("{} -- position error against ground truth".format(name))
        axis.set_ylabel("error (m)")
        axis.grid(alpha=0.25)
        axis.legend(loc="upper left")
        axis.set_ylim(bottom=0.0)

    axes[-1][0].set_xlabel("time (s)")
    figure.tight_layout()

    if output:
        figure.savefig(output, dpi=140)
        print("\nwrote {}".format(output))
    else:
        plt.show()
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument(
        "csv",
        nargs="*",
        help="CSV files written by the metrics node (default: everything in ~/bfmc_runs)",
    )
    parser.add_argument("-o", "--output", help="save the figure instead of showing it")
    args = parser.parse_args(argv)

    paths = list(args.csv)
    if not paths:
        directory = os.path.join(os.path.expanduser("~"), "bfmc_runs")
        if os.path.isdir(directory):
            paths = sorted(
                os.path.join(directory, f)
                for f in os.listdir(directory)
                if f.endswith("_localization.csv")
            )

    if not paths:
        print("No CSV files found. Run the demo first:", file=sys.stderr)
        print("    ./scripts/run_demo.sh", file=sys.stderr)
        return 1

    for path in paths:
        if not os.path.exists(path):
            print("missing: {}".format(path), file=sys.stderr)
            return 1
        summarise(os.path.basename(path), read_csv(path))

    return plot(paths, args.output)


if __name__ == "__main__":
    raise SystemExit(main())
