# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Grouped live LiDAR chunk timings from runtime benchmark artifacts."""

import argparse
import json
import math
from pathlib import Path
from statistics import median, quantiles


def summarize(payload: dict, *, warmup: int, allow_single_sweep: bool = False) -> dict:
    """Summarize measured timings without mixing output channels or warmup."""
    if warmup < 0:
        raise ValueError("warmup must be nonnegative")
    if payload.get("artifact_type") != "flashdreams.runtime.demo.benchmark_stats":
        raise ValueError("Expected a flashdreams --stats-path artifact")
    steps = {}
    for sample in payload["samples"]:
        index = sample["step_index"]
        if index is None or sample["result_index"] != 0:
            continue
        if sample["name"] == "lidar_sweeps" or sample["unit"] == "s":
            value = sample["value"]
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError(f"Nonfinite or nonnumeric metric in step {index}")
            metrics = steps.setdefault(index, {})
            if sample["name"] in metrics:
                raise ValueError(f"Duplicate metric {sample['name']} in step {index}")
            metrics[sample["name"]] = value
    measured = [steps[index] for index in sorted(steps)][warmup:]
    if not measured:
        raise ValueError("No measured steps remain after warmup")
    for metrics in measured:
        if metrics.get("lidar_sweeps") not in (1, 2):
            raise ValueError("Expected short joint chunks containing one or two sweeps")
        if "total_s" not in metrics:
            raise ValueError("Missing backend total_s timing")
        if any(value < 0 for value in metrics.values()):
            raise ValueError("Timings and sweep counts must be nonnegative")
    names = set.intersection(*(set(metrics) for metrics in measured)) - {"lidar_sweeps"}
    groups = {}
    for label, rows in (
        ("all", measured),
        ("one_sweep", [row for row in measured if row["lidar_sweeps"] == 1]),
        ("two_sweeps", [row for row in measured if row["lidar_sweeps"] == 2]),
    ):
        if not rows:
            if label == "two_sweeps" and allow_single_sweep:
                continue
            raise ValueError(f"No {label} samples; collect complete three-chunk cycles")
        timings = {}
        for name in sorted(names):
            values = [row[name] * 1000 for row in rows]
            timings[name.removesuffix("_s") + "_ms"] = {
                "median": median(values),
                "p90": quantiles(values, n=10, method="inclusive")[8]
                if len(values) > 1
                else values[0],
            }
        groups[label] = {"count": len(rows), "timings": timings}
    delta = {
        name: {
            statistic: groups["two_sweeps"]["timings"][name][statistic]
            - groups["one_sweep"]["timings"][name][statistic]
            for statistic in ("median", "p90")
        }
        for name in groups["all"]["timings"]
        if "two_sweeps" in groups
    }
    return {
        "discarded_steps": warmup,
        "groups": groups,
        "two_minus_one_ms": delta,
        "scope": "Backend timings, not browser/network/input-to-visible latency. "
        "Overlapping stage times must not be summed.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Group short-chunk --stats-path timings by LiDAR sweep count."
    )
    parser.add_argument("stats", type=Path)
    parser.add_argument("--warmup", type=int, default=7)
    parser.add_argument(
        "--allow-single-sweep",
        action="store_true",
        help="Allow the fixed one-sweep chunks of the 7.5 Hz mode",
    )
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.stats.read_text())
    report = summarize(
        payload, warmup=args.warmup, allow_single_sweep=args.allow_single_sweep
    )
    report["source"] = str(args.stats.resolve())
    report["session"] = payload["session"]
    encoded = json.dumps(report, indent=2, sort_keys=True) + "\n"
    if args.output is not None:
        args.output.write_text(encoded)
    print(encoded, end="")


if __name__ == "__main__":
    main()
