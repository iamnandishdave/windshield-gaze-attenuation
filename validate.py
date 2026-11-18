#!/usr/bin/env python3
"""Integrity checks for the supplementary package.

Verifies required files exist, CSV shapes match the packaged runs, and the
seed list is consistent with parameters.json. Run from the package root:

    python3 validate.py
"""
import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

EXPECTED = {
    "outputs/per_seed_metrics.csv": (14401, 15),
    "outputs/summary_statistics.csv": (127, 8),
    "outputs/paired_comparisons.csv": (29, 7),
    "outputs/tradeoff_sweep.csv": (7201, 16),
    "outputs/containment_montecarlo.csv": (27, 7),
    "outputs/groundtruth_sensitivity.csv": (79, 4),
    "outputs/groundtruth_comparison.csv": (358, 7),
    "outputs/alignment_diagnostics.csv": (7, 6),
    "outputs/latency_checks.csv": (13, 2),
    "outputs/rule_base.csv": (28, 8),
    "REQUIRED_FILES": [
        "README.md", "CITATION.cff", "LICENSE", "parameters.json",
        "requirements.txt",
        "code/run_analysis.py", "code/make_figures.py", "code/src/core.py",
        "outputs/environment.json", "outputs/seed_list.json",
        "outputs/example_traces.npz",
        "outputs/groundtruth_sensitivity.csv",
        "outputs/groundtruth_comparison.csv",
        "outputs/alignment_diagnostics.csv",
    ],
    "FIGURES": [f"figures/figure_{i}_{name}.{ext}"
                for i, name in [(1, "geometry"), (2, "memberships"),
                                (3, "architecture"), (4, "multiscenario"),
                                (5, "tradeoff"), (6, "containment")]
                for ext in ("eps", "png")],
}

FAMILIES = ["S1", "S2", "S3", "S4", "S5", "S6"]
SEEDS_PER_FAMILY = 200

failures = []


def check(cond, msg):
    status = "ok" if cond else "FAIL"
    print(f"[{status}] {msg}")
    if not cond:
        failures.append(msg)


for rel in EXPECTED["REQUIRED_FILES"]:
    check(os.path.isfile(os.path.join(ROOT, rel)), f"exists: {rel}")

for rel in EXPECTED["FIGURES"]:
    check(os.path.isfile(os.path.join(ROOT, rel)), f"exists: {rel}")

for rel, shape in EXPECTED.items():
    if rel in ("REQUIRED_FILES", "FIGURES"):
        continue
    nrows, ncols = shape
    path = os.path.join(ROOT, rel)
    if not os.path.isfile(path):
        check(False, f"csv shape: {rel} (missing)")
        continue
    with open(path, newline="") as fh:
        rows = list(csv.reader(fh))
    check(len(rows) == nrows, f"csv shape: {rel} rows {len(rows)} == {nrows}")
    if ncols is not None:
        check(len(rows[0]) == ncols,
              f"csv shape: {rel} cols {len(rows[0])} == {ncols}")

params = json.load(open(os.path.join(ROOT, "parameters.json")))
seeds = json.load(open(os.path.join(ROOT, "outputs/seed_list.json")))
check(set(seeds) == set(FAMILIES), "seed families == S1..S6")
check(all(len(seeds[f]) == SEEDS_PER_FAMILY for f in FAMILIES),
      f"seeds per family == {SEEDS_PER_FAMILY}")
check(params["random_seed"] == 20261027,
      "parameters.json family base seed == 20261027")
check(params["sample_rate_hz"] == 50.0 and params["duration_s"] == 60.0,
      "parameters.json sample rate / duration unchanged")
motion = params["containment_error_model"]["vehicle_yaw_rate_deg_s"]
check(motion["ordinary"] == [1, 8] and motion["aggressive"] == [10, 40],
      "ordinary/aggressive yaw ranges match corrected manuscript")
basis = params["containment_error_model"]["ordinary_motion_basis"]
check(basis["curve_radius_m"] == [180, 900]
      and basis["vehicle_speed_m_s"] == [15, 25],
      "ordinary curve-radius/speed context matches corrected manuscript")

print()
if failures:
    print(f"{len(failures)} check(s) FAILED")
    sys.exit(1)
print("All integrity checks passed.")
