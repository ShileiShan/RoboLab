#!/usr/bin/env python3
"""Plot Piper reference/baseline/candidate qpos tracking curves."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REFERENCE_COLOR = "#4c4c4c"
BASELINE_COLOR = "#dc742d"
CANDIDATE_COLOR = "#2c82f6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-timeseries", type=Path, required=True)
    parser.add_argument("--candidate-timeseries", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="qpos_tracking")
    parser.add_argument("--lead-sim-steps", type=int, default=0,
                        help="Shift simulated states earlier by this many timesteps before plotting.")
    parser.add_argument("--lead-sim-one-step", action="store_true",
                        help="Shift simulated states one timestep earlier before plotting.")
    args = parser.parse_args()
    if args.lead_sim_one_step and args.lead_sim_steps == 0:
        args.lead_sim_steps = 1
    if args.lead_sim_steps < 0:
        parser.error("--lead-sim-steps must be non-negative.")
    return args


def _load_npz(path: Path) -> dict:
    with np.load(path, allow_pickle=False) as handle:
        return {key: np.asarray(handle[key]) for key in handle.files}


def _joint_rmse(reference: np.ndarray, actual: np.ndarray) -> float:
    error = np.asarray(actual - reference, dtype=np.float64)
    return float(np.sqrt(np.mean(np.square(error))))


def _lead_steps(values: np.ndarray, steps: int) -> np.ndarray:
    values = np.asarray(values)
    if steps <= 0 or values.shape[0] <= 1:
        return values.copy()
    steps = min(int(steps), values.shape[0] - 1)
    return np.concatenate((values[steps:], np.repeat(values[-1:], steps, axis=0)), axis=0)


def _plot_group(
    *,
    time_s: np.ndarray,
    reference: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    labels: list[str],
    unit: str,
    title: str,
    output_path: Path,
) -> None:
    num_plots = len(labels)
    cols = 2
    rows = int(np.ceil(num_plots / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(16, 3.7 * rows), sharex=True)
    axes = np.atleast_1d(axes).reshape(rows, cols)

    for joint_idx, label in enumerate(labels):
        ax = axes[joint_idx // cols, joint_idx % cols]
        ref = reference[:, joint_idx]
        base = baseline[:, joint_idx]
        cand = candidate[:, joint_idx]
        base_rmse = _joint_rmse(ref, base)
        cand_rmse = _joint_rmse(ref, cand)

        ax.plot(time_s, ref, color=REFERENCE_COLOR, linewidth=1.8, linestyle="--", label="Reference")
        ax.plot(time_s, base, color=BASELINE_COLOR, linewidth=1.6, label=f"Baseline (RMSE={base_rmse:.4f})")
        ax.plot(time_s, cand, color=CANDIDATE_COLOR, linewidth=1.6, label=f"Candidate (RMSE={cand_rmse:.4f})")
        ax.set_title(label)
        ax.set_ylabel(unit)
        ax.grid(True, alpha=0.25, linestyle=":")

    for joint_idx in range(num_plots, rows * cols):
        axes[joint_idx // cols, joint_idx % cols].axis("off")

    for ax in axes[-1, :]:
        if ax.has_data():
            ax.set_xlabel("Time (s)")

    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 0.985))
    fig.suptitle(title, fontsize=16, y=0.998)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    baseline_data = _load_npz(args.baseline_timeseries)
    candidate_data = _load_npz(args.candidate_timeseries)

    common_steps = min(
        len(baseline_data["time_s"]),
        len(candidate_data["time_s"]),
        baseline_data["reference_state"].shape[0],
        candidate_data["actual_state"].shape[0],
        baseline_data["actual_state"].shape[0],
    )
    time_s = np.asarray(baseline_data["time_s"][:common_steps], dtype=np.float64)
    reference_state = np.asarray(baseline_data["reference_state"][:common_steps], dtype=np.float64)
    baseline_state = np.asarray(baseline_data["actual_state"][:common_steps], dtype=np.float64)
    candidate_state = np.asarray(candidate_data["actual_state"][:common_steps], dtype=np.float64)
    if args.lead_sim_steps > 0:
        baseline_state = _lead_steps(baseline_state, args.lead_sim_steps)
        candidate_state = _lead_steps(candidate_state, args.lead_sim_steps)

    left_labels = [f"Left Joint {idx}" for idx in range(1, 7)]
    right_labels = [f"Right Joint {idx}" for idx in range(1, 7)]
    gripper_labels = ["Left Gripper", "Right Gripper"]

    output_dir = args.output_dir
    _plot_group(
        time_s=time_s,
        reference=reference_state[:, 0:6],
        baseline=baseline_state[:, 0:6],
        candidate=candidate_state[:, 0:6],
        labels=left_labels,
        unit="rad",
        title="Left Arm Qpos Tracking",
        output_path=output_dir / f"{args.prefix}_left_arm.png",
    )
    _plot_group(
        time_s=time_s,
        reference=reference_state[:, 6:12],
        baseline=baseline_state[:, 6:12],
        candidate=candidate_state[:, 6:12],
        labels=right_labels,
        unit="rad",
        title="Right Arm Qpos Tracking",
        output_path=output_dir / f"{args.prefix}_right_arm.png",
    )
    _plot_group(
        time_s=time_s,
        reference=reference_state[:, 12:14],
        baseline=baseline_state[:, 12:14],
        candidate=candidate_state[:, 12:14],
        labels=gripper_labels,
        unit="m",
        title="Gripper Qpos Tracking",
        output_path=output_dir / f"{args.prefix}_gripper.png",
    )

    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
