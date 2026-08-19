#!/usr/bin/env python3
"""Extract a replayable Piper joint-target trace from raw real-robot HDF5."""

from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "robolab" / "eval" / "piper_joint_target_trace.py"
MODULE_SPEC = importlib.util.spec_from_file_location("piper_joint_target_trace", MODULE_PATH)
if MODULE_SPEC is None or MODULE_SPEC.loader is None:
    raise RuntimeError(f"Could not load module spec from {MODULE_PATH}.")
_module = importlib.util.module_from_spec(MODULE_SPEC)
sys.modules[MODULE_SPEC.name] = _module
MODULE_SPEC.loader.exec_module(_module)

extract_joint_target_trace_from_real_hdf5 = _module.extract_joint_target_trace_from_real_hdf5
save_joint_target_trace = _module.save_joint_target_trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True, help="Raw real-robot HDF5 episode.")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz joint-trace file.")
    parser.add_argument("--control-hz", type=float, default=30.0,
                        help="Control rate used by the dataset (default: 30 Hz).")
    parser.add_argument("--left-gripper-closed", type=float, default=None,
                        help="Optional raw left-gripper value treated as fully closed.")
    parser.add_argument("--left-gripper-open", type=float, default=None,
                        help="Optional raw left-gripper value treated as fully open.")
    parser.add_argument("--right-gripper-closed", type=float, default=None,
                        help="Optional raw right-gripper value treated as fully closed.")
    parser.add_argument("--right-gripper-open", type=float, default=None,
                        help="Optional raw right-gripper value treated as fully open.")
    parser.add_argument("--gripper-quantile-low", type=float, default=0.01,
                        help="Lower quantile for automatic gripper range fitting (default: 0.01).")
    parser.add_argument("--gripper-quantile-high", type=float, default=0.99,
                        help="Upper quantile for automatic gripper range fitting (default: 0.99).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace = extract_joint_target_trace_from_real_hdf5(
        args.hdf5,
        control_hz=args.control_hz,
        left_gripper_closed=args.left_gripper_closed,
        left_gripper_open=args.left_gripper_open,
        right_gripper_closed=args.right_gripper_closed,
        right_gripper_open=args.right_gripper_open,
        gripper_quantile_low=args.gripper_quantile_low,
        gripper_quantile_high=args.gripper_quantile_high,
    )
    output = save_joint_target_trace(trace, args.output)
    print(f"Wrote {trace.steps} reference states ({trace.action_dim}-D env targets) to {output}")
    print(f"Replay duration: {trace.duration_s:.3f} s at {trace.control_hz:g} Hz")
    print(
        "Gripper mapping: "
        f"left raw [{trace.gripper_real_min[0]:.6g}, {trace.gripper_real_max[0]:.6g}] -> [0, 0.035] m; "
        f"right raw [{trace.gripper_real_min[1]:.6g}, {trace.gripper_real_max[1]:.6g}] -> [0, 0.035] m"
    )


if __name__ == "__main__":
    main()
