#!/usr/bin/env python3
"""Extract a replayable Piper action trace from a RoboLab episode HDF5 file."""

from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--hdf5", type=Path, required=True, help="Completed RoboLab run_N.hdf5 file.")
    parser.add_argument("--demo", default="demo_0", help="Demo group below data/ (default: demo_0).")
    parser.add_argument("--output", type=Path, required=True, help="Output .npz action-trace file.")
    parser.add_argument("--control-hz", type=float, default=30.0,
                        help="Control rate used when recording (default: 30 Hz).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.control_hz <= 0:
        raise ValueError("--control-hz must be positive.")
    if not args.hdf5.is_file():
        raise FileNotFoundError(f"HDF5 file does not exist: {args.hdf5}")

    dataset_path = f"data/{args.demo}/actions"
    with h5py.File(args.hdf5, "r") as handle:
        if dataset_path not in handle:
            raise KeyError(f"No action dataset at {dataset_path} in {args.hdf5}.")
        actions = np.asarray(handle[dataset_path], dtype=np.float32)

    if actions.ndim != 2 or actions.shape[0] < 1:
        raise ValueError(f"Expected a non-empty 2-D action array, got {actions.shape}.")
    if actions.shape[1] != 14:
        raise ValueError(
            f"Expected a 14-D Double Piper environment action, got {actions.shape[1]} dimensions."
        )
    if not np.isfinite(actions).all():
        raise ValueError("The recorded actions contain NaN or infinity.")

    output = args.output.with_suffix(".npz")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        actions=actions,
        control_hz=np.asarray(args.control_hz, dtype=np.float64),
        source_hdf5=np.asarray(str(args.hdf5.resolve())),
        demo=np.asarray(args.demo),
    )
    print(f"Wrote {actions.shape[0]} replay actions ({actions.shape[1]}-D) to {output}")
    print(f"Replay duration: {actions.shape[0] / args.control_hz:.3f} s at {args.control_hz:g} Hz")


if __name__ == "__main__":
    main()
