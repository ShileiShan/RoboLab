# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a recorded Double Piper action trace without a policy server."""

import argparse
import os
import sys
import traceback

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
from isaaclab.app import AppLauncher

from robolab.eval.action_replay import load_action_trace

parser = argparse.ArgumentParser(
    description="Run an open-loop Piper evaluation from a recorded environment-action trace.",
    allow_abbrev=False,
)
parser.add_argument("--action-trace", type=str, required=True,
                    help="Path to a .npz trace created by extract_piper_action_trace.py.")
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")

from robolab.constants import DEFAULT_TASK_SUBFOLDERS  # noqa: E402
from robolab.eval.runner import add_common_eval_args, run_evaluation  # noqa: E402

add_common_eval_args(parser)
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
trace = load_action_trace(args_cli.action_trace)

# Piper registrations use 30 Hz (dt=1/240, decimation=8).  Replaying at a
# different frequency changes the physical meaning of every saved action.
if abs(trace.control_hz - 30.0) > 1.0e-6:
    parser.error(
        f"Action trace control_hz is {trace.control_hz:g}; Piper replay requires 30 Hz."
    )
if trace.action_dim != 14:
    parser.error(f"Action trace has {trace.action_dim} action dimensions; Double Piper requires 14.")
if args_cli.num_envs != 1:
    parser.error("Open-loop replay currently requires --num-envs 1 so one trace maps to one environment.")
if args_cli.num_runs != 1:
    parser.error("Open-loop replay currently requires --num-runs 1; make separate traces for separate episodes.")

# Make the environment time out exactly when the trace ends.  IsaacLab uses
# ceil(episode_length_s / step_dt), so leave a quarter-control-step margin to
# make floating-point rounding deterministically yield ``trace.steps``.
# This guarantees that a candidate that has not reached task termination still
# exports a complete, equally long trajectory instead of invented tail actions.
replay_timeout_s = (trace.steps - 0.25) / trace.control_hz
os.environ["ROBOLAB_EPISODE_LENGTH_S"] = f"{replay_timeout_s:.12g}"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import robolab.constants  # noqa: E402
from robolab.eval.action_replay import ActionReplayClient  # noqa: E402
from robolab.registrations.piper.auto_env_registrations_jointpos import auto_register_piper_envs  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

if args_cli.task is None and args_cli.task_dirs == DEFAULT_TASK_SUBFOLDERS:
    args_cli.task_dirs = ["piper"]

auto_register_piper_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)


def make_client(args: argparse.Namespace) -> ActionReplayClient:
    del args
    return ActionReplayClient(trace)


def main() -> None:
    print(
        f"[RoboLab] Open-loop replay: {trace.steps} actions, "
        f"{trace.duration_s:.3f}s at {trace.control_hz:g} Hz; source={trace.source_hdf5 or 'unknown'}"
    )
    run_evaluation(args_cli, policy="action_replay", client_factory=make_client)
    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\033[96m[RoboLab] Terminated with error: {exc}\033[0m")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
