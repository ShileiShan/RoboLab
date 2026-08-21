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
parser.add_argument("--piper-action-format", "--piper_action_format",
                    choices=["auto", "env", "model"], default="auto",
                    help=("Action trace format. 'env' is RoboLab order "
                          "[left_arm, right_arm, left_gripper_m, right_gripper_m]; "
                          "'model' is raw server/real-robot order "
                          "[left_arm, left_gripper_norm, right_arm, right_gripper_norm]. "
                          "Default: auto."))
parser.add_argument("--enable-piper-action-filter", "--enable_piper_action_filter",
                    dest="piper_action_filter", action="store_true", default=None,
                    help=("Force-enable Piper action filtering during replay. By default, replay filters "
                          "raw model/server traces and does not re-filter RoboLab HDF5 env traces."))
parser.add_argument("--disable-piper-action-filter", "--disable_piper_action_filter",
                    dest="piper_action_filter", action="store_false",
                    help=("Disable Piper action filtering during replay, still converting "
                          "model-format actions to env format."))
parser.add_argument("--piper-action-filter-ema", "--piper_action_filter_ema",
                    dest="piper_action_filter_ema", action="store_true", default=True,
                    help="Enable EMA in the Piper real-robot-style action filter (default: on).")
parser.add_argument("--disable-piper-action-filter-ema", "--disable_piper_action_filter_ema",
                    dest="piper_action_filter_ema", action="store_false",
                    help="Disable EMA while keeping Piper max_delta action limits enabled.")
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
from robolab.eval.piper_action_filter import (  # noqa: E402
    PiperActionFilterClient,
    PiperActionFilterConfig,
    infer_piper_action_format,
)
from robolab.registrations.piper.auto_env_registrations_jointpos import auto_register_piper_envs  # noqa: E402

robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = args_cli.enable_subtask
robolab.constants.VERBOSE = args_cli.enable_verbose
robolab.constants.DEBUG = args_cli.enable_debug

if args_cli.task is None and args_cli.task_dirs == DEFAULT_TASK_SUBFOLDERS:
    args_cli.task_dirs = ["piper"]

auto_register_piper_envs(task_dirs=args_cli.task_dirs, task=args_cli.task)


def make_client(args: argparse.Namespace) -> PiperActionFilterClient:
    action_format = (
        infer_piper_action_format(trace.actions, source_hdf5=trace.source_hdf5)
        if args.piper_action_format == "auto"
        else args.piper_action_format
    )
    filter_enabled = args.piper_action_filter if args.piper_action_filter is not None else action_format == "model"
    filter_config = PiperActionFilterConfig(use_ema=args.piper_action_filter_ema)
    print(
        "\033[96m[RoboLab] Piper action replay format/filter: "
        f"format={action_format}, filter={'on' if filter_enabled else 'off'}, "
        f"EMA={'on' if filter_config.use_ema else 'off'}, "
        f"alpha={filter_config.alpha:g}, joint_max_delta={filter_config.max_joint_delta:g}, "
        f"gripper_max_delta={filter_config.max_gripper_delta:g}, "
        f"gripper_scale={filter_config.model_gripper_scale:g}, "
        f"gripper_opening={filter_config.max_gripper_opening:g}\033[0m"
    )
    return PiperActionFilterClient(
        ActionReplayClient(trace),
        config=filter_config,
        action_format=action_format,
        enabled=filter_enabled,
    )


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
