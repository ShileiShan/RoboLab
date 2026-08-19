# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Replay a real-robot Piper joint trajectory as absolute joint targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from pathlib import Path

import cv2  # noqa: F401 -- must import this before isaaclab. Do not remove
import numpy as np
import torch
from isaaclab.app import AppLauncher

from robolab.eval.piper_joint_target_trace import load_joint_target_trace
from robolab.eval.piper_comparison_video import (
    ComparisonVideoRecorder,
    enable_task_comparison_semantics,
)

parser = argparse.ArgumentParser(
    description="Run a Piper joint-target replay from a real-robot qpos trace.",
    allow_abbrev=False,
)
parser.add_argument("--joint-trace", "--joint_trace", type=str, required=True,
                    help="Path to a .npz trace created by extract_piper_joint_target_trace.py.")
parser.add_argument("--task", type=str, default="MakeBreakfastTask",
                    help="Registered Piper task/environment name (default: MakeBreakfastTask).")
parser.add_argument("--num-envs", "--num_envs", type=int, default=1,
                    help="Number of environments to spawn (must be 1).")
parser.add_argument("--output-folder-name", "--output_folder_name", type=str, default=None,
                    help="Output folder name under <repo>/output.")
parser.add_argument("--video-mode", "--video_mode", choices=["viewport", "none"], default="viewport",
                    help="Whether to save the comparison viewport video (default: viewport).")
parser.add_argument(
    "--comparison-object-set",
    "--comparison_object_set",
    choices=["none", "breakfast", "pour_water", "all"],
    default="none",
    help="Optional tracked object set to export as a separate semantic mask video.",
)
parser.add_argument("--renderer", type=str, default="realtime", choices=["realtime", "pathtracing"])
parser.add_argument("--rendering-type", "--rendering_type", type=str, default="performance",
                    choices=["performance", "balanced", "quality"])
parser.add_argument("--enable-verbose", "--enable_verbose", action="store_true")
parser.add_argument("--enable-debug", "--enable_debug", action="store_true")
AppLauncher.add_app_launcher_args(parser)

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
trace = load_joint_target_trace(args_cli.joint_trace)

if args_cli.num_envs != 1:
    parser.error("Joint-target replay currently requires --num-envs 1.")
if trace.steps < 2:
    parser.error("Joint-target replay requires at least 2 reference states.")

replay_steps = trace.steps - 1
replay_timeout_s = (replay_steps - 0.25) / trace.control_hz
os.environ["ROBOLAB_EPISODE_LENGTH_S"] = f"{replay_timeout_s:.12g}"

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.envs.mdp as mdp  # noqa: E402
import omni.kit.app  # noqa: E402
import omni.timeline  # noqa: E402
from isaaclab.managers import TerminationTermCfg as DoneTerm  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

import robolab.constants  # noqa: E402
from robolab.constants import PACKAGE_DIR, get_timestamp, set_output_dir  # noqa: E402
from robolab.core.environments.config import parse_env_cfg  # noqa: E402
from robolab.core.environments.runtime import create_env  # noqa: E402
from robolab.core.observations.observation_utils import unpack_viewport_cams  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.registrations.piper.auto_env_registrations_jointpos import auto_register_piper_envs  # noqa: E402
from robolab.robots.piper import (  # noqa: E402
    PIPER_ARM_ARMATURE,
    PIPER_ARM_FRICTION,
    PIPER_ARM_KD,
    PIPER_ARM_KP,
)


@configclass
class TimeoutOnlyTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)


def _build_output_paths(args: argparse.Namespace) -> tuple[str, str]:
    output_folder_name = args.output_folder_name or (get_timestamp() + "_piper_joint_target_replay")
    output_dir = os.path.join(PACKAGE_DIR, "output", output_folder_name)
    scene_output_dir = os.path.join(output_dir, args.task)
    os.makedirs(scene_output_dir, exist_ok=True)
    return output_dir, scene_output_dir


def _extract_env_state(obs) -> np.ndarray:
    proprio = obs["proprio_obs"]
    left_arm = proprio["left_arm_joint_pos"][0].detach().cpu().numpy()
    right_arm = proprio["right_arm_joint_pos"][0].detach().cpu().numpy()
    left_gripper = proprio["left_gripper_pos"][0].detach().cpu().numpy()
    right_gripper = proprio["right_gripper_pos"][0].detach().cpu().numpy()
    left_opening = float(np.max(np.abs(left_gripper)) * 0.035)
    right_opening = float(np.max(np.abs(right_gripper)) * 0.035)
    return np.concatenate(
        (
            np.asarray(left_arm, dtype=np.float32).reshape(-1),
            np.asarray(right_arm, dtype=np.float32).reshape(-1),
            np.asarray([left_opening, right_opening], dtype=np.float32),
        ),
        axis=0,
    ).astype(np.float32)


def _write_initial_reference_pose(env, initial_target: np.ndarray) -> None:
    robot = env.scene["robot"]
    joint_name_to_index = {name: index for index, name in enumerate(robot.data.joint_names)}
    env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = torch.zeros_like(joint_pos)

    initial_target = np.asarray(initial_target, dtype=np.float32)
    left_arm = torch.as_tensor(initial_target[0:6], device=env.device).unsqueeze(0)
    right_arm = torch.as_tensor(initial_target[6:12], device=env.device).unsqueeze(0)
    left_gripper = float(initial_target[12])
    right_gripper = float(initial_target[13])

    for idx, name in enumerate([f"joint{i}_l" for i in range(1, 7)]):
        joint_pos[:, joint_name_to_index[name]] = left_arm[:, idx]
    for idx, name in enumerate([f"joint{i}_r" for i in range(1, 7)]):
        joint_pos[:, joint_name_to_index[name]] = right_arm[:, idx]

    joint_pos[:, joint_name_to_index["finger_joint_left_l"]] = left_gripper
    joint_pos[:, joint_name_to_index["finger_joint_right_l"]] = -left_gripper
    joint_pos[:, joint_name_to_index["finger_joint_left_r"]] = right_gripper
    joint_pos[:, joint_name_to_index["finger_joint_right_r"]] = -right_gripper

    robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
    robot.set_joint_position_target(joint_pos, env_ids=env_ids)
    robot.set_joint_velocity_target(joint_vel, env_ids=env_ids)


def _build_metrics(
    *,
    trace_path: str,
    summary_path: str,
    timeseries_path: str,
    hdf5_path: str,
    video_path: str | None,
    time_s: np.ndarray,
    reference_state: np.ndarray,
    actual_state: np.ndarray,
) -> tuple[dict, dict[str, np.ndarray]]:
    arm_reference = reference_state[:, 0:12]
    gripper_reference = reference_state[:, 12:14]
    arm_actual = actual_state[:, 0:12]
    gripper_actual = actual_state[:, 12:14]

    arm_error = np.asarray(arm_actual - arm_reference, dtype=np.float64)
    gripper_error = np.asarray(gripper_actual - gripper_reference, dtype=np.float64)
    arm_abs = np.abs(arm_error)
    gripper_abs = np.abs(gripper_error)

    timeseries = {
        "time_s": np.asarray(time_s, dtype=np.float32),
        "reference_state": np.asarray(reference_state, dtype=np.float32),
        "actual_state": np.asarray(actual_state, dtype=np.float32),
        "arm_error": np.asarray(arm_error, dtype=np.float32),
        "gripper_error": np.asarray(gripper_error, dtype=np.float32),
        "arm_mean_abs_error": np.mean(arm_abs, axis=1).astype(np.float32),
        "arm_max_abs_error": np.max(arm_abs, axis=1).astype(np.float32),
        "left_arm_mean_abs_error": np.mean(arm_abs[:, 0:6], axis=1).astype(np.float32),
        "right_arm_mean_abs_error": np.mean(arm_abs[:, 6:12], axis=1).astype(np.float32),
        "left_gripper_abs_error": gripper_abs[:, 0].astype(np.float32),
        "right_gripper_abs_error": gripper_abs[:, 1].astype(np.float32),
        "reference_gripper": np.asarray(gripper_reference, dtype=np.float32),
        "actual_gripper": np.asarray(gripper_actual, dtype=np.float32),
    }

    joint_labels = [f"left_joint{i}" for i in range(1, 7)] + [f"right_joint{i}" for i in range(1, 7)]
    arm_rmse_per_joint = np.sqrt(np.mean(np.square(arm_error), axis=0))

    summary = {
        "trace_path": trace_path,
        "summary_path": summary_path,
        "timeseries_path": timeseries_path,
        "video_path": video_path,
        "hdf5_path": hdf5_path,
        "num_reference_states": int(reference_state.shape[0]),
        "num_control_steps": int(reference_state.shape[0] - 1),
        "duration_s": float(time_s[-1]) if len(time_s) else 0.0,
        "arm_mae": float(np.mean(arm_abs)),
        "arm_rmse": float(np.sqrt(np.mean(np.square(arm_error)))),
        "arm_max_abs": float(np.max(arm_abs)),
        "left_arm_mae": float(np.mean(arm_abs[:, 0:6])),
        "right_arm_mae": float(np.mean(arm_abs[:, 6:12])),
        "left_arm_rmse": float(np.sqrt(np.mean(np.square(arm_error[:, 0:6])))),
        "right_arm_rmse": float(np.sqrt(np.mean(np.square(arm_error[:, 6:12])))),
        "gripper_mae": float(np.mean(gripper_abs)),
        "gripper_rmse": float(np.sqrt(np.mean(np.square(gripper_error)))),
        "gripper_max_abs": float(np.max(gripper_abs)),
        "left_gripper_mae": float(np.mean(gripper_abs[:, 0])),
        "right_gripper_mae": float(np.mean(gripper_abs[:, 1])),
        "left_gripper_rmse": float(np.sqrt(np.mean(np.square(gripper_error[:, 0])))),
        "right_gripper_rmse": float(np.sqrt(np.mean(np.square(gripper_error[:, 1])))),
        "arm_rmse_per_joint": {
            label: float(value) for label, value in zip(joint_labels, arm_rmse_per_joint)
        },
        "actuator_params": {
            "stiffness": float(PIPER_ARM_KP),
            "damping": float(PIPER_ARM_KD),
            "armature": float(PIPER_ARM_ARMATURE),
            "friction": float(PIPER_ARM_FRICTION),
        },
    }
    return summary, timeseries


def _write_metrics_artifacts(
    *,
    summary_path: Path,
    timeseries_path: Path,
    summary: dict,
    timeseries: dict[str, np.ndarray],
    trace,
) -> None:
    summary = dict(summary)
    summary["control_hz"] = float(trace.control_hz)
    summary["source_hdf5"] = trace.source_hdf5
    summary["gripper_real_min"] = (
        np.asarray(trace.gripper_real_min, dtype=np.float32).tolist()
        if trace.gripper_real_min is not None else None
    )
    summary["gripper_real_max"] = (
        np.asarray(trace.gripper_real_max, dtype=np.float32).tolist()
        if trace.gripper_real_max is not None else None
    )
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    np.savez_compressed(timeseries_path, **timeseries)


def _ensure_timeline_started() -> None:
    timeline = omni.timeline.get_timeline_interface()
    if timeline.is_playing():
        return

    kit_app = omni.kit.app.get_app()
    timeline.play()
    deadline = time.monotonic() + 30.0
    while not timeline.is_playing():
        kit_app.update()
        if time.monotonic() >= deadline:
            raise RuntimeError("Isaac timeline did not start within 30 seconds after reset.")


def main() -> None:
    robolab.constants.VERBOSE = args_cli.enable_verbose
    robolab.constants.DEBUG = args_cli.enable_debug
    robolab.constants.ENABLE_SUBTASK_PROGRESS_CHECKING = False

    auto_register_piper_envs(task_dirs=["piper"], task=args_cli.task)
    output_dir, scene_output_dir = _build_output_paths(args_cli)
    set_output_dir(scene_output_dir)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device, num_envs=1)
    env_cfg.instruction = "Track the recorded Piper joint trajectory."
    env_cfg.terminations = TimeoutOnlyTerminations()
    env_cfg.subtasks = None

    env = None
    video_recorder = None
    try:
        env, env_cfg = create_env(
            scene=env_cfg,
            device=args_cli.device,
            num_envs=1,
            policy="joint_target_replay",
            renderer=args_cli.renderer,
            rendering_mode=args_cli.rendering_type,
        )
        tracked_object_labels = enable_task_comparison_semantics(env, env_cfg, args_cli.comparison_object_set)

        obs, _ = env.reset()
        if env.recorder_manager is not None and hasattr(env.recorder_manager, "set_hdf5_file"):
            env.recorder_manager.set_hdf5_file("run_0.hdf5")
            env.recorder_manager.set_episode_index(0, env_ids=[0])

        _write_initial_reference_pose(env, trace.env_targets[0])
        for _ in range(3):
            env.sim.render()
            env.scene.update(dt=0.0)
        obs = env.observation_manager.compute(update_history=False)

        if args_cli.video_mode == "viewport":
            video_fps = 1.0 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
            video_recorder = ComparisonVideoRecorder.create(
                env=env,
                output_dir=scene_output_dir,
                video_stem="joint_target_replay_viewport",
                video_fps=video_fps,
                tracked_object_labels=tracked_object_labels,
            )
            video_recorder.write_frame(obs, env_id=0)
            video_path = Path(video_recorder.rgb_path)
        else:
            video_path = None

        _ensure_timeline_started()

        reference_state = np.concatenate((trace.arm_reference, trace.gripper_reference), axis=1)
        actual_state = np.empty_like(reference_state, dtype=np.float32)
        actual_state[0] = _extract_env_state(obs)

        for step_idx in range(1, trace.steps):
            action = torch.as_tensor(trace.env_targets[step_idx], device=env.device, dtype=torch.float32)
            obs, _, _, _, _ = env.step(action.unsqueeze(0))
            actual_state[step_idx] = _extract_env_state(obs)
            if video_recorder is not None:
                video_recorder.write_frame(obs, env_id=0)
            if env.all_terminated and step_idx != trace.steps - 1:
                raise RuntimeError(f"Replay terminated early at step {step_idx} before the reference finished.")

        if not env.all_terminated:
            print(
                "\033[93m[RoboLab] Warning: the environment did not time out exactly at the final replay step; "
                "exported artifacts may miss recorder output.\033[0m"
            )

        time_s = np.arange(trace.steps, dtype=np.float32) / np.float32(trace.control_hz)
        hdf5_path = Path(scene_output_dir) / "run_0.hdf5"
        summary_path = Path(scene_output_dir) / "joint_tracking_summary.json"
        timeseries_path = Path(scene_output_dir) / "joint_tracking_timeseries.npz"
        summary, timeseries = _build_metrics(
            trace_path=str(Path(args_cli.joint_trace).resolve()),
            summary_path=str(summary_path.resolve()),
            timeseries_path=str(timeseries_path.resolve()),
            hdf5_path=str(hdf5_path.resolve()),
            video_path=str(video_path.resolve()) if video_path is not None else None,
            time_s=time_s,
            reference_state=reference_state,
            actual_state=actual_state,
        )
        _write_metrics_artifacts(
            summary_path=summary_path,
            timeseries_path=timeseries_path,
            summary=summary,
            timeseries=timeseries,
            trace=trace,
        )

        print(
            f"[RoboLab] Joint-target replay complete: {trace.steps} reference states, "
            f"{replay_steps} control steps at {trace.control_hz:g} Hz."
        )
        print(f"[RoboLab] Output dir: {output_dir}")
        print(f"[RoboLab] Tracking summary: {summary_path}")
        print(f"[RoboLab] Tracking timeseries: {timeseries_path}")
        if video_path is not None:
            print(f"[RoboLab] Viewport video: {video_path}")
        if video_recorder is not None and video_recorder.robot_mask_path is not None:
            print(f"[RoboLab] Robot mask video: {video_recorder.robot_mask_path}")
        if video_recorder is not None and video_recorder.object_mask_path is not None:
            print(f"[RoboLab] Tracked-object mask video: {video_recorder.object_mask_path}")
    finally:
        if video_recorder is not None:
            video_recorder.release()
        if env is not None:
            env.close()
        simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\033[96m[RoboLab] Terminated with error: {exc}\033[0m")
        traceback.print_exc()
        try:
            simulation_app.close()
        except Exception:
            pass
        sys.exit(1)
