# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# isort: skip_file

"""
Run a gripper-toggle episode against a registered Piper (dual-arm) task.

Holds both arms at their current joint positions while toggling both grippers
between open and closed every `--toggle-every` steps. Useful for sanity-
checking the Piper robot/scene/action path end to end (arms spawn and hold
pose, both grippers move, cameras render, task terminations don't error).

Usage:
    Basic usage (default task: PiperSingleObjectPickPlaceTask):
    $ python examples/run_piper_gripper_toggle.py

    Headless (no viewer, no on-screen rendering):
    $ python examples/run_piper_gripper_toggle.py --headless
"""

import argparse
import cv2  # noqa: F401  must be imported before isaaclab
import os
import sys
import traceback

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run gripper-toggle episode on a registered Piper task.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
AppLauncher.add_app_launcher_args(parser)
parser.add_argument("--task", nargs="+", default=None,
                    help="List of tasks to run on (default: PiperSingleObjectPickPlaceTask).")
parser.add_argument("--num-steps", type=int, default=100, help="Number of steps per episode.")
parser.add_argument("--toggle-every", type=int, default=15, help="Toggle grippers every N steps.")
parser.add_argument("--video-mode", "--video_mode", type=str, default="all",
                    choices=["all", "viewport", "sensor", "none"],
                    help="Which videos to save: 'all' (sensor + viewport), 'viewport' only, "
                         "'sensor' only, or 'none' (default: all)")

args_cli, _ = parser.parse_known_args()
args_cli.enable_cameras = True
args_cli.save_videos = args_cli.video_mode != "none"
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402
from tqdm import tqdm  # noqa: E402

from robolab.constants import PACKAGE_DIR, get_output_dir, set_output_dir  # noqa: E402
from robolab.core.environments.factory import get_envs  # noqa: E402
from robolab.core.environments.runtime import create_env, end_episode  # noqa: E402
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams  # noqa: E402
from robolab.core.utils.video_utils import VideoWriter  # noqa: E402
from robolab.registrations.piper.auto_env_registrations_jointpos import auto_register_piper_envs  # noqa: E402

auto_register_piper_envs()

_LEFT_ARM_JOINTS = ["joint1_l", "joint2_l", "joint3_l", "joint4_l", "joint5_l", "joint6_l"]
_RIGHT_ARM_JOINTS = ["joint1_r", "joint2_r", "joint3_r", "joint4_r", "joint5_r", "joint6_r"]


def run_piper_gripper_toggle_episode(env, env_cfg=None, *, save_videos=True, video_mode="all",
                                      num_steps=100, toggle_every=15):
    """Toggle both grippers open/closed every `toggle_every` steps while holding both arms fixed."""
    robot = env.scene["robot"]
    obs, _ = env.reset()

    joint_names = list(robot.data.joint_names)
    left_arm_idx = [joint_names.index(n) for n in _LEFT_ARM_JOINTS]
    right_arm_idx = [joint_names.index(n) for n in _RIGHT_ARM_JOINTS]

    instruction = getattr(env_cfg, "instruction", None) or "piper_gripper_toggle"
    if isinstance(instruction, dict):
        instruction = instruction.get("default", "piper_gripper_toggle")

    if env_cfg is not None:
        video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt)
    else:
        video_fps = 15

    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")

    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_env{env_id}" if env.num_envs > 1 else ""
            if save_sensor:
                p = os.path.join(get_output_dir(), f"piper_gripper_toggle{suffix}.mp4")
                video_writers_obs.append(VideoWriter(p, video_fps))
            if save_viewport:
                p = os.path.join(get_output_dir(), f"piper_gripper_toggle{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(p, video_fps))

    gripper_open = False
    try:
        for count in tqdm(range(num_steps)):
            if count % toggle_every == 0:
                gripper_open = not gripper_open
                print(f"[Step {count:04d}] Gripper state: {'open' if gripper_open else 'closed'}")

            left_arm_pos = robot.data.joint_pos[:, left_arm_idx]
            right_arm_pos = robot.data.joint_pos[:, right_arm_idx]
            gripper_val = 0.035 if gripper_open else 0.0
            gripper_action = torch.full((env.num_envs, 1), gripper_val, device=env.device)

            actions = torch.cat([left_arm_pos, right_arm_pos, gripper_action, gripper_action], dim=1)
            obs, _, term, trunc, info = env.step(actions)

            if save_videos:
                for env_id in range(env.num_envs):
                    if save_sensor:
                        frame = unpack_image_obs(obs, scale=0.5, env_id=env_id).get("combined_image")
                        if frame is not None:
                            video_writers_obs[env_id].write(frame)
                    if save_viewport:
                        frame_vp = unpack_viewport_cams(obs, env_id=env_id).get("combined_image")
                        if frame_vp is not None:
                            video_writers_viewport[env_id].write(frame_vp)
    finally:
        for w in video_writers_obs:
            w.release()
        for w in video_writers_viewport:
            w.release()


def main():
    output_dir = os.path.join(PACKAGE_DIR, "output", "run_piper_gripper_toggle")
    os.makedirs(output_dir, exist_ok=True)

    if args_cli.task:
        task_envs = get_envs(task=args_cli.task)
    else:
        task_envs = get_envs(task="PiperSingleObjectPickPlaceTask")
    print(f"Running gripper toggle on {len(task_envs)} environments: {task_envs}")

    for task_env in task_envs:
        scene_output_dir = os.path.join(output_dir, task_env)
        os.makedirs(scene_output_dir, exist_ok=True)
        set_output_dir(scene_output_dir)

        env, env_cfg = create_env(task_env,
                                  device=args_cli.device,
                                  num_envs=args_cli.num_envs,
                                  use_fabric=True)
        try:
            print(f"Running {task_env}: '{env_cfg.instruction}'")
            run_piper_gripper_toggle_episode(
                env,
                env_cfg,
                save_videos=args_cli.save_videos,
                video_mode=args_cli.video_mode,
                num_steps=args_cli.num_steps,
                toggle_every=args_cli.toggle_every,
            )
            end_episode(env)
        finally:
            env.close()

    simulation_app.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Terminated with error: {e}")
        traceback.print_exc()
        simulation_app.close()
        sys.exit(1)
