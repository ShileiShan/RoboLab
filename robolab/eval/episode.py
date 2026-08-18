# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Policy episode runner for RoboLab.

This module contains the run_episode function that executes a single
policy-controlled episode given any :class:`InferenceClient` subclass.
The function stays policy-agnostic — concrete clients live under
``policies/<policy>/client.py``.

Supports multi-env: one PolicyClient per env, per-env video writers,
actions inferred per active env and stacked for env.step().
"""

import logging
import os
import re
import time
from collections import defaultdict

import cv2
import numpy as np
import torch
from tqdm import tqdm

logger = logging.getLogger(__name__)


# These labels are deliberately comparison-specific rather than generic
# ``bread``/``toaster`` classes.  The breakfast USD is shared by normal task
# runs, while this visualisation needs an unambiguous, opt-in segmentation
# contract for just the two moving bread slices and the breakfast toaster
# geometry (including its button-equipped model).
_BREAKFAST_COMPARISON_SEMANTICS = {
    "mianbaopian_0_130": "comparison_bread",
    "mianbaopian_0_131": "comparison_bread",
    "mianbaojia_0_175": "comparison_toaster",
    "Toaster014": "comparison_toaster",
}


def _enable_breakfast_comparison_semantics(env, env_cfg) -> bool:
    """Tag the MakeBreakfastTask objects before its first rendered reset.

    The fixed comparison camera uses raw semantic IDs.  Scene USDs normally do
    not carry labels for task objects, so adding tags here is preferable to
    permanently changing shared assets.  This is called before ``env.reset()``
    so Replicator builds its ID map with the labels already present.
    """
    requested = os.environ.get("ROBOLAB_COMPARISON_INCLUDE_BREAKFAST_OBJECTS", "0") == "1"
    if not requested:
        return False

    task_name = getattr(env_cfg, "_task_name", type(env_cfg).__name__)
    if task_name != "MakeBreakfastTask":
        raise ValueError(
            "ROBOLAB_COMPARISON_INCLUDE_BREAKFAST_OBJECTS=1 is currently "
            "supported only for MakeBreakfastTask."
        )

    import omni.usd
    # Isaac Sim 5 moved this utility from the retired ``omni.isaac.core``
    # namespace.  LabelsAPI is also what Replicator's semantic annotator
    # consumes natively.
    from isaacsim.core.utils.semantics import add_labels

    stage = omni.usd.get_context().get_stage()
    for object_name, semantic_label in _BREAKFAST_COMPARISON_SEMANTICS.items():
        # IsaacLab assets expose their concrete paths after scene creation.
        # Keep the explicit fallback for AssetBaseCfg objects on IsaacLab
        # versions that do not retain ``_prim_paths`` publicly.  Toaster014
        # is not a contact-tracked asset, so it intentionally uses the same
        # concrete scene-path fallback.
        asset = getattr(env.scene, object_name, None)
        prim_paths = tuple(getattr(asset, "_prim_paths", ()) or ()) if asset is not None else ()
        if not prim_paths:
            prim_paths = tuple(
                f"/World/envs/env_{env_id}/scene/{object_name}"
                for env_id in range(env.num_envs)
            )
        for prim_path in prim_paths:
            prim = stage.GetPrimAtPath(prim_path)
            if not prim.IsValid():
                raise RuntimeError(
                    f"Could not add comparison semantics: scene prim {prim_path!r} does not exist."
                )
            add_labels(prim, labels=[semantic_label], instance_name="class", overwrite=True)
    return True

class TimingStats:
    """Simple timing utility for profiling code sections."""

    def __init__(self):
        self.times = defaultdict(list)
        self._start_times = {}

    def start(self, name: str):
        self._start_times[name] = time.perf_counter()

    def stop(self, name: str):
        if name in self._start_times:
            elapsed = time.perf_counter() - self._start_times[name]
            self.times[name].append(elapsed)
            del self._start_times[name]

    def to_dict(self, num_steps: int) -> dict:
        """Return timing summary as a dict for results logging."""
        d = {}
        for name, times in self.times.items():
            d[f"{name}_s"] = round(sum(times), 3)
            d[f"{name}_avg_ms"] = round(sum(times) / len(times) * 1000, 1) if times else 0
        d["wall_total_s"] = round(sum(sum(t) for t in self.times.values()), 3)
        d["it_per_sec"] = round(num_steps / d["wall_total_s"], 2) if d["wall_total_s"] > 0 else 0
        return d

from robolab.constants import VISUALIZE, get_output_dir
from robolab.core.logging.results import get_all_env_subtask_infos
from robolab.core.observations.observation_utils import unpack_image_obs, unpack_viewport_cams
from robolab.core.utils.video_utils import VideoWriter
from robolab.core.world.world_state import get_world
from robolab.eval.base_client import InferenceClient


def run_episode(
    env,
    env_cfg,
    episode,
    client: InferenceClient,
    *,
    headless=False,
    save_videos=True,
    video_mode="all",
    sensor_video_camera: str | None = None,
):
    """Run a policy-controlled episode across all parallel envs.

    The policy client is constructed by the caller (typically a per-policy
    runner under ``policies/<policy>/run.py``). This function stays
    policy-agnostic.

    Args:
        env: The environment instance (RobolabEnv with num_envs >= 1)
        env_cfg: Environment configuration
        episode: Run index (each run produces num_envs episodes)
        client: Constructed inference client. One connection shared across envs
            with per-env chunk state keyed by ``env_id``.
        headless: If True, don't display video
        save_videos: If True, save per-env episode videos
        video_mode: Which videos to save: 'all', 'viewport', 'sensor', or 'none'
        sensor_video_camera: Optional image_obs camera key to record instead of
            the concatenated sensor camera strip.

    Returns:
        tuple: (env_results, subtask_status, timing)
            env_results: per-env dicts with {env_id, success, step}
            subtask_status: list of per-step subtask info dicts
            timing: dict with wall-clock timing breakdown
    """
    timer = TimingStats()

    # Must run before the warm-up reset below: semantic annotators construct
    # their ID-to-label table while rendering that first reset frame.
    include_breakfast_objects = _enable_breakfast_comparison_semantics(env, env_cfg)

    # Keep the historical warm-up reset, but suppress a generated task's
    # expensive reset pre-roll here.  The second reset below is the actual
    # episode reset and is the only one allowed to perform/record setup.
    env._dynamic_setup_enabled = False
    obs, _ = env.reset()
    max_steps = env.max_episode_length
    video_fps = 1 / (env_cfg.sim.render_interval * env_cfg.sim.dt) # Hz
    # The policy loop can run faster than the configured render cadence.  Write
    # only at that cadence so one second of policy time remains one second of
    # MP4 time (and matches reset pre-roll frames, which are rendered at the
    # same cadence).
    video_write_stride = max(1, round(1 / (env.step_dt * video_fps)))
    instruction = env_cfg.instruction
    # Pull action dim from the env's action manager (IsaacLab canonical),
    # falling back to the gym action space if the manager isn't available.
    action_dim = getattr(
        getattr(env, "action_manager", None),
        "total_action_dim",
        None,
    ) or env.action_space.shape[-1]

    subtask_status = []

    # Setup per-env streaming video writers
    save_sensor = save_videos and video_mode in ("all", "sensor")
    save_viewport = save_videos and video_mode in ("all", "viewport")
    cleaned_instruction = re.sub(r'[^\w\s]', '', instruction).replace(' ', '_')
    # Define unconditionally so the finally clause below can iterate them either way.
    video_writers_obs: list[VideoWriter] = []
    video_writers_viewport: list[VideoWriter] = []
    video_writers_robot_mask: list[VideoWriter] = []
    video_writers_breakfast_objects_mask: list[VideoWriter] = []
    comparison_camera_name = "piper_comparison_camera"
    comparison_mask_key = f"{comparison_camera_name}_semantic_segmentation"
    comparison_sensor = getattr(env.scene, "sensors", {}).get(comparison_camera_name)
    write_robot_masks = bool(
        save_viewport
        and comparison_sensor is not None
        and "semantic_segmentation" in getattr(comparison_sensor.cfg, "data_types", ())
    )
    write_breakfast_objects_masks = include_breakfast_objects and write_robot_masks
    if save_videos:
        for env_id in range(env.num_envs):
            suffix = f"_{episode}_env{env_id}" if env.num_envs > 1 else f"_{episode}"
            if save_sensor:
                camera_suffix = f"_{sensor_video_camera}" if sensor_video_camera else ""
                video_path = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}{camera_suffix}.mp4")
                video_writers_obs.append(VideoWriter(video_path, video_fps))
            if save_viewport:
                video_path_viewport = os.path.join(get_output_dir(), f"{cleaned_instruction}{suffix}_viewport.mp4")
                video_writers_viewport.append(VideoWriter(video_path_viewport, video_fps))

            # This stream is available only for the fixed Piper comparison
            # camera.  It is intentionally a separate grayscale-as-RGB MP4 so
            # offline compositing never has to infer the robot from RGB.
            if write_robot_masks:
                mask_path = os.path.join(
                    get_output_dir(), f"{cleaned_instruction}{suffix}_viewport_robot_mask.mp4"
                )
                video_writers_robot_mask.append(VideoWriter(mask_path, video_fps))
            if write_breakfast_objects_masks:
                mask_path = os.path.join(
                    get_output_dir(), f"{cleaned_instruction}{suffix}_viewport_breakfast_objects_mask.mp4"
                )
                video_writers_breakfast_objects_mask.append(VideoWriter(mask_path, video_fps))

    robot_semantic_ids: set[int] | None = None
    breakfast_object_semantic_ids: set[int] | None = None

    def semantic_labels_describe_robot(labels) -> bool:
        """Accept Isaac Replicator's version-dependent semantic label forms."""
        if isinstance(labels, dict):
            return any(
                str(key).lower() == "class" and str(value).lower() == "robot"
                for key, value in labels.items()
            )
        return "robot" in str(labels).lower()

    def semantic_labels_describe_breakfast_object(labels) -> bool:
        """Match only the opt-in bread/toaster labels created above."""
        wanted = {"comparison_bread", "comparison_toaster"}
        if isinstance(labels, dict):
            return any(
                str(key).lower() == "class" and str(value).lower() in wanted
                for key, value in labels.items()
            )
        labels_text = str(labels).lower()
        return any(label in labels_text for label in wanted)

    def segmentation_from_frame(frame_obs, env_id: int) -> np.ndarray:
        if comparison_mask_key not in frame_obs.get("viewport_cam", {}):
            raise RuntimeError(
                "Piper comparison camera did not provide semantic segmentation "
                f"term {comparison_mask_key!r}."
            )
        segmentation = frame_obs["viewport_cam"][comparison_mask_key][env_id].detach().cpu().numpy()
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[..., 0]
        if segmentation.ndim != 2:
            raise RuntimeError(
                "Expected raw single-channel semantic segmentation, got "
                f"shape {segmentation.shape} from {comparison_mask_key!r}."
            )
        return segmentation

    def robot_mask_from_frame(frame_obs, env_id: int):
        """Return the fixed camera's binary robot mask for one environment."""
        nonlocal robot_semantic_ids
        if robot_semantic_ids is None:
            segmentation_info = comparison_sensor.data.info.get("semantic_segmentation", {})
            id_to_labels = segmentation_info.get("idToLabels", {})
            robot_semantic_ids = {
                int(semantic_id)
                for semantic_id, labels in id_to_labels.items()
                if semantic_labels_describe_robot(labels)
            }
            if not robot_semantic_ids:
                raise RuntimeError(
                    "The Piper comparison camera found no semantic class=robot IDs. "
                    f"Available semantic labels: {id_to_labels!r}"
                )

        segmentation = segmentation_from_frame(frame_obs, env_id)
        mask = np.isin(segmentation, tuple(robot_semantic_ids))
        return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)

    def breakfast_objects_mask_from_frame(frame_obs, env_id: int):
        """Return one binary mask containing both bread slices and the toaster."""
        nonlocal breakfast_object_semantic_ids
        if breakfast_object_semantic_ids is None:
            segmentation_info = comparison_sensor.data.info.get("semantic_segmentation", {})
            id_to_labels = segmentation_info.get("idToLabels", {})
            breakfast_object_semantic_ids = {
                int(semantic_id)
                for semantic_id, labels in id_to_labels.items()
                if semantic_labels_describe_breakfast_object(labels)
            }
            if not breakfast_object_semantic_ids:
                raise RuntimeError(
                    "The Piper comparison camera found no breakfast-object semantic IDs. "
                    f"Available semantic labels: {id_to_labels!r}"
                )

        segmentation = segmentation_from_frame(frame_obs, env_id)
        mask = np.isin(segmentation, tuple(breakfast_object_semantic_ids))
        return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)

    def write_video_frames(frame_obs, *, skip_frozen: bool = False) -> None:
        """Append a synchronized frame to each enabled per-env video stream."""
        if not save_videos:
            return
        for env_id in range(env.num_envs):
            if skip_frozen and env._frozen_envs[env_id]:
                continue
            if save_sensor:
                sensor_images = unpack_image_obs(frame_obs, scale=0.5, env_id=env_id)
                sensor_frame = (
                    sensor_images[sensor_video_camera]
                    if sensor_video_camera
                    else sensor_images["combined_image"]
                )
                video_writers_obs[env_id].write(sensor_frame)
            if save_viewport:
                viewport_frame = unpack_viewport_cams(frame_obs, env_id=env_id).get("combined_image")
                video_writers_viewport[env_id].write(viewport_frame)
            if write_robot_masks:
                video_writers_robot_mask[env_id].write(robot_mask_from_frame(frame_obs, env_id))
            if write_breakfast_objects_masks:
                video_writers_breakfast_objects_mask[env_id].write(
                    breakfast_objects_mask_from_frame(frame_obs, env_id)
                )

    # The reset event owns setup physics.  It calls this lightweight callback
    # only after it rendered a setup frame; the same writers are then retained
    # for policy frames, yielding one continuous dashboard MP4.
    def capture_setup_frame(reset_env) -> None:
        setup_obs = reset_env.observation_manager.compute(update_history=False)
        write_video_frames(setup_obs)

    record_setup_video = bool(
        save_videos
        and getattr(env_cfg, "record_setup_video", False)
        and (save_sensor or save_viewport)
    )
    env._dynamic_setup_capture = capture_setup_frame if record_setup_video else None
    env._dynamic_setup_enabled = True
    obs, _ = env.reset()
    # Do not retain a callback into closed video writers on later incidental
    # resets (e.g. teardown or a caller's next episode warm-up).
    env._dynamic_setup_capture = None

    # Most evaluation videos historically begin after the first control step.
    # Parameter-comparison runs opt in to an explicit reset frame so the first
    # tile can verify that the two trajectories start from the same pose.
    if save_videos and os.environ.get("ROBOLAB_WRITE_INITIAL_VIDEO_FRAME") == "1":
        # Semantic segmentation is produced asynchronously by the RTX camera.
        # A reset observation can therefore contain a partially populated ID
        # buffer even though its RGB frame is complete.  Render/update a few
        # times without stepping physics, then recompute observations: this is
        # still t=0, but produces a settled RGB/semantic frame in both runs.
        if comparison_sensor is not None:
            for _ in range(3):
                env.sim.render()
                env.scene.update(dt=0.0)
            obs = env.observation_manager.compute(update_history=False)
        write_video_frames(obs)

    # Set up per-run HDF5 file and per-env demo indices only after reset.  The
    # reset pre-roll is intentionally not a policy trajectory/action history.
    if env.recorder_manager is not None and hasattr(env.recorder_manager, 'set_hdf5_file'):
        env.recorder_manager.set_hdf5_file(f"run_{episode}.hdf5")
        for env_id in range(env.num_envs):
            env.recorder_manager.set_episode_index(env_id, env_ids=[env_id])

    import omni.kit.app
    import omni.timeline
    timeline = omni.timeline.get_timeline_interface()
    kit_app = omni.kit.app.get_app()

    # A GUI session normally starts the timeline itself.  In a headless
    # process this is not guaranteed, and polling without requesting play can
    # leave an evaluation spinning forever before its first policy action.
    if not timeline.is_playing():
        logger.info("Isaac timeline is stopped after reset; starting it for evaluation.")
        timeline.play()
        try:
            timeline_timeout_s = float(os.environ.get("ROBOLAB_TIMELINE_START_TIMEOUT_S", "30"))
        except ValueError as exc:
            raise ValueError("ROBOLAB_TIMELINE_START_TIMEOUT_S must be a positive number.") from exc
        if timeline_timeout_s <= 0:
            raise ValueError("ROBOLAB_TIMELINE_START_TIMEOUT_S must be a positive number.")
        timeline_deadline = time.monotonic() + timeline_timeout_s
        while not timeline.is_playing():
            kit_app.update()
            if time.monotonic() >= timeline_deadline:
                raise RuntimeError(
                    "Isaac timeline did not start within "
                    f"{timeline_timeout_s:g} seconds after reset."
                )

    actual_steps = 0
    try:
        for step in tqdm(range(max_steps)):
            timer.start("policy_inference")
            # Infer actions for all active (non-frozen) envs in ONE call.
            # Batching-capable clients send a single request for every env
            # needing a replan; the InferenceClient default is a serial
            # loop, so other policies behave exactly as before.
            actions = torch.zeros(env.num_envs, action_dim, device=env.device)
            last_viz = None
            rets = client.infer_batch(
                obs, instruction, env_ids=list(env.active_env_ids)
            )
            for env_id in sorted(rets):
                ret = rets[env_id]
                actions[env_id] = torch.tensor(ret["action"], device=env.device)
                if env_id == 0 or last_viz is None:
                    last_viz = ret.get("viz")
            timer.stop("policy_inference")

            if not headless and last_viz is not None:
                cv2.imshow(f"{instruction}", cv2.cvtColor(last_viz, cv2.COLOR_RGB2BGR))
                cv2.waitKey(1)

            if VISUALIZE:
                get_world(env).visualize()

            timer.start("env_step")
            obs, reward, term, trunc, info = env.step(actions)
            timer.stop("env_step")

            # Collect per-env subtask info (list of dicts, one per env)
            per_env_infos = get_all_env_subtask_infos(env)
            subtask_status.append(per_env_infos)

            # Write per-env video frames (skip frozen envs)
            if save_videos and actual_steps % video_write_stride == 0:
                timer.start("video_write")
                # Keep the old frozen-env behavior for policy frames. Setup
                # frames are written by the reset callback before freezing can
                # occur, so both streams remain continuous and synchronized.
                write_video_frames(obs, skip_frozen=True)
                timer.stop("video_write")

            actual_steps += 1

            # RobolabEnv freezes terminated envs and exports recordings automatically
            if env.all_terminated:
                break
    finally:
        for vw in (
            video_writers_obs
            + video_writers_viewport
            + video_writers_robot_mask
            + video_writers_breakfast_objects_mask
        ):
            try:
                vw.release()
            except Exception:
                logger.exception("Failed to release video writer")
        try:
            client.reset()
        except Exception:
            logger.exception("Failed to reset client after episode")

    timing = timer.to_dict(actual_steps)
    setup_sim = getattr(env, "_dynamic_setup_sim_duration_s", {})
    setup_video = getattr(env, "_dynamic_setup_video_duration_s", {})
    if setup_sim:
        timing["setup_sim_s"] = round(max(setup_sim.values()), 3)
    if setup_video:
        timing["setup_video_s"] = round(max(setup_video.values()), 3)
    return env.get_env_results(), subtask_status, timing
