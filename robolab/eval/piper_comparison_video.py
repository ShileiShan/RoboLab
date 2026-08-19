"""Shared viewport/mask export helpers for Piper comparison replays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


COMPARISON_CAMERA_NAME = "piper_comparison_camera"
COMPARISON_MASK_KEY = f"{COMPARISON_CAMERA_NAME}_semantic_segmentation"

_MAKE_BREAKFAST_SEMANTICS = {
    "breakfast": {
        "mianbaopian_0_130": "comparison_bread",
        "mianbaopian_0_131": "comparison_bread",
        "mianbaojia_0_175": "comparison_toaster",
        "Toaster014": "comparison_toaster",
    },
    "pour_water": {
        "shuihu_0_186": "comparison_kettle",
        "shuibei_0_098": "comparison_cup",
    },
}
_MAKE_BREAKFAST_SEMANTICS["all"] = {
    **_MAKE_BREAKFAST_SEMANTICS["breakfast"],
    **_MAKE_BREAKFAST_SEMANTICS["pour_water"],
}


def enable_task_comparison_semantics(env, env_cfg, object_set: str) -> tuple[str, ...]:
    """Attach semantic labels needed for offline mask compositing."""
    normalized = (object_set or "none").strip().lower()
    if normalized == "none":
        return ()

    task_name = getattr(env_cfg, "_task_name", type(env_cfg).__name__)
    if task_name != "MakeBreakfastTask":
        raise ValueError(
            f"Comparison object set {object_set!r} is currently supported only for MakeBreakfastTask, "
            f"got {task_name!r}."
        )
    if normalized not in _MAKE_BREAKFAST_SEMANTICS:
        raise ValueError(
            f"Unknown comparison object set {object_set!r}; expected one of "
            f"{sorted(_MAKE_BREAKFAST_SEMANTICS)}."
        )

    import omni.usd
    from isaacsim.core.utils.semantics import add_labels

    stage = omni.usd.get_context().get_stage()
    label_map = _MAKE_BREAKFAST_SEMANTICS[normalized]
    applied_labels: list[str] = []
    for object_name, semantic_label in label_map.items():
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
        applied_labels.append(semantic_label)
    return tuple(sorted(set(applied_labels)))


def _label_matches_robot(labels) -> bool:
    if isinstance(labels, dict):
        return any(
            str(key).lower() == "class" and str(value).lower() == "robot"
            for key, value in labels.items()
        )
    return "robot" in str(labels).lower()


def _label_matches_any(labels, wanted: set[str]) -> bool:
    wanted = {label.lower() for label in wanted}
    if isinstance(labels, dict):
        return any(
            str(key).lower() == "class" and str(value).lower() in wanted
            for key, value in labels.items()
        )
    labels_text = str(labels).lower()
    return any(label in labels_text for label in wanted)


@dataclass
class ComparisonVideoRecorder:
    """Write RGB, robot mask, and tracked-object mask MP4s from one env."""

    rgb_writer: Any
    robot_mask_writer: Any | None
    object_mask_writer: Any | None
    comparison_sensor: Any | None
    tracked_object_labels: tuple[str, ...]
    rgb_path: str
    robot_mask_path: str | None
    object_mask_path: str | None

    _robot_semantic_ids: set[int] | None = None
    _object_semantic_ids: set[int] | None = None

    @classmethod
    def create(
        cls,
        *,
        env,
        output_dir: str | Path,
        video_stem: str,
        video_fps: float,
        tracked_object_labels: tuple[str, ...] = (),
    ) -> "ComparisonVideoRecorder":
        from robolab.core.utils.video_utils import VideoWriter

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        rgb_path = output_dir / f"{video_stem}.mp4"
        robot_mask_path = output_dir / f"{video_stem}_robot_mask.mp4"
        object_mask_path = output_dir / f"{video_stem}_tracked_objects_mask.mp4"
        comparison_sensor = getattr(env.scene, "sensors", {}).get(COMPARISON_CAMERA_NAME)
        robot_writer = VideoWriter(str(robot_mask_path), video_fps) if comparison_sensor is not None else None
        object_writer = (
            VideoWriter(str(object_mask_path), video_fps)
            if comparison_sensor is not None and tracked_object_labels else None
        )
        return cls(
            rgb_writer=VideoWriter(str(rgb_path), video_fps),
            robot_mask_writer=robot_writer,
            object_mask_writer=object_writer,
            comparison_sensor=comparison_sensor,
            tracked_object_labels=tuple(tracked_object_labels),
            rgb_path=str(rgb_path),
            robot_mask_path=str(robot_mask_path) if robot_writer is not None else None,
            object_mask_path=str(object_mask_path) if object_writer is not None else None,
        )

    def _segmentation_from_frame(self, obs, env_id: int) -> np.ndarray:
        if COMPARISON_MASK_KEY not in obs.get("viewport_cam", {}):
            raise RuntimeError(
                "Piper comparison camera did not provide semantic segmentation "
                f"term {COMPARISON_MASK_KEY!r}."
            )
        segmentation = obs["viewport_cam"][COMPARISON_MASK_KEY][env_id].detach().cpu().numpy()
        if segmentation.ndim == 3 and segmentation.shape[-1] == 1:
            segmentation = segmentation[..., 0]
        if segmentation.ndim != 2:
            raise RuntimeError(
                "Expected raw single-channel semantic segmentation, got "
                f"shape {segmentation.shape} from {COMPARISON_MASK_KEY!r}."
            )
        return segmentation

    def _resolve_robot_ids(self) -> set[int]:
        if self._robot_semantic_ids is not None:
            return self._robot_semantic_ids
        if self.comparison_sensor is None:
            raise RuntimeError("Piper comparison camera is unavailable; cannot build robot mask.")
        segmentation_info = self.comparison_sensor.data.info.get("semantic_segmentation", {})
        id_to_labels = segmentation_info.get("idToLabels", {})
        self._robot_semantic_ids = {
            int(semantic_id)
            for semantic_id, labels in id_to_labels.items()
            if _label_matches_robot(labels)
        }
        if not self._robot_semantic_ids:
            raise RuntimeError(
                "The Piper comparison camera found no semantic class=robot IDs. "
                f"Available semantic labels: {id_to_labels!r}"
            )
        return self._robot_semantic_ids

    def _resolve_object_ids(self) -> set[int]:
        if self._object_semantic_ids is not None:
            return self._object_semantic_ids
        if self.comparison_sensor is None:
            raise RuntimeError("Piper comparison camera is unavailable; cannot build object mask.")
        segmentation_info = self.comparison_sensor.data.info.get("semantic_segmentation", {})
        id_to_labels = segmentation_info.get("idToLabels", {})
        wanted = set(self.tracked_object_labels)
        self._object_semantic_ids = {
            int(semantic_id)
            for semantic_id, labels in id_to_labels.items()
            if _label_matches_any(labels, wanted)
        }
        if not self._object_semantic_ids:
            raise RuntimeError(
                "The Piper comparison camera found no tracked-object semantic IDs. "
                f"Wanted labels: {sorted(wanted)!r}; available semantic labels: {id_to_labels!r}"
            )
        return self._object_semantic_ids

    def _mask_rgb(self, segmentation: np.ndarray, semantic_ids: set[int]) -> np.ndarray:
        mask = np.isin(segmentation, tuple(semantic_ids))
        return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)

    def write_frame(self, obs, *, env_id: int = 0) -> None:
        from robolab.core.observations.observation_utils import unpack_viewport_cams

        self.rgb_writer.write(unpack_viewport_cams(obs, env_id=env_id).get("combined_image"))
        if self.comparison_sensor is None:
            return
        segmentation = None
        if self.robot_mask_writer is not None:
            segmentation = self._segmentation_from_frame(obs, env_id)
            self.robot_mask_writer.write(self._mask_rgb(segmentation, self._resolve_robot_ids()))
        if self.object_mask_writer is not None:
            if segmentation is None:
                segmentation = self._segmentation_from_frame(obs, env_id)
            self.object_mask_writer.write(self._mask_rgb(segmentation, self._resolve_object_ids()))

    def release(self) -> None:
        self.rgb_writer.release()
        if self.robot_mask_writer is not None:
            self.robot_mask_writer.release()
        if self.object_mask_writer is not None:
            self.object_mask_writer.release()
