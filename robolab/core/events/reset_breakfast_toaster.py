# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import numpy as np
import torch
from pxr import Gf, Usd, UsdGeom, UsdPhysics

from robolab.constants import SCENE_DIR
from robolab.core.scenes.utils import find_scene_file


_SCENE_FILE = "breakfast.usda"
_SCENE_ROOT_PATH = "/world"
_JOINT_STATE_CHANNELS = {
    "PhysicsPrismaticJoint": "linear",
    "PhysicsRevoluteJoint": "angular",
}


def _extract_world_pose_wxyz(
    prim: Usd.Prim,
    xform_cache: UsdGeom.XformCache,
) -> tuple[np.ndarray, np.ndarray]:
    """Return world position + quaternion (wxyz) for a composed USD prim."""
    world_xform = xform_cache.GetLocalToWorldTransform(prim)
    pos = np.array(world_xform.ExtractTranslation(), dtype=np.float32)

    rot_mat = world_xform.ExtractRotationMatrix()
    col0 = Gf.Vec3d(rot_mat[0][0], rot_mat[1][0], rot_mat[2][0])
    col1 = Gf.Vec3d(rot_mat[0][1], rot_mat[1][1], rot_mat[2][1])
    col2 = Gf.Vec3d(rot_mat[0][2], rot_mat[1][2], rot_mat[2][2])
    sx, sy, sz = col0.GetLength(), col1.GetLength(), col2.GetLength()
    if sx > 1e-9 and sy > 1e-9 and sz > 1e-9:
        norm_mat = Gf.Matrix3d(
            col0[0] / sx,
            col1[0] / sy,
            col2[0] / sz,
            col0[1] / sx,
            col1[1] / sy,
            col2[1] / sz,
            col0[2] / sx,
            col1[2] / sy,
            col2[2] / sz,
        )
        quat = norm_mat.ExtractRotation().GetQuat()
    else:
        quat = Gf.Quatd(1.0, 0.0, 0.0, 0.0)

    orient = np.array([quat.GetReal(), *quat.GetImaginary()], dtype=np.float32)
    return pos, orient


def _get_float_attr(prim: Usd.Prim, attr_name: str) -> float | None:
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return None
    value = attr.Get()
    if value is None:
        return None
    return float(value)


def _first_not_none(*values: float | None) -> float:
    for value in values:
        if value is not None:
            return float(value)
    return 0.0


def _get_applied_drive_names(prim: Usd.Prim) -> tuple[str, ...]:
    return tuple(
        schema.split(":", 1)[1]
        for schema in prim.GetAppliedSchemas()
        if schema.startswith("PhysicsDriveAPI:")
    )


def _extract_joint_defaults(prim: Usd.Prim, root_prim: Usd.Prim) -> dict:
    joint_type = prim.GetTypeName()
    state_channel = _JOINT_STATE_CHANNELS[joint_type]
    axis_name = prim.GetAttribute("physics:axis").Get()

    authored_position = _get_float_attr(prim, f"state:{state_channel}:physics:position")
    authored_velocity = _get_float_attr(prim, f"state:{state_channel}:physics:velocity")

    drives: list[dict] = []
    drive_defaults: dict[str, tuple[float | None, float | None]] = {}
    for drive_name in _get_applied_drive_names(prim):
        target_position = _get_float_attr(prim, f"drive:{drive_name}:physics:targetPosition")
        target_velocity = _get_float_attr(prim, f"drive:{drive_name}:physics:targetVelocity")
        drives.append(
            {
                "name": drive_name,
                "target_position": target_position,
                "target_velocity": target_velocity,
            }
        )
        drive_defaults[drive_name] = (target_position, target_velocity)

    type_drive_position, type_drive_velocity = drive_defaults.get(state_channel, (None, None))
    axis_drive_position, axis_drive_velocity = drive_defaults.get(axis_name, (None, None))

    return {
        "suffix": str(prim.GetPath()).removeprefix(str(root_prim.GetPath())),
        "state_channel": state_channel,
        "position": _first_not_none(authored_position, type_drive_position, axis_drive_position),
        "velocity": _first_not_none(authored_velocity, type_drive_velocity, axis_drive_velocity),
        "drives": tuple(drives),
    }


def _extract_root_xform_defaults(root_prim: Usd.Prim) -> tuple[tuple[str, object], ...]:
    xform_defaults: list[tuple[str, object]] = []

    xform_op_order_attr = root_prim.GetAttribute("xformOpOrder")
    if xform_op_order_attr and xform_op_order_attr.IsValid():
        xform_defaults.append(("xformOpOrder", xform_op_order_attr.Get()))

    for attr in root_prim.GetAttributes():
        name = attr.GetName()
        if not name.startswith("xformOp:") or name == "xformOpOrder":
            continue
        xform_defaults.append((name, attr.Get()))

    return tuple(xform_defaults)


@lru_cache(maxsize=1)
def _load_jointed_asset_defaults() -> dict[str, dict]:
    """Load authored defaults for every breakfast scene asset that contains joints."""
    scene_path = Path(find_scene_file(_SCENE_FILE, SCENE_DIR))
    stage = Usd.Stage.Open(str(scene_path))
    if stage is None:
        raise FileNotFoundError(f"Could not open breakfast scene at {scene_path!s}.")

    world_prim = stage.GetPrimAtPath(_SCENE_ROOT_PATH)
    if not world_prim.IsValid():
        raise ValueError(f"Could not find {_SCENE_ROOT_PATH} in {scene_path!s}.")

    xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
    asset_defaults: dict[str, dict] = {}

    for root_prim in world_prim.GetChildren():
        rigid_suffixes: list[str] = []
        rigid_positions: list[np.ndarray] = []
        rigid_orientations: list[np.ndarray] = []
        joints: list[dict] = []

        for prim in Usd.PrimRange(root_prim):
            joint_type = prim.GetTypeName()
            if joint_type in _JOINT_STATE_CHANNELS:
                joints.append(_extract_joint_defaults(prim, root_prim))

            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
            if rigid_body_api and rigid_body_api.GetRigidBodyEnabledAttr().Get():
                suffix = str(prim.GetPath()).removeprefix(str(root_prim.GetPath()))
                pos, orient = _extract_world_pose_wxyz(prim, xform_cache)
                rigid_suffixes.append(suffix)
                rigid_positions.append(pos)
                rigid_orientations.append(orient)

        if not joints:
            continue

        asset_defaults[root_prim.GetName()] = {
            "root_xform": _extract_root_xform_defaults(root_prim),
            "rigid_suffixes": tuple(rigid_suffixes),
            "rigid_positions": np.stack(rigid_positions, axis=0),
            "rigid_orientations": np.stack(rigid_orientations, axis=0),
            "joints": tuple(joints),
        }

    return asset_defaults


def _ensure_runtime_views(env) -> None:
    if hasattr(env, "_breakfast_jointed_asset_reset_views"):
        return

    try:
        from isaacsim.core.prims import RigidPrim

        def _make_rigid_view(paths: list[str]):
            return RigidPrim(paths, reset_xform_properties=False)

    except ImportError:
        from isaacsim.core.experimental.prims import RigidPrim

        def _make_rigid_view(paths: list[str]):
            return RigidPrim(paths, resolve_paths=False)

    defaults_by_asset = _load_jointed_asset_defaults()
    asset_views: dict[str, dict] = {}
    for asset_name, defaults in defaults_by_asset.items():
        root_paths = [
            f"/World/envs/env_{env_id}/scene/{asset_name}"
            for env_id in range(env.num_envs)
        ]
        rigid_paths = [
            f"{root_path}{suffix}"
            for root_path in root_paths
            for suffix in defaults["rigid_suffixes"]
        ]
        asset_views[asset_name] = {
            "rigid": _make_rigid_view(rigid_paths),
            "num_rigid_per_env": len(defaults["rigid_suffixes"]),
        }

    env._breakfast_jointed_asset_reset_views = asset_views


def _set_runtime_joint_state(joint_prim: Usd.Prim, state_channel: str, position: float, velocity: float) -> None:
    from pxr import PhysxSchema

    joint_state_api = PhysxSchema.JointStateAPI.Get(joint_prim, state_channel)
    if not joint_state_api:
        joint_state_api = PhysxSchema.JointStateAPI.Apply(joint_prim, state_channel)

    if not joint_state_api.GetPositionAttr():
        joint_state_api.CreatePositionAttr(position)
    else:
        joint_state_api.GetPositionAttr().Set(position)

    if not joint_state_api.GetVelocityAttr():
        joint_state_api.CreateVelocityAttr(velocity)
    else:
        joint_state_api.GetVelocityAttr().Set(velocity)


def _set_runtime_drive_targets(joint_prim: Usd.Prim, drives: tuple[dict, ...]) -> None:
    for drive in drives:
        drive_api = UsdPhysics.DriveAPI.Get(joint_prim, drive["name"])
        if not drive_api:
            drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, drive["name"])

        if drive["target_position"] is not None:
            if not drive_api.GetTargetPositionAttr():
                drive_api.CreateTargetPositionAttr(drive["target_position"])
            else:
                drive_api.GetTargetPositionAttr().Set(drive["target_position"])

        if drive["target_velocity"] is not None:
            if not drive_api.GetTargetVelocityAttr():
                drive_api.CreateTargetVelocityAttr(drive["target_velocity"])
            else:
                drive_api.GetTargetVelocityAttr().Set(drive["target_velocity"])


def _set_runtime_root_xform(root_prim: Usd.Prim, xform_defaults: tuple[tuple[str, object], ...]) -> None:
    for attr_name, attr_value in xform_defaults:
        attr = root_prim.GetAttribute(attr_name)
        if not attr or not attr.IsValid():
            continue
        attr.Set(attr_value)


def reset_breakfast_scene_jointed_assets(env, env_ids: torch.Tensor) -> None:
    """Restore all breakfast scene assets with authored joints to their default state."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.scene.device)
    if len(env_ids) == 0:
        return

    defaults_by_asset = _load_jointed_asset_defaults()
    if not defaults_by_asset:
        return

    _ensure_runtime_views(env)
    import omni.usd

    stage = omni.usd.get_context().get_stage()
    env_ids = env_ids.to(dtype=torch.long, device=env.scene.device)
    env_ids_cpu = env_ids.detach().cpu().tolist()
    env_origins = env.scene.env_origins[env_ids].detach().to(dtype=torch.float32)

    for asset_name, defaults in defaults_by_asset.items():
        views = env._breakfast_jointed_asset_reset_views[asset_name]
        for env_id in env_ids_cpu:
            runtime_root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/scene/{asset_name}")
            if runtime_root.IsValid():
                _set_runtime_root_xform(runtime_root, defaults["root_xform"])

        num_rigid = views["num_rigid_per_env"]
        rigid_positions_default = torch.as_tensor(
            defaults["rigid_positions"], dtype=torch.float32, device=env.scene.device
        )
        rigid_orientations_default = torch.as_tensor(
            defaults["rigid_orientations"], dtype=torch.float32, device=env.scene.device
        )
        rigid_positions = torch.cat(
            [rigid_positions_default + origin.unsqueeze(0) for origin in env_origins],
            dim=0,
        )
        rigid_orientations = rigid_orientations_default.repeat(len(env_ids_cpu), 1)
        rigid_indices = torch.as_tensor(
            [
                env_id * num_rigid + rigid_idx
                for env_id in env_ids_cpu
                for rigid_idx in range(num_rigid)
            ],
            dtype=torch.long,
            device=env.scene.device,
        )
        zero_velocities = torch.zeros((rigid_indices.numel(), 6), dtype=torch.float32, device=env.scene.device)
        views["rigid"].set_world_poses(
            positions=rigid_positions,
            orientations=rigid_orientations,
            indices=rigid_indices,
        )
        views["rigid"].set_velocities(
            velocities=zero_velocities,
            indices=rigid_indices,
        )

        for env_id in env_ids_cpu:
            runtime_root = f"/World/envs/env_{env_id}/scene/{asset_name}"
            for joint_defaults in defaults["joints"]:
                joint_prim = stage.GetPrimAtPath(runtime_root + joint_defaults["suffix"])
                if not joint_prim.IsValid():
                    continue
                _set_runtime_joint_state(
                    joint_prim,
                    state_channel=joint_defaults["state_channel"],
                    position=joint_defaults["position"],
                    velocity=joint_defaults["velocity"],
                )
                _set_runtime_drive_targets(joint_prim, joint_defaults["drives"])


# Backward-compatible alias for the previous toaster-only event name.
reset_breakfast_toaster014 = reset_breakfast_scene_jointed_assets
