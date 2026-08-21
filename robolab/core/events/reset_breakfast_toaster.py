# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import torch
from pxr import Gf, Usd, UsdPhysics

from robolab.constants import SCENE_DIR
from robolab.core.scenes.utils import find_scene_file


_SCENE_FILE = "breakfast.usda"
_SCENE_ROOT_PATH = "/world"
_JOINT_STATE_CHANNELS = {
    "PhysicsPrismaticJoint": "linear",
    "PhysicsRevoluteJoint": "angular",
}


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


def _get_vec3_attr(prim: Usd.Prim, attr_name: str) -> tuple[float, float, float] | None:
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return None
    value = attr.Get()
    if value is None:
        return None
    return tuple(float(component) for component in value)


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


def _extract_xform_defaults(prim: Usd.Prim) -> tuple[tuple[str, object], ...]:
    xform_defaults: list[tuple[str, object]] = []

    xform_op_order_attr = prim.GetAttribute("xformOpOrder")
    if xform_op_order_attr and xform_op_order_attr.IsValid():
        xform_defaults.append(("xformOpOrder", xform_op_order_attr.Get()))

    for attr in prim.GetAttributes():
        name = attr.GetName()
        if not name.startswith("xformOp:") or name == "xformOpOrder":
            continue
        xform_defaults.append((name, attr.Get()))

    return tuple(xform_defaults)


def _extract_rigid_body_defaults(prim: Usd.Prim, root_prim: Usd.Prim) -> dict:
    return {
        "suffix": str(prim.GetPath()).removeprefix(str(root_prim.GetPath())),
        "xform": _extract_xform_defaults(prim),
        "linear_velocity": _get_vec3_attr(prim, "physics:velocity"),
        "angular_velocity": _get_vec3_attr(prim, "physics:angularVelocity"),
    }


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

    asset_defaults: dict[str, dict] = {}

    for root_prim in world_prim.GetChildren():
        rigid_bodies: list[dict] = []
        joints: list[dict] = []

        for prim in Usd.PrimRange(root_prim):
            joint_type = prim.GetTypeName()
            if joint_type in _JOINT_STATE_CHANNELS:
                joints.append(_extract_joint_defaults(prim, root_prim))

            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
            if rigid_body_api and rigid_body_api.GetRigidBodyEnabledAttr().Get():
                rigid_bodies.append(_extract_rigid_body_defaults(prim, root_prim))

        if not joints:
            continue

        asset_defaults[root_prim.GetName()] = {
            "root_xform": _extract_xform_defaults(root_prim),
            "rigid_bodies": tuple(rigid_bodies),
            "joints": tuple(joints),
        }

    return asset_defaults


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


def _set_runtime_vector_attr(prim: Usd.Prim, attr_name: str, value: tuple[float, float, float] | None) -> None:
    if value is None:
        return
    attr = prim.GetAttribute(attr_name)
    if not attr or not attr.IsValid():
        return
    attr.Set(Gf.Vec3f(*value))


def reset_breakfast_scene_jointed_assets(env, env_ids: torch.Tensor) -> None:
    """Restore all breakfast scene assets with authored joints to their default state."""
    if env_ids is None:
        env_ids = torch.arange(env.num_envs, device=env.scene.device)
    if len(env_ids) == 0:
        return

    defaults_by_asset = _load_jointed_asset_defaults()
    if not defaults_by_asset:
        return

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    env_ids_cpu = env_ids.to(dtype=torch.long, device=env.scene.device).detach().cpu().tolist()

    for asset_name, defaults in defaults_by_asset.items():
        for env_id in env_ids_cpu:
            runtime_root = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/scene/{asset_name}")
            if runtime_root.IsValid():
                _set_runtime_root_xform(runtime_root, defaults["root_xform"])
                for rigid_body_defaults in defaults["rigid_bodies"]:
                    rigid_prim = stage.GetPrimAtPath(str(runtime_root.GetPath()) + rigid_body_defaults["suffix"])
                    if not rigid_prim.IsValid():
                        continue
                    _set_runtime_root_xform(rigid_prim, rigid_body_defaults["xform"])
                    _set_runtime_vector_attr(
                        rigid_prim, "physics:velocity", rigid_body_defaults["linear_velocity"]
                    )
                    _set_runtime_vector_attr(
                        rigid_prim, "physics:angularVelocity", rigid_body_defaults["angular_velocity"]
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
