# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Piper action-space conversion and real-robot-style command filtering."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np

PiperActionFormat = Literal["env", "model"]


@dataclass(frozen=True)
class PiperActionFilterConfig:
    """Low-level command filter that mirrors the Piper deployment path.

    The filter operates in RoboLab environment action space:
    ``[left_arm(6), right_arm(6), left_gripper_m, right_gripper_m]``.
    """

    use_ema: bool = True
    alpha: float = 0.6
    max_joint_delta: float = 0.15
    max_gripper_delta: float = 0.0167
    model_gripper_scale: float = 0.035
    max_gripper_opening: float = 0.035

    def __post_init__(self) -> None:
        if not (0.0 < self.alpha <= 1.0):
            raise ValueError(f"alpha must be in (0, 1], got {self.alpha!r}")
        if self.max_joint_delta <= 0:
            raise ValueError(f"max_joint_delta must be positive, got {self.max_joint_delta!r}")
        if self.max_gripper_delta <= 0:
            raise ValueError(f"max_gripper_delta must be positive, got {self.max_gripper_delta!r}")
        if self.model_gripper_scale <= 0:
            raise ValueError(f"model_gripper_scale must be positive, got {self.model_gripper_scale!r}")
        if self.max_gripper_opening <= 0:
            raise ValueError(f"max_gripper_opening must be positive, got {self.max_gripper_opening!r}")


def model_action_to_env_action(
    action: np.ndarray,
    *,
    model_gripper_scale: float = 0.035,
    max_gripper_opening: float = 0.035,
) -> np.ndarray:
    """Convert Piper model/server order to RoboLab env order and gripper units."""
    action = np.asarray(action, dtype=np.float32)
    if action.shape[-1] != 14:
        raise ValueError(f"Expected 14-D Piper action, got shape {action.shape}")

    env_action = np.empty_like(action)
    env_action[..., 0:6] = action[..., 0:6]
    env_action[..., 6:12] = action[..., 7:13]
    env_action[..., 12] = np.clip(np.clip(action[..., 6], 0.0, 1.0) * model_gripper_scale, 0.0, max_gripper_opening)
    env_action[..., 13] = np.clip(np.clip(action[..., 13], 0.0, 1.0) * model_gripper_scale, 0.0, max_gripper_opening)
    return env_action


def clamp_env_grippers(action: np.ndarray, *, max_gripper_opening: float = 0.035) -> np.ndarray:
    """Clamp a RoboLab env-space Piper action's gripper commands."""
    action = np.asarray(action, dtype=np.float32).copy()
    if action.shape[-1] != 14:
        raise ValueError(f"Expected 14-D Piper action, got shape {action.shape}")
    action[..., 12:14] = np.clip(action[..., 12:14], 0.0, max_gripper_opening)
    return action


def infer_piper_action_format(actions: np.ndarray, *, source_hdf5: str | None = None) -> PiperActionFormat:
    """Infer whether a 14-D Piper trace is in env or model/server order.

    Traces extracted from RoboLab HDF5 are known env-space actions.  For external
    raw traces, the right-arm final joint in column 12 usually makes env-space
    gripper columns invalid, while model/server grippers live in columns 6 and 13.
    Ambiguous traces default to env-space to preserve historical replay files.
    """
    actions = np.asarray(actions, dtype=np.float32)
    if actions.ndim != 2 or actions.shape[1] != 14:
        raise ValueError(f"Expected Piper trace with shape (steps, 14), got {actions.shape}")
    if source_hdf5:
        return "env"

    tol = 1.0e-5
    env_grippers = actions[:, 12:14]
    model_grippers = actions[:, [6, 13]]
    env_valid = bool(np.all((env_grippers >= -tol) & (env_grippers <= 0.035 + tol)))
    model_valid = bool(np.all((model_grippers >= -tol) & (model_grippers <= 1.0 + tol)))

    if model_valid and not env_valid:
        return "model"
    return "env"


class PiperActionFilter:
    """Stateful per-env filter for Piper absolute position commands."""

    def __init__(self, config: PiperActionFilterConfig | None = None, *, action_format: PiperActionFormat = "env"):
        self.config = config or PiperActionFilterConfig()
        if action_format not in ("env", "model"):
            raise ValueError(f"action_format must be 'env' or 'model', got {action_format!r}")
        self.action_format = action_format
        self._last_actions: dict[int, np.ndarray] = {}

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._last_actions.clear()
        else:
            self._last_actions.pop(env_id, None)

    def filter(self, action: np.ndarray, obs: Any, *, env_id: int = 0) -> np.ndarray:
        target = self.to_env_action(action)
        last = self._last_actions.get(env_id)
        if last is None:
            last = self._initial_action_from_obs(obs, env_id=env_id)

        if self.config.use_ema:
            target = self.config.alpha * target + (1.0 - self.config.alpha) * last

        max_delta = np.asarray(
            [self.config.max_joint_delta] * 12 + [self.config.max_gripper_delta] * 2,
            dtype=np.float32,
        )
        filtered = last + np.clip(target - last, -max_delta, max_delta)
        filtered = clamp_env_grippers(filtered, max_gripper_opening=self.config.max_gripper_opening)
        self._last_actions[env_id] = filtered
        return filtered.copy()

    def to_env_action(self, action: np.ndarray) -> np.ndarray:
        if self.action_format == "model":
            return model_action_to_env_action(
                action,
                model_gripper_scale=self.config.model_gripper_scale,
                max_gripper_opening=self.config.max_gripper_opening,
            )
        return clamp_env_grippers(action, max_gripper_opening=self.config.max_gripper_opening)

    def _initial_action_from_obs(self, obs: Any, *, env_id: int) -> np.ndarray:
        proprio = obs["proprio_obs"]
        left_arm = _obs_value_to_numpy(proprio["left_arm_joint_pos"], env_id=env_id)
        right_arm = _obs_value_to_numpy(proprio["right_arm_joint_pos"], env_id=env_id)
        left_gripper = _gripper_obs_to_opening(
            proprio["left_gripper_pos"], env_id=env_id, max_gripper_opening=self.config.max_gripper_opening
        )
        right_gripper = _gripper_obs_to_opening(
            proprio["right_gripper_pos"], env_id=env_id, max_gripper_opening=self.config.max_gripper_opening
        )
        return np.concatenate(
            [
                np.asarray(left_arm, dtype=np.float32).reshape(-1),
                np.asarray(right_arm, dtype=np.float32).reshape(-1),
                np.asarray([left_gripper, right_gripper], dtype=np.float32),
            ]
        )


class PiperActionFilterClient:
    """Client wrapper that filters Piper actions before env.step()."""

    def __init__(
        self,
        client: Any,
        *,
        config: PiperActionFilterConfig | None = None,
        action_format: PiperActionFormat = "env",
        enabled: bool = True,
    ) -> None:
        self.client = client
        self.filter = PiperActionFilter(config=config, action_format=action_format)
        self.enabled = bool(enabled)

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        ret = self.client.infer(obs, instruction, env_id=env_id)
        ret = dict(ret)
        if self.enabled:
            ret["action"] = self.filter.filter(ret["action"], obs, env_id=env_id)
        else:
            ret["action"] = self.filter.to_env_action(ret["action"])
        return ret

    def infer_batch(self, obs: Any, instruction: str, *, env_ids: list[int]) -> dict[int, dict]:
        rets = self.client.infer_batch(obs, instruction, env_ids=env_ids)
        return {
            env_id: {
                **ret,
                "action": (
                    self.filter.filter(ret["action"], obs, env_id=env_id)
                    if self.enabled
                    else self.filter.to_env_action(ret["action"])
                ),
            }
            for env_id, ret in rets.items()
        }

    def reset(self, *, env_id: int | None = None) -> None:
        self.filter.reset(env_id=env_id)
        self.client.reset(env_id=env_id)

    def close(self) -> None:
        self.client.close()

    def visualize(self, obs: Any, *, env_id: int = 0) -> np.ndarray | None:
        return self.client.visualize(obs, env_id=env_id)


def _obs_value_to_numpy(value: Any, *, env_id: int) -> np.ndarray:
    env_value = value[env_id]
    if hasattr(env_value, "detach"):
        env_value = env_value.detach().cpu().numpy()
    return np.asarray(env_value, dtype=np.float32)


def _gripper_obs_to_opening(value: Any, *, env_id: int, max_gripper_opening: float) -> float:
    gripper_obs = _obs_value_to_numpy(value, env_id=env_id)
    normalized = float(np.max(np.abs(gripper_obs)))
    return float(np.clip(normalized, 0.0, 1.0) * max_gripper_opening)
