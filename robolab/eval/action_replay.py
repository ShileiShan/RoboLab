"""Deterministic, environment-space action-trace loading for open-loop evals."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ActionTrace:
    """A fixed action sequence sampled at a known control rate."""

    actions: np.ndarray
    control_hz: float
    source_hdf5: str | None = None
    demo: str | None = None

    @property
    def steps(self) -> int:
        return int(self.actions.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.actions.shape[1])

    @property
    def duration_s(self) -> float:
        return self.steps / self.control_hz


def load_action_trace(path: str | Path) -> ActionTrace:
    """Load and validate an action trace written by the extraction utility."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Action trace does not exist: {path}")
    with np.load(path, allow_pickle=False) as trace_file:
        if "actions" not in trace_file:
            raise ValueError(f"Action trace {path} has no 'actions' array.")
        actions = np.asarray(trace_file["actions"], dtype=np.float32)
        control_hz = float(np.asarray(trace_file.get("control_hz", 30.0)).item())
        source_hdf5 = str(np.asarray(trace_file["source_hdf5"]).item()) if "source_hdf5" in trace_file else None
        demo = str(np.asarray(trace_file["demo"]).item()) if "demo" in trace_file else None

    if actions.ndim != 2 or actions.shape[0] < 1 or actions.shape[1] < 1:
        raise ValueError(f"Action trace {path} must have shape (steps, action_dim), got {actions.shape}.")
    if not np.isfinite(actions).all():
        raise ValueError(f"Action trace {path} contains NaN or infinity.")
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError(f"Action trace {path} has invalid control_hz={control_hz!r}.")
    return ActionTrace(actions=actions, control_hz=control_hz, source_hdf5=source_hdf5, demo=demo)


class ActionReplayClient:
    """Minimal policy-client replacement that returns one saved action per step.

    Actions must already be in RoboLab's environment action order and units.
    In particular, Piper traces use the post-processed 14-D order recorded by
    ``PreStepActionsRecorder``; this client never calls a policy server.
    """

    def __init__(self, trace: ActionTrace):
        self.trace = trace
        self.max_steps = trace.steps
        self._indices: dict[int, int] = {}

    def infer_batch(
        self, obs: Any, instruction: str, *, env_ids: list[int]
    ) -> dict[int, dict[str, np.ndarray | None]]:
        del obs, instruction
        result: dict[int, dict[str, np.ndarray | None]] = {}
        for env_id in env_ids:
            index = self._indices.get(env_id, 0)
            if index >= self.trace.steps:
                raise RuntimeError(
                    "Action trace exhausted before the replay environment terminated. "
                    "Use a trace with the intended duration."
                )
            result[env_id] = {"action": self.trace.actions[index].copy(), "viz": None}
            self._indices[env_id] = index + 1
        return result

    def reset(self, *, env_id: int | None = None) -> None:
        if env_id is None:
            self._indices.clear()
        else:
            self._indices.pop(env_id, None)

    def close(self) -> None:
        return None
