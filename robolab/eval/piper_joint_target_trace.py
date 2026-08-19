"""Load and extract real-robot Piper joint-target traces for simulation replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np


PIPER_ENV_ACTION_DIM = 14
PIPER_GRIPPER_MAX_OPENING_M = 0.035
_DEFAULT_GRIPPER_RANGE_EPS = 1.0e-4


@dataclass(frozen=True)
class JointTargetTrace:
    """Reference joint trajectory plus the mapped RoboLab env-space targets."""

    env_targets: np.ndarray
    arm_reference: np.ndarray
    gripper_reference: np.ndarray
    raw_qpos: np.ndarray
    control_hz: float
    source_hdf5: str | None = None
    demo: str | None = None
    gripper_real_min: np.ndarray | None = None
    gripper_real_max: np.ndarray | None = None

    @property
    def steps(self) -> int:
        return int(self.env_targets.shape[0])

    @property
    def action_dim(self) -> int:
        return int(self.env_targets.shape[1])

    @property
    def duration_s(self) -> float:
        return self.steps / self.control_hz


def real_piper_qpos_to_env_arm_reference(raw_qpos: np.ndarray) -> np.ndarray:
    """Convert raw real-robot qpos order to RoboLab's 12-D arm order."""
    raw_qpos = np.asarray(raw_qpos, dtype=np.float32)
    if raw_qpos.ndim != 2 or raw_qpos.shape[1] != PIPER_ENV_ACTION_DIM:
        raise ValueError(f"Expected raw qpos with shape (steps, 14), got {raw_qpos.shape}.")
    return np.concatenate((raw_qpos[:, 0:6], raw_qpos[:, 7:13]), axis=1).astype(np.float32, copy=False)


def _resolve_gripper_range(
    values: np.ndarray,
    *,
    closed_override: float | None,
    open_override: float | None,
    quantile_low: float,
    quantile_high: float,
    min_range: float,
) -> tuple[float, float]:
    if closed_override is not None and open_override is not None:
        closed = float(closed_override)
        opened = float(open_override)
    else:
        q_low, q_high = np.quantile(values, [quantile_low, quantile_high]).tolist()
        closed = float(min(np.min(values), q_low)) if closed_override is None else float(closed_override)
        opened = float(max(np.max(values), q_high)) if open_override is None else float(open_override)

    if opened < closed:
        closed, opened = opened, closed
    return closed, opened


def _map_real_gripper_to_opening(
    values: np.ndarray,
    *,
    closed: float,
    opened: float,
    max_opening_m: float,
    min_range: float,
) -> np.ndarray:
    if opened - closed < min_range:
        if opened <= 0.0:
            return np.zeros_like(values, dtype=np.float32)
        return np.full_like(values, fill_value=max_opening_m, dtype=np.float32)
    normalized = (values - closed) / (opened - closed)
    return np.clip(normalized, 0.0, 1.0).astype(np.float32) * float(max_opening_m)


def convert_real_piper_qpos_to_joint_trace(
    raw_qpos: np.ndarray,
    *,
    control_hz: float = 30.0,
    left_gripper_closed: float | None = None,
    left_gripper_open: float | None = None,
    right_gripper_closed: float | None = None,
    right_gripper_open: float | None = None,
    gripper_quantile_low: float = 0.01,
    gripper_quantile_high: float = 0.99,
    max_gripper_opening_m: float = PIPER_GRIPPER_MAX_OPENING_M,
    min_gripper_range: float = _DEFAULT_GRIPPER_RANGE_EPS,
    source_hdf5: str | None = None,
    demo: str | None = None,
) -> JointTargetTrace:
    """Convert raw real-robot qpos into a replayable RoboLab joint-target trace."""
    raw_qpos = np.asarray(raw_qpos, dtype=np.float32)
    if raw_qpos.ndim != 2 or raw_qpos.shape[0] < 2 or raw_qpos.shape[1] != PIPER_ENV_ACTION_DIM:
        raise ValueError(f"Expected raw qpos with shape (steps>=2, 14), got {raw_qpos.shape}.")
    if not np.isfinite(raw_qpos).all():
        raise ValueError("Raw qpos contains NaN or infinity.")
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError(f"control_hz must be positive, got {control_hz!r}.")
    if not (0.0 <= gripper_quantile_low < gripper_quantile_high <= 1.0):
        raise ValueError(
            "Gripper quantiles must satisfy 0 <= low < high <= 1, got "
            f"{gripper_quantile_low} and {gripper_quantile_high}."
        )

    arm_reference = real_piper_qpos_to_env_arm_reference(raw_qpos)
    left_closed, left_opened = _resolve_gripper_range(
        raw_qpos[:, 6],
        closed_override=left_gripper_closed,
        open_override=left_gripper_open,
        quantile_low=gripper_quantile_low,
        quantile_high=gripper_quantile_high,
        min_range=min_gripper_range,
    )
    right_closed, right_opened = _resolve_gripper_range(
        raw_qpos[:, 13],
        closed_override=right_gripper_closed,
        open_override=right_gripper_open,
        quantile_low=gripper_quantile_low,
        quantile_high=gripper_quantile_high,
        min_range=min_gripper_range,
    )
    gripper_reference = np.stack(
        (
            _map_real_gripper_to_opening(
                raw_qpos[:, 6],
                closed=left_closed,
                opened=left_opened,
                max_opening_m=max_gripper_opening_m,
                min_range=min_gripper_range,
            ),
            _map_real_gripper_to_opening(
                raw_qpos[:, 13],
                closed=right_closed,
                opened=right_opened,
                max_opening_m=max_gripper_opening_m,
                min_range=min_gripper_range,
            ),
        ),
        axis=1,
    ).astype(np.float32)

    env_targets = np.empty_like(raw_qpos)
    env_targets[:, 0:6] = raw_qpos[:, 0:6]
    env_targets[:, 6:12] = raw_qpos[:, 7:13]
    env_targets[:, 12:14] = gripper_reference

    return JointTargetTrace(
        env_targets=env_targets.astype(np.float32),
        arm_reference=arm_reference.astype(np.float32),
        gripper_reference=gripper_reference.astype(np.float32),
        raw_qpos=raw_qpos.astype(np.float32),
        control_hz=float(control_hz),
        source_hdf5=source_hdf5,
        demo=demo,
        gripper_real_min=np.asarray([left_closed, right_closed], dtype=np.float32),
        gripper_real_max=np.asarray([left_opened, right_opened], dtype=np.float32),
    )


def extract_joint_target_trace_from_real_hdf5(
    hdf5_path: str | Path,
    *,
    control_hz: float = 30.0,
    **kwargs,
) -> JointTargetTrace:
    """Read a raw real-robot HDF5 episode and convert it to a joint-target trace."""
    hdf5_path = Path(hdf5_path)
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"HDF5 file does not exist: {hdf5_path}")

    with h5py.File(hdf5_path, "r") as handle:
        dataset_path = "observations/qpos"
        if dataset_path not in handle:
            raise KeyError(f"No dataset {dataset_path!r} in {hdf5_path}.")
        raw_qpos = np.asarray(handle[dataset_path], dtype=np.float32)

    return convert_real_piper_qpos_to_joint_trace(
        raw_qpos,
        control_hz=control_hz,
        source_hdf5=str(hdf5_path.resolve()),
        **kwargs,
    )


def save_joint_target_trace(trace: JointTargetTrace, path: str | Path) -> Path:
    """Persist a replayable joint-target trace as a compressed NPZ file."""
    path = Path(path).with_suffix(".npz")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        env_targets=trace.env_targets.astype(np.float32),
        arm_reference=trace.arm_reference.astype(np.float32),
        gripper_reference=trace.gripper_reference.astype(np.float32),
        raw_qpos=trace.raw_qpos.astype(np.float32),
        control_hz=np.asarray(trace.control_hz, dtype=np.float64),
        source_hdf5=np.asarray(trace.source_hdf5 or ""),
        demo=np.asarray(trace.demo or ""),
        gripper_real_min=(
            np.asarray(trace.gripper_real_min, dtype=np.float32)
            if trace.gripper_real_min is not None else np.zeros(2, dtype=np.float32)
        ),
        gripper_real_max=(
            np.asarray(trace.gripper_real_max, dtype=np.float32)
            if trace.gripper_real_max is not None else np.ones(2, dtype=np.float32)
        ),
    )
    return path


def load_joint_target_trace(path: str | Path) -> JointTargetTrace:
    """Load and validate a saved joint-target trace."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Joint target trace does not exist: {path}")
    with np.load(path, allow_pickle=False) as trace_file:
        required = ("env_targets", "arm_reference", "gripper_reference", "raw_qpos")
        missing = [key for key in required if key not in trace_file]
        if missing:
            raise ValueError(f"Joint target trace {path} is missing arrays: {missing}")
        env_targets = np.asarray(trace_file["env_targets"], dtype=np.float32)
        arm_reference = np.asarray(trace_file["arm_reference"], dtype=np.float32)
        gripper_reference = np.asarray(trace_file["gripper_reference"], dtype=np.float32)
        raw_qpos = np.asarray(trace_file["raw_qpos"], dtype=np.float32)
        control_hz = float(np.asarray(trace_file.get("control_hz", 30.0)).item())
        source_hdf5 = str(np.asarray(trace_file.get("source_hdf5", "")).item()) or None
        demo = str(np.asarray(trace_file.get("demo", "")).item()) or None
        gripper_real_min = np.asarray(trace_file.get("gripper_real_min", np.zeros(2)), dtype=np.float32)
        gripper_real_max = np.asarray(trace_file.get("gripper_real_max", np.ones(2)), dtype=np.float32)

    if env_targets.ndim != 2 or env_targets.shape[0] < 2 or env_targets.shape[1] != PIPER_ENV_ACTION_DIM:
        raise ValueError(f"Joint target trace {path} has invalid env_targets shape {env_targets.shape}.")
    if arm_reference.shape != (env_targets.shape[0], 12):
        raise ValueError(
            f"Joint target trace {path} has invalid arm_reference shape {arm_reference.shape}; "
            f"expected ({env_targets.shape[0]}, 12)."
        )
    if gripper_reference.shape != (env_targets.shape[0], 2):
        raise ValueError(
            f"Joint target trace {path} has invalid gripper_reference shape {gripper_reference.shape}; "
            f"expected ({env_targets.shape[0]}, 2)."
        )
    if raw_qpos.shape != env_targets.shape:
        raise ValueError(
            f"Joint target trace {path} has invalid raw_qpos shape {raw_qpos.shape}; "
            f"expected {env_targets.shape}."
        )
    if not (np.isfinite(env_targets).all() and np.isfinite(arm_reference).all() and np.isfinite(gripper_reference).all()):
        raise ValueError(f"Joint target trace {path} contains NaN or infinity.")
    if not np.isfinite(control_hz) or control_hz <= 0:
        raise ValueError(f"Joint target trace {path} has invalid control_hz={control_hz!r}.")

    return JointTargetTrace(
        env_targets=env_targets,
        arm_reference=arm_reference,
        gripper_reference=gripper_reference,
        raw_qpos=raw_qpos,
        control_hz=control_hz,
        source_hdf5=source_hdf5,
        demo=demo,
        gripper_real_min=gripper_real_min,
        gripper_real_max=gripper_real_max,
    )
