# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import logging
import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Any

import numpy as np
from openpi_client import image_tools, websocket_client_policy

from robolab.eval.base_client import InferenceClient

logger = logging.getLogger(__name__)


class Pi0PiperDualArmClient(InferenceClient):
    """Dual-arm client for the "Double Piper" robot.

    The AgileX/OpenPI server expects ordinary-policy observations in this shape:
    ``state`` = [left_arm(6), right_arm(6)], ``gripper_position`` = [left, right],
    and three RGB images under ``images`` in CHW uint8 layout.

    Server actions arrive as [left_arm(6), left_gripper, right_arm(6), right_gripper].
    RoboLab's Piper env expects [left_arm(6), right_arm(6), left_gripper, right_gripper],
    with gripper commands in meters rather than normalized [0, 1].
    """

    DEFAULT_HORIZONS: dict[str, int] = {
        "pi0": 10,
        "pi0_fast": 10,
        "paligemma": 10,
        "paligemma_fast": 10,
        "pi05": 15,
    }
    FALLBACK_HORIZON: int = 15
    MAX_GRIPPER_OPENING: float = 0.035

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8000,
        open_loop_horizon: int | None = None,
        remote_uri: str | None = None,
        policy_variant: str = "pi05",
    ) -> None:
        super().__init__()
        if open_loop_horizon is None:
            open_loop_horizon = self.DEFAULT_HORIZONS.get(policy_variant, self.FALLBACK_HORIZON)
        self.open_loop_horizon = int(open_loop_horizon)
        self.policy_variant = policy_variant
        self._remote_uri = remote_uri
        self._remote_host = remote_host
        self._remote_port = remote_port
        self._display = remote_uri if remote_uri is not None else f"{remote_host}:{remote_port}"

        print(f"[{self.__class__.__name__}] Awaiting for server on {self._display} to be ready...")
        self.client = self._connect()
        print(f"[{self.__class__.__name__}] Connected to {self._display}.")

    def _connect(self):
        if self._remote_uri is not None:
            return websocket_client_policy.WebsocketClientPolicy(self._remote_uri)
        return websocket_client_policy.WebsocketClientPolicy(self._remote_host, self._remote_port)

    def _infer_with_retry(self, request: dict, max_retries: int = 3) -> dict:
        """Call server, reconnecting up to ``max_retries`` times on connection drop."""
        import websockets.exceptions

        for attempt in range(max_retries):
            try:
                return self.client.infer(request)
            except (
                websockets.exceptions.ConnectionClosedError,
                websockets.exceptions.ConnectionClosedOK,
                OSError,
            ) as e:
                if attempt + 1 >= max_retries:
                    raise
                logger.warning(
                    "[%s] Connection lost (%s), reconnecting (attempt %d/%d)...",
                    self.__class__.__name__, e, attempt + 1, max_retries,
                )
                self.client = self._connect()
                self._chunks.clear()
                self._counters.clear()

    # ---- required hooks -----------------------------------------------

    def _extract_observation(self, raw_obs: dict, *, env_id: int = 0) -> dict:
        image_obs = raw_obs["image_obs"]
        left_hand_image = image_obs["left_hand_camera"][env_id].clone().detach().cpu().numpy()
        right_hand_image = image_obs["right_hand_camera"][env_id].clone().detach().cpu().numpy()
        first_person_image = image_obs["first_person_camera"][env_id].clone().detach().cpu().numpy()

        robot_state = raw_obs["proprio_obs"]
        left_arm_joint_pos = robot_state["left_arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        right_arm_joint_pos = robot_state["right_arm_joint_pos"][env_id].clone().detach().cpu().numpy()
        left_gripper_pos = robot_state["left_gripper_pos"][env_id].clone().detach().cpu().numpy()
        right_gripper_pos = robot_state["right_gripper_pos"][env_id].clone().detach().cpu().numpy()

        return {
            "left_hand_image": left_hand_image,
            "right_hand_image": right_hand_image,
            "first_person_image": first_person_image,
            "left_arm_joint_pos": left_arm_joint_pos,
            "right_arm_joint_pos": right_arm_joint_pos,
            "left_gripper_pos": left_gripper_pos,
            "right_gripper_pos": right_gripper_pos,
        }

    def _pack_request(self, extracted_obs: dict, instruction: str) -> dict:
        left_gripper = self._gripper_scalar(extracted_obs["left_gripper_pos"])
        right_gripper = self._gripper_scalar(extracted_obs["right_gripper_pos"])

        return {
            "state": np.concatenate(
                [
                    extracted_obs["left_arm_joint_pos"],
                    extracted_obs["right_arm_joint_pos"],
                ]
            ).astype(np.float32),
            "gripper_position": np.asarray([left_gripper, right_gripper], dtype=np.float32),
            "images": {
                "cam_top": self._to_chw_uint8(extracted_obs["first_person_image"]),
                "cam_left_wrist": self._to_chw_uint8(extracted_obs["left_hand_image"]),
                "cam_right_wrist": self._to_chw_uint8(extracted_obs["right_hand_image"]),
            },
            "prompt": instruction,
        }

    def _query_server(self, request: dict) -> dict:
        return self._infer_with_retry(request)

    def _unpack_response(self, response: dict) -> np.ndarray:
        return np.asarray(response["actions"])

    # ---- optional hooks -----------------------------------------------

    def _postprocess_chunk(self, chunk: np.ndarray) -> np.ndarray:
        # Server: [left_arm(6), left_gripper, right_arm(6), right_gripper]
        # Env:    [left_arm(6), right_arm(6), left_gripper, right_gripper]
        chunk = np.asarray(chunk, dtype=np.float32)
        if chunk.shape[-1] != 14:
            raise ValueError(f"Expected Piper action chunks with 14 dims, got shape {chunk.shape}")

        env_chunk = np.empty_like(chunk)
        env_chunk[..., 0:6] = chunk[..., 0:6]
        env_chunk[..., 6:12] = chunk[..., 7:13]
        env_chunk[..., 12] = np.clip(chunk[..., 6], 0.0, 1.0) * self.MAX_GRIPPER_OPENING
        env_chunk[..., 13] = np.clip(chunk[..., 13], 0.0, 1.0) * self.MAX_GRIPPER_OPENING
        return env_chunk

    @staticmethod
    def _to_chw_uint8(image: np.ndarray) -> np.ndarray:
        """Convert IsaacLab HWC RGB/RGBA images to the AgileX server's CHW uint8 RGB layout."""
        image = np.asarray(image)
        if image.ndim != 3:
            raise ValueError(f"Expected image with 3 dims, got shape {image.shape}")
        if image.shape[-1] not in (3, 4):
            raise ValueError(f"Expected HWC RGB/RGBA image, got shape {image.shape}")
        image = image[..., :3]
        if image.dtype != np.uint8:
            if np.issubdtype(image.dtype, np.floating) and image.size and np.nanmax(image) <= 1.0:
                image = image * 255.0
            image = np.clip(image, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(image.transpose(2, 0, 1))

    @staticmethod
    def _gripper_scalar(gripper_obs: np.ndarray) -> float:
        """Collapse one side's finger observations to a single normalized opening ratio."""
        value = np.max(np.abs(np.asarray(gripper_obs, dtype=np.float32)))
        return float(np.clip(value, 0.0, 1.0))

    def _build_visualization(self, extracted_obs: dict) -> np.ndarray:
        img1 = image_tools.resize_with_pad(extracted_obs["first_person_image"], 224, 224)
        img2 = image_tools.resize_with_pad(extracted_obs["left_hand_image"], 224, 224)
        img3 = image_tools.resize_with_pad(extracted_obs["right_hand_image"], 224, 224)
        return np.concatenate([img1, img2, img3], axis=1)


@dataclass
class _PendingRTCChunk:
    """One background replan and the old model-space prefix it conditions on."""

    future: Future[dict]
    start_index: int
    chunk_index: int
    prefix_model_actions: np.ndarray
    prefix_steps: int
    ready_before_prefix_logged: bool = False


@dataclass
class _RTCChunkState:
    """Client-side execution state for one Piper environment."""

    robot_actions: np.ndarray
    model_actions: np.ndarray
    next_index: int = 0
    chunk_index: int = 0
    pending: _PendingRTCChunk | None = None
    last_action: np.ndarray | None = None
    last_viz: np.ndarray | None = None
    executed_actions: int = 0


class Pi0RTCPiperDualArmClient(Pi0PiperDualArmClient):
    """Training-time action-conditioning RTC client for the dual-arm Piper.

    The deployed RTC service returns a full robot-space action chunk and its
    paired model-space chunk.  The client executes the robot chunk at the
    control rate, then asynchronously replans after ``execution_horizon``
    actions.  The replan carries the still-unexecuted *model-space* actions so
    the server can hard-condition its new chunk on the committed prefix.

    This intentionally differs from a one-step RTC loop: inference never
    blocks a running chunk, so a 30 Hz action stream can continue while a
    slower server prepares the next chunk.
    """

    DEFAULT_CONTROL_HZ: float = 30.0
    DEFAULT_EXECUTION_HORIZON: int = 10
    DEFAULT_INFERENCE_DELAY_STEPS: int = 5
    MODEL_ACTION_DIM: int = 32

    def __init__(
        self,
        remote_host: str = "localhost",
        remote_port: int = 8001,
        remote_uri: str | None = None,
        control_hz: float = DEFAULT_CONTROL_HZ,
        execution_horizon: int = DEFAULT_EXECUTION_HORIZON,
        inference_delay_steps: int = DEFAULT_INFERENCE_DELAY_STEPS,
        rtc_debug: bool = False,
        rtc_debug_action_interval: int = 10,
    ) -> None:
        if control_hz <= 0:
            raise ValueError(f"control_hz must be positive, got {control_hz}")
        if execution_horizon <= 0:
            raise ValueError(f"execution_horizon must be positive, got {execution_horizon}")
        if inference_delay_steps <= 0:
            raise ValueError(f"inference_delay_steps must be positive, got {inference_delay_steps}")
        if inference_delay_steps > execution_horizon:
            raise ValueError(
                "inference_delay_steps must not exceed execution_horizon, got "
                f"{inference_delay_steps} > {execution_horizon}"
            )
        if rtc_debug_action_interval <= 0:
            raise ValueError(
                "rtc_debug_action_interval must be positive, got "
                f"{rtc_debug_action_interval}"
            )

        # Reuse the established OpenPI transport and reconnect behavior.  The
        # base action cache is deliberately unused; this class owns paired
        # robot/model-space chunks instead.
        super().__init__(
            remote_host=remote_host,
            remote_port=remote_port,
            remote_uri=remote_uri,
            open_loop_horizon=1,
            policy_variant="rtc",
        )
        self.control_hz = float(control_hz)
        self.execution_horizon = int(execution_horizon)
        self.inference_delay_steps = int(inference_delay_steps)
        self._control_period_s = 1.0 / self.control_hz
        self._last_control_start: float | None = None
        self._last_server_timing: dict | None = None
        self._rtc_states: dict[int, _RTCChunkState] = {}
        self.rtc_debug = bool(rtc_debug)
        self.rtc_debug_action_interval = int(rtc_debug_action_interval)
        # A WebsocketClientPolicy connection is not safe for overlapping infer
        # calls.  RTC is intentionally single-env; one worker hides inference
        # latency without concurrent use of that connection.
        self._replan_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pi0-rtc-replan")
        self._debug(
            "config: "
            f"remote={self._display} control_hz={self.control_hz:g} "
            f"execution_horizon={self.execution_horizon} "
            f"inference_delay_steps={self.inference_delay_steps} "
            f"action_debug_interval={self.rtc_debug_action_interval}"
        )

    def infer(self, obs: Any, instruction: str, *, env_id: int = 0) -> dict:
        """Return the next buffered action without blocking on a replan."""
        if not instruction or not instruction.strip():
            raise AssertionError("Pi0 RTC requires a non-empty language instruction")

        self._wait_for_control_slot()
        state = self._rtc_states.get(env_id)
        extracted = None

        if state is None:
            # A first chunk is needed before the episode can move.  Every
            # later request is asynchronous and cannot stall env.step().
            extracted = self._extract_observation(obs, env_id=env_id)
            response = self._query_server(self._pack_rtc_request(extracted, instruction))
            state = self._make_chunk_state(response)
            state.last_viz = self._build_visualization(extracted)
            self._rtc_states[env_id] = state
            self._debug(
                "initial chunk: "
                f"env={env_id} robot_shape={state.robot_actions.shape} "
                f"model_shape={state.model_actions.shape} "
                f"server_timing={self._format_server_timing(self._last_server_timing)}"
            )
            # The initial request may take far longer than one control period.
            # Start rate limiting from when its first action is actually ready.
            self._last_control_start = time.monotonic()
        else:
            self._collect_ready_replan(state)

        if state.pending is None and state.next_index >= self.execution_horizon:
            # The observation is from immediately before the first committed
            # prefix action.  The worker starts while those prefix actions are
            # sent to the environment at subsequent 30 Hz ticks.
            extracted = self._extract_observation(obs, env_id=env_id)
            state.last_viz = self._build_visualization(extracted)
            self._start_replan(state, extracted, instruction)

        selected_index = state.next_index
        previous_action = state.last_action
        action = self._next_buffered_action(state)
        state.executed_actions += 1
        self._debug_action(state, selected_index, previous_action, action)
        return {"action": action, "viz": state.last_viz}

    def infer_batch(self, obs: Any, instruction: str, *, env_ids: list[int]) -> dict[int, dict]:
        """RTC uses one ordered WebSocket stream and therefore requires one env."""
        if len(env_ids) > 1:
            raise ValueError(
                "Pi0 training-time RTC supports one environment per client; use --num-envs 1."
            )
        return super().infer_batch(obs, instruction, env_ids=env_ids)

    def reset(self, *, env_id: int | None = None) -> None:
        # Do not start a new episode while the old request is using the shared
        # WebSocket.  Episode reset is outside the control loop, so waiting
        # here is safe; late responses are discarded with their old state.
        states = self._rtc_states.values() if env_id is None else [self._rtc_states.get(env_id)]
        for state in states:
            if state is not None and state.pending is not None:
                try:
                    state.pending.future.result()
                except Exception:
                    logger.debug("Discarding failed RTC replan during reset", exc_info=True)

        super().reset(env_id=env_id)
        if env_id is None:
            self._rtc_states.clear()
        else:
            self._rtc_states.pop(env_id, None)
        self._last_control_start = None
        self._last_server_timing = None

    def close(self) -> None:
        self.reset()
        self._replan_executor.shutdown(wait=True)

    def _wait_for_control_slot(self) -> None:
        now = time.monotonic()
        if self._last_control_start is not None:
            remaining = self._control_period_s - (now - self._last_control_start)
            if remaining > 0:
                time.sleep(remaining)
                now = time.monotonic()
        self._last_control_start = now

    def _make_chunk_state(self, response: dict) -> _RTCChunkState:
        robot_actions, model_actions = self._unpack_chunk_response(response)
        self._last_server_timing = response.get("server_timing")
        postprocessed_robot_actions = self._postprocess_chunk(robot_actions)
        self._debug_response_chunks("initial response", robot_actions, model_actions, postprocessed_robot_actions)
        return _RTCChunkState(
            robot_actions=postprocessed_robot_actions,
            model_actions=model_actions,
        )

    def _start_replan(self, state: _RTCChunkState, extracted_obs: dict, instruction: str) -> None:
        start_index = state.next_index
        if start_index >= len(state.model_actions):
            return

        leftover = np.ascontiguousarray(state.model_actions[start_index:].copy())
        prefix_steps = min(self.inference_delay_steps, len(leftover))
        if prefix_steps == 0:
            return

        request = self._pack_rtc_request(
            extracted_obs,
            instruction,
            prev_leftover_model=leftover,
            prefix_steps=prefix_steps,
            client_step=start_index,
            local_chunk_index=state.chunk_index,
        )
        state.pending = _PendingRTCChunk(
            future=self._replan_executor.submit(self._query_server, request),
            start_index=start_index,
            chunk_index=state.chunk_index,
            prefix_model_actions=leftover[:prefix_steps].copy(),
            prefix_steps=prefix_steps,
        )
        self._debug(
            "replan request: "
            f"global_step={state.executed_actions} chunk={state.chunk_index} "
            f"start_index={start_index} leftover_shape={leftover.shape} "
            f"prefix_steps={prefix_steps} "
            f"leftover={self._array_summary(leftover)}"
        )

    def _collect_ready_replan(self, state: _RTCChunkState) -> None:
        pending = state.pending
        if pending is None or not pending.future.done():
            return

        # Paper RTC hard-conditions the new chunk's first ``prefix_steps`` to
        # the old leftover prefix.  If inference returns earlier than that,
        # keep executing the old chunk until the conditioned prefix has really
        # been consumed, then switch to the same index in the new chunk.
        elapsed = max(0, state.next_index - pending.start_index)
        if elapsed < pending.prefix_steps:
            if not pending.ready_before_prefix_logged:
                pending.ready_before_prefix_logged = True
                self._debug(
                    "replan ready early: "
                    f"global_step={state.executed_actions} chunk={pending.chunk_index} "
                    f"elapsed={elapsed} prefix_steps={pending.prefix_steps}; "
                    "continuing old chunk until the hard-conditioned prefix is consumed"
                )
            return

        state.pending = None
        try:
            response = pending.future.result()
            robot_actions, model_actions = self._unpack_chunk_response(response)
        except Exception:
            logger.exception("[%s] RTC replan failed; continuing the current chunk", self.__class__.__name__)
            return

        self._last_server_timing = response.get("server_timing")
        response_prefix = model_actions[:pending.prefix_steps]
        model_prefix_max_abs = float(np.max(np.abs(response_prefix - pending.prefix_model_actions)))
        if not np.allclose(response_prefix, pending.prefix_model_actions, rtol=1e-4, atol=1e-5):
            logger.warning(
                "[%s] RTC response prefix differs from the hard-conditioned prefix; "
                "assuming the service response is still aligned to the request timeline.",
                self.__class__.__name__,
            )

        # The service's returned chunk starts at the request's first leftover
        # action.  Actions executed while the worker ran are already in the
        # past, so drop exactly that elapsed prefix before switching chunks.
        if elapsed >= len(robot_actions):
            raise RuntimeError(
                "RTC replan arrived after its entire action chunk had expired; "
                "increase the chunk horizon or reduce inference latency."
            )

        postprocessed_robot_actions = self._postprocess_chunk(robot_actions)
        self._debug_response_chunks(
            f"replan response chunk={pending.chunk_index + 1}",
            robot_actions,
            model_actions,
            postprocessed_robot_actions,
        )
        old_robot_prefix = state.robot_actions[
            pending.start_index : pending.start_index + pending.prefix_steps
        ]
        robot_prefix_max_abs = float(
            np.max(np.abs(postprocessed_robot_actions[:pending.prefix_steps] - old_robot_prefix))
        )
        seam_max_abs = (
            float(np.max(np.abs(postprocessed_robot_actions[elapsed] - state.last_action)))
            if state.last_action is not None
            else float("nan")
        )
        self._debug(
            "replan switch: "
            f"global_step={state.executed_actions} old_chunk={pending.chunk_index} "
            f"new_chunk={pending.chunk_index + 1} elapsed={elapsed} "
            f"model_prefix_max_abs={model_prefix_max_abs:.6g} "
            f"robot_prefix_max_abs={robot_prefix_max_abs:.6g} "
            f"seam_max_abs={seam_max_abs:.6g} "
            f"server_timing={self._format_server_timing(self._last_server_timing)}"
        )

        state.robot_actions = postprocessed_robot_actions
        state.model_actions = model_actions
        state.next_index = elapsed
        state.chunk_index = pending.chunk_index + 1

    def _next_buffered_action(self, state: _RTCChunkState) -> np.ndarray:
        if state.next_index < len(state.robot_actions):
            action = state.robot_actions[state.next_index]
            state.next_index += 1
            state.last_action = action
            return action

        # A late response may still arrive with a valid tail (its timeline
        # began at the replan point), but until then holding the last joint
        # target is safer than issuing an arbitrary action or blocking.
        if state.last_action is None:
            raise RuntimeError("RTC action buffer is empty before an initial chunk was received")
        logger.warning("[%s] RTC action chunk exhausted; holding the last action", self.__class__.__name__)
        state.next_index += 1
        return state.last_action

    def _debug_action(
        self,
        state: _RTCChunkState,
        selected_index: int,
        previous_action: np.ndarray | None,
        action: np.ndarray,
    ) -> None:
        """Emit sparse action diagnostics without changing control behavior."""
        if not self.rtc_debug or state.executed_actions % self.rtc_debug_action_interval:
            return
        delta = float(np.max(np.abs(action - previous_action))) if previous_action is not None else float("nan")
        self._debug(
            "action: "
            f"global_step={state.executed_actions} chunk={state.chunk_index} "
            f"local_index={selected_index} max_delta={delta:.6g} "
            f"arms={self._format_vector(action[:12])} "
            f"grippers_m={self._format_vector(action[12:])}"
        )

    def _debug(self, message: str) -> None:
        if self.rtc_debug:
            print(f"[Pi0RTC] {message}", flush=True)

    @staticmethod
    def _format_vector(values: np.ndarray, *, precision: int = 4) -> str:
        return np.array2string(np.asarray(values), precision=precision, suppress_small=True)

    @staticmethod
    def _array_summary(values: np.ndarray) -> str:
        array = np.asarray(values)
        if array.size == 0:
            return f"shape={array.shape} dtype={array.dtype} empty"
        finite = array[np.isfinite(array)]
        if finite.size == 0:
            return f"shape={array.shape} dtype={array.dtype} all_nonfinite"
        return (
            f"shape={array.shape} dtype={array.dtype} "
            f"min={float(np.min(finite)):.6g} max={float(np.max(finite)):.6g} "
            f"mean={float(np.mean(finite)):.6g} std={float(np.std(finite)):.6g}"
        )

    def _debug_rtc_request(self, request: dict, *, request_kind: str) -> None:
        if not self.rtc_debug:
            return

        images = request.get("images", {})
        image_summaries = ", ".join(
            f"{name}:shape={np.asarray(image).shape},dtype={np.asarray(image).dtype}"
            for name, image in sorted(images.items())
        )
        self._debug(
            "request: "
            f"kind={request_kind} "
            f"state_summary={self._array_summary(request['state'])} "
            f"state_values={self._format_vector(request['state'])} "
            f"gripper_position={self._format_vector(request['gripper_position'])} "
            f"prompt={request.get('prompt')!r} "
            f"images=[{image_summaries}]"
        )

        prev_leftover = request.get("_paper_rtc_prev_leftover_model")
        if prev_leftover is not None:
            self._debug(
                "request rtc conditioning: "
                f"client_step={request.get('_paper_rtc_client_step')} "
                f"local_chunk_index={request.get('_paper_rtc_local_chunk_index')} "
                f"prefix_steps={request.get('_paper_rtc_inference_delay_steps')} "
                f"attention_horizon={request.get('_paper_rtc_prefix_attention_horizon')} "
                f"prev_leftover={self._array_summary(prev_leftover)} "
                f"prefix_first={self._format_vector(prev_leftover[0])} "
                f"prefix_last={self._format_vector(prev_leftover[request['_paper_rtc_inference_delay_steps'] - 1])}"
            )

    def _debug_response_chunks(
        self,
        label: str,
        raw_robot_actions: np.ndarray,
        model_actions: np.ndarray,
        env_robot_actions: np.ndarray,
    ) -> None:
        if not self.rtc_debug:
            return

        self._debug(
            f"{label}: "
            f"raw_robot={self._array_summary(raw_robot_actions)} "
            f"model={self._array_summary(model_actions)} "
            f"env_robot={self._array_summary(env_robot_actions)}"
        )
        self._debug(
            f"{label} samples: "
            f"raw_first={self._format_vector(raw_robot_actions[0])} "
            f"raw_exec_horizon={self._format_vector(raw_robot_actions[self.execution_horizon])} "
            f"raw_last={self._format_vector(raw_robot_actions[-1])}"
        )
        self._debug(
            f"{label} postprocess: "
            f"env_first={self._format_vector(env_robot_actions[0])} "
            f"env_exec_horizon={self._format_vector(env_robot_actions[self.execution_horizon])} "
            f"env_last={self._format_vector(env_robot_actions[-1])}"
        )
        self._debug(
            f"{label} grippers: "
            f"raw_left_dim6_norm_range=({float(np.min(raw_robot_actions[:, 6])):.6g},"
            f"{float(np.max(raw_robot_actions[:, 6])):.6g}) "
            f"raw_right_dim13_norm_range=({float(np.min(raw_robot_actions[:, 13])):.6g},"
            f"{float(np.max(raw_robot_actions[:, 13])):.6g}) "
            f"env_left_m_range=({float(np.min(env_robot_actions[:, 12])):.6g},"
            f"{float(np.max(env_robot_actions[:, 12])):.6g}) "
            f"env_right_m_range=({float(np.min(env_robot_actions[:, 13])):.6g},"
            f"{float(np.max(env_robot_actions[:, 13])):.6g})"
        )

    @staticmethod
    def _format_server_timing(server_timing: dict | None) -> str:
        """Return a compact, stable timing summary for RTC debug output."""
        if not server_timing:
            return "none"
        preferred_keys = (
            "request_kind",
            "conditioned_prefix_steps",
            "prev_leftover_len",
            "get_action_ms",
            "model_inference_ms",
            "infer_ms",
            "prev_total_ms",
            "server_recv_wait_ms",
            "server_prepare_ms",
            "server_pack_ms",
        )
        values = [
            f"{key}={server_timing[key]}"
            for key in preferred_keys
            if key in server_timing
        ]
        return ",".join(values) if values else "keys=" + ",".join(sorted(server_timing))

    def _pack_rtc_request(
        self,
        extracted_obs: dict,
        instruction: str,
        *,
        prev_leftover_model: np.ndarray | None = None,
        prefix_steps: int | None = None,
        client_step: int | None = None,
        local_chunk_index: int | None = None,
    ) -> dict:
        """Pack the training-time action-conditioning RTC protocol."""
        request = super()._pack_request(extracted_obs, instruction)
        request["_paper_rtc_client_chunk_request"] = True

        if prev_leftover_model is not None:
            if prev_leftover_model.ndim != 2 or prev_leftover_model.shape[1] != self.MODEL_ACTION_DIM:
                raise ValueError(
                    "Expected model-space RTC leftover with shape (H, "
                    f"{self.MODEL_ACTION_DIM}), got {prev_leftover_model.shape}"
                )
            assert prefix_steps is not None
            request.update(
                {
                    "_paper_rtc_prev_leftover_model": prev_leftover_model,
                    "_paper_rtc_inference_delay_steps": prefix_steps,
                    "_paper_rtc_prefix_attention_horizon": len(prev_leftover_model),
                    "_training_time_rtc_prefix_steps": prefix_steps,
                    "_paper_rtc_client_step": client_step,
                    "_paper_rtc_local_chunk_index": local_chunk_index,
                }
            )
        request_kind = "replan" if prev_leftover_model is not None else "initial"
        self._debug_rtc_request(request, request_kind=request_kind)
        return request

    def _unpack_chunk_response(self, response: dict) -> tuple[np.ndarray, np.ndarray]:
        """Validate the paired full chunks required by client-executed RTC."""
        robot_actions = np.asarray(response["actions"], dtype=np.float32)
        model_actions = np.asarray(response["model_space_actions"], dtype=np.float32)
        if robot_actions.ndim != 2 or robot_actions.shape[0] == 0 or robot_actions.shape[1] != 14:
            raise ValueError(f"Expected RTC robot action chunk with shape (H, 14), got {robot_actions.shape}")
        expected_model_shape = (robot_actions.shape[0], self.MODEL_ACTION_DIM)
        if model_actions.shape != expected_model_shape:
            raise ValueError(
                "Expected paired RTC model action chunk with shape "
                f"{expected_model_shape}, got {model_actions.shape}"
            )
        if self.execution_horizon >= len(robot_actions):
            raise ValueError(
                f"RTC execution_horizon={self.execution_horizon} must be shorter than "
                f"the returned action chunk length {len(robot_actions)}"
            )
        return robot_actions, model_actions

    @classmethod
    def _resize_to_chw_uint8(cls, image: np.ndarray) -> np.ndarray:
        return cls._to_chw_uint8(image_tools.resize_with_pad(image, 224, 224))


if __name__ == "__main__":
    import time

    import torch

    client = Pi0PiperDualArmClient()
    fake_obs = {
        "image_obs": {
            "left_hand_camera": [torch.zeros((480, 640, 3), dtype=torch.uint8)],
            "right_hand_camera": [torch.zeros((480, 640, 3), dtype=torch.uint8)],
            "first_person_camera": [torch.zeros((480, 640, 3), dtype=torch.uint8)],
        },
        "proprio_obs": {
            "left_arm_joint_pos": torch.zeros((1, 6), dtype=torch.float32),
            "right_arm_joint_pos": torch.zeros((1, 6), dtype=torch.float32),
            "left_gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
            "right_gripper_pos": torch.zeros((1, 1), dtype=torch.float32),
        },
    }
    fake_instruction = "pick up the rubiks cube and place it in the box"

    start = time.time()
    client.infer(fake_obs, fake_instruction)  # warm up
    num = 20
    for _ in range(num):
        ret = client.infer(fake_obs, fake_instruction)
        print(ret["action"].shape)
    end = time.time()

    print(f"Average inference time: {(end - start) / num}")
