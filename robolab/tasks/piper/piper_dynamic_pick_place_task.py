# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial
import json
import os

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_grabbed, object_moved_to_container, object_outside_of
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task
from robolab.tasks.piper.dynamic_scene_utils import (
    GENERATED_SCENE_ENV,
    DYNAMIC_DROP_PLAN_ENV,
    DYNAMIC_EPISODE_LENGTH_ENV,
    INSTRUCTION_ENV,
    OBJECT_NAME_ENV,
    OBJECT_NAMES_ENV,
    build_instruction,
)
from robolab.tasks.piper.dynamic_drop_events import sequential_drop_reset
from robolab.tasks.piper.piper_single_object_pick_place_task import PIPER_FINGER_CONTACTS


OBJECT_NAME = os.environ.get(OBJECT_NAME_ENV, "banana")
OBJECT_NAMES = json.loads(os.environ.get(OBJECT_NAMES_ENV, f'["{OBJECT_NAME}"]'))
TASK_OBJECT_NAMES = OBJECT_NAMES if len(OBJECT_NAMES) > 1 else [OBJECT_NAME]
SCENE_PATH = os.environ.get(GENERATED_SCENE_ENV)
INSTRUCTION = os.environ.get(
    INSTRUCTION_ENV,
    build_instruction(OBJECT_NAME, all_objects=len(TASK_OBJECT_NAMES) > 1),
)
DROP_PLAN_RAW = os.environ.get(DYNAMIC_DROP_PLAN_ENV)
DROP_PLAN = json.loads(DROP_PLAN_RAW) if DROP_PLAN_RAW else None
EPISODE_LENGTH_S = float(os.environ.get(DYNAMIC_EPISODE_LENGTH_ENV, "20"))

if not SCENE_PATH:
    raise RuntimeError(
        "PiperDynamicPickPlaceTask requires a generated scene. "
        "Use policies/pi0_family/run_piper.py with --dynamic-object or --dynamic-object-usd."
    )


@configclass
class PiperDynamicPickAndPlaceTerminations:
    """Termination configuration for runtime-generated Piper pick-place tasks."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_moved_to_container,
        params={
            # Generated scenes with multiple objects are all-object transfer
            # tasks: every object must leave pick_box and reach place_box.
            "object": TASK_OBJECT_NAMES,
            "target_container": "place_box",
            "source_container": "pick_box",
            "gripper_name": PIPER_FINGER_CONTACTS,
            "tolerance": 0.05,
            "require_contact_with": True,
            "require_gripper_detached": True,
            "logical": "all",
        },
    )


@configclass
class PiperDynamicPickPlaceEvents:
    """Default reset followed by generated-scene runtime sequential drop."""

    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    sequential_drop = EventTerm(
        func=sequential_drop_reset,
        mode="reset",
        params={"plan": DROP_PLAN},
    )


@dataclass
class PiperDynamicPickPlaceTask(Task):
    task_name = "PiperDynamicPickPlaceTask"
    contact_object_list = [*OBJECT_NAMES, "pick_box", "place_box", "table"]
    # All-object transfer needs object↔container contact information for every
    # generated body, not just the historical single target object.
    contact_sensor_body_object_list = TASK_OBJECT_NAMES
    scene = import_scene(SCENE_PATH, contact_object_list)
    terminations = PiperDynamicPickAndPlaceTerminations
    # Defining task events replaces the generic BaseEventCfg, so the event
    # configuration above explicitly retains reset_scene_to_default.
    events = PiperDynamicPickPlaceEvents if DROP_PLAN is not None else None
    record_setup_video = bool(DROP_PLAN and DROP_PLAN.get("record_setup_video", False))
    instruction: str = INSTRUCTION
    # This is policy-control time only.  Runtime sequential-drop setup occurs
    # inside env.reset() and intentionally does not consume this budget.
    episode_length_s: float = EPISODE_LENGTH_S

    subtasks = [
        Subtask(
            name="pick_and_place_all_objects" if len(TASK_OBJECT_NAMES) > 1 else "pick_and_place",
            conditions={
                object_name: [
                    (
                        partial(
                            object_grabbed,
                            object=object_name,
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        0.0,
                    ),
                    (
                        partial(
                            object_outside_of,
                            object=object_name,
                            container="pick_box",
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        0.0,
                    ),
                    (
                        partial(
                            object_moved_to_container,
                            object=object_name,
                            target_container="place_box",
                            source_container="pick_box",
                            require_contact_with=False,
                            require_gripper_detached=True,
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        1.0,
                    ),
                ]
                for object_name in TASK_OBJECT_NAMES
            },
            logical="all",
            score=1.0,
        )
    ]
