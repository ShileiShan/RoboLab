# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from functools import partial

import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from robolab.core.events.reset_breakfast_toaster import reset_breakfast_scene_jointed_assets
from robolab.core.scenes.utils import import_scene
from robolab.core.task.conditionals import object_on_top
from robolab.core.task.subtask import Subtask
from robolab.core.task.task import Task
from robolab.tasks.piper.piper_single_object_pick_place_task import PIPER_FINGER_CONTACTS


BREADS = ["mianbaopian_0_130", "mianbaopian_0_131"]
TOASTER = "mianbaojia_0_175"
TOASTER014 = "Toaster014"
PLATE = "clay_plates"
KETTLE = "shuihu_0_186"
CUP = "shuibei_0_098"
CONTACT_OBJECTS = BREADS + [TOASTER, PLATE, KETTLE, CUP, "table"]
SCENE_OBJECTS = CONTACT_OBJECTS + [TOASTER014]


@configclass
class MakeBreakfastTerminations:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=object_on_top,
        params={
            "object": BREADS,
            "reference_object": PLATE,
            "logical": "all",
            "require_gripper_detached": True,
            "gripper_name": PIPER_FINGER_CONTACTS,
        },
    )


@configclass
class MakeBreakfastEvents:
    reset_scene = EventTerm(func=mdp.reset_scene_to_default, mode="reset")
    reset_scene_jointed_assets = EventTerm(func=reset_breakfast_scene_jointed_assets, mode="reset")


@dataclass
class MakeBreakfastTask(Task):
    contact_object_list = CONTACT_OBJECTS
    scene = import_scene("breakfast.usda", SCENE_OBJECTS)
    events = MakeBreakfastEvents
    terminations = MakeBreakfastTerminations
    instruction = {
        "default": "Put the two bread slices into the toaster, then place the toasted bread slices onto the plate.",
        "vague": "Make breakfast.",
        "specific": "Pick up each of the two bread slices and insert them into the toaster on the table, then take the two slices out of the toaster and place them onto the plate.",
    }
    episode_length_s: int = 20
    attributes = ["conjunction", "semantics"]
    subtasks = [
        Subtask(
            name="breads_on_toaster",
            conditions={
                bread: [
                    (
                        partial(
                            object_on_top,
                            object=bread,
                            reference_object=TOASTER,
                            require_gripper_detached=True,
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        1.0,
                    ),
                ]
                for bread in BREADS
            },
            logical="all",
            score=0.5,
        ),
        Subtask(
            name="breads_on_plate",
            conditions={
                bread: [
                    (
                        partial(
                            object_on_top,
                            object=bread,
                            reference_object=PLATE,
                            require_gripper_detached=True,
                            gripper_name=PIPER_FINGER_CONTACTS,
                        ),
                        1.0,
                    ),
                ]
                for bread in BREADS
            },
            logical="all",
            score=0.5,
        ),
    ]
