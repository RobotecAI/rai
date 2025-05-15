# Copyright (C) 2025 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
from typing import List, Tuple, Union

from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_bench.manipulation_o3de.interfaces import (
    ManipulationTask,
)
from rai_sim.simulation_bridge import Entity, SceneConfig

loggers_type = Union[RcutilsLogger, logging.Logger]


class MoveObjectsToLeftTask(ManipulationTask):
    def __init__(self, obj_types: List[str], logger: loggers_type | None = None):
        """
        This task requires moving all objects of specified types to the left side of the table (positive y).

        Parameters
        ----------
        obj_types : List[str]
            A list of object types to be moved.
        """
        super().__init__(logger=logger)
        self.obj_types = obj_types

    @property
    def task_prompt(self) -> str:
        obj_names = ", ".join(obj + "s" for obj in self.obj_types).replace("_", " ")
        # NOTE (jmatejcz) specifing (positive y) might not be the best way to tell agent what to do,
        # but 'left side' is depending on where camera is positioned so it might not be enough
        return f"Manipulate objects, so that all of the {obj_names} are on the left side of the table (positive y)"

    def check_if_required_objects_present(self, simulation_config: SceneConfig) -> bool:
        """Validate if any object present"""
        object_types_present = self.group_entities_by_type(
            entities=simulation_config.entities
        )
        return set(self.obj_types) <= object_types_present.keys()

    def calculate_correct(self, entities: List[Entity]) -> Tuple[int, int]:
        """
        Calculate the number of objects correctly moved to the left side of the table.

        An object is considered correctly placed if its y-coordinate is positive.

        Parameters
        ----------
        entities : List[Entity]
            List of all entities present in the simulation scene.

        Returns
        -------
        Tuple[int, int]
            A tuple where the first element is the number of correctly placed objects (with positive y)
            and the second element is the number of incorrectly placed objects.
        """
        selected_type_objects = self.filter_entities_by_object_type(
            entities=entities, object_types=self.obj_types
        )
        correct = sum(
            1 for ent in selected_type_objects if ent.pose.pose.position.y > 0.0
        )
        incorrect: int = len(selected_type_objects) - correct
        return correct, incorrect
