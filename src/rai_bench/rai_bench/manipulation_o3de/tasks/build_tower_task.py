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


class BuildCubeTowerTask(ManipulationTask):
    ALLOWED_OBJECTS = {"red_cube", "blue_cube", "yellow_cube"}
    # fixed upper limit for allowable displacement for every object type
    # it should ensure that this displacement is not greater than half of the object size
    MAXIMUM_DISPLACEMENT = {"red_cube": 0.02, "blue_cube": 0.02, "yellow_cube": 0.02}

    def __init__(
        self,
        obj_types: List[str],
        allowable_displacement: float = 0.02,
        logger: loggers_type | None = None,
    ):
        """
        This task requires that cubes of the specified types are arranged into a single vertical tower.
        Only objects with types specified in `obj_types` (which must be a subset of the allowed objects)
        are considered. Cubes are grouped by their z-coordinate using a horizontal tolerance, and only
        groups with more than one cube are considered towers. The height of the tallest tower determines
        the number of correctly placed cubes.

        Parameters
        ----------
        obj_types : List[str]
            A list of cube types (e.g., ["red_cube", "blue_cube"]) to be used for building the tower.
            Each type must be one of the allowed objects: {"red_cube", "blue_cube", "yellow_cube"}.
        allowable_displacement : float, optional
            The allowable horizontal displacement (tolerance, in meters) used when grouping cubes by their
            z-coordinate. Default is 0.02.


        Raises
        ------
        TypeError
            If any of the provided object types is not allowed.
        """
        # NOTE (jmatejcz) what if allowable_displament is greater then the size of object?
        # we could check the z distance between entities
        # or trust user with this
        super().__init__(logger)
        if not set(obj_types).issubset(self.ALLOWED_OBJECTS):
            raise TypeError(
                f"Invalid obj_types provided: {obj_types}. Allowed objects: {self.ALLOWED_OBJECTS}"
            )
        for obj_type in obj_types:
            if allowable_displacement > self.MAXIMUM_DISPLACEMENT[obj_type]:
                raise ValueError(
                    f"allowable_displacement too large. For object type: {obj_type} maximum is {self.MAXIMUM_DISPLACEMENT[obj_type]}"
                )
        self.obj_types = obj_types
        self.allowable_displacement = allowable_displacement

    @property
    def task_prompt(self) -> str:
        cube_names = ", ".join(obj + "s" for obj in self.obj_types).replace("_", " ")
        return f"Manipulate objects so that all {cube_names} form a single vertical tower. Other types of objects cannot be included in a tower."

    def check_if_required_objects_present(self, simulation_config: SceneConfig) -> bool:
        """
        Validate that at least two cubes of the specified types are present.

        Returns
        -------
        bool
            True if at least two cubes of the allowed types are present, False otherwise.
        """
        cube_count = sum(
            1 for ent in simulation_config.entities if ent.prefab_name in self.obj_types
        )
        return cube_count > 1

    def calculate_correct(self, entities: List[Entity]) -> Tuple[int, int]:
        """
        Calculate the number of correctly and incorrectly placed cubes.

        This task does not consider a single cube as correctly placed.
        Cubes are grouped by their z-coordinate using a horizontal tolerance.
        The highest tower (the group with the most cubes) is considered correct,
        and all other cubes are counted as incorrect.

        Parameters
        ----------
        entities : List[Entity]
            List of all entities present in the simulation scene.

        Returns
        -------
        Tuple[int, int]
            A tuple where the first element is the number of correctly placed cubes (from the tallest tower)
            and the second element is the number of incorrectly placed cubes.
        """

        # Group entities by z-coordinate
        grouped_entities = self.group_entities_along_z_axis(
            entities, self.allowable_displacement
        )
        selected_type_objects = self.filter_entities_by_object_type(
            entities=entities, object_types=self.obj_types
        )

        correct = 0
        incorrect = 0
        for group in grouped_entities:
            if len(group) > 1:
                # we treat single standing cubes as incorrect
                if all(entity.prefab_name in self.obj_types for entity in group):
                    # highest tower is number of correctly placed objects
                    correct = max(correct, len(group))
        incorrect = len(selected_type_objects) - correct
        return correct, incorrect
