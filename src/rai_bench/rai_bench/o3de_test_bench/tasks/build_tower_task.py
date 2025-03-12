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

from rai_bench.o3de_test_bench.tasks.manipulation_task import (  # type: ignore
    ManipulationTask,
)
from rai_sim.simulation_bridge import Entity, SimulationConfig  # type: ignore

loggers_type = Union[RcutilsLogger, logging.Logger]


class BuildCubeTowerTask(ManipulationTask):
    ALLOWED_OBJECTS = {"red_cube", "blue_cube", "yellow_cube"}

    def __init__(self, obj_types: List[str], logger: loggers_type | None = None):
        """
        Initialize the BuildCubeTowerTask.

        Raises
        ------
        TypeError
            If any of the provided object types is not allowed.
        """
        super().__init__(logger)

        if not set(obj_types).issubset(self.ALLOWED_OBJECTS):
            raise TypeError(
                f"Invalid obj_types provided: {obj_types}. Allowed objects: {self.ALLOWED_OBJECTS}"
            )

        self.obj_types = obj_types

    def get_prompt(self) -> str:
        cube_names = ", ".join(obj + "s" for obj in self.obj_types).replace("_", " ")
        return f"Manipulate objects so that all {cube_names} form a single vertical tower. Other types of objects cannot be included in a tower."

    def check_if_required_objects_present(
        self, simulation_config: SimulationConfig
    ) -> bool:
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

    def calculate_correct(
        self, entities: List[Entity], allowable_displacment: float = 0.02
    ) -> Tuple[int, int]:
        """
        Calculate the number of correctly and incorrectly placed cubes.

        This task does not consider a single cube as correctly placed.
        Cubes are grouped by their z-coordinate using a horizontal tolerance.
        The highest tower (i.e., the group with the most cubes) is considered correct,
        and all other cubes are counted as incorrect.

        Parameters
        ----------
        entities : List[Entity]
            List of all entities present in the simulation scene.
        allowable_displacment : float, optional
            The allowable horizontal displacement (tolerance) used when grouping cubes by their
            z-coordinate. Default is 0.02.

        Returns
        -------
        Tuple[int, int]
            A tuple where the first element is the number of correctly placed cubes (from the tallest tower)
            and the second element is the number of incorrectly placed cubes.
        """

        # Group entities by z-coordinate
        grouped_entities = self.group_entities_along_z_axis(
            entities, allowable_displacment
        )

        correct = 0
        incorrect = 0
        for group in grouped_entities:
            if len(group) > 1:
                # we treat single standing cubes as incorrect
                if all(entity.prefab_name in self.obj_types for entity in group):
                    # highest tower is number of correctly placed objects
                    # TODO (jm) should we check z distance between entities?
                    correct = max(correct, len(group))
        incorrect = len(entities) - correct
        return correct, incorrect
