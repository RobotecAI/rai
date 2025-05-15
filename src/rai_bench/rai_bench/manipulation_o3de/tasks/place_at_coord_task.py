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
import math
from typing import List, Tuple, Union

from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_bench.manipulation_o3de.interfaces import (
    ManipulationTask,
)
from rai_sim.simulation_bridge import Entity, SceneConfig

loggers_type = Union[RcutilsLogger, logging.Logger]


class PlaceObjectAtCoordTask(ManipulationTask):
    def __init__(
        self,
        obj_type: str,
        target_position: Tuple[float, float],
        allowable_displacement: float = 0.02,
        logger: loggers_type | None = None,
    ):
        """
        This task requires placing one object of specified type into specified coords.

        Parameters
        ----------
        obj_type : str
            The type of object to be placed.
        target_position : Tuple[float, float]
            The target (x, y) coordinates (in meters) where one object of the specified type should be placed.
            The z coordinate is not enforced.
        allowable_displacement : float, optional
            The acceptable deviation (in meters) from the target (x, y) coordinates.
            Defaults to 0.02.
        """
        super().__init__(logger)
        self.obj_type = obj_type
        self.target_position = target_position
        self.allowable_displacement = allowable_displacement

    @property
    def task_prompt(self) -> str:
        x, y = self.target_position
        return (
            f"Manipulate one {self.obj_type.replace('_', ' ')} so that it is placed at "
            f"the coordinates (x: {x}, y: {y})."
        )

    def check_if_required_objects_present(self, simulation_config: SceneConfig) -> bool:
        count = sum(
            1 for ent in simulation_config.entities if ent.prefab_name == self.obj_type
        )
        return count >= 1

    def calculate_correct(self, entities: List[Entity]) -> Tuple[int, int]:
        """
        Calculate the number of correctly and incorrectly placed objects.

        This task is successful if exactly one object of the specified type is placed
        at the target (x, y) coordinates (within the allowable displacement). If more than one
        object exists, only one counts as correct or incorrect

        Parameters
        ----------
        entities : List[Entity]
            List of all entities present in the simulation scene.

        Returns
        -------
        Tuple[int, int]
            A tuple where the first element number of correctly placed objects, second number of incorrect
        """
        target_objects = [ent for ent in entities if ent.prefab_name == self.obj_type]
        correct = 0

        for ent in target_objects:
            dx = ent.pose.pose.position.x - self.target_position[0]
            dy = ent.pose.pose.position.y - self.target_position[1]
            distance = math.sqrt(dx**2 + dy**2)
            if distance <= self.allowable_displacement:
                correct = 1  # Only one correct placement is needed.
                break

        incorrect = 1 - correct
        return correct, incorrect
