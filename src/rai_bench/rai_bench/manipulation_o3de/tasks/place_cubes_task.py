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


class PlaceCubesTask(ManipulationTask):
    obj_types = ["red_cube", "blue_cube", "yellow_cube"]

    def __init__(
        self,
        threshold_distance: float = 0.15,
        logger: loggers_type | None = None,
    ):
        """
        This task requires that evry cube is placed to at least one other cube.

        Parameters
        ----------
        threshold_distance : float, optional
            The distance threshold (in meters) used to determine if two cubes are adjacent. If the Euclidean
            distance between two cubes is less than or equal to this value, they are considered adjacent.
            Defaults to 0.15.
        """
        super().__init__(logger)
        self.threshold_distance = threshold_distance

    @property
    def task_prompt(self) -> str:
        return "Manipulate objects, so that all cubes are adjacent to at least one cube"

    def check_if_required_objects_present(self, simulation_config: SceneConfig) -> bool:
        """
        Returns
        -------
        bool
            True if at least two cubes are present; otherwise, False.
        """
        cubes_num = 0
        for ent in simulation_config.entities:
            if ent.prefab_name in self.obj_types:
                cubes_num += 1
                if cubes_num > 1:
                    return True

        return False

    def calculate_correct(self, entities: List[Entity]) -> Tuple[int, int]:
        """
        Calculate the number of correctly and incorrectly placed cubes based on adjacency.

        An object is considered correctly placed if it is adjacent to at least one other cube
        within the given threshold distance.

        Parameters
        ----------
        entities : List[Entity]
            List of all entities (cubes) present in the simulation scene.

        Returns
        -------
        Tuple[int, int]
            A tuple where the first element is the number of correctly placed cubes (i.e., cubes that
            are adjacent to at least one other cube) and the second element is the number of
            incorrectly placed cubes.
        """
        correct = sum(
            1
            for ent in entities
            if self.is_adjacent_to_any(
                ent.pose.pose,
                [e.pose.pose for e in entities if e != ent],
                self.threshold_distance,
            )
        )
        incorrect: int = len(entities) - correct
        return correct, incorrect
