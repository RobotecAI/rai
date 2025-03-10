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

from typing import List, Literal, Tuple, Union
import logging

from rai_bench.o3de_test_bench.tasks.manipulation_task import (  # type: ignore
    ManipulationTask,
)
from rai_sim.simulation_bridge import SimulationConfig, SpawnedEntity  # type: ignore
from rclpy.impl.rcutils_logger import RcutilsLogger

loggers_type = Union[RcutilsLogger, logging.Logger]


class BuildCubeTowerTask(ManipulationTask):
    obj_types = ["red_cube", "blue_cube", "yellow_cube"]

    def get_prompt(self) -> str:
        return (
            "Manipulate objects so that all cubes form a single vertical tower. "
            "Other types of objects cannot be included in a tower."
        )

    def validate_config(self, simulation_config: SimulationConfig) -> bool:
        """Validate that at least two cubes are present."""
        cubes_num = 0
        for ent in simulation_config.entities:
            if ent.prefab_name in self.obj_types:
                cubes_num += 1
                if cubes_num > 1:
                    return True

        return False

    def calculate_correct(self, entities: List[SpawnedEntity]) -> Tuple[int, int]:
        # TODO (jm) not sure how to treat single standing cube, as correctly or incorrectly placed?
        # for now when cubes are separated, one of them is treated as correctly placed
        # assuming it's the foundation of the tower, the rest of them as incorrect
        """
        This task does not consider single cube as correctly placed,
        only cubes placed on other cube are counted as correctly placed
        """

        tolerance_xy = 0.05  # Allowable horizontal displacement
        margin_z = 0.1  # Allowable height difference to be in the same group

        # Group entities by z-coordinate
        grouped_entities = self.group_entities_by_z_coordinate(entities, margin_z)

        correct = 0
        incorrect = 0
        tower_consists_only_of_valid_types = True

        for group in grouped_entities:
            previous_entity = None
            for entity in group:
                if entity.prefab_name not in self.obj_types:
                    tower_consists_only_of_valid_types = False
                    incorrect += 1
                    continue  # Skip non-valid objects

                if previous_entity is None:
                    correct += 1  # The first valid entity in the group is correct
                else:
                    # Check if the object is aligned with the previous one
                    if (
                        abs(
                            previous_entity.pose.translation.x
                            - entity.pose.translation.x
                        )
                        <= tolerance_xy
                        and abs(
                            previous_entity.pose.translation.y
                            - entity.pose.translation.y
                        )
                        <= tolerance_xy
                    ):
                        correct += 1  # Object is correctly stacked
                    else:
                        incorrect += 1  # Misaligned object

                previous_entity = entity

        # If any non-allowed objects were found in the tower, mark everything incorrect
        if not tower_consists_only_of_valid_types:
            correct = 0
            incorrect = len(entities)

        return correct, incorrect
