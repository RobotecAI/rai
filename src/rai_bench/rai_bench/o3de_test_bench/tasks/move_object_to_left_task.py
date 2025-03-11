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


class MoveObjectsToLeftTask(ManipulationTask):
    def __init__(self, obj_types: List[str], logger: loggers_type | None = None):
        super().__init__(logger=logger)
        self.obj_types = obj_types

    def get_prompt(self) -> str:
        obj_names = ", ".join(obj + "s" for obj in self.obj_types).replace("_", " ")
        return f"Manipulate objects, so that all of the following objects are on the left side of the table (positive y): {obj_names}."

    def check_if_required_objects_present(
        self, simulation_config: SimulationConfig
    ) -> bool:
        """Validate if any object present"""
        object_types_present = self.group_entities_by_type(
            entities=simulation_config.entities
        )
        return set(self.obj_types) <= object_types_present.keys()

    def calculate_correct(self, entities: List[Entity]) -> Tuple[int, int]:
        correct = sum(1 for ent in entities if ent.pose.translation.y > 0.0)
        incorrect: int = len(entities) - correct
        return correct, incorrect
