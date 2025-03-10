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
from typing import List, Tuple

from rai_bench.o3de_test_bench.tasks.manipulation_task import (  # type: ignore
    ManipulationTask,
)
from rai_sim.simulation_bridge import Entity, SimulationConfig  # type: ignore


class PlaceCubesTask(ManipulationTask):
    obj_types = ["red_cube", "blue_cube", "yellow_cube"]

    def get_prompt(self) -> str:
        return "Manipulate objects, so that all cubes are adjacent to at least one cube"

    def check_if_required_objects_present(
        self, simulation_config: SimulationConfig
    ) -> bool:
        """validate if at least 2 cubes are present"""
        cubes_num = 0
        for ent in simulation_config.entities:
            if ent.prefab_name in self.obj_types:
                cubes_num += 1
                if cubes_num > 1:
                    return True

        return False

    def calculate_correct(self, entities: List[Entity]) -> Tuple[int, int]:
        """Calculate how many objects are positioned correct and incorrect"""
        correct = sum(
            1
            for ent in entities
            if self.is_adjacent_to_any(
                ent.pose, [e.pose for e in entities if e != ent], 0.15
            )
        )
        incorrect: int = len(entities) - correct
        return correct, incorrect
