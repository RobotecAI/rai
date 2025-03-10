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

from rai_bench.o3de_test_bench.tasks.manipulation_task import (
    ManipulationTask,  # type: ignore
)
from rai_sim.simulation_bridge import SimulationConfig, SpawnedEntity  # type: ignore


class GroupCubesByColorTask(ManipulationTask):
    obj_types = ["red_cube", "blue_cube", "yellow_cube"]

    def get_prompt(self):
        return "Group the cubes by color. Each group must contain all cubes of the same color."

    def validate_config(self, simulation_config: SimulationConfig):
        """
        Ensure that at least two cubes are present.
        """
        cubes = [
            entity
            for entity in simulation_config.entities
            if entity.prefab_name in self.obj_types
        ]
        return len(cubes) > 1

    def calculate_correct(self, entities: List[SpawnedEntity]) -> Tuple[int, int]:
        """Count correctly and incorrectly placed objects based on clustering rules."""
        properly_clustered: List[SpawnedEntity] = []
        misclustered: List[SpawnedEntity] = []

        for cube_type, cubes in self.group_entities_by_type(entities).items():
            neighbourhood_list = self.build_neighbourhood_list(cubes)
            clusters = self.find_clusters(neighbourhood_list)
            if len(clusters) == 1:
                if all(
                    self.check_neighbourhood_types(
                        neighbourhood=neighbourhood_list[cube],
                        allowed_types=[cube_type],
                    )
                    for cube in clusters[0]
                ):
                    properly_clustered.extend(clusters[0])
                else:
                    misclustered.extend(clusters[0])
            else:
                misclustered.extend(cubes)

        return len(properly_clustered), len(misclustered)
