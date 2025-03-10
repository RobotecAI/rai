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
from rai_sim.simulation_bridge import (  # type: ignore
    SimulationConfig,
    SpawnedEntity,
)


class SeparateCarrotsTask(ManipulationTask):
    obj_types = ["carrot"]

    def get_prompt(self):
        return "Separate the carrots far away from the other objects of the table and put all the carrots next to each other."

    def validate_config(self, simulation_config: SimulationConfig) -> bool:
        for ent in simulation_config.entities:
            if ent.prefab_name in self.obj_types:
                return True

        return False

    def calculate_correct(self, entities: List[SpawnedEntity]) -> Tuple[int, int]:
        properly_clustered: List[SpawnedEntity] = []
        misclustered: List[SpawnedEntity] = []

        entities_by_type = self.group_entities_by_type(entities)

        neighbourhood_list = self.build_neighbourhood_list(entities_by_type["carrot"])
        clusters = self.find_clusters(neighbourhood_list)

        if len(clusters) == 1:
            if all(
                self.check_neighbourhood_types(
                    neighbourhood=neighbourhood_list[carrot],
                    allowed_types=["carrot"],
                )
                for carrot in clusters[0]
            ):
                properly_clustered.extend(clusters[0])
            else:
                misclustered.extend(clusters[0])
        else:
            misclustered.extend(entities_by_type["carrot"])

        return len(properly_clustered), len(misclustered)
