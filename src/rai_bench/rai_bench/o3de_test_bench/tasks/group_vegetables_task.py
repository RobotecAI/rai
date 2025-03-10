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


class GroupVegetablesTask(ManipulationTask):
    obj_types = ["tomato", "apple", "corn", "carrot"]

    def get_prompt(self) -> str:
        return (
            "Manipulate objects so that vegetables form separate clusters based on their types. "
            "Each cluster must: "
            "1. Contain ALL vegetables of the same type "
            "2. Contain ONLY vegetables of the same type "
            "3. Form a single connected group "
            "4. Be completely separated from other clusters "
        )

    def check_if_required_objects_present(
        self, simulation_config: SimulationConfig
    ) -> bool:
        """Ensure that at least two types of vegetables are present."""
        veg_types = {
            ent.prefab_name
            for ent in simulation_config.entities
            if ent.prefab_name in self.obj_types
        }
        return len(veg_types) > 1

    def calculate_correct(self, entities: List[Entity]) -> Tuple[int, int]:
        """Count correctly and incorrectly placed objects based on clustering rules."""
        properly_clustered: List[Entity] = []
        misclustered: List[Entity] = []

        entities_by_type = self.group_entities_by_type(entities)

        for veg_type, veggies in entities_by_type.items():
            neighbourhood_list = self.build_neighbourhood_list(veggies)
            clusters = self.find_clusters(neighbourhood_list)
            if len(clusters) == 1:
                if all(
                    self.check_neighbourhood_types(
                        neighbourhood=neighbourhood_list[v], allowed_types=[veg_type]
                    )
                    for v in clusters[0]
                ):
                    properly_clustered.extend(clusters[0])
                else:
                    misclustered.extend(clusters[0])
            else:
                misclustered.extend(veggies)

        return len(properly_clustered), len(misclustered)
