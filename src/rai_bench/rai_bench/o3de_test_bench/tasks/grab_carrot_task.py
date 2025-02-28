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

from rai_bench.benchmark_model import (
    EntitiesMismatchException,
)
from rai_bench.o3de_test_bench.tasks.manipulation_task import ManipulationTask
from rai_sim.o3de.o3de_bridge import (
    SimulationBridge,
)
from rai_sim.simulation_bridge import SimulationConfig, SpawnedEntity, SimulationConfigT


class GrabCarrotTask(ManipulationTask):
    obj_types = ["carrot"]

    def get_prompt(self) -> str:
        return "Manipulate objects, so that all carrots to the left side of the table (positive y)"

    def validate_config(self, simulation_config: SimulationConfig) -> bool:
        for ent in simulation_config.entities:
            if ent.prefab_name in self.obj_types:
                return True

        return False

    def calculate_correct(self, entities: List[SpawnedEntity]) -> Tuple[int, int]:
        """Calculate how many objects are positioned correct and incorrect"""
        correct = sum(1 for ent in entities if ent.pose.translation.y > 0.0)
        incorrect: int = len(entities) - correct
        return correct, incorrect

    def calculate_initial_placements(
        self, simulation_bridge: SimulationBridge[SimulationConfigT]
    ) -> tuple[int, int]:
        """
        Calculates the number of objects that are correctly and incorrectly placed initially.
        """
        initial_carrots = self.filter_entities_by_prefab_type(
            simulation_bridge.spawned_entities, prefab_types=self.obj_types
        )
        initially_correct, initially_incorrect = self.calculate_correct(
            entities=initial_carrots
        )

        self.logger.info(  # type: ignore
            f"Initially correctly placed carrots: {initially_correct}, Initially incorrectly placed carrots: {initially_incorrect}"
        )
        return initially_correct, initially_incorrect

    def calculate_final_placements(
        self, simulation_bridge: SimulationBridge[SimulationConfigT]
    ) -> tuple[int, int]:
        """
        Calculates the number of objects that are correctly and incorrectly placed at the end of the simulation.
        """
        scene_state = simulation_bridge.get_scene_state()
        final_carrots = self.filter_entities_by_prefab_type(
            scene_state.entities, prefab_types=self.obj_types
        )
        final_correct, final_incorrect = self.calculate_correct(entities=final_carrots)

        self.logger.info(  # type: ignore
            f"Finally correctly placed carrots: {final_correct}, Finally incorrectly placed carrots: {final_incorrect}"
        )
        return final_correct, final_incorrect

    def calculate_result(
        self, simulation_bridge: SimulationBridge[SimulationConfig]
    ) -> float:
        """
        Calculates a score from 0.0 to 1.0, where 0.0 represents the initial placements or worse and 1.0 represents perfect final placements.
        """
        initially_correct, initially_incorrect = self.calculate_initial_placements(
            simulation_bridge
        )
        final_correct, final_incorrect = self.calculate_final_placements(
            simulation_bridge
        )

        total_objects = initially_correct + initially_incorrect
        if total_objects == 0:
            return 1.0
        elif (initially_correct + initially_incorrect) != (
            final_correct + final_incorrect
        ):
            raise EntitiesMismatchException(
                "number of initial entities does not match final entities number."
            )
        elif initially_incorrect == 0:
            pass
            # NOTE all objects are placed correctly
            # no point in running task
            raise ValueError("All objects are placed correctly at the start.")
        else:
            corrected = final_correct - initially_correct
            score = max(0.0, corrected / initially_incorrect)

            self.logger.info(f"Calculated score: {score:.2f}")  # type: ignore
            return score
