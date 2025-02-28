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
    Task,
)
from rai_sim.o3de.o3de_bridge import SimulationBridge
from rai_sim.simulation_bridge import SimulationConfig, SimulationConfigT, SpawnedEntity


class PlaceCubesTask(Task):
    # TODO (jm) extract common logic to some parent manipulation task
    obj_types = ["red_cube", "blue_cube", "yellow_cube"]

    def get_prompt(self) -> str:
        return "Manipulate objects, so that all cubes are adjacent to at least one cube"

    def validate_config(self, simulation_config: SimulationConfig) -> bool:
        cubes_num = 0
        for ent in simulation_config.entities:
            if ent.prefab_name in self.obj_types:
                cubes_num += 1
                if cubes_num > 1:
                    return True

        return False

    def calculate_correct(self, entities: List[SpawnedEntity]) -> Tuple[int, int]:
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

    def calculate_initial_placements(
        self, simulation_bridge: SimulationBridge[SimulationConfigT]
    ) -> tuple[int, int]:
        """
        Calculates the number of objects that are correctly and incorrectly placed initially.
        """
        initial_cubes = self.filter_entities_by_prefab_type(
            simulation_bridge.spawned_entities, prefab_types=self.obj_types
        )
        initially_correct, initially_incorrect = self.calculate_correct(
            entities=initial_cubes
        )

        self.logger.info(  # type: ignore
            f"Initially correctly placed cubes: {initially_correct}, Initially incorrectly placed cubes: {initially_incorrect}"
        )
        return initially_correct, initially_incorrect

    def calculate_final_placements(
        self, simulation_bridge: SimulationBridge[SimulationConfigT]
    ) -> tuple[int, int]:
        """
        Calculates the number of objects that are correctly and incorrectly placed at the end of the simulation.
        """
        scene_state = simulation_bridge.get_scene_state()
        final_cubes = self.filter_entities_by_prefab_type(
            scene_state.entities, prefab_types=self.obj_types
        )
        final_correct, final_incorrect = self.calculate_correct(entities=final_cubes)

        self.logger.info(  # type: ignore
            f"Finally correctly placed cubes: {final_correct}, Finally incorrectly placed cubes: {final_incorrect}"
        )
        return final_correct, final_incorrect

    def calculate_result(
        self, simulation_bridge: SimulationBridge[SimulationConfigT]
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
            raise ValueError("All objects are placed correctly at the start.")
        else:
            corrected = final_correct - initially_correct
            score = max(0.0, corrected / initially_incorrect)

            self.logger.info(f"Calculated score: {score:.2f}")  # type: ignore
            return score
