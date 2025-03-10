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
from abc import ABC, abstractmethod
from typing import List, Tuple, Union

from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_bench.benchmark_model import (  # type: ignore
    EntitiesMismatchException,
    Task,  # type: ignore
)
from rai_sim.simulation_bridge import (  # type: ignore
    SimulationBridge,
    SimulationConfig,
    SimulationConfigT,
    SpawnedEntity,
)

loggers_type = Union[RcutilsLogger, logging.Logger]


class ManipulationTask(Task, ABC):
    obj_types: List[str] = []

    def __init__(self, logger: loggers_type | None = None):
        super().__init__(logger)
        self.initially_misplaced_now_correct = 0
        self.initially_misplaced_still_incorrect = 0
        self.initially_correct_still_correct = 0
        self.initially_correct_now_incorrect = 0

    def reset_values(self):
        self.initially_misplaced_now_correct = 0
        self.initially_misplaced_still_incorrect = 0
        self.initially_correct_still_correct = 0
        self.initially_correct_now_incorrect = 0

    @abstractmethod
    def calculate_correct(self, entities: List[SpawnedEntity]) -> Tuple[int, int]:
        """
        This method should implement calculation of how many objects
        are positioned correctly and incorrectly

        first int of the tuple must be number of correctly placed objects
        second int s- number of incorrectly placed objects
        """
        pass

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
