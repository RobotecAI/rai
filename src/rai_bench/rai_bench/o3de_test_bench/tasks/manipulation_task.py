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
    EntityT,
    Task,
)
from rai_sim.simulation_bridge import (  # type: ignore
    SimulationBridge,
    SimulationConfig,
    SimulationConfigT,
)

loggers_type = Union[RcutilsLogger, logging.Logger]


class ManipulationTask(Task, ABC):
    # TODO (jm) is this clear, what obj_types is?
    """
    Common class for manipulaiton tasks
    obj_types are objects types that will be considered as the subject of the task.
    That means that based on these objects simulation config will be evaluated
    and score will be calculated.

    Example
    -------
        MoveObjectsToLeftTask with 'carrot' as objects type, will check if carrtos are present
        and then calculated score based on how many carrots were moved to the left side
    """

    obj_types: List[str] = []

    @abstractmethod
    def check_if_required_objects_present(
        self, simulation_config: SimulationConfig
    ) -> bool:
        """
        Check if the required objects are present in the simulation configuration.

        Returns
        -------
        bool
            True if the required objects are present, False otherwise.
        """
        return True

    def check_if_any_placed_incorrectly(
        self, simulation_config: SimulationConfig
    ) -> bool:
        """
        Check if any object is placed incorrectly in the simulation configuration.
        Save number of initially correctly and incorrectly placed objects for
        future calculations

        Returns
        -------
        bool
            True if at least one object is placed incorrectly, False otherwise.
        """
        initial_entities = self.filter_entities_by_prefab_type(
            simulation_config.entities, object_types=self.obj_types
        )
        correct, incorrect = self.calculate_correct(entities=initial_entities)
        self.initially_correct = correct
        self.initially_incorrect = incorrect
        self.logger.info(  # type: ignore
            f"Objects placed correctly in simulation config: {correct}, incorrectly: {incorrect}"
        )
        return incorrect > 0

    def validate_config(self, simulation_config: SimulationConfig) -> bool:
        """
        Validate the simulation configuration.

        Checks whether the required objects are present and if any of them is placed incorrectly.
        If these conditions are not met, the task should not be run with this configuration.

        Parameters
        ----------
        simulation_config : SimulationConfig
            The simulation configuration to validate.

        Returns
        -------
        bool
            True if the configuration is valid, False otherwise.
        """

        if self.check_if_required_objects_present(
            simulation_config=simulation_config
        ) and self.check_if_any_placed_incorrectly(simulation_config=simulation_config):
            return True
        else:
            return False

    @abstractmethod
    def calculate_correct(self, entities: List[EntityT]) -> Tuple[int, int]:
        """Method to calculate how many objects are placed correctly

        Parameters
        ----------
        entities : List[EntityT]
            list of ALL entities present in the simulaiton scene

        Returns
        -------
        Tuple[int, int]
            first int HAVE TO be number of correctly placed objects, second int - number of incorrectly placed objects
        """
        pass

    def calculate_current_placements(
        self, simulation_bridge: SimulationBridge[SimulationConfigT]
    ) -> tuple[int, int]:
        """
        Calculate the current placements of objects in the simulation.

        Filters the current scene entities by the allowed object types and determines
        the number of correctly and incorrectly placed objects at the end of the simulation.

        Parameters
        ----------
        simulation_bridge : SimulationBridge[SimulationConfigT]
            The simulation bridge containing the current scene state.

        Returns
        -------
        tuple[int, int]
            A tuple where the first element is the number of currently correctly placed objects
            and the second element is the number of currently incorrectly placed objects.
        """
        scene_state = simulation_bridge.get_scene_state()
        current_objects = self.filter_entities_by_prefab_type(
            scene_state.entities, object_types=self.obj_types
        )
        current_correct, current_incorrect = self.calculate_correct(
            entities=current_objects
        )

        self.logger.info(  # type: ignore
            f"Currently correctly placed objects: {current_correct}, Currenlty incorrectly placed objects: {current_incorrect}"
        )
        return current_correct, current_incorrect

    def calculate_result(
        self, simulation_bridge: SimulationBridge[SimulationConfig]
    ) -> float:
        """
        Calculate the task score based on the difference between initial and current placements.

        The score ranges from 0.0 to 1.0, where 0.0 indicates that the initial placements
        remain unchanged (or got worse), and 1.0 indicates perfect placements relative to the initial ones.
        The score is computed as the improvement in the number of correctly placed objects
        divided by the number of initially incorrectly placed objects.

        Parameters
        ----------
        simulation_bridge : SimulationBridge[SimulationConfig]
            The simulation bridge that provides access to the current scene state.

        Returns
        -------
        float
            The calculated score, ranging from 0.0 to 1.0.

        Raises
        ------
        EntitiesMismatchException
            If the total number of initial entities does not match the total number of current entities.
        """
        # TODO (jm) probably redundant as we chack number of incorrect when creating scenario
        initially_correct, initially_incorrect = self.calculate_correct(
            entities=simulation_bridge.spawned_entities
        )
        self.logger.info(  # type: ignore
            f"Objects placed correctly in simulation config: {initially_correct}, incorrectly: {initially_incorrect}"
        )
        current_correct, current_incorrect = self.calculate_current_placements(
            simulation_bridge
        )

        initial_objects_num = initially_correct + initially_incorrect
        current_objects_num = current_correct + current_incorrect
        if initial_objects_num == 0:
            return 1.0
        elif initial_objects_num != current_objects_num:
            raise EntitiesMismatchException(
                f"number of initial entities does not match current entities number, initially: {initially_correct + initially_incorrect}, current: {current_correct + current_incorrect}"
            )
        else:
            corrected = current_correct - initially_correct
            score = max(0.0, corrected / initially_incorrect)

            self.logger.info(f"Calculated score: {score:.2f}")  # type: ignore
            return score
