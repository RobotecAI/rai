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

import time
import logging
from abc import ABC, abstractmethod
from typing import TypeVar, Union, List

from rclpy.impl.rcutils_logger import RcutilsLogger

from langchain_core.messages import BaseMessage, HumanMessage

from rai.messages import HumanMultimodalMessage
from rai_sim.simulation_bridge import (
    SimulationBridge,
    SimulationConfig,
    PoseModel,
    SpawnedEntity,
)


SimulationBridgeT = TypeVar("SimulationBridgeT", bound=SimulationBridge)
loggers_type = Union[RcutilsLogger, logging.Logger]


class EntitiesMismatchException(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)


class Task(ABC):
    """
    Task to perform.
    Specyfic implementation should implement a way to calculate results.
    Abstract provides utility functions for common calculations, that can be usefull when
    creating metrics
    """

    def __init__(
        self,
        logger: loggers_type | None = None,
    ) -> None:
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    @abstractmethod
    def get_prompt(self) -> str:
        pass

    @abstractmethod
    def validate_scene(self, simulation_config: SimulationConfig) -> bool:
        """Task should be able to verify if given scene is suitable for specific task
        for example: GrabCarrotTask should verify if there is any carrots in the scene

        Args:
            simulation_config (SimulationConfig): initial scene setup
        Returns:
            bool: True is suitable, False otherwise
        """
        pass

    @abstractmethod
    def calculate_result(self, simulation_bridge: SimulationBridge) -> float:
        """
        Calculate result of the task
        """
        pass

    def filter_entities_by_prefab_type(
        self, entities: List[SpawnedEntity], prefab_types: List[str]
    ) -> List[SpawnedEntity]:
        """Filter and return only these entities that match provided prefab types"""
        return [ent for ent in entities if ent.prefab_name in prefab_types]

    def euclidean_distance(self, pos1: PoseModel, pos2: PoseModel) -> float:
        """Calculate euclidean distance between 2 positions"""
        return (
            (pos1.translation.x - pos2.translation.x) ** 2
            + (pos1.translation.y - pos2.translation.y) ** 2
            + (pos1.translation.z - pos2.translation.z) ** 2
        ) ** 0.5

    def is_adjacent(self, pos1: PoseModel, pos2: PoseModel, threshold_distance: float):
        """
        Check if positions are adjacent to each other, the threshold_distance is a distance
        in simulation, refering to how close they have to be to classify them as adjacent
        """
        self.logger.debug(
            f"Euclidean distance: {self.euclidean_distance(pos1, pos2)}, pos1: {pos1}, pos2: {pos2}"
        )
        return self.euclidean_distance(pos1, pos2) < threshold_distance

    def is_adjacent_to_any(
        self, pos1: PoseModel, positions: List[PoseModel], threshold_distance: float
    ) -> bool:
        """
        Check if given position is adjacent to any position in the given list.
        """

        return any(
            self.is_adjacent(pos1, pos2, threshold_distance) for pos2 in positions
        )

    def count_adjacent(
        self, positions: List[PoseModel], threshold_distance: float
    ) -> int:
        """
        Count how many adjacent positions are in the given list.
        Note that position has to be adjacent to only 1 other position
        to be counted, not all of them
        """
        adjacent_count = 0

        for i, p1 in enumerate(positions):
            for j, p2 in enumerate(positions):
                if i != j:
                    if self.is_adjacent(p1, p2, threshold_distance):
                        adjacent_count += 1
                        break

        return adjacent_count


class Scenario:
    """Single instances are run separatly by benchmark"""

    def __init__(self, task: Task, simulation_config: SimulationConfig) -> None:
        if not task.validate_scene(simulation_config):
            raise ValueError("This scene is invalid for this task.")
        self.task = task
        self.simulation_config = simulation_config


class Benchmark:
    """
    Defined by a set of scenarios to be done
    """

    def __init__(
        self,
        simulation_bridge: SimulationBridge,
        scenarios: list[Scenario],
        logger: loggers_type | None = None,
    ) -> None:
        self.simulation_bridge = simulation_bridge
        self.scenarios = enumerate(iter(scenarios))
        self.results = []
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

    @classmethod
    def create_scenarios(
        cls, tasks: List[Task], simulation_configs: List[SimulationConfig]
    ):

        scenarios = []
        for task in tasks:
            for sim_conf in simulation_configs:
                try:
                    scenarios.append(Scenario(task=task, simulation_config=sim_conf))
                except ValueError as e:
                    print(
                        f"Could not create Scenario from task: {task.get_prompt()} and simulation_config: {sim_conf}, {e}"
                    )
        return scenarios

    def run_next(self, agent):
        """
        Runs the next scenario
        """
        try:
            i, scenario = next(self.scenarios)  # Get the next scenario

            self.simulation_bridge.setup_scene(scenario.simulation_config)
            self._logger.info(
                "======================================================================================"
            )
            self._logger.info(
                f"RUNNING SCENARIO NUMBER {i+1}, TASK: {scenario.task.get_prompt()}"
            )
            initial_result = scenario.task.calculate_result(self.simulation_bridge)
            self._logger.info(f"RESULT OF THE INITIAL SETUP: {initial_result}")
            ts = time.perf_counter()
            for state in agent.stream(
                {"messages": [HumanMessage(content=scenario.task.get_prompt())]}
            ):
                graph_node_name = list(state.keys())[0]
                msg = state[graph_node_name]["messages"][-1]

                if isinstance(msg, HumanMultimodalMessage):
                    last_msg = msg.text
                elif isinstance(msg, BaseMessage):
                    if isinstance(msg.content, list):
                        if len(msg.content) == 1:
                            if type(msg.content[0]) is dict:
                                last_msg = msg.content[0].get("text", "")
                    else:
                        last_msg = msg.content
                        self._logger.debug(f"{graph_node_name}: {last_msg}")
                else:
                    raise ValueError(f"Unexpected type of message: {type(msg)}")

                self._logger.info(f"AI Message: {msg}")

            te = time.perf_counter()

            result = scenario.task.calculate_result(self.simulation_bridge)

            total_time = te - ts
            self._logger.info(f"TASK SCORE: {result}, TOTAL TIME: {total_time:.3f}")

            self.results.append(
                {
                    "task": scenario.task.get_prompt(),
                    "initial_score": initial_result,
                    "final_score": result,
                    "total_time": f"{total_time:.3f}",
                    # TODO (jm) figure out how to get number of tool calls
                    "tool_calls": None,
                }
            )

        except StopIteration:
            print("No more scenarios left to run.")

    def get_results(self) -> list[dict]:
        return self.results
