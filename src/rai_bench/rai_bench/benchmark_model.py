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

import csv
import logging
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rclpy.impl.rcutils_logger import RcutilsLogger

from rai.messages import HumanMultimodalMessage
from rai_sim.simulation_bridge import (
    PoseModel,
    SimulationBridge,
    SimulationConfig,
    SimulationConfigT,
    SpawnedEntity,
)

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
    def validate_config(self, simulation_config: SimulationConfig) -> bool:
        """Task should be able to verify if given config is suitable for specific task

        Args:
            simulation_config (SimulationConfig): initial scene setup
        Returns:
            bool: True is suitable, False otherwise
        """
        pass

    @abstractmethod
    def calculate_result(
        self, simulation_bridge: SimulationBridge[SimulationConfigT]
    ) -> float:
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
        self.logger.debug(  # type: ignore
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


class Scenario(Generic[SimulationConfigT]):
    """Single instances are run separatly by benchmark"""

    def __init__(
        self,
        task: Task,
        simulation_config: SimulationConfigT,
        simulation_config_path: str,
    ) -> None:
        if not task.validate_config(simulation_config):
            raise ValueError("This scene is invalid for this task.")
        self.task = task
        self.simulation_config = simulation_config
        # NOTE (jm) needed for logging which config was used,
        # there probably is better method to do it
        self.simulation_config_path = simulation_config_path


class Benchmark:
    """
    Defined by a set of scenarios to be done
    """

    def __init__(
        self,
        simulation_bridge: SimulationBridge[SimulationConfigT],
        scenarios: List[Scenario[SimulationConfigT]],
        logger: loggers_type | None = None,
    ) -> None:
        self.simulation_bridge = simulation_bridge
        self.num_of_scenarios = len(scenarios)
        self.scenarios = enumerate(iter(scenarios))
        self.results: List[Dict[str, Any]] = []
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

    @classmethod
    def create_scenarios(
        cls,
        tasks: List[Task],
        simulation_configs: List[SimulationConfigT],
        simulation_configs_paths: List[str],
    ) -> List[Scenario[SimulationConfigT]]:
        # TODO (jm) hacky_fix, taking paths as args here, not the best solution,
        # but more changes to code would be required
        scenarios: List[Scenario[SimulationConfigT]] = []
        for task in tasks:
            for sim_conf, sim_path in zip(simulation_configs, simulation_configs_paths):
                try:
                    scenarios.append(
                        Scenario(
                            task=task,
                            simulation_config=sim_conf,
                            simulation_config_path=sim_path,
                        )
                    )
                except ValueError as e:
                    print(
                        f"Could not create Scenario from task: {task.get_prompt()} and simulation_config: {sim_conf}, {e}"
                    )
        return scenarios

    def run_next(self, agent) -> None:
        """
        Runs the next scenario
        """
        try:
            i, scenario = next(self.scenarios)  # Get the next scenario

            self.simulation_bridge.setup_scene(scenario.simulation_config)
            self._logger.info(  # type: ignore
                "======================================================================================"
            )
            self._logger.info(  # type: ignore
                f"RUNNING SCENARIO NUMBER {i + 1} / {self.num_of_scenarios}, TASK: {scenario.task.get_prompt()}"
            )
            initial_result = scenario.task.calculate_result(self.simulation_bridge)
            self._logger.info(f"RESULT OF THE INITIAL SETUP: {initial_result}")  # type: ignore
            tool_calls_num = 0

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
                        self._logger.debug(f"{graph_node_name}: {last_msg}")  # type: ignore

                else:
                    raise ValueError(f"Unexpected type of message: {type(msg)}")

                if isinstance(msg, AIMessage):
                    # TODO (jm) figure out more robust way of counting tool calls
                    tool_calls_num += len(msg.tool_calls)

                self._logger.info(f"AI Message: {msg}")  # type: ignore

            te = time.perf_counter()

            result = scenario.task.calculate_result(self.simulation_bridge)
            total_time = te - ts
            self._logger.info(  # type: ignore
                f"TASK SCORE: {result}, TOTAL TIME: {total_time:.3f}, NUM_OF_TOOL_CALLS: {tool_calls_num}"
            )

            self.results.append(
                {
                    "task": scenario.task.get_prompt(),
                    "simulation_config": scenario.simulation_config_path,
                    "initial_score": initial_result,
                    "final_score": result,
                    "total_time": f"{total_time:.3f}",
                    "number_of_tool_calls": tool_calls_num,
                }
            )

        except StopIteration:
            print("No more scenarios left to run.")

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results

    def dump_results_to_csv(self, filename: str) -> None:
        if not self.results:
            self._logger.warning("No results to save.")  # type: ignore
            return

        fieldnames = [
            "task",
            "initial_score",
            "simulation_config",
            "final_score",
            "total_time",
            "number_of_tool_calls",
        ]

        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.results)

        self._logger.info(f"Results saved to {filename}")  # type: ignore
