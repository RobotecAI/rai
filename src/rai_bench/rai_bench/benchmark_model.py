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
from typing import Any, Dict, Generic, List, Union, Set

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from rai.messages import HumanMultimodalMessage
from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_sim.simulation_bridge import (
    Pose,
    SimulationBridge,
    SimulationConfig,
    SimulationConfigT,
    SpawnedEntity,
)

loggers_type = Union[RcutilsLogger, logging.Logger]


class EntitiesMismatchException(Exception):
    pass


class Task(ABC):
    """
    Abstract of a Task. Provides utility functions for common calculations
    that can be helfull when creating metrics.
    Specific child classes should implement:
    - get_prompt method
    - validate_config
    - calculate_result
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
        """Returns the task instruction - the prompt that will be passed to agent"""
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
        Calculates result of the task, based on info retrieved from simulation.
        Should return score between 0.0 and 1.
        """
        pass

    def get_initial_and_current_positions(
        self,
        simulation_bridge: SimulationBridge[SimulationConfig],
        object_types: List[str],
    ):
        scene_state = simulation_bridge.get_scene_state()
        initial_objects = self.filter_entities_by_prefab_type(
            simulation_bridge.spawned_entities, prefab_types=object_types
        )
        final_objects = self.filter_entities_by_prefab_type(
            scene_state.entities, prefab_types=object_types
        )

        if len(initial_objects) != len(final_objects):
            raise EntitiesMismatchException(
                "Number of initially spawned entities does not match number of entities present at the end."
            )
        return initial_objects, final_objects

    def filter_entities_by_prefab_type(
        self, entities: List[SpawnedEntity], prefab_types: List[str]
    ) -> List[SpawnedEntity]:
        """Filter and return only these entities that match provided prefab types"""
        return [ent for ent in entities if ent.prefab_name in prefab_types]

    def euclidean_distance(self, pos1: Pose, pos2: Pose) -> float:
        """Calculate euclidean distance between 2 positions"""
        return (
            (pos1.translation.x - pos2.translation.x) ** 2
            + (pos1.translation.y - pos2.translation.y) ** 2
            + (pos1.translation.z - pos2.translation.z) ** 2
        ) ** 0.5

    def is_adjacent(self, pos1: Pose, pos2: Pose, threshold_distance: float):
        """
        Check if positions are adjacent to each other, the threshold_distance is a distance
        in simulation, refering to how close they have to be to classify them as adjacent
        """
        self.logger.debug(  # type: ignore
            f"Euclidean distance: {self.euclidean_distance(pos1, pos2)}, pos1: {pos1}, pos2: {pos2}"
        )
        return self.euclidean_distance(pos1, pos2) < threshold_distance

    def is_adjacent_to_any(
        self, pos1: Pose, positions: List[Pose], threshold_distance: float
    ) -> bool:
        """
        Check if given position is adjacent to any position in the given list.
        """

        return any(
            self.is_adjacent(pos1, pos2, threshold_distance) for pos2 in positions
        )

    def count_adjacent(self, positions: List[Pose], threshold_distance: float) -> int:
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

    def build_neighbourhood_list(
        self, entities: List[SpawnedEntity]
    ) -> Dict[SpawnedEntity, List[SpawnedEntity]]:
        """Assignes a list of neighbours to every object based on threshold distance"""
        neighbourhood_graph: Dict[SpawnedEntity, List[SpawnedEntity]] = {
            entity: [] for entity in entities
        }
        for entity in entities:
            neighbourhood_graph[entity] = [
                other
                for other in entities
                if entity != other and self.is_adjacent(entity.pose, other.pose, 0.15)
            ]
        return neighbourhood_graph

    def group_entities_by_type(
        self, entities: List[SpawnedEntity]
    ) -> Dict[str, List[SpawnedEntity]]:
        """Returns dictionary of entities grouped by type"""
        entities_by_type: Dict[str, List[SpawnedEntity]] = {}
        for entity in entities:
            entities_by_type.setdefault(entity.prefab_name, []).append(entity)
        return entities_by_type

    def check_neighbourhood_types(
        self,
        neighbourhood: List[SpawnedEntity],
        allowed_types: List[str],
    ) -> bool:
        """Check if ALL neighbours are given types"""
        return not neighbourhood or all(
            adj.prefab_name in allowed_types for adj in neighbourhood
        )

    def find_clusters(
        self, neighbourhood_list: Dict[SpawnedEntity, List[SpawnedEntity]]
    ) -> List[List[SpawnedEntity]]:
        """Find clusters of entities using DFS algorithm, lone entities are counted as a cluster"""
        visited: Set[SpawnedEntity] = set()
        clusters: List[List[SpawnedEntity]] = []

        def dfs(node: SpawnedEntity, cluster: List[SpawnedEntity]):
            visited.add(node)
            cluster.append(node)
            for neighbor in neighbourhood_list.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for node in neighbourhood_list.keys():
            if node not in visited:
                component: List[SpawnedEntity] = []
                dfs(node, component)
                clusters.append(component)

        return clusters


class Scenario(Generic[SimulationConfigT]):
    """
    A Scenarios are defined by a pair of Task and Simlation Config.
    Each Scenario is executed separatly by a Benchmark.
    """

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
    Benchmark represents a set of Scenarios to be executed and evaluated.
    It manages the execution, logs results, and provides functionality
    for tracking and exporting performance metrics.
    """

    def __init__(
        self,
        simulation_bridge: SimulationBridge[SimulationConfigT],
        scenarios: List[Scenario[SimulationConfigT]],
        logger: loggers_type | None = None,
        results_filename: str = "benchmark_results.csv",
    ) -> None:
        self.simulation_bridge = simulation_bridge
        self.num_of_scenarios = len(scenarios)
        self.scenarios = enumerate(iter(scenarios))
        self.results: List[Dict[str, Any]] = []
        self.results_filename = results_filename
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self.fieldnames = [
            "task",
            "simulation_config",
            "final_score",
            "total_time",
            "number_of_tool_calls",
        ]
        self._initialize_results_file()

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

    def _initialize_results_file(self):
        """Initialize the CSV file with headers."""
        with open(
            self.results_filename, mode="w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

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
            scenario_result: Dict[str, Any] = {
                "task": scenario.task.get_prompt(),
                "simulation_config": scenario.simulation_config_path,
                "final_score": result,
                "total_time": f"{total_time:.3f}",
                "number_of_tool_calls": tool_calls_num,
            }
            self.results.append(scenario_result)
            self._save_scenario_result_to_csv(scenario_result)

        except StopIteration:
            print("No more scenarios left to run.")

    def _save_scenario_result_to_csv(self, result: Dict[str, Any]) -> None:
        """Save a single scenario result to the CSV file."""
        with open(
            self.results_filename, mode="a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(result)

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results
