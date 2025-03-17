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
import math
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Generic, List, Set, TypeVar, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph  # type: ignore
from rai.messages import HumanMultimodalMessage  # type: ignore
from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_sim.simulation_bridge import (  # type: ignore
    Entity,
    Pose,
    SimulationBridge,
    SimulationConfig,
    SimulationConfigT,
)

loggers_type = Union[RcutilsLogger, logging.Logger]
EntityT = TypeVar("EntityT", bound=Entity)


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
        """
        Validate whether the provided simulation configuration is suitable for this task.

        Returns
        -------
        bool
            True if the configuration is suitable, False otherwise.
        """
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
        Calculate the task result (score) based on the simulation information.

        Parameters
        ----------
        simulation_bridge : SimulationBridge[SimulationConfigT]
            The simulation bridge used to retrieve simulation data.

        Returns
        -------
        float
            A score between 0.0 and 1.0.
        """
        pass

    def filter_entities_by_object_type(
        self, entities: List[EntityT], object_types: List[str]
    ) -> List[EntityT]:
        """
        Filter and return only the entities that match the provided prefab types.

        Parameters
        ----------
        entities : List[EntityT]
            The list of entities to filter.
        object_types : List[str]
            The allowed object types.

        Returns
        -------
        List[EntityT]
            A list of entities whose prefab_name is in object_types.
        """
        return [ent for ent in entities if ent.prefab_name in object_types]

    def euclidean_distance(self, pos1: Pose, pos2: Pose) -> float:
        """Calculate euclidean distance between 2 positions"""
        return (
            (pos1.translation.x - pos2.translation.x) ** 2
            + (pos1.translation.y - pos2.translation.y) ** 2
            + (pos1.translation.z - pos2.translation.z) ** 2
        ) ** 0.5

    def is_adjacent(self, pos1: Pose, pos2: Pose, threshold_distance: float):
        """
        Check if two positions are adjacent, based on a threshold distance.

        Parameters
        ----------
        pos1 : Pose
            The first position.
        pos2 : Pose
            The second position.
        threshold_distance : float
            The maximum allowed distance for the positions to be considered adjacent.

        Returns
        -------
        bool
            True if the Euclidean distance between pos1 and pos2 is less than threshold_distance, False otherwise.
        """
        return self.euclidean_distance(pos1, pos2) < threshold_distance

    def is_adjacent_to_any(
        self, pos1: Pose, positions: List[Pose], threshold_distance: float
    ) -> bool:
        """
        Check if a position is adjacent to any position in a given list.

        Parameters
        ----------
        pos1 : Pose
            The position to check.
        positions : List[Pose]
            A list of positions to compare against.
        threshold_distance : float
            The distance threshold for adjacency.

        Returns
        -------
        bool
            True if pos1 is adjacent to any position in positions, False otherwise.
        """

        return any(
            self.is_adjacent(pos1, pos2, threshold_distance) for pos2 in positions
        )

    def count_adjacent(self, positions: List[Pose], threshold_distance: float) -> int:
        """
        Count how many positions in the list are adjacent to at least one other position.

        Parameters
        ----------
        positions : List[Pose]
            A list of positions.
        threshold_distance : float
            The distance threshold to determine adjacency.

        Returns
        -------
        int
            The count of positions that are adjacent to at least one other position.
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
        self, entities: List[EntityT], threshold_distance: float = 0.15
    ) -> Dict[EntityT, List[EntityT]]:
        """
        Build a neighbourhood list assigning a list of neighbours to every entity based on a threshold distance.

        Parameters
        ----------
        entities : List[EntityT]
            The list of entities.
        threshold_distance : float, optional
            The maximum distance between entities to consider them neighbours. Default is 0.15.

        Returns
        -------
        Dict[EntityT, List[EntityT]]
            A dictionary mapping each entity to a list of neighbouring entities.
        """
        neighbourhood_graph: Dict[EntityT, List[EntityT]] = {
            entity: [] for entity in entities
        }
        for entity in entities:
            neighbourhood_graph[entity] = [
                other
                for other in entities
                if entity != other
                and self.is_adjacent(entity.pose, other.pose, threshold_distance)
            ]
        return neighbourhood_graph

    def group_entities_by_type(
        self, entities: List[EntityT]
    ) -> Dict[str, List[EntityT]]:
        """
        Group entities by their prefab type.

        Parameters
        ----------
        entities : List[EntityT]
            The list of entities to group.

        Returns
        -------
        Dict[str, List[EntityT]]
            A dictionary with keys as prefab names and values as lists of entities of that type.
        """
        entities_by_type: Dict[str, List[EntityT]] = {}
        for entity in entities:
            entities_by_type.setdefault(entity.prefab_name, []).append(entity)
        return entities_by_type

    def check_neighbourhood_types(
        self,
        neighbourhood: List[EntityT],
        allowed_types: List[str],
    ) -> bool:
        """
        Check if all entities in the neighbourhood are of the allowed types.

        Parameters
        ----------
        neighbourhood : List[EntityT]
            The list of neighbouring entities.
        allowed_types : List[str]
            The allowed prefab types.

        Returns
        -------
        bool
            True if the neighbourhood is empty or if all neighbours have a prefab_name in allowed_types, False otherwise.
        """
        return not neighbourhood or all(
            adj.prefab_name in allowed_types for adj in neighbourhood
        )

    def find_clusters(
        self, neighbourhood_list: Dict[EntityT, List[EntityT]]
    ) -> List[List[EntityT]]:
        """
        Identify clusters of entities using a DFS algorithm.

        Each connected component in the neighbourhood graph is considered a cluster.
        Lone entities are counted as their own cluster.

        Parameters
        ----------
        neighbourhood_list : Dict[EntityT, List[EntityT]]
            A dictionary mapping entities to their list of neighbours.

        Returns
        -------
        List[List[EntityT]]
            A list of clusters, where each cluster is a list of connected entities.
        """
        visited: Set[EntityT] = set()
        clusters: List[List[EntityT]] = []

        def dfs(node: EntityT, cluster: List[EntityT]):
            visited.add(node)
            cluster.append(node)
            for neighbor in neighbourhood_list.get(node, []):
                if neighbor not in visited:
                    dfs(neighbor, cluster)

        for node in neighbourhood_list.keys():
            if node not in visited:
                component: List[EntityT] = []
                dfs(node, component)
                clusters.append(component)

        return clusters

    def group_entities_along_z_axis(
        # TODO (jm) figure out how to group by other coords and orientation, without reapeting code
        self,
        entities: List[EntityT],
        margin: float,
    ) -> List[List[EntityT]]:
        """
        Group entities that are aligned along the z axis based on their x and y coordinates.

        Entities are first sorted by their x and y coordinates. Then, each entity is added to an existing group
        if its (x, y) distance from the first entity in the group is within the specified margin.
        Otherwise, a new group is created.

        Parameters
        ----------
        entities : List[EntityT]
            The list of entities to group.
        margin : float
            The maximum allowable Euclidean distance in the x-y plane to consider entities as part of the same group.

        Returns
        -------
        List[List[EntityT]]
            A list of groups (clusters) of entities.
        """

        entities = sorted(
            entities, key=lambda ent: (ent.pose.translation.x, ent.pose.translation.y)
        )

        groups: List[List[EntityT]] = []
        for entity in entities:
            placed = False
            for group in groups:
                dx = group[0].pose.translation.x - entity.pose.translation.x
                dy = group[0].pose.translation.y - entity.pose.translation.y
                if math.sqrt(dx * dx + dy * dy) <= margin:
                    group.append(entity)
                    placed = True
                    break
            if not placed:
                groups.append([entity])
        return groups


class Scenario(Generic[SimulationConfigT]):
    """
    A Scenario are defined by a pair of Task and Simlation Config.
    Each Scenario is executed separatly by a Benchmark.
    """

    def __init__(
        self,
        task: Task,
        simulation_config: SimulationConfigT,
        simulation_config_path: str,
    ) -> None:
        """
        Initialize a Scenario.

        Parameters
        ----------
        task : Task
            The task to be executed.
        simulation_config : SimulationConfigT
            The simulation configuration for the scenario.
        simulation_config_path : str
            The file path to the simulation configuration.

        Raises
        ------
        ValueError
            If the provided simulation configuration is not valid for the task.
        """
        if not task.validate_config(simulation_config):
            raise ValueError("This scene is invalid for this task.")
        self.task = task
        self.simulation_config = simulation_config
        # NOTE (jm) needed for logging which config was used,
        # there probably is better way to do it
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
        """
        Create scenarios by pairing each task with each suitable simulation configuration.

        Parameters
        ----------
        tasks : List[Task]
            The list of tasks.
        simulation_configs : List[SimulationConfigT]
            The list of simulation configurations.
        simulation_configs_paths : List[str]
            The corresponding file paths for the simulation configurations.

        Returns
        -------
        List[Scenario[SimulationConfigT]]
            A list of scenarios generated from the given tasks and simulation configurations.
        """
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

    def run_next(self, agent: CompiledStateGraph) -> None:
        """
        Run the next scenario in the benchmark.

        Parameters
        ----------
        agent : CompiledStateGraph
            The agent used to execute the scenario.

        This method sets up the scene, streams the agent's responses, logs messages,
        counts tool calls, calculates the final task score, and writes the result to a CSV file.
        """
        try:
            i, scenario = next(self.scenarios)  # Get the next scenario

            self.simulation_bridge.setup_scene(scenario.simulation_config)
            self._logger.info(  # type: ignore
                "======================================================================================"
            )
            self._logger.info(  # type: ignore
                f"RUNNING SCENARIO NUMBER {i + 1} / {self.num_of_scenarios}\n TASK: {scenario.task.get_prompt()}\n SIMULATION_CONFIG: {scenario.simulation_config_path}"
            )
            tool_calls_num = 0

            ts = time.perf_counter()
            for state in agent.stream(
                {"messages": [HumanMessage(content=scenario.task.get_prompt())]},
                {"recursion_limit": 100},  # TODO (jm) what should be recursion limit?
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
            try:
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
            except EntitiesMismatchException as e:
                self._logger.error(e)  # type:ignore

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
