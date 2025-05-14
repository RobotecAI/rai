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
import math
from abc import ABC, abstractmethod
from collections import defaultdict
from typing import Dict, List, Set, Tuple, TypeVar, Union

from rai.types import Pose
from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_sim.simulation_bridge import (
    Entity,
    SceneConfig,
    SimulationBridge,
    SimulationConfigT,
    SpawnedEntity,
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
    - validate_config
    - calculate_score
    """

    def __init__(
        self,
        logger: loggers_type | None = None,
    ) -> None:
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        pass

    @property
    @abstractmethod
    def task_prompt(self) -> str:
        """
        Returns the task instruction - the prompt that will be passed to agent
        """
        pass

    @abstractmethod
    def validate_config(self, simulation_config: SceneConfig) -> bool:
        """Task should be able to verify if given config is suitable for specific task

        Args:
            simulation_config (SimulationConfig): initial scene setup
        Returns:
            bool: True is suitable, False otherwise
        """
        pass

    @abstractmethod
    def calculate_score(
        self, simulation_bridge: SimulationBridge[SimulationConfigT]
    ) -> float:
        """
        Calculate the task score based on the simulation information.

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
            (pos1.position.x - pos2.position.x) ** 2
            + (pos1.position.y - pos2.position.y) ** 2
            + (pos1.position.z - pos2.position.z) ** 2
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
                and self.is_adjacent(
                    entity.pose.pose, other.pose.pose, threshold_distance
                )
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
        entities_by_type: Dict[str, List[EntityT]] = defaultdict(list)
        for entity in entities:
            entities_by_type[entity.prefab_name].append(entity)
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
        # NOTE (jmatejcz) figure out how to group by other coords and orientation, without reapeting code
        self,
        entities: List[EntityT],
        margin: float,
    ) -> List[List[EntityT]]:
        """
        Group entities that are aligned along the z axis based on their x and y coordinates.

        Entities are first sorted by their x and y coordinates. Then, each entity is added to an existing group
        if its (x, y) distance from the first entity in the group is within the specified margin.
        Otherwise, a new group is created.

        Example
        ----------
        You have 2 separate vertical towers of cubes.
        In that case method will return 2 groups of entities, one for each tower.

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
            entities,
            key=lambda ent: (ent.pose.pose.position.x, ent.pose.pose.position.y),
        )

        groups: List[List[EntityT]] = []
        for entity in entities:
            placed = False
            for group in groups:
                dx = group[0].pose.pose.position.x - entity.pose.pose.position.x
                dy = group[0].pose.pose.position.y - entity.pose.pose.position.y
                if math.sqrt(dx * dx + dy * dy) <= margin:
                    group.append(entity)
                    placed = True
                    break
            if not placed:
                groups.append([entity])
        return groups


class ManipulationTask(Task, ABC):
    """
    Common class for manipulaiton tasks
    obj_types variable represents object types that will be considered as the subject of the task.
    That means that based on positions of these objects simulation config will be evaluated
    and score will be calculated.

    Example
    -------
        MoveObjectsToLeftTask with 'carrot' as objects type, will check if carrtos are present
        and then calculated score based on how many carrots were moved to the left side
    """

    obj_types: List[str] = []

    @property
    def system_prompt(self) -> str:
        return """
    You are a robotic arm with interfaces to detect and manipulate objects.
    Here are the coordinates information:
    x - front to back (positive is forward)
    y - left to right (positive is right)
    z - up to down (positive is up)
    Before starting the task, make sure to grab the camera image to understand the environment.
    """

    @abstractmethod
    def check_if_required_objects_present(self, simulation_config: SceneConfig) -> bool:
        """
        Check if the required objects are present in the simulation configuration.

        Returns
        -------
        bool
            True if the required objects are present, False otherwise.
        """
        return True

    def check_if_any_placed_incorrectly(self, simulation_config: SceneConfig) -> bool:
        """
        Check if any object is placed incorrectly in the simulation configuration.
        Save number of initially correctly and incorrectly placed objects for
        future calculations

        Returns
        -------
        bool
            True if at least one object is placed incorrectly, False otherwise.
        """
        _, incorrect = self.calculate_correct(entities=simulation_config.entities)
        return incorrect > 0

    def validate_config(self, simulation_config: SceneConfig) -> bool:
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
    def calculate_correct(
        self, entities: List[Entity] | List[SpawnedEntity]
    ) -> Tuple[int, int]:
        """Method to calculate how many objects are placed correctly

        Parameters
        ----------
        entities : List[Entity]
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
        Get the current placements of objects in the simulation
        and calculated their current placements

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
        current_correct, current_incorrect = self.calculate_correct(
            entities=scene_state.entities
        )

        self.logger.info(  # type: ignore
            f"Currently correctly placed objects: {current_correct}, Currenlty incorrectly placed objects: {current_incorrect}"
        )
        return current_correct, current_incorrect

    def calculate_score(
        self, simulation_bridge: SimulationBridge[SceneConfig]
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
