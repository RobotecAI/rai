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
from typing import Generic, TypeVar, Union, List
from rclpy.impl.rcutils_logger import RcutilsLogger

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from rai.messages import HumanMultimodalMessage
from rai_sim.simulation_bridge import (
    SimulationBridge,
    SimulationConfig,
    PoseModel,
    Entity,
    SpawnedEntity,
)


SimulationConnectorT = TypeVar("SimulationConnectorT", bound=SimulationBridge)
loggers_type = Union[RcutilsLogger, logging.Logger]


class Task(ABC, Generic[SimulationConnectorT]):
    """
    Task to perform.
    Specyfic implementation should implement a way to calculate results.
    Abstract provides utility functions for common calculations, that can be usefull when
    creating metrics
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        pass

    @abstractmethod
    def calculate_result(
        self,
        engine_connector: SimulationConnectorT,
        initial_scene_setup: SimulationConfig,
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
        return self.euclidean_distance(pos1, pos2) < threshold_distance

    def is_adjacent_to_any(
        self, pos1: PoseModel, positions: List[PoseModel], threshold_distance: float
    ) -> bool:
        """
        Check if pos1 is adjacent to any position in the given list.
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


class Scenario(BaseModel):
    """Single instances are run separatly by benchmark"""

    task: Task
    scene_config: SimulationConfig
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # pydantic does not support ABC classes


class Benchmark:
    """
    Defined by a set of scenarios to be done
    """

    def __init__(
        self,
        scenarios: list[Scenario],
        logger: loggers_type | None = None,
    ) -> None:
        self.engine_connector: SimulationBridge
        self.tasks: list[Task] = []
        self.scenarios = enumerate(iter(scenarios))
        self.results = []
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

    def run_next(self, agent):
        """
        Runs the next scenario
        """
        try:
            i, scenario = next(self.scenarios)  # Get the next scenario

            self.engine_connector.setup_scene(scenario.scene_config)
            self._logger.info(
                f"RUNNING SCENARIO NUMBER {i+1}, TASK: {scenario.task.get_prompt()}"
            )

            for state in agent.stream(
                {"messages": [HumanMessage(content=scenario.task.get_prompt())]}
            ):
                graph_node_name = list(state.keys())[0]
                msg = state[graph_node_name]["messages"][-1]

                if isinstance(msg, HumanMultimodalMessage):
                    last_msg = msg.text
                elif isinstance(msg, BaseMessage):
                    if isinstance(msg.content, list):
                        assert len(msg.content) == 1
                        last_msg = msg.content[0].get("text", "")
                    else:
                        last_msg = msg.content
                else:
                    raise ValueError(f"Unexpected type of message: {type(msg)}")

                self._logger.debug(f"{graph_node_name}: {last_msg}")
                self._logger.info(f"AI Message: {msg}")

            self.engine_connector.get_scene_state()
            result = scenario.task.calculate_result(
                self.engine_connector, scenario.scene_config
            )

            self._logger.info(f"TASK SCORE: {result}")

        except StopIteration:
            print("No more scenarios left to run.")
