# Copyright (C) 2024 Robotec.AI
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

from abc import ABC, abstractmethod
from os import system

from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from rai_sim.engine_connector import EngineConnector, SceneConfig, SceneSetup
from rai.agents.conversational_agent import create_conversational_agent

from pydantic import ConfigDict


class Task(ABC):
    """ "
    Specific task to perform with different scene setups.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        pass

    @abstractmethod
    def calculate_progress(
        self, engine_connector: EngineConnector, initial_scene_setup: SceneSetup
    ) -> float:
        """
        Calculate progress of the task
        """
        pass


class Scenario(BaseModel):
    task: Task
    scene_config: SceneConfig
    model_config = ConfigDict(
        arbitrary_types_allowed=True
    )  # pydantic does not support ABC classes


class Benchmark:
    """
    Set of tasks to be done.
    """

    def __init__(self, scenarios: list[Scenario]) -> None:
        self.engine_connector: EngineConnector
        self.tasks: list[Task] = []
        self.scenarios = iter(scenarios)
        self.results = []

    def run_next(self, agent):
        """
        Runs the next scenario in the iterator manually.
        """
        try:
            scenario = next(self.scenarios)  # Get the next scenario

            initial_scene_setup = self.engine_connector.setup_scene(
                scenario.scene_config
            )
            task = scenario.task
            print(
                f"========================================= RUNNING TASK ===================================="
            )
            print(f"RUNNING TASK: {task.get_prompt()}")
            print(
                "==========================================================================================="
            )
            output = agent.invoke(
                {"messages": [HumanMessage(content=task.get_prompt())]}
            )
            result = task.calculate_progress(self.engine_connector, initial_scene_setup)
            self.results.append(result)
            output["messages"][-1].pretty_print()

        except StopIteration:
            print("No more scenarios left to run.")
