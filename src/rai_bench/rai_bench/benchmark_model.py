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

from langchain_core.messages import BaseMessage, HumanMessage
from pydantic import BaseModel, ConfigDict

from rai.messages import HumanMultimodalMessage
from rai_sim.engine_connector import EngineConnector, SceneConfig, SceneSetup


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
                "========================================= RUNNING TASK ===================================="
            )
            print(f"RUNNING TASK: {task.get_prompt()}")
            print(
                "==========================================================================================="
            )
            for state in agent.stream(
                {"messages": [HumanMessage(content=task.get_prompt())]}
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

                print(f"{graph_node_name}: {last_msg}")

            result = task.calculate_progress(self.engine_connector, initial_scene_setup)
            self.results.append(result)
            msg.pretty_print()

        except StopIteration:
            print("No more scenarios left to run.")
