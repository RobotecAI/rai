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

from geometry_msgs.msg import Pose
from pydantic import BaseModel

from rai.agents.conversational_agent import create_conversational_agent


class Entity:
    # name: str
    # prefab_name: str
    # pose: Pose
    pass


class SceneConfig(BaseModel):
    """
    Setup of scene - arrangmenet of objects, interactions, environment etc.
    """

    entities: list[Entity]


class SceneSetup(ABC):
    """
    Info about entities in the scene (positions, collisions, etc.)
    """

    entities: list[Entity]


class EngineConnector(ABC):
    """
    Responsible for communication with simulation.
    """

    def __init__(self):
        pass

    @abstractmethod
    def setup_scene(self, scene_config: SceneConfig) -> SceneSetup:
        pass

    @abstractmethod
    def _spawn_entity(self, entity: Entity):
        pass

    @abstractmethod
    def despawn_entity(self, entity: Entity):
        pass

    @abstractmethod
    def get_object_position(self, object_name: str) -> Pose:
        pass


class O3DEEngineConnector(EngineConnector):
    def _spawn_entity(self, entity: Entity):
        # connector.service_call('spawn', entity)
        pass

    def _despawn_entity(self, entity: Entity):
        pass

    def setup_scene(self, scene_config: SceneConfig) -> SceneSetup:
        pass
        # 8 times despawn_entity
        # 10 times spawn_entity


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


class Scenario(BaseModel):
    task: Task
    scene_config: SceneConfig


class Benchmark:
    """
    Set of tasks to be done.
    """

    def __init__(self, agent, scenarios: list[Scenario]) -> None:
        self.engine_connector: EngineConnector
        self.tasks: list[Task] = []
        self.agent = agent
        self.scenarios = scenarios
        self.results = []

    def run(self):
        """
        Run benchmark
        """
        for scenario in self.scenarios:
            initial_scene_setup = self.engine_connector.setup_scene(
                scenario.scene_config
            )
            task = scenario.task
            self.agent.invoke({"messages": [task.get_prompt()]})
            result = task.calculate_progress(self.engine_connector, initial_scene_setup)
            self.results.append(result)


########### EXAMPLE USAGE ###########


class O3DEInterface(EngineConnector):
    def __init__(self) -> None:
        pass

    def spawn_entity(self, prefab_name: str, name: str, pose: Pose) -> None:
        """
        Spawns an entity in the simulation.

        Args:
            prefab_name: The name of the prefab to spawn.
            name: The name of the entity, by which it can be later referenced.
            pose: The pose of the entity.
        """


class O3DESceneConfig(SceneConfig):
    binary_path: str
    objects_positions: dict[str, Pose]


class BuildTowerTask(Task):
    def get_prompt(self) -> str:
        return "Build tower"

    def calculate_progress(self, engine_connector: EngineConnector) -> float:
        return engine_connector.get_object_position(
            "cube1"
        ) - engine_connector.get_object_position("cube2")


agent = create_conversational_agent(llm, tools, "You are Bob Budowniczy.")

scene_config = O3DESceneConfig(binary_path="/path/to/scene", objects_positions={})
task = BuildTowerTask()
scenarios = [Scenario(task=BuildTowerTask(), scene_config=scene_config)]

benchmark = Benchmark(agent, scenarios)
benchmark.run()
