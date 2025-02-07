from abc import ABC, abstractmethod
from geometry_msgs.msg import Pose
from rai.agents.conversational_agent import create_conversational_agent
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class SceneConfig(ABC):
    """
    Setup of scene - arrangmenet of objects, interactions, environment etc.
    """
    def __init__(self) -> None:
        pass


class Entity:
    # name: str
    # prefab_name: str
    # pose: Pose
    pass

class EngineConnector(ABC):
    """
    Responsible for communication with simulation.
    """
    def __init__(self):
        pass

    @abstractmethod
    def setup_scene(self, scene_config: SceneConfig) -> None:
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

    def setup_scene(self, scene_config: SceneConfig):
        pass
        # 8 times despawn_entity
        # 10 times spawn_entity

class Task(ABC):
    """"
    Specific task to perform with different scene setups.
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def get_prompt(self) -> str:
        pass

    @abstractmethod
    def calculate_progress(self, engine_connector: EngineConnector) -> float:
        """
        Calculate progress of the task
        """
        pass


class Scenario(BaseModel):
    task: Task
    scene_config: SceneConfig

class Benchmark():
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
            self.engine_connector.setup_scene(scenario.scene_config)
            task = scenario.task
            self.agent.invoke({"messages": [task.get_prompt()]})
            result = task.calculate_progress(self.engine_connector)
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
        pass

class O3DESceneConfig(SceneConfig):
    def __init__(self, binary_path: str, objects_positions: dict[str, Pose]) -> None:
        self.binary_path = binary_path
        self.objects_positions = objects_positions

class BuildTowerTask(Task):
    def get_prompt(self) -> str:
        return "Build tower"
    
    def calculate_progress(self, engine_connector: EngineConnector) -> float:
        return engine_connector.get_object_position("cube1") - engine_connector.get_object_position("cube2")


agent = create_conversational_agent(
    llm, tools, "You are Bob Budowniczy."
)

scene_config = O3DESceneConfig(binary_path="/path/to/scene", objects_positions={})
task = BuildTowerTask()
scenarios = [Scenario(task=BuildTowerTask(), scene_config=scene_config)]

benchmark = Benchmark(agent, scenarios)
benchmark.run()









