from abc import ABC, abstractmethod
from geometry_msgs.msg import Pose
from rai.agents.conversational_agent import create_conversational_agent
from langchain_core.tools import BaseTool


class SceneConfig(ABC):
    """
    Setup of scene - arrangmenet of objects, interactions, environment etc.
    """
    def __init__(self) -> None:
        pass


class Interface(ABC):
    """
    Responsible for communication with simulation.
    """
    def __init__(self):
        pass

    @abstractmethod
    def load_scenario(self, scenario_config: SceneConfig) -> None:
        pass

    @abstractmethod
    def spawn_entity(self, prefab_name: str, name: str, pose: Pose) -> None:
        pass

    @abstractmethod
    def delete_entity(self, name: str) -> None:
        pass

    @abstractmethod
    def get_entity_pose(self, name: str) -> Pose:
        pass

    @abstractmethod
    def move_entity(self, name: str, target_pose: Pose) -> None:
        pass


class Scene(ABC):
    """
    Responsible for simualtion control.
    """
    def __init__(self, interface: Interface):
        self.interface = interface


    @abstractmethod
    def run(self):
        pass

    @abstractmethod
    def stop(self):
        pass

    def init_scene_config(self, scenario_config: SceneConfig):
        self.interface.load_scenario(scenario_config)


    def execute_task(self, llm_model: str, prompt: str, system_prompt: str):
        """
        Crete llm agent with the tools and system prompt
        """
        agent = create_conversational_agent(llm_model, self.get_tools(), system_prompt)
        agent.invoke({"messages": [prompt]})


    @abstractmethod
    def get_tools(self) -> list[BaseTool]:
        """
        Return list of tools used in the scene
        """
        pass


class Task(ABC):
    """"
    Specific task to perform with different scene setups.
    """
    def __init__(self, prompt: str, scene: Scene, scene_configs: list[SceneConfig], llm_model: str, system_prompt: str) -> None:
        self.system_prompt = system_prompt
        self.prompt = prompt
        self.scene = scene
        # TODO consider passing just one config
        self.scene_configs = scene_configs
        self.llm_model = llm_model


    def run(self):
        for scene_config in self.scene_configs:
            self.scene.init_scene_config(scene_config)
            self.scene.execute_task(self.llm_model, self.prompt, self.system_prompt)
            #TODO calculate progress
            #TODO calculate final result

    @abstractmethod
    def calculate_progress(self):
        """
        Calculate progress of the task
        """
        pass


    @abstractmethod
    def final_result(self):
        """
        Return final result of the task
        """
        pass


class Benchmark(ABC):
    """
    Set of tasks to be done.
    """
    def __init__(self, llm: str) -> None:
        self.llm = llm
        self.tasks: list[Task] = []
    def run_tasks(self):
        for task in self.tasks:
            task.run()

########### EXAMPLE USAGE ###########
    
class O3DEInterface(Interface):
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


class FiveRedCubesSceneConfig(SceneConfig):
    def __init__(self) -> None:
        pass

class FiveGreenCubesSceneConfig(SceneConfig):
    def __init__(self) -> None:
        pass


class ManipulationScene(Scene):
    def __init__(self, interface: Interface):
        super().__init__(interface)
    

class BuildTowerTask(Task):
    def __init__(self, llm_model: str) -> None:
        system_prompt = """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up)

        Before starting the task, make sure to grab the camera image to understand the environment.
        """
        prompt = "build tower from cubes"
        scene = ManipulationScene(O3DEInterface())
        scene_configs: list[SceneConfig] = [FiveRedCubesSceneConfig(), FiveGreenCubesSceneConfig()]
        super().__init__(prompt, scene, scene_configs, llm_model, system_prompt)


class ExampleBenchmark(Benchmark):
    def __init__(self, model: str) -> None:
        super().__init__(model)
        self.tasks = [BuildTowerTask(model)]

benchmark = ExampleBenchmark(model = "model")
benchmark.run_tasks()











