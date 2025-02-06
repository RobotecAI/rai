########### EXAMPLE USAGE ###########
import time
from rai.agents.conversational_agent import create_conversational_agent
from rai.tools.ros.manipulation import GetObjectPositionsTool, MoveToPointTool
import threading

from wandb import agent

from rai_hmi import base
import rclpy
import rclpy.qos
from rai.node import RaiBaseNode
from rai.tools.ros.native import GetCameraImage, Ros2GetTopicsNamesAndTypesTool
from rai.utils.model_initialization import get_llm_model
from rai.communication.ros2.connectors import ROS2ARIConnector
from benchmark_model import Scenario, Task, Benchmark
from rai_sim.engine_connector import (
    load_config,
)
from rai_sim.o3de.o3de_connector import O3DEngineConnector


class GrabCarrotTask(Task):
    def get_prompt(self) -> str:
        return "Move one carrot to the left side of the table"

    def calculate_progress(self, engine_connector, initial_scene_setup) -> float:
        # TODO
        return 1


class CollectCornsTask(Task):
    def get_prompt(self) -> str:
        return "Put red cubes next to each other"

    def calculate_progress(self, engine_connector, initial_scene_setup) -> float:
        # TODO
        return 1


def create_base_node():
    node = RaiBaseNode(node_name="manipulation_demo")
    node.declare_parameter("conversion_ratio", 1.0)
    return node


def create_tools(node):
    return [
        GetObjectPositionsTool(
            node=node,
            target_frame="panda_link0",
            source_frame="RGBDCamera5",
            camera_topic="/color_image5",
            depth_topic="/depth_image5",
            camera_info_topic="/color_camera_info5",
        ),
        MoveToPointTool(node=node, manipulator_frame="panda_link0"),
        GetCameraImage(node=node),
        Ros2GetTopicsNamesAndTypesTool(node=node),
    ]


if __name__ == "__main__":

    llm = get_llm_model(model_type="complex_model", streaming=True)

    system_prompt = """
    You are a robotic arm with interfaces to detect and manipulate objects.
    Here are the coordinates information:
    x - front to back (positive is forward)
    y - left to right (positive is right)
    z - up to down (positive is up)
    Before starting the task, make sure to grab the camera image to understand the environment.
    """

    # load different scenes
    one_carrot_scene_config = load_config("src/rai_bench/rai_bench/scene_config.yaml")
    multiple_carrot_scene_config = load_config(
        "src/rai_bench/rai_bench/scene2_config.yaml"
    )
    no_carrot_scene_config = load_config(
        "src/rai_bench/rai_bench/scene_no_carrot_config.yaml"
    )

    # combine different scene configs with the tasks to create various scenarios

    scenarios = [
        # Scenario(task=GrabCarrotTask(), scene_config=one_carrot_scene_config),
        Scenario(task=CollectCornsTask(), scene_config=one_carrot_scene_config),
        # Scenario(task=GrabCarrotTask(), scene_config=multiple_carrot_scene_config),
        # Scenario(task=GrabCarrotTask(), scene_config=no_carrot_scene_config),
        # Scenario(task=CollectCornsTask(), scene_config=multiple_carrot_scene_config),
        Scenario(task=CollectCornsTask(), scene_config=no_carrot_scene_config),
    ]

    # connector = ROS2ARIConnector()
    # o3de = O3DEngineConnector(connector)

    # benchmark = Benchmark(scenarios)
    # benchmark.engine_connector = o3de
    # node = create_base_node()
    # tools = create_tools(node)
    # base_node_thread = threading.Thread(target=node.spin)
    # base_node_thread.start()

    # for i in range(len(scenarios)):
    #     agent = create_conversational_agent(llm, tools, system_prompt)
    #     benchmark.run_next(agent=agent)

    # time.sleep(3)

    # connector.shutdown()
    # rclpy.shutdown()

    for s in scenarios:
        rclpy.init()
        node = create_base_node()
        base_node_thread = threading.Thread(target=rclpy.spin, args=(node,))
        base_node_thread.start()
        tools = create_tools(node)

        connector = ROS2ARIConnector()
        o3de = O3DEngineConnector(connector)

        benchmark = Benchmark([s])
        benchmark.engine_connector = o3de

        agent = create_conversational_agent(llm, tools, system_prompt)
        benchmark.run_next(agent=agent)

        connector.shutdown()
        rclpy.shutdown()
        base_node_thread.join()
