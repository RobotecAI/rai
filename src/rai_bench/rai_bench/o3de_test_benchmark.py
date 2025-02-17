########### EXAMPLE USAGE ###########
import rclpy
import time
import rclpy.qos
from benchmark_model import Benchmark, Scenario, Task
from rai_open_set_vision.tools import GetGrabbingPointTool

from rai.agents.conversational_agent import create_conversational_agent
from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.tools.ros.manipulation import GetObjectPositionsTool, MoveToPointTool
from rai.tools.ros2.topics import GetROS2ImageTool, GetROS2TopicsNamesAndTypesTool
from rai.utils.model_initialization import get_llm_model
from rai_sim.engine_connector import (
    load_config,
)
from rai_sim.o3de.o3de_connector import O3DEngineConnector
import rclpy
from rclpy.node import Node
from rai_interfaces.srv import ManipulatorMoveTo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from builtin_interfaces.msg import Time
from rai.utils.ros_async import get_future_result


class GrabCarrotTask(Task):
    def get_prompt(self) -> str:
        return "Move one carrot to the left side of the table"

    def calculate_progress(self, engine_connector, initial_scene_setup) -> float:
        # TODO
        return 1


class RedCubesTask(Task):
    def get_prompt(self) -> str:
        return "Put red cubes side by side"

    def calculate_progress(self, engine_connector, initial_scene_setup) -> float:
        # TODO
        return 1


def create_tools(connector):
    return [
        GetObjectPositionsTool(
            connector=connector,
            target_frame="panda_link0",
            source_frame="RGBDCamera5",
            camera_topic="/color_image5",
            depth_topic="/depth_image5",
            camera_info_topic="/color_camera_info5",
            get_grabbing_point_tool=GetGrabbingPointTool(connector=connector),
        ),
        MoveToPointTool(connector=connector, manipulator_frame="panda_link0"),
        GetROS2ImageTool(connector=connector),
        GetROS2TopicsNamesAndTypesTool(connector=connector),
    ]


class ManipulatorClient(Node):
    def __init__(self):
        super().__init__("manipulator_client")
        self.client = self.create_client(ManipulatorMoveTo, "/manipulator_move_to")

    def send_request(self):
        request = ManipulatorMoveTo.Request()
        request.initial_gripper_state = True
        request.final_gripper_state = False

        request.target_pose = PoseStamped()
        request.target_pose.header = Header()
        request.target_pose.header.stamp = Time(sec=0, nanosec=0)
        request.target_pose.header.frame_id = "panda_link0"

        request.target_pose.pose.position.x = 0.3
        request.target_pose.pose.position.y = 0.0
        request.target_pose.pose.position.z = 0.5

        request.target_pose.pose.orientation.x = 1.0
        request.target_pose.pose.orientation.y = 0.0
        request.target_pose.pose.orientation.z = 0.0
        request.target_pose.pose.orientation.w = 0.0

        while not self.client.wait_for_service(timeout_sec=5.0):
            self.get_logger().info("Service not available, waiting...")

        self.get_logger().info("Making request to manipulator...")

        future = self.client.call_async(request)
        result = get_future_result(future, timeout_sec=5.0)

        if result is not None:
            self.get_logger().info(f"Result: {result}")
        else:
            self.get_logger().error("Service call failed")


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
        Scenario(task=GrabCarrotTask(), scene_config=one_carrot_scene_config),
        Scenario(task=RedCubesTask(), scene_config=one_carrot_scene_config),
        # Scenario(task=GrabCarrotTask(), scene_config=multiple_carrot_scene_config),
        Scenario(task=GrabCarrotTask(), scene_config=no_carrot_scene_config),
        Scenario(task=RedCubesTask(), scene_config=multiple_carrot_scene_config),
        Scenario(task=RedCubesTask(), scene_config=no_carrot_scene_config),
    ]

    rclpy.init()
    connector = ROS2ARIConnector()
    node = connector.node
    node.declare_parameter("conversion_ratio", 1.0)

    tools = create_tools(connector)

    o3de = O3DEngineConnector(connector)
    client = ManipulatorClient()

    benchmark = Benchmark(scenarios)
    benchmark.engine_connector = o3de
    for i, s in enumerate(scenarios):
        agent = create_conversational_agent(llm, tools, system_prompt)
        benchmark.run_next(agent=agent)
        print(f"Scenario {i} done")

        client.send_request()

    time.sleep(30)
    connector.shutdown()
    o3de.shutdown()
    rclpy.shutdown()
