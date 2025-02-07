import random
import time
from threading import Thread

from gazebo_msgs.srv import DeleteEntity, SpawnEntity
from geometry_msgs.msg import Point, Quaternion
from langchain_core.messages import HumanMessage
from rclpy.node import Node
from rclpy.task import Future
from scenarios.scenario_base import ScenarioBase
from tf2_ros import Buffer, TransformListener

from rai.agents.conversational_agent import create_conversational_agent
from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.node import RaiBaseNode
from rai.tools.ros2.topics import GetROS2ImageTool
from rai.tools.ros.manipulation import GetObjectPositionsTool, MoveToPointTool
from rai.tools.ros.native import Ros2GetTopicsNamesAndTypesTool
from rai.utils.model_initialization import get_llm_model
from rai_interfaces.srv import ManipulatorMoveTo


class ScenarioManager(Node):
    """
    A class responsible for playing the scenarios
    """

    def __init__(self, scenario_types, seeds=[]):
        """
        Initializes the ScenarioManager

        Args:
            scenario_types: A list of scenario classes to play
            seeds: A list of seeds to use for each scenario
        """
        super().__init__("scenario_manager")
        self.scenario_types = scenario_types
        self.seeds = seeds

        self.spawn_client = self.create_client(SpawnEntity, "/spawn_entity")
        self.delete_client = self.create_client(DeleteEntity, "/delete_entity")
        self.manipulator_client = self.create_client(
            ManipulatorMoveTo, "/manipulator_move_to"
        )
        self.tf2_buffer = Buffer()
        self.tf2_listener = TransformListener(self.tf2_buffer, self)

        while not self.spawn_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")
        while not self.delete_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("service not available, waiting again...")

        self.timer = self.create_timer(1.0, self.timer_callback)
        self.scenario: ScenarioBase = None
        self.current_scenario = 0
        self.agent_thread: Thread = None
        self.manipulator_ready = False
        self.scores = []

    def _init_scenario(self):
        self.scenario = self.scenario_types[self.current_scenario](
            self.spawn_client, self.delete_client, self.manipulator_client, self
        )
        self.manipulator_ready = False
        request = ManipulatorMoveTo.Request()
        request.target_pose.pose.orientation = Quaternion(
            x=0.923880, y=-0.382683, z=0.0, w=0.0
        )
        request.target_pose.pose.position = Point(x=0.2, y=0.0, z=0.2)
        if self.current_scenario < len(self.seeds):
            random.seed(self.seeds[self.current_scenario])
        else:
            random.seed(42)

        def callback(future: Future):
            self.manipulator_ready = True
            self.scenario.reset()

        self.scenario.manipulator_client.call_async(request).add_done_callback(callback)

    def _terminate_scenario(self):
        self.get_logger().info(f"Scenario terminated with score {self.scores[-1]}")
        self.scenario = None
        self.tf2_buffer = Buffer()
        self.tf2_listener = TransformListener(self.tf2_buffer, self)
        if self.current_scenario == len(self.scenario_types) - 1:
            self.get_logger().info(
                f"All scenarios are completed, with scores: {self.scores}"
            )
            request = ManipulatorMoveTo.Request()
            request.target_pose.pose.orientation = Quaternion(
                x=0.923880, y=-0.382683, z=0.0, w=0.0
            )
            request.target_pose.pose.position = Point(x=0.2, y=0.0, z=0.2)

            def callback(future: Future):
                self.manipulator_ready = True
                self.executor.shutdown()

            self.manipulator_client.call_async(request).add_done_callback(callback)
            self.timer.cancel()
            return
        self.current_scenario = (self.current_scenario + 1) % len(self.scenario_types)
        self.manipulator_ready = False

    def timer_callback(self):
        if self.scenario is None:
            self._init_scenario()

        if not self.manipulator_ready:
            return

        progress, terminated = self.scenario.step()
        if terminated and not (self.agent_thread and self.agent_thread.is_alive()):
            self.scores.append(progress)
            self._terminate_scenario()
            return
        if self.agent_thread and not self.agent_thread.is_alive():
            self.get_logger().info(
                "Agent failed to fulfill the task, terminating the scenario."
            )
            self.scores.append(progress)
            self._terminate_scenario()


class RaiBenchmarkManager(ScenarioManager):
    """
    A class responsible for playing the scenarios and running the conversational agent for each scenario
    """

    def __init__(self, scenario_types, seeds=[]):
        super().__init__(scenario_types, seeds)
        self.agent = None

    def _init_scenario(self):
        super()._init_scenario()

        self.rai_node = RaiBaseNode(node_name="manipulation_demo")
        self.rai_node.declare_parameter("conversion_ratio", 1.0)

        connector = ROS2ARIConnector()
        tools = [
            GetObjectPositionsTool(
                connector=connector,
                node=self.rai_node,
                target_frame="panda_link0",
                source_frame="RGBDCamera5",
                camera_topic="/color_image5",
                depth_topic="/depth_image5",
                camera_info_topic="/color_camera_info5",
            ),
            MoveToPointTool(node=self.rai_node, manipulator_frame="panda_link0"),
            GetROS2ImageTool(node=self.rai_node, connector=connector),
            Ros2GetTopicsNamesAndTypesTool(node=self.rai_node),
        ]

        llm = get_llm_model(model_type="complex_model")

        system_prompt = """
        You are a robotic arm with interfaces to detect and manipulate objects.
        Here are the coordinates information:
        x - front to back (positive is forward)
        y - left to right (positive is right)
        z - up to down (positive is up)

        Before starting the task, make sure to grab the camera image to understand the environment.
        """

        self.agent = create_conversational_agent(
            llm=llm,
            tools=tools,
            system_prompt=system_prompt,
        )

        def run_agent():
            time.sleep(1)
            self.agent.invoke(
                {"messages": [HumanMessage(content=self.scenario.get_prompt())]}
            )["messages"][-1].pretty_print()

        self.agent_thread = Thread(target=run_agent)
        self.agent_thread.start()
