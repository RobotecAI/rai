########### EXAMPLE USAGE ###########
import rclpy
import logging
import time
import rclpy.qos

from rai_bench.benchmark_model import Benchmark, Scenario, Task
from rai_open_set_vision.tools import GetGrabbingPointTool

from rai.agents.conversational_agent import create_conversational_agent
from rai.communication.ros2.connectors import ROS2ARIConnector
from rai.tools.ros.manipulation import GetObjectPositionsTool, MoveToPointTool
from rai.tools.ros2.topics import GetROS2ImageTool, GetROS2TopicsNamesAndTypesTool
from rai.utils.model_initialization import get_llm_model
from rai_sim.o3de.o3de_bridge import (
    O3DEngineArmManipulationBridge,
    O3DExROS2SimulationConfig,
    SimulationConfig,
)

from rai_interfaces.srv import ManipulatorMoveTo
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Header
from pathlib import Path


class GrabCarrotTask(Task):
    def get_prompt(self) -> str:
        return "Move all carrots to the left side of the table (positive y)"

    def calculate_result(
        self,
        engine_connector: O3DEngineArmManipulationBridge,
        initial_scene_setup: SimulationConfig,
    ) -> float:
        result = 0.0
        corrected_objects = 0  # when the object which was in the incorrect place at the start, is in a correct place at the end
        misplaced_objects = 0  # when the object which was in the incorrect place at the start, is in a incorrect place at the end
        unchanged_correct = 0  # when the object which was in the correct place at the start, is in a correct place at the end
        displaced_objects = 0  # when the object which was in the correct place at the start, is in a incorrect place at the end

        scene_state = engine_connector.get_scene_state()
        entities = scene_state.entities

        print(entities)
        print(initial_scene_setup)
        for entity in entities:
            if entity.prefab_name == "carrot":  # check all carrots
                for ini_ent in initial_scene_setup.entities:
                    if (
                        entity.name in ini_ent.name
                    ):  # check for the entity in the initial setup
                        if (
                            ini_ent.pose.translation.y <= 0.0
                        ):  # only check carrots that have been on the right side of the table at the start
                            if entity.pose.translation.y >= 0.0:
                                corrected_objects += 1
                            else:
                                misplaced_objects += 1

                        else:  # also check if carrots that were already on the right side are still there
                            if entity.pose.translation.y >= 0.0:
                                unchanged_correct += 1
                            else:
                                displaced_objects += 1
        all_obj = (
            corrected_objects
            + misplaced_objects
            + unchanged_correct
            + displaced_objects
        )
        print(
            corrected_objects, misplaced_objects, unchanged_correct, displaced_objects
        )
        if all_obj == 0:
            print("No objects to manipulate, returning 1...")
            return 1
        else:
            result = corrected_objects + unchanged_correct / (
                corrected_objects
                + misplaced_objects
                + unchanged_correct
                + displaced_objects
            )
            print(
                f"corrected_objects: {corrected_objects}, misplaced_objects: {misplaced_objects}, unchanged_correct: {unchanged_correct}, displaced_objects: {displaced_objects}"
            )
            return result


class RedCubesTask(Task):
    def get_prompt(self) -> str:
        return "Put cubes next to each other"

    def calculate_result(
        self,
        engine_connector: O3DEngineArmManipulationBridge,
        initial_scene_setup: SimulationConfig,
    ) -> float:
        cube_types = ["red_cube", "blue_cube", "yellow_cube"]
        num_of_objects = sum(
            1
            for entity in initial_scene_setup.entities
            if entity.prefab_name in cube_types
        )
        scene_state = engine_connector.get_scene_state()
        entities = scene_state.entities

        initial_positions = [
            ent.pose
            for ent in initial_scene_setup.entities
            if ent.prefab_name in cube_types
        ]
        final_red_cubes = [
            ent.pose for ent in entities if ent.prefab_name in cube_types
        ]
        initial_adjacent = self.count_adjacent(initial_positions, 0.1)
        final_adjacent = self.count_adjacent(final_red_cubes, 0.1)

        required_moves = num_of_objects - initial_adjacent
        if required_moves > 0:
            return (final_adjacent - initial_adjacent) / (
                num_of_objects - initial_adjacent
            )
        else:
            # if there is no cube that is supposed to be moved, but was moved
            # score should punish
            return 1 - (initial_adjacent - final_adjacent)


def request_to_base_position() -> ManipulatorMoveTo.Request:
    request = ManipulatorMoveTo.Request()
    request.initial_gripper_state = True
    request.final_gripper_state = False

    request.target_pose = PoseStamped()
    request.target_pose.header = Header()
    request.target_pose.header.frame_id = "panda_link0"

    request.target_pose.pose.position.x = 0.3
    request.target_pose.pose.position.y = 0.0
    request.target_pose.pose.position.z = 0.5

    request.target_pose.pose.orientation.x = 1.0
    request.target_pose.pose.orientation.y = 0.0
    request.target_pose.pose.orientation.z = 0.0
    request.target_pose.pose.orientation.w = 0.0
    return request


if __name__ == "__main__":
    rclpy.init()
    connector = ROS2ARIConnector()
    node = connector.node
    node.declare_parameter("conversion_ratio", 1.0)

    o3de = O3DEngineArmManipulationBridge(connector)

    # define model
    llm = get_llm_model(model_type="complex_model", streaming=True)

    system_prompt = """
    You are a robotic arm with interfaces to detect and manipulate objects.
    Here are the coordinates information:
    x - front to back (positive is forward)
    y - left to right (positive is right)  
    z - up to down (positive is up)
    Before starting the task, make sure to grab the camera image to understand the environment.
    """
    # define tools
    tools = [
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

    # load different scenes
    one_carrot_scene_config = O3DExROS2SimulationConfig.load_config(
        base_config_path=Path("src/rai_bench/o3de_test_bench/scene1.yaml"),
        connector_config_path=Path("src/rai_bench/o3de_test_bench/o3de_config.yaml"),
    )
    multiple_carrot_scene_config = O3DExROS2SimulationConfig.load_config(
        base_config_path=Path("src/rai_bench/o3de_test_bench/scene2.yaml"),
        connector_config_path=Path("src/rai_bench/o3de_test_bench/o3de_config.yaml"),
    )
    red_cubes_scene_config = O3DExROS2SimulationConfig.load_config(
        base_config_path=Path("src/rai_bench/o3de_test_bench/scene3.yaml"),
        connector_config_path=Path("src/rai_bench/o3de_test_bench/o3de_config.yaml"),
    )
    # combine different scene configs with the tasks to create various scenarios
    scenarios = [
        Scenario(task=GrabCarrotTask(), scene_config=one_carrot_scene_config),
        Scenario(task=GrabCarrotTask(), scene_config=multiple_carrot_scene_config),
        Scenario(task=GrabCarrotTask(), scene_config=red_cubes_scene_config),
        Scenario(task=RedCubesTask(), scene_config=one_carrot_scene_config),
        Scenario(task=RedCubesTask(), scene_config=red_cubes_scene_config),
    ]
    # define logger
    log_file = "src/rai_bench/o3de_test_bench/benchamrk_agent.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.DEBUG)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    bench_logger = logging.getLogger("Benchmark logger")
    bench_logger.setLevel(logging.INFO)
    bench_logger.addHandler(file_handler)

    agent_logger = logging.getLogger("Agent logger")
    agent_logger.setLevel(logging.DEBUG)
    agent_logger.addHandler(file_handler)

    # custom request to arm
    request = request_to_base_position()

    # define benchamrk
    benchmark = Benchmark(scenarios, logger=bench_logger)
    benchmark.engine_connector = o3de
    for i, s in enumerate(scenarios):
        agent = create_conversational_agent(
            llm, tools, system_prompt, logger=agent_logger
        )
        benchmark.run_next(agent=agent)
        print(f"{i+1} Scenario done")
        o3de.move_arm(request=request)  # return to case position

    time.sleep(3)
    connector.shutdown()
    o3de.shutdown()
    rclpy.shutdown()
