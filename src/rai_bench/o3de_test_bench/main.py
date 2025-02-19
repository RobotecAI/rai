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

########### EXAMPLE USAGE ###########
import rclpy
import logging
import time
import rclpy.qos

from rai_bench.benchmark_model import (
    Benchmark,
    Scenario,
    Task,
    EntitiesMismatchException,
)
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
    ) -> float:
        corrected_objects = 0  # when the object which was in the incorrect place at the start, is in a correct place at the end
        misplaced_objects = 0  # when the object which was in the incorrect place at the start, is in a incorrect place at the end
        unchanged_correct = 0  # when the object which was in the correct place at the start, is in a correct place at the end
        displaced_objects = 0  # when the object which was in the correct place at the start, is in a incorrect place at the end

        scene_state = engine_connector.get_scene_state()

        initial_carrots = self.filter_entities_by_prefab_type(
            engine_connector.spawned_entities, prefab_types=["carrot"]
        )
        final_carrots = self.filter_entities_by_prefab_type(
            scene_state.entities, prefab_types=["carrot"]
        )
        num_of_objects = len(initial_carrots)

        if num_of_objects != len(final_carrots):
            raise EntitiesMismatchException(
                f"Number of initially spawned entities does not match number of entities present at the end."
            )

        for ini_carrot in initial_carrots:
            for final_carrot in final_carrots:
                if ini_carrot.name == final_carrot.name:
                    initial_y = ini_carrot.pose.translation.y
                    final_y = final_carrot.pose.translation.y
                    if (
                        initial_y <= 0.0
                    ):  # Carrot started in the incorrect place (right side)
                        if final_y >= 0.0:
                            corrected_objects += 1  # Moved to correct side
                        else:
                            misplaced_objects += 1  # Stayed on incorrect side
                    else:  # Carrot started in the correct place (left side)
                        if final_y >= 0.0:
                            unchanged_correct += 1  # Stayed on correct side
                        else:
                            displaced_objects += (
                                1  # Moved incorrectly to the wrong side
                            )
                    break
            else:
                raise EntitiesMismatchException(
                    f"Entity with name: {ini_carrot.name} which was present in initial scene, not found in final scene."
                )

        if num_of_objects == 0:
            self.logger.info("No objects to manipulate, returning 1.0")
            return 1.0
        else:
            self.logger.info(
                f"corrected_objects: {corrected_objects}, misplaced_objects: {misplaced_objects}, unchanged_correct: {unchanged_correct}, displaced_objects: {displaced_objects}"
            )
            return (corrected_objects + unchanged_correct) / num_of_objects


class RedCubesTask(Task):
    def get_prompt(self) -> str:
        return "Put all cubes next to each other"

    def calculate_result(
        self,
        engine_connector: O3DEngineArmManipulationBridge,
    ) -> float:
        corrected_objects = 0  # when the object which was in the incorrect place at the start, is in a correct place at the end
        misplaced_objects = 0  # when the object which was in the incorrect place at the start, is in a incorrect place at the end
        unchanged_correct = 0  # when the object which was in the correct place at the start, is in a correct place at the end
        displaced_objects = 0  # when the object which was in the correct place at the start, is in a incorrect place at the end

        cube_types = ["red_cube", "blue_cube", "yellow_cube"]
        scene_state = engine_connector.get_scene_state()

        initial_cubes = self.filter_entities_by_prefab_type(
            engine_connector.spawned_entities, prefab_types=cube_types
        )
        final_cubes = self.filter_entities_by_prefab_type(
            scene_state.entities, prefab_types=cube_types
        )
        num_of_objects = len(initial_cubes)

        if num_of_objects != len(final_cubes):
            raise EntitiesMismatchException(
                f"Number of initially spawned entities does not match number of entities present at the end."
            )

        ini_poses = [cube.pose for cube in initial_cubes]
        final_poses = [cube.pose for cube in final_cubes]

        for ini_cube in initial_cubes:
            for final_cube in final_cubes:
                if ini_cube.name == final_cube.name:
                    was_adjacent_initially = self.is_adjacent_to_any(
                        ini_cube.pose, ini_poses, 0.1
                    )
                    is_adjacent_finally = self.is_adjacent_to_any(
                        final_cube.pose, final_poses, 0.1
                    )
                    if not was_adjacent_initially and is_adjacent_finally:
                        corrected_objects += 1
                    elif not was_adjacent_initially and not is_adjacent_finally:
                        misplaced_objects += 1
                    elif was_adjacent_initially and is_adjacent_finally:
                        unchanged_correct += 1
                    elif was_adjacent_initially and not is_adjacent_finally:
                        displaced_objects += 1

                    break
            else:
                raise EntitiesMismatchException(
                    f"Entity with name: {ini_cube.name} which was present in initial scene, not found in final scene."
                )

        if num_of_objects == 0:
            self.logger.info("No objects to manipulate, returning score 1.0")
            return 1
        else:
            self.logger.info(
                f"corrected_objects: {corrected_objects}, misplaced_objects: {misplaced_objects}, unchanged_correct: {unchanged_correct}, displaced_objects: {displaced_objects}"
            )
            return (corrected_objects + unchanged_correct) / num_of_objects


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
    # define loggers
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
    agent_logger.setLevel(logging.INFO)
    agent_logger.addHandler(file_handler)

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
    multiple_cubes_scene_config = O3DExROS2SimulationConfig.load_config(
        base_config_path=Path("src/rai_bench/o3de_test_bench/scene4.yaml"),
        connector_config_path=Path("src/rai_bench/o3de_test_bench/o3de_config.yaml"),
    )
    # combine different scene configs with the tasks to create various scenarios
    scenarios = [
        # Scenario(task=GrabCarrotTask(), scene_config=one_carrot_scene_config),
        # Scenario(task=GrabCarrotTask(), scene_config=multiple_carrot_scene_config),
        # Scenario(task=GrabCarrotTask(), scene_config=red_cubes_scene_config),
        # # Scenario(task=RedCubesTask(), scene_config=one_carrot_scene_config),
        Scenario(
            task=RedCubesTask(logger=bench_logger), scene_config=red_cubes_scene_config
        ),
        Scenario(
            task=RedCubesTask(logger=bench_logger),
            scene_config=multiple_cubes_scene_config,
        ),
    ]

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
