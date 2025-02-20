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
from rai_sim.simulation_bridge import Translation, Rotation
from rai_sim.o3de.o3de_bridge import (
    O3DEngineArmManipulationBridge,
    O3DExROS2SimulationConfig,
    SimulationBridge,
    SimulationConfig,
    PoseModel,
)

from pathlib import Path


class GrabCarrotTask(Task):
    def get_prompt(self) -> str:
        return "Manipulate objects, so that all carrots to the left side of the table (positive y)"

    def calculate_result(
        self, engine_connector: SimulationBridge, simulation_config: SimulationConfig
    ) -> float:
        corrected_objects = 0  # when the object which was in the incorrect place at the start, is in a correct place at the end
        misplaced_objects = 0  # when the object which was in the incorrect place at the start, is in a incorrect place at the end
        unchanged_correct = 0  # when the object which was in the correct place at the start, is in a correct place at the end
        displaced_objects = 0  # when the object which was in the correct place at the start, is in a incorrect place at the end

        scene_state = engine_connector.get_scene_state()

        initial_carrots = self.filter_entities_by_prefab_type(
            simulation_config.entities, prefab_types=["carrot"]
        )
        final_carrots = self.filter_entities_by_prefab_type(
            scene_state.entities, prefab_types=["carrot"]
        )
        num_initial_carrots = len(initial_carrots)

        if num_initial_carrots != len(final_carrots):
            raise EntitiesMismatchException(
                "Number of initially spawned entities does not match number of entities present at the end."
            )

        if num_initial_carrots == 0:
            self.logger.info("No objects to manipulate, returning 1.0")
            return 1.0
        else:

            for ini_carrot in initial_carrots:
                for final_carrot in final_carrots:
                    if ini_carrot.name == final_carrot.name:
                        initial_y = ini_carrot.pose.translation.y
                        final_y = final_carrot.pose.translation.y
                        # NOTE the specific coords that refer to for example
                        # middle of the table can differ across simulations,
                        # take that into consideration
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

                self.logger.info(
                    f"corrected_objects: {corrected_objects}, misplaced_objects: {misplaced_objects}, unchanged_correct: {unchanged_correct}, displaced_objects: {displaced_objects}"
                )
                return (corrected_objects + unchanged_correct) / num_initial_carrots


class PlaceCubesTask(Task):
    def get_prompt(self) -> str:
        return "Manipulate objects, so that  all cubes are next to each other"

    def calculate_result(
        self,
        engine_connector: SimulationBridge,
        simulation_config: SimulationConfig,
    ) -> float:
        corrected_objects = 0  # when the object which was in the incorrect place at the start, is in a correct place at the end
        misplaced_objects = 0  # when the object which was in the incorrect place at the start, is in a incorrect place at the end
        unchanged_correct = 0  # when the object which was in the correct place at the start, is in a correct place at the end
        displaced_objects = 0  # when the object which was in the correct place at the start, is in a incorrect place at the end

        cube_types = ["red_cube", "blue_cube", "yellow_cube"]
        scene_state = engine_connector.get_scene_state()

        initial_cubes = self.filter_entities_by_prefab_type(
            simulation_config.entities, prefab_types=cube_types
        )
        final_cubes = self.filter_entities_by_prefab_type(
            scene_state.entities, prefab_types=cube_types
        )
        num_of_objects = len(initial_cubes)

        if num_of_objects != len(final_cubes):
            raise EntitiesMismatchException(
                "Number of initially spawned entities does not match number of entities present at the end."
            )
        if num_of_objects == 0:
            self.logger.info("No objects to manipulate, returning score 1.0")
            return 1.0
        else:
            ini_poses = [cube.pose for cube in initial_cubes]
            final_poses = [cube.pose for cube in final_cubes]
            # NOTE the specific coords that refer to for example
            # middle of the table can differ across simulations,
            # take that into consideration
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

                self.logger.info(
                    f"corrected_objects: {corrected_objects}, misplaced_objects: {misplaced_objects}, unchanged_correct: {unchanged_correct}, displaced_objects: {displaced_objects}"
                )
                return (corrected_objects + unchanged_correct) / num_of_objects


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
    log_file = "src/rai_bench/o3de_test_bench/benchamrk.log"
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
    configs_dir = "src/rai_bench/o3de_test_bench/configs/"
    connector_path = configs_dir + "o3de_config.yaml"
    one_carrot_scene_config = O3DExROS2SimulationConfig.load_config(
        base_config_path=Path(configs_dir + "scene1.yaml"),
        connector_config_path=Path(connector_path),
    )
    multiple_carrot_scene_config = O3DExROS2SimulationConfig.load_config(
        base_config_path=Path(configs_dir + "scene2.yaml"),
        connector_config_path=Path(connector_path),
    )
    red_cubes_scene_config = O3DExROS2SimulationConfig.load_config(
        base_config_path=Path(configs_dir + "scene3.yaml"),
        connector_config_path=Path(connector_path),
    )
    multiple_cubes_scene_config = O3DExROS2SimulationConfig.load_config(
        base_config_path=Path(configs_dir + "scene4.yaml"),
        connector_config_path=Path(connector_path),
    )
    # combine different scene configs with the tasks to create various scenarios
    scenarios = [
        Scenario(
            task=GrabCarrotTask(logger=bench_logger),
            scene_config=one_carrot_scene_config,
        ),
        Scenario(
            task=GrabCarrotTask(logger=bench_logger),
            scene_config=multiple_carrot_scene_config,
        ),
        Scenario(
            task=GrabCarrotTask(logger=bench_logger),
            scene_config=red_cubes_scene_config,
        ),
        Scenario(
            task=PlaceCubesTask(logger=bench_logger),
            scene_config=red_cubes_scene_config,
        ),
        Scenario(
            task=PlaceCubesTask(logger=bench_logger),
            scene_config=multiple_cubes_scene_config,
        ),
    ]

    # custom request to arm
    base_arm_pose = PoseModel(translation=Translation(x=0.3, y=0.0, z=0.4))

    # define benchamrk
    benchmark = Benchmark(
        simulation_bridge=o3de,
        scenarios=scenarios,
        logger=bench_logger,
    )
    for i, s in enumerate(scenarios):
        agent = create_conversational_agent(
            llm, tools, system_prompt, logger=agent_logger
        )
        benchmark.run_next(agent=agent)
        o3de.move_arm(
            pose=base_arm_pose,
            initial_gripper_state=True,
            final_gripper_state=False,
            frame_id="panda_link0",
        )  # return to case position
        time.sleep(2)  # admire the end position for a second ;)

    connector.shutdown()
    o3de.shutdown()
    rclpy.shutdown()
