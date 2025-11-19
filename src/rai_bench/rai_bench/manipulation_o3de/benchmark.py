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
import logging
import statistics
import time
import uuid
from pathlib import Path
from typing import List, TypeVar

import rclpy
from langchain.tools import BaseTool
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from rai.agents.langchain.core import (
    create_conversational_agent,
)
from rai.communication.ros2.connectors import ROS2Connector
from rai.messages import HumanMultimodalMessage
from rai.tools.ros2 import (
    GetObjectPositionsTool,
    GetROS2ImageTool,
    GetROS2TopicsNamesAndTypesTool,
    MoveToPointTool,
)
from rai_perception.tools import GetGrabbingPointTool

from rai_bench.base_benchmark import BaseBenchmark, RunSummary, TimeoutException
from rai_bench.manipulation_o3de.interfaces import Task
from rai_bench.manipulation_o3de.results_tracking import (
    ScenarioResult,
)
from rai_bench.results_processing.langfuse_scores_tracing import ScoreTracingHandler
from rai_bench.utils import get_llm_model_name
from rai_sim.o3de.o3de_bridge import (
    O3DEngineArmManipulationBridge,
    O3DExROS2SimulationConfig,
)
from rai_sim.simulation_bridge import (
    Entity,
    SceneConfig,
)

EntityT = TypeVar("EntityT", bound=Entity)


class EntitiesMismatchException(Exception):
    pass


class Scenario:
    """
    A Scenario are defined by a pair of Task and Simlation Config.
    Each Scenario is executed separatly by a Benchmark.
    """

    def __init__(
        self,
        task: Task,
        scene_config: SceneConfig,
        scene_config_path: str,
        level: str | None = None,
    ) -> None:
        """
        Initialize a Scenario.

        Parameters
        ----------
        task : Task
            The task to be executed.
        scene_config : SceneConfig
            The scene configuration for the scenario.
        scene_config_path : str
            The file path to the scene configuration.
        level : str
            The difficulty level of this scenario

        Raises
        ------
        ValueError
            If the provided scene configuration is not valid for the task.
        """
        self.task = task
        self.scene_config = scene_config
        # NOTE (jmatejcz) needed for logging which config was used,
        # there probably is better way to do it
        self.scene_config_path = scene_config_path
        if not level:
            self.level = "not_declared"
        else:
            self.level = level


class ManipulationO3DEBenchmark(BaseBenchmark):
    """
    ManipulationO3DEBenchmark represents a set of Scenarios to be executed and evaluated.
    It manages the execution, logs results, and provides functionality
    for tracking and exporting performance metrics.
    """

    def __init__(
        self,
        model_name: str,
        simulation_bridge: O3DEngineArmManipulationBridge,
        simulation_config: O3DExROS2SimulationConfig,
        scenarios: List[Scenario],
        results_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            results_dir=results_dir,
            logger=logger,
        )
        self.simulation_bridge = simulation_bridge
        self.simulation_bridge.init_simulation(simulation_config=simulation_config)
        self.simulation_bridge.launch_robotic_stack(
            required_robotic_ros2_interfaces=simulation_config.required_robotic_ros2_interfaces,
            launch_description=self.launch_description(),
        )
        self.num_of_scenarios = len(scenarios)
        self.scenarios = enumerate(iter(scenarios))

        self.scenario_results: List[ScenarioResult] = []
        self.score_tracing_handler = ScoreTracingHandler()
        self.csv_initialize(self.results_filename, ScenarioResult)

    def launch_description(self):
        launch_moveit = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [
                    "src/examples/rai-manipulation-demo/Project/Examples/panda_moveit_config_demo.launch.py",
                ]
            )
        )

        launch_robotic_manipulation = Node(
            package="robotic_manipulation",
            executable="robotic_manipulation",
            output="screen",
            parameters=[
                {"use_sim_time": True},
            ],
        )

        launch_openset = IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [
                    FindPackageShare("rai_bringup"),
                    "/launch/openset.launch.py",
                ]
            ),
        )

        return LaunchDescription(
            [
                launch_openset,
                launch_moveit,
                launch_robotic_manipulation,
            ]
        )

    @classmethod
    def create_scenarios(
        cls,
        tasks: List[Task],
        scene_configs: List[SceneConfig],
        scene_configs_paths: List[str],
        logger: logging.Logger | None = None,
        level: str | None = None,
    ) -> List[Scenario]:
        """
        Create scenarios by pairing each task with each suitable simulation configuration.

        Parameters
        ----------
        tasks : List[Task]
            The list of tasks.
        simulation_configs : List[SimulationConfigT]
            The list of simulation configurations.
        simulation_configs_paths : List[str]
            The corresponding file paths for the simulation configurations.

        Returns
        -------
        List[Scenario[SimulationConfigT]]
            A list of scenarios generated from the given tasks and simulation configurations.
        """
        # HACK (jmatejcz) taking paths as args here, not the best solution,
        # but more changes to code would be required
        scenarios: List[Scenario] = []
        if not logger:
            logger = logging.getLogger(__name__)
        for task in tasks:
            for scene_conf, scene_path in zip(scene_configs, scene_configs_paths):
                if task.validate_config(simulation_config=scene_conf):
                    scenarios.append(
                        Scenario(
                            task=task,
                            scene_config=scene_conf,
                            scene_config_path=scene_path,
                            level=level,
                        )
                    )
                else:
                    logger.debug(
                        f"Simulation config: {scene_path} is not suitable for task: {task.task_prompt}"
                    )
        return scenarios

    def run_next(self, agent: CompiledStateGraph, experiment_id: uuid.UUID) -> None:
        """
        Run the next scenario in the benchmark.

        Parameters
        ----------
        agent : CompiledStateGraph
            The agent used to execute the scenario.

        This method sets up the scene, streams the agent's responses, logs messages,
        counts tool calls, calculates the final task score, and writes the result to a CSV file.
        """
        try:
            i, scenario = next(self.scenarios)  # Get the next scenario
            try:
                with self.time_limit(30):
                    # NOTE (jmatejcz) sometimes spawning objects freezes
                    self.simulation_bridge.setup_scene(scenario.scene_config)
            except TimeoutException as e:
                self.logger.error(msg=f"Setup scene timeout: {e}")
                return
            self.logger.info(
                "======================================================================================"
            )
            self.logger.info(
                f"RUNNING SCENARIO NUMBER {i + 1} / {self.num_of_scenarios}\n TASK: {scenario.task.task_prompt}\n SIMULATION_CONFIG: {scenario.scene_config_path}"
            )
            callbacks = self.score_tracing_handler.get_callbacks()
            run_id = uuid.uuid4()
            config: RunnableConfig = {
                "run_id": run_id,
                "callbacks": callbacks,
                "tags": [
                    f"experiment-id:{experiment_id}",
                    "benchmark:manipulation-o3de",
                    self.model_name,
                    f"scenario-difficulty:{scenario.level}",
                ],
                "recursion_limit": 50,
            }
            tool_calls_num = 0

            ts = time.perf_counter()
            prev_count: int = 0
            try:
                with self.time_limit(210):
                    for state in agent.stream(
                        {"messages": [HumanMessage(content=scenario.task.task_prompt)]},
                        config=config,
                    ):
                        node = next(iter(state))
                        new_messages = state[node]["messages"][prev_count:]
                        prev_count = len(state[node]["messages"])

                        for msg in new_messages:
                            if isinstance(msg, HumanMultimodalMessage):
                                last_msg = msg.text()
                            elif isinstance(msg, BaseMessage):
                                if isinstance(msg.content, list):
                                    if len(msg.content) == 1:
                                        if type(msg.content[0]) is dict:
                                            last_msg = msg.content[0].get("text", "")
                                else:
                                    last_msg = msg.content
                                    self.logger.debug(f"{node}: {last_msg}")

                            else:
                                raise ValueError(
                                    f"Unexpected type of message: {type(msg)}"
                                )

                            if isinstance(msg, AIMessage):
                                tool_calls_num += len(msg.tool_calls)

                            self.logger.info(f"AI Message: {msg}")
            except TimeoutException as e:
                self.logger.error(msg=f"Task timeout: {e}")
            except GraphRecursionError as e:
                self.logger.error(msg=f"Reached recursion limit {e}")

            te = time.perf_counter()
            try:
                score = scenario.task.calculate_score(self.simulation_bridge)
                total_time = te - ts
                self.logger.info(
                    f"TASK SCORE: {score}, TOTAL TIME: {total_time:.3f}, NUM_OF_TOOL_CALLS: {tool_calls_num}"
                )

                scenario_result = ScenarioResult(
                    task_prompt=scenario.task.task_prompt,
                    system_prompt=scenario.task.system_prompt,
                    scene_config_path=scenario.scene_config_path,
                    model_name=self.model_name,
                    score=score,
                    level=scenario.level,
                    total_time=total_time,
                    number_of_tool_calls=tool_calls_num,
                )
                self.scenario_results.append(scenario_result)
                self.csv_writerow(self.results_filename, scenario_result)
                # computing after every iteration in case of early stopping
                self.compute_and_save_summary()

                for callback in callbacks:
                    self.score_tracing_handler.send_score(
                        callback=callback,
                        run_id=run_id,
                        score=score,
                    )

            except EntitiesMismatchException as e:
                self.logger.error(e)

        except StopIteration:
            print("No more scenarios left to run.")

    def compute_and_save_summary(self) -> None:
        """Compute summary statistics and save them to the summary file."""
        self.logger.info("Computing and saving average results...")

        success_count = sum(1 for r in self.scenario_results if r.score == 1.0)
        success_rate = (
            success_count / len(self.scenario_results) * 100
            if self.scenario_results
            else 0
        )
        avg_time = (
            statistics.mean(r.total_time for r in self.scenario_results)
            if self.scenario_results
            else 0
        )

        summary = RunSummary(
            model_name=self.model_name,
            success_rate=round(success_rate, 2),
            avg_time=round(avg_time, 3),
            total_tasks=len(self.scenario_results),
        )
        self.csv_initialize(self.summary_filename, RunSummary)
        self.csv_writerow(self.summary_filename, summary)


def _setup_benchmark_environment(
    o3de_config_path: str,
    model_name: str,
    scenarios: List[Scenario],
    out_dir: Path,
    bench_logger: logging.Logger,
):
    """Setup common benchmark environment"""
    rclpy.init()
    connector = ROS2Connector()
    node = connector.node
    node.declare_parameter("conversion_ratio", 1.0)

    # define tools
    tools: List[BaseTool] = [
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

    # define o3de bridge
    simulation_config = O3DExROS2SimulationConfig.load_config(
        config_path=Path(o3de_config_path)
    )
    o3de = O3DEngineArmManipulationBridge(connector, logger=bench_logger)

    # define benchmark
    benchmark = ManipulationO3DEBenchmark(
        model_name=model_name,
        simulation_bridge=o3de,
        simulation_config=simulation_config,
        scenarios=scenarios,
        logger=bench_logger,
        results_dir=out_dir,
    )

    return connector, o3de, benchmark, tools


def run_benchmark(
    llm: BaseChatModel,
    out_dir: Path,
    scenarios: List[Scenario],
    bench_logger: logging.Logger,
    o3de_config_path: str = "src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml",
    experiment_id: uuid.UUID = uuid.uuid4(),
):
    connector, o3de, benchmark, tools = _setup_benchmark_environment(
        o3de_config_path, get_llm_model_name(llm), scenarios, out_dir, bench_logger
    )
    try:
        for scenario in scenarios:
            # create new agent for each scenario so its independent from previous ones.
            agent = create_conversational_agent(
                llm, tools, scenario.task.system_prompt, logger=bench_logger
            )
            benchmark.run_next(agent=agent, experiment_id=experiment_id)
            o3de.reset_arm()
            time.sleep(0.2)  # admire the end position for a second ;)

        time.sleep(3)
        bench_logger.info(
            "==============================================================="
        )
        bench_logger.info("ALL SCENARIOS DONE. BENCHMARK COMPLETED!")
        bench_logger.info(
            "==============================================================="
        )
    finally:
        connector.shutdown()
        o3de.shutdown()
        rclpy.shutdown()
