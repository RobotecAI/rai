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
import csv
import logging
import time
from typing import Any, Dict, Generic, List, TypeVar, Union

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from rai.messages import HumanMultimodalMessage
from rclpy.impl.rcutils_logger import RcutilsLogger

from rai_bench.manipulation_o3de.interfaces import Task
from rai_sim.simulation_bridge import Entity, SimulationBridge, SimulationConfigT

loggers_type = Union[RcutilsLogger, logging.Logger]
EntityT = TypeVar("EntityT", bound=Entity)


class EntitiesMismatchException(Exception):
    pass


class Scenario(Generic[SimulationConfigT]):
    """
    A Scenario are defined by a pair of Task and Simlation Config.
    Each Scenario is executed separatly by a Benchmark.
    """

    def __init__(
        self,
        task: Task,
        simulation_config: SimulationConfigT,
        simulation_config_path: str,
    ) -> None:
        """
        Initialize a Scenario.

        Parameters
        ----------
        task : Task
            The task to be executed.
        simulation_config : SimulationConfigT
            The simulation configuration for the scenario.
        simulation_config_path : str
            The file path to the simulation configuration.

        Raises
        ------
        ValueError
            If the provided simulation configuration is not valid for the task.
        """
        self.task = task
        self.simulation_config = simulation_config
        # NOTE (jmatejcz) needed for logging which config was used,
        # there probably is better way to do it
        self.simulation_config_path = simulation_config_path


class Benchmark:
    """
    Benchmark represents a set of Scenarios to be executed and evaluated.
    It manages the execution, logs results, and provides functionality
    for tracking and exporting performance metrics.
    """

    def __init__(
        self,
        simulation_bridge: SimulationBridge[SimulationConfigT],
        scenarios: List[Scenario[SimulationConfigT]],
        logger: loggers_type | None = None,
        results_filename: str = "benchmark_results.csv",
    ) -> None:
        self.simulation_bridge = simulation_bridge
        self.num_of_scenarios = len(scenarios)
        self.scenarios = enumerate(iter(scenarios))
        self.results: List[Dict[str, Any]] = []
        self.results_filename = results_filename
        if logger:
            self._logger = logger
        else:
            self._logger = logging.getLogger(__name__)

        self.fieldnames = [
            "task",
            "simulation_config",
            "final_score",
            "total_time",
            "number_of_tool_calls",
        ]
        self._initialize_results_file()

    @classmethod
    def create_scenarios(
        cls,
        tasks: List[Task],
        simulation_configs: List[SimulationConfigT],
        simulation_configs_paths: List[str],
        logger: loggers_type | None = None,
    ) -> List[Scenario[SimulationConfigT]]:
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
        # NOTE (jmatejcz) hacky_fix, taking paths as args here, not the best solution,
        # but more changes to code would be required
        scenarios: List[Scenario[SimulationConfigT]] = []
        if not logger:
            logger = logging.getLogger(__name__)
        for task in tasks:
            for sim_conf, sim_path in zip(simulation_configs, simulation_configs_paths):
                if task.validate_config(simulation_config=sim_conf):
                    scenarios.append(
                        Scenario(
                            task=task,
                            simulation_config=sim_conf,
                            simulation_config_path=sim_path,
                        )
                    )
                else:
                    logger.debug(  # type: ignore
                        f"Simulation config: {sim_path} is not suitable for task: {task.get_prompt()}"
                    )
        return scenarios

    def _initialize_results_file(self):
        """Initialize the CSV file with headers."""
        with open(
            self.results_filename, mode="w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()

    def run_next(self, agent: CompiledStateGraph) -> None:
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

            self.simulation_bridge.setup_scene(scenario.simulation_config)
            self._logger.info(  # type: ignore
                "======================================================================================"
            )
            self._logger.info(  # type: ignore
                f"RUNNING SCENARIO NUMBER {i + 1} / {self.num_of_scenarios}\n TASK: {scenario.task.get_prompt()}\n SIMULATION_CONFIG: {scenario.simulation_config_path}"
            )
            tool_calls_num = 0

            ts = time.perf_counter()
            prev_count: int = 0
            for state in agent.stream(
                {"messages": [HumanMessage(content=scenario.task.get_prompt())]},
                {
                    "recursion_limit": 100
                },  # NOTE (jmatejcz) what should be recursion limit?
            ):
                node = next(iter(state))
                new_messages = state[node]["messages"][prev_count:]
                prev_count = len(state[node]["messages"])

                for msg in new_messages:
                    if isinstance(msg, HumanMultimodalMessage):
                        last_msg = msg.text
                    elif isinstance(msg, BaseMessage):
                        if isinstance(msg.content, list):
                            if len(msg.content) == 1:
                                if type(msg.content[0]) is dict:
                                    last_msg = msg.content[0].get("text", "")
                        else:
                            last_msg = msg.content
                            self._logger.debug(f"{node}: {last_msg}")  # type: ignore

                    else:
                        raise ValueError(f"Unexpected type of message: {type(msg)}")

                    if isinstance(msg, AIMessage):
                        tool_calls_num += len(msg.tool_calls)

                    self._logger.info(f"AI Message: {msg}")  # type: ignore

            te = time.perf_counter()
            try:
                result = scenario.task.calculate_result(self.simulation_bridge)
                total_time = te - ts
                self._logger.info(  # type: ignore
                    f"TASK SCORE: {result}, TOTAL TIME: {total_time:.3f}, NUM_OF_TOOL_CALLS: {tool_calls_num}"
                )
                scenario_result: Dict[str, Any] = {
                    "task": scenario.task.get_prompt(),
                    "simulation_config": scenario.simulation_config_path,
                    "final_score": result,
                    "total_time": f"{total_time:.3f}",
                    "number_of_tool_calls": tool_calls_num,
                }
                self.results.append(scenario_result)
                self._save_scenario_result_to_csv(scenario_result)
            except EntitiesMismatchException as e:
                self._logger.error(e)  # type: ignore

        except StopIteration:
            print("No more scenarios left to run.")

    def _save_scenario_result_to_csv(self, result: Dict[str, Any]) -> None:
        """Save a single scenario result to the CSV file."""
        with open(
            self.results_filename, mode="a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(result)

    def get_results(self) -> List[Dict[str, Any]]:
        return self.results
