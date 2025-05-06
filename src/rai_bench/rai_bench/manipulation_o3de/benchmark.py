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
from pathlib import Path
from typing import Generic, List, TypeVar

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from rai.messages import HumanMultimodalMessage

from rai_bench.base_benchmark import BaseBenchmark, BenchmarkSummary
from rai_bench.manipulation_o3de.interfaces import Task
from rai_bench.manipulation_o3de.results_tracking import ScenarioResult
from rai_sim.simulation_bridge import (
    Entity,
    SimulationBridge,
    SimulationConfigT,
)

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


class ManipulationO3DEBenchmark(BaseBenchmark):
    """
    ManipulationO3DEBenchmark represents a set of Scenarios to be executed and evaluated.
    It manages the execution, logs results, and provides functionality
    for tracking and exporting performance metrics.
    """

    def __init__(
        self,
        model_name: str,
        simulation_bridge: SimulationBridge[SimulationConfigT],
        scenarios: List[Scenario[SimulationConfigT]],
        results_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            results_dir=results_dir,
            logger=logger,
        )
        self.simulation_bridge = simulation_bridge
        self.num_of_scenarios = len(scenarios)
        self.scenarios = enumerate(iter(scenarios))

        self.scenario_results: List[ScenarioResult] = []
        self.csv_initialize(self.results_filename, ScenarioResult)

    @classmethod
    def create_scenarios(
        cls,
        tasks: List[Task],
        simulation_configs: List[SimulationConfigT],
        simulation_configs_paths: List[str],
        logger: logging.Logger | None = None,
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
                    logger.debug(
                        f"Simulation config: {sim_path} is not suitable for task: {task.task_prompt}"
                    )
        return scenarios

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
            self.logger.info(
                "======================================================================================"
            )
            self.logger.info(
                f"RUNNING SCENARIO NUMBER {i + 1} / {self.num_of_scenarios}\n TASK: {scenario.task.task_prompt}\n SIMULATION_CONFIG: {scenario.simulation_config_path}"
            )
            tool_calls_num = 0

            ts = time.perf_counter()
            prev_count: int = 0
            try:
                for state in agent.stream(
                    {"messages": [HumanMessage(content=scenario.task.task_prompt)]},
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
                                self.logger.debug(f"{node}: {last_msg}")

                        else:
                            raise ValueError(f"Unexpected type of message: {type(msg)}")

                        if isinstance(msg, AIMessage):
                            tool_calls_num += len(msg.tool_calls)

                        self.logger.info(f"AI Message: {msg}")
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
                    simulation_config_path=scenario.simulation_config_path,
                    model_name=self.model_name,
                    score=score,
                    total_time=total_time,
                    number_of_tool_calls=tool_calls_num,
                )
                self.scenario_results.append(scenario_result)
                self.csv_writerow(self.results_filename, scenario_result)
                # computing after every iteration in case of early stopping
                self.compute_and_save_summary()
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

        # TODO (jm) extend this bechmark to implement extra tool calls
        # since this benchmark doesn't have the concept of "extra tool calls",
        # we use the total number of tool calls instead
        total_tool_calls = sum(r.number_of_tool_calls for r in self.scenario_results)

        summary = BenchmarkSummary(
            model_name=self.model_name,
            success_rate=round(success_rate, 2),
            avg_time=round(avg_time, 3),
            total_extra_tool_calls_used=total_tool_calls,
            total_tasks=len(self.scenario_results),
        )
        self.csv_initialize(self.summary_filename, BenchmarkSummary)
        self.csv_writerow(self.summary_filename, summary)

        self.logger.info(
            f"Summary for model {self.model_name}: Success rate {success_rate:.2f}%, "
            f"Average time {avg_time:.3f}s, Total tasks: {len(self.scenario_results)}"
        )
