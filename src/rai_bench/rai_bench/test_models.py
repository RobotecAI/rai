# # Copyright (C) 2025 Robotec.AI
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #         http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
import uuid
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Literal

from git import Optional
from pydantic import BaseModel

from rai_bench.agents import AgentFactory
from rai_bench.utils import (
    define_benchmark_logger,
)


class BenchmarkConfig(BaseModel):
    repeats: int = 1

    @property
    @abstractmethod
    def name(self) -> str:
        pass


class ManipulationO3DEBenchmarkConfig(BenchmarkConfig):
    """Configuration for Manipulation O3DE Benchmark.

    Parameters
    ----------
    o3de_config_path : str
        path to O3DE configuration file
    levels : List[Literal["trivial", "easy", "medium", "hard", "very_hard"]], optional
        difficulty levels to include in benchmark, by default all levels are included:
        ["trivial", "easy", "medium", "hard", "very_hard"]
    """

    o3de_config_path: str
    levels: List[Literal["trivial", "easy", "medium", "hard", "very_hard"]] = [
        "trivial",
        "easy",
        "medium",
        "hard",
        "very_hard",
    ]

    @property
    def name(self) -> str:
        return "manipulation_o3de"


class ToolCallingAgentBenchmarkConfig(BenchmarkConfig):
    """Configuration for Tool Calling Agent Benchmark.

    Parameters
    ----------
    extra_tool_calls : List[int], optional
        how many extra tool calls allowed to still pass, by default [0]
    prompt_detail : List[Literal["brief", "descriptive"]], optional
        how descriptive should task prompt be, by default all levels are included:
        ["brief", "descriptive"]
    N_shots : List[Literal[0, 2, 5]], optional
        how many examples are in system prompt, by default all are included: [0, 2, 5]
    complexities : List[Literal["easy", "medium", "hard"]], optional
        complexity levels of tasks to include in the benchmark, by default all levels are included:
        ["easy", "medium", "hard"]
    task_types : List[Literal["basic", "manipulation", "navigation", "custom_interfaces", "spatial_reasoning"]], optional
        types of tasks to include in the benchmark, by default all types are included:
        ["basic", "manipulation", "navigation", "custom_interfaces", "spatial_reasoning"]

    For more detailed explanation of parameters, see the documentation:
    (https://robotecai.github.io/rai/simulation_and_benchmarking/rai_bench/)
    """

    extra_tool_calls: List[int] = [0]
    complexities: List[Literal["easy", "medium", "hard"]] = ["easy", "medium", "hard"]
    N_shots: List[Literal[0, 2, 5]] = [0, 2, 5]
    prompt_detail: List[Literal["brief", "descriptive"]] = ["brief", "descriptive"]
    task_types: List[
        Literal[
            "basic",
            "manipulation",
            "custom_interfaces",
            "spatial_reasoning",
        ]
    ] = [
        "basic",
        "manipulation",
        "custom_interfaces",
        "spatial_reasoning",
    ]

    @property
    def name(self) -> str:
        return "tool_calling_agent"


def test_agents(
    agent_factories: List[AgentFactory],
    benchmark_configs: List[BenchmarkConfig],
    out_dir: str,
    experiment_name: Optional[str] = None,
):
    """
    Test multiple agent factories on multiple benchmark configurations.

    Args:
        agent_factories: List of agent factories to benchmark
        benchmark_configs: List of benchmark configurations to run
        out_dir: Output directory for results
        experiment_name: Optional name for the experiment
    """
    experiment_id = uuid.uuid4()

    # Generate experiment name containing current datetime, if not provided
    now = datetime.now()
    if experiment_name is None:
        now = datetime.now()
        experiment_name = f"run_{now.strftime('%Y-%m-%d_%H-%M-%S')}"

    for bench_conf in benchmark_configs:
        for agent_factory in agent_factories:
            for repeat in range(bench_conf.repeats):
                curr_out_dir = (
                    Path(out_dir)
                    / experiment_name
                    / bench_conf.name
                    / agent_factory.model_name
                    / str(repeat)
                )
                curr_out_dir.mkdir(parents=True, exist_ok=True)

                bench_logger = define_benchmark_logger(out_dir=curr_out_dir)

                try:
                    run_single_benchmark(
                        agent_factory=agent_factory,
                        bench_conf=bench_conf,
                        curr_out_dir=curr_out_dir,
                        experiment_id=experiment_id,
                        bench_logger=bench_logger,
                    )
                except Exception as e:
                    bench_logger.critical(f"BENCHMARK RUN FAILED: {e}")
                    bench_logger.critical(
                        f"{bench_conf.name} benchmark for {agent_factory.model_name}, repeat: {repeat + 1}"
                    )


def run_single_benchmark(
    agent_factory: AgentFactory,
    bench_conf: BenchmarkConfig,
    curr_out_dir: Path,
    experiment_id: uuid.UUID,
    bench_logger,
):
    """Run a single benchmark configuration with a single agent factory."""
    if isinstance(bench_conf, ToolCallingAgentBenchmarkConfig):
        import rai_bench.tool_calling_agent as tool_calling_agent

        tasks = tool_calling_agent.get_tasks(
            extra_tool_calls=bench_conf.extra_tool_calls,
            complexities=bench_conf.complexities,
            task_types=bench_conf.task_types,
        )
        tool_calling_agent.run_benchmark(
            agent_factory=agent_factory,
            out_dir=curr_out_dir,
            tasks=tasks,
            experiment_id=experiment_id,
            bench_logger=bench_logger,
        )

    elif isinstance(bench_conf, ManipulationO3DEBenchmarkConfig):
        import rai_bench.manipulation_o3de as manipulation_o3de

        scenarios = manipulation_o3de.get_scenarios(
            levels=bench_conf.levels,
            logger=bench_logger,
        )
        manipulation_o3de.run_benchmark(
            agent_factory=agent_factory,
            out_dir=curr_out_dir,
            o3de_config_path=bench_conf.o3de_config_path,
            scenarios=scenarios,
            experiment_id=experiment_id,
            bench_logger=bench_logger,
        )
