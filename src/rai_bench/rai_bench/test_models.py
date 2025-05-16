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
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import List, Literal

from pydantic import BaseModel

import rai_bench.examples.manipulation_o3de as manipulation_o3de_bench
import rai_bench.examples.tool_calling_agent as tool_calling_agent_bench
from rai_bench.manipulation_o3de.predefined.scenarios import get_all_scenarios
from rai_bench.tool_calling_agent.predefined.tasks import get_all_tasks
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
    # by default include all
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
    extra_tool_calls: int = 0
    ...
    # TODO (jmatejcz)
    # by type
    # by complexity

    @property
    def name(self) -> str:
        return "tool_calling_agent"


def test_models(
    model_names: List[str], vendors: List[str], benchmark_configs: List[BenchmarkConfig]
):
    now = datetime.now()
    out_dir = (
        f"src/rai_bench/rai_bench/experiments/run_{now.strftime('%Y-%m-%d_%H-%M-%S')}/"
    )

    tool_calling_tasks = get_all_tasks(extra_tool_calls=5)
    if len(model_names) != len(vendors):
        raise ValueError("Number of passed models must match number of passed vendors")
    else:
        for bench_conf in benchmark_configs:
            for i, model_name in enumerate(model_names):
                # for extra_calls in extra_tool_calls:
                for u in range(bench_conf.repeats):
                    curr_out_dir = (
                        out_dir + bench_conf.name + "/" + model_name + "/" + str(u)
                    )
                    bench_logger = define_benchmark_logger(out_dir=Path(curr_out_dir))
                    try:
                        if isinstance(bench_conf, ToolCallingAgentBenchmarkConfig):
                            tool_calling_agent_bench.run_benchmark(
                                model_name=model_name,
                                vendor=vendors[i],
                                out_dir=curr_out_dir,
                                tasks=tool_calling_tasks,
                                bench_logger=bench_logger,
                            )
                        elif isinstance(bench_conf, ManipulationO3DEBenchmarkConfig):
                            manipulation_o3de_scenarios = get_all_scenarios(
                                connector_path=bench_conf.o3de_config_path,
                                levels=bench_conf.levels,
                                logger=bench_logger,
                            )
                            manipulation_o3de_bench.run_benchmark(
                                model_name=model_name,
                                vendor=vendors[i],
                                out_dir=Path(curr_out_dir),
                                scenarios=manipulation_o3de_scenarios,
                                bench_logger=bench_logger,
                            )
                    except Exception as e:
                        print(
                            f"Failed to run {bench_conf.name} benchmark for {model_name}, vendor: {vendors[i]}, execution number: {u + 1}, because: {str(e)}"
                        )
                    continue
