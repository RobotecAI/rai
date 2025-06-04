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
from typing import Any, Dict, List, Literal

from git import Optional
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel

import rai_bench.manipulation_o3de as manipulation_o3de
import rai_bench.tool_calling_agent as tool_calling_agent
from rai_bench.utils import (
    define_benchmark_logger,
    get_llm_for_benchmark,
    get_llm_model_name,
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
    complexities: List[Literal["easy", "medium", "hard"]] = ["easy", "medium", "hard"]
    task_types: List[
        Literal[
            "basic",
            "manipulation",
            "navigation",
            "custom_interfaces",
            "spatial_reasoning",
        ]
    ] = [
        "basic",
        "manipulation",
        "navigation",
        "custom_interfaces",
        "spatial_reasoning",
    ]

    @property
    def name(self) -> str:
        return "tool_calling_agent"


def test_dual_agents(
    multimodal_llms: List[BaseChatModel],
    tool_calling_models: List[BaseChatModel],
    benchmark_configs: List[BenchmarkConfig],
    out_dir: str,
    m_system_prompt: Optional[str] = None,
    tool_system_prompt: Optional[str] = None,
):
    if len(multimodal_llms) != len(tool_calling_models):
        raise ValueError(
            "Number of passed multimodal models must match number of passed tool calling models"
        )
    experiment_id = uuid.uuid4()
    for bench_conf in benchmark_configs:
        # for each bench configuration seperate run folder
        now = datetime.now()
        run_name = f"run_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
        for i, m_llm in enumerate(multimodal_llms):
            tool_llm = tool_calling_models[i]
            for u in range(bench_conf.repeats):
                curr_out_dir = (
                    out_dir
                    + "/"
                    + run_name
                    + "/"
                    + bench_conf.name
                    + "/"
                    + get_llm_model_name(m_llm)
                    + "/"
                    + str(u)
                )
                bench_logger = define_benchmark_logger(out_dir=Path(curr_out_dir))
                try:
                    if isinstance(bench_conf, ToolCallingAgentBenchmarkConfig):
                        tool_calling_tasks = tool_calling_agent.get_tasks(
                            extra_tool_calls=bench_conf.extra_tool_calls,
                            complexities=bench_conf.complexities,
                            task_types=bench_conf.task_types,
                        )
                        tool_calling_agent.run_benchmark_dual_agent(
                            multimodal_llm=m_llm,
                            tool_calling_llm=tool_llm,
                            m_system_prompt=m_system_prompt,
                            tool_system_prompt=tool_system_prompt,
                            out_dir=Path(curr_out_dir),
                            tasks=tool_calling_tasks,
                            experiment_id=experiment_id,
                            bench_logger=bench_logger,
                        )
                    elif isinstance(bench_conf, ManipulationO3DEBenchmarkConfig):
                        manipulation_o3de_scenarios = manipulation_o3de.get_scenarios(
                            levels=bench_conf.levels,
                            logger=bench_logger,
                        )
                        manipulation_o3de.run_benchmark_dual_agent(
                            multimodal_llm=m_llm,
                            tool_calling_llm=tool_llm,
                            out_dir=Path(curr_out_dir),
                            o3de_config_path=bench_conf.o3de_config_path,
                            scenarios=manipulation_o3de_scenarios,
                            experiment_id=experiment_id,
                            bench_logger=bench_logger,
                        )
                except Exception as e:
                    bench_logger.critical(f"BENCHMARK RUN FAILED: {e}")
                    raise e


def test_models(
    model_names: List[str],
    vendors: List[str],
    benchmark_configs: List[BenchmarkConfig],
    out_dir: str,
    additional_model_args: Optional[List[Dict[str, Any]]] = None,
):
    if additional_model_args is None:
        additional_model_args = [{} for _ in model_names]

    experiment_id = uuid.uuid4()
    if len(model_names) != len(vendors):
        raise ValueError("Number of passed models must match number of passed vendors")
    else:
        for bench_conf in benchmark_configs:
            # for each bench configuration seperate run folder
            now = datetime.now()
            run_name = f"run_{now.strftime('%Y-%m-%d_%H-%M-%S')}"
            for i, model_name in enumerate(model_names):
                for u in range(bench_conf.repeats):
                    curr_out_dir = (
                        out_dir
                        + "/"
                        + run_name
                        + "/"
                        + bench_conf.name
                        + "/"
                        + model_name
                        + "/"
                        + str(u)
                    )
                    llm = get_llm_for_benchmark(
                        model_name=model_name,
                        vendor=vendors[i],
                        **additional_model_args[i],
                    )
                    bench_logger = define_benchmark_logger(out_dir=Path(curr_out_dir))
                    try:
                        if isinstance(bench_conf, ToolCallingAgentBenchmarkConfig):
                            tool_calling_tasks = tool_calling_agent.get_tasks(
                                extra_tool_calls=bench_conf.extra_tool_calls,
                                complexities=bench_conf.complexities,
                                task_types=bench_conf.task_types,
                            )
                            tool_calling_agent.run_benchmark(
                                llm=llm,
                                out_dir=Path(curr_out_dir),
                                tasks=tool_calling_tasks,
                                experiment_id=experiment_id,
                                bench_logger=bench_logger,
                            )
                        elif isinstance(bench_conf, ManipulationO3DEBenchmarkConfig):
                            manipulation_o3de_scenarios = (
                                manipulation_o3de.get_scenarios(
                                    levels=bench_conf.levels,
                                    logger=bench_logger,
                                )
                            )
                            manipulation_o3de.run_benchmark(
                                llm=llm,
                                out_dir=Path(curr_out_dir),
                                o3de_config_path=bench_conf.o3de_config_path,
                                scenarios=manipulation_o3de_scenarios,
                                experiment_id=experiment_id,
                                bench_logger=bench_logger,
                            )
                    except Exception as e:
                        bench_logger.critical(f"BENCHMARK RUN FAILED: {e}")
                        bench_logger.critical(
                            f"{bench_conf.name} benchmark for {model_name}, vendor: {vendors[i]}, execution number: {u + 1}"
                        )
