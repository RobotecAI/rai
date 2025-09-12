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
import csv
import uuid
from abc import abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Literal

import numpy as np
from git import Optional
from langchain.chat_models.base import BaseChatModel
from pydantic import BaseModel

import rai_bench.manipulation_o3de as manipulation_o3de
import rai_bench.tool_calling_agent as tool_calling_agent
import rai_bench.vlm_benchmark as vlm_benchmark
from rai_bench.base_benchmark import ModelSummary, RunSummary, TasksSummary
from rai_bench.results_processing.data_loading import (
    DETAILED_FILE_NAME,
    SUMMARY_FILE_NAME,
)
from rai_bench.utils import (
    define_benchmark_logger,
    get_llm_for_benchmark,
    get_llm_model_name,
)

MODEL_SUMMARY_FILE_NAME = "model_summary.csv"
TASKS_SUMMARY_FILE_NAME = "tasks_summary.csv"
BENCHMARK_SUMMARY = "benchmark_summary.csv"


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
        ]
    ] = [
        "basic",
        "manipulation",
        "navigation",
        "custom_interfaces",
    ]

    @property
    def name(self) -> str:
        return "tool_calling_agent"


class VLMBenchmarkConfig(BenchmarkConfig):
    complexities: List[Literal["easy", "medium", "hard"]] = ["easy", "medium", "hard"]
    task_types: List[
        Literal[
            "bool_response_image_task",
            "quantity_response_image_task",
            "multiple_choice_image_task",
        ]
    ] = [
        "bool_response_image_task",
        "quantity_response_image_task",
        "multiple_choice_image_task",
    ]

    @property
    def name(self) -> str:
        return "vlm"


def merge_model_repeats_summary(
    bench_name: str, model_name: str, run_dir: Path
) -> None:
    """Merge summary results across all repeats for a single model.

    Parameters
    ----------
    bench_name : str
        Name of the benchmark
    model_name : str
        Name of the model
    run_dir : Path
        Directory containing the benchmark run results
    """
    model_dir = run_dir / bench_name / model_name
    if not model_dir.exists():
        return

    summaries: List[RunSummary] = []
    for repeat_dir in model_dir.iterdir():
        if repeat_dir.is_dir() and repeat_dir.name.isdigit():
            summary_file = repeat_dir / SUMMARY_FILE_NAME
            if summary_file.exists():
                with open(summary_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        summaries.append(RunSummary.model_validate(row))

    if not summaries:
        return

    success_rates = [s.success_rate for s in summaries]
    times = [s.avg_time for s in summaries]
    total_tasks_list = [s.total_tasks for s in summaries]

    avg_success_rate = np.mean(success_rates)
    std_success_rate = np.std(success_rates)
    avg_time = np.mean(times)
    std_time = np.std(times)
    total_tasks = np.mean(total_tasks_list)

    merged_summary = ModelSummary(
        model_name=model_name,
        avg_success_rate=round(float(avg_success_rate), 2),
        std_success_rate=round(float(std_success_rate), 3),
        avg_time=round(float(avg_time), 3),
        std_time=round(float(std_time), 3),
        avg_total_tasks=round(float(total_tasks), 3),
        repeats=len(summaries),
    )

    merged_file = model_dir / MODEL_SUMMARY_FILE_NAME
    with open(merged_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ModelSummary.model_fields.keys())
        writer.writeheader()
        writer.writerow(merged_summary.model_dump())


def merge_benchmark_summary(
    bench_name: str, run_dir: Path, model_names: List[str]
) -> None:
    """Merge summary results across all models for a single benchmark.

    Parameters
    ----------
    bench_name : str
        Name of the benchmark
    run_dir : Path
        Directory containing the benchmark run results
    model_names : List[str]
        List of model names to include in the summary
    """
    bench_dir = run_dir / bench_name
    if not bench_dir.exists():
        return

    all_summaries: List[ModelSummary] = []
    for model_name in model_names:
        model_dir = bench_dir / model_name
        merged_file = model_dir / MODEL_SUMMARY_FILE_NAME

        if merged_file.exists():
            with open(merged_file, "r") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    all_summaries.append(ModelSummary.model_validate(row))

    if not all_summaries:
        return

    benchmark_summary_file = bench_dir / BENCHMARK_SUMMARY
    with open(benchmark_summary_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ModelSummary.model_fields.keys())
        writer.writeheader()
        for summary in all_summaries:
            writer.writerow(summary.model_dump())


def merge_tasks_summary(bench_name: str, model_name: str, run_dir: Path) -> None:
    """Merge task results across all repeats for a single model, aggregating by task.

    Parameters
    ----------
    bench_name : str
        Name of the benchmark
    model_name : str
        Name of the model
    run_dir : Path
        Directory containing the benchmark run results
    """
    model_dir = run_dir / bench_name / model_name
    if not model_dir.exists():
        return

    task_data_by_id: Dict[str, Dict[str, Any]] = {}

    for repeat_dir in model_dir.iterdir():
        if repeat_dir.is_dir() and repeat_dir.name.isdigit():
            results_file = repeat_dir / DETAILED_FILE_NAME
            if results_file.exists():
                with open(results_file, "r") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        task_id = row["task_id"]
                        task_prompt = row["task_prompt"]
                        score = float(row["score"])
                        total_time = float(row["total_time"])

                        if task_id not in task_data_by_id:
                            task_data_by_id[task_id] = {
                                "scores": [],
                                "times": [],
                                "task_prompt": task_prompt,
                            }

                        task_data_by_id[task_id]["scores"].append(score)
                        task_data_by_id[task_id]["times"].append(total_time)

    if not task_data_by_id:
        return

    # Calculate statistics for each task
    task_summaries: List[TasksSummary] = []
    for task_id, data in task_data_by_id.items():
        scores = np.array(data["scores"])
        times = np.array(data["times"])
        task_prompt = data["task_prompt"]

        task_summary = TasksSummary(
            model_name=model_name,
            task_id=task_id,
            task_prompt=task_prompt,
            avg_success_rate=round(float(scores.mean()), 3),
            std_success_rate=round(float(scores.std()), 3),
            avg_time=round(float(times.mean()), 3),
            std_time=round(float(times.std()), 3),
            repeats=len(scores),
        )
        task_summaries.append(task_summary)

    tasks_summary_file = model_dir / TASKS_SUMMARY_FILE_NAME
    with open(tasks_summary_file, "w", newline="") as f:
        if task_summaries:
            writer = csv.DictWriter(f, fieldnames=TasksSummary.model_fields.keys())
            writer.writeheader()
            for task_summary in task_summaries:
                writer.writerow(task_summary.model_dump())


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
            run_dir = Path(out_dir) / run_name
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

                        elif isinstance(bench_conf, VLMBenchmarkConfig):
                            vlm_tasks = vlm_benchmark.get_spatial_tasks()
                            vlm_benchmark.run_benchmark(
                                llm=llm,
                                out_dir=Path(curr_out_dir),
                                tasks=vlm_tasks,
                                bench_logger=bench_logger,
                            )

                    except Exception as e:
                        bench_logger.critical(f"BENCHMARK RUN FAILED: {e}")
                        bench_logger.critical(
                            f"{bench_conf.name} benchmark for {model_name}, vendor: {vendors[i]}, execution number: {u + 1}"
                        )
            merge_results_logger = define_benchmark_logger(out_dir=Path(out_dir))
            merge_results_logger.info(
                f"Merging summaries for benchmark: {bench_conf.name}"
            )

            for model_name in model_names:
                merge_model_repeats_summary(bench_conf.name, model_name, run_dir)
                merge_tasks_summary(bench_conf.name, model_name, run_dir)

            merge_benchmark_summary(bench_conf.name, run_dir, model_names)

            merge_results_logger.info(
                f"Summary merging completed for benchmark: {bench_conf.name}"
            )
