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
import ast
import os
import re
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from pydantic import BaseModel

from rai_bench.base_benchmark import RunSummary
from rai_bench.manipulation_o3de.results_tracking import ScenarioResult
from rai_bench.tool_calling_agent.results_tracking import (
    SubTaskResult,
    TaskResult,
    ToolCallingAgentRunSummary,
    ValidatorResult,
)

EXPERIMENT_DIR = "./src/rai_bench/rai_bench/experiments"
DETAILED_FILE_NAME: str = "results.csv"
SUMMARY_FILE_NAME: str = "results_summary.csv"


def convert_row_to_task_result(row: pd.Series) -> TaskResult:
    """
    Convert a DataFrame row to a TaskResult object.

    Parameters
    ----------
    row : pd.Series
        A row from the detailed results DataFrame

    Returns
    -------
    TaskResult
        A TaskResult object
    """
    validation_info_raw = safely_parse_json_like_string(row["validation_info"])

    validator_results: List[ValidatorResult] = []
    for val_info in validation_info_raw:
        subtasks: List[SubTaskResult] = []
        for subtask in val_info["subtasks"]:
            subtask_result = SubTaskResult(
                args=subtask.get("args", {}),
                errors=subtask.get("errors", []),
                passed=subtask.get("passed", False),
            )
            subtasks.append(subtask_result)

        validator_result = ValidatorResult(
            type=val_info.get("type", ""),
            subtasks=subtasks,
            extra_tool_calls_used=val_info.get("extra_tool_calls_used", 0),
            passed=val_info.get("passed", False),
        )
        validator_results.append(validator_result)

    return TaskResult(
        task_prompt=row["task_prompt"],
        system_prompt=row["system_prompt"],
        complexity=row["complexity"],
        type=row["type"],
        model_name=row["model_name"],
        validation_info=validator_results,
        extra_tool_calls=int(row["extra_tool_calls"]),
        extra_tool_calls_used=int(row["extra_tool_calls_used"]),
        score=float(row["score"]),
        total_time=float(row["total_time"]),
        run_id=uuid.UUID(row["run_id"]),
    )


def convert_row_to_scenario_result(row: pd.Series) -> ScenarioResult:
    """
    Convert a DataFrame row to a ScenarioResult object.

    Parameters
    ----------
    row : pd.Series
        A row from the scenario results DataFrame

    Returns
    -------
    ScenarioResult
        A ScenarioResult object
    """
    return ScenarioResult(
        task_prompt=row["task_prompt"],
        system_prompt=row["system_prompt"],
        model_name=row["model_name"],
        scene_config_path=row["scene_config_path"],
        score=float(row["score"]),
        level=row["level"],
        total_time=float(row["total_time"]),
        number_of_tool_calls=int(row["number_of_tool_calls"]),
    )


BENCHMARKS_CONVERTERS: Dict[str, Any] = {
    "tool_calling_agent": convert_row_to_task_result,
    "manipulation_o3de": convert_row_to_scenario_result,
}


class ModelRepeatResults:
    """Container for results of a single repeat of one model."""

    def __init__(self, repeat_num: int):
        self.repeat_num = repeat_num
        self.task_results: List[BaseModel] = []
        self.summary: Optional[RunSummary] = None


class ModelRunResults:
    """Container for results of one model from a single run with multiple repeats."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.repeats: List[ModelRepeatResults] = []

    @property
    def task_results(self) -> List[BaseModel]:
        """Get all task results from all repeats."""
        results: List[BaseModel] = []
        for repeat in self.repeats:
            results.extend(repeat.task_results)
        return results

    @property
    def summaries(self) -> List[RunSummary]:
        """Get all summaries from all repeats."""
        results: List[RunSummary] = []
        for repeat in self.repeats:
            if repeat.summary:
                results.append(repeat.summary)
        return results


class ModelResults:
    """Results for a specific model across multiple runs."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.runs: List[ModelRunResults] = []

    def check_if_summaries_present(self) -> bool:
        if not self.runs or not any(run.summaries for run in self.runs):
            return False
        return True

    @property
    def count(self):
        return sum(len(run.summaries) for run in self.runs)

    @property
    def total_tasks(self) -> int:
        if not self.check_if_summaries_present():
            return 0

        return sum(
            summary.total_tasks for run in self.runs for summary in run.summaries
        )

    @property
    def avg_success_rate(self) -> float:
        """Calculate the average success rate across all runs."""
        if not self.check_if_summaries_present():
            return 0.0

        total = sum(
            summary.success_rate for run in self.runs for summary in run.summaries
        )
        return total / self.count

    @property
    def avg_time(self) -> float:
        """Calculate the average completion time across all runs."""
        if not self.check_if_summaries_present():
            return 0.0

        total = sum(summary.avg_time for run in self.runs for summary in run.summaries)
        return total / self.count

    @property
    def avg_extra_tool_calls(self) -> float:
        """Calculate the average extra tool calls used across all runs."""
        if not self.check_if_summaries_present():
            return 0.0

        total = 0.0
        count = 0
        for run in self.runs:
            for summary in run.summaries:
                if isinstance(summary, ToolCallingAgentRunSummary):
                    total += summary.total_extra_tool_calls_used
                    count += 1
                else:
                    raise NotImplementedError

        return total / count


class BenchmarkResults:
    """Results for a specific benchmark across multiple models and runs."""

    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.models: Dict[str, ModelResults] = {}

    def get_model_results(self, model_name: str) -> Optional[ModelResults]:
        """Get results for a specific model."""
        return self.models.get(model_name)

    def merge(self, other: "BenchmarkResults") -> None:
        """
        Merge another BenchmarkResults object into this one.

        Parameters
        ----------
        other : BenchmarkResults
            The other BenchmarkResults object to merge
        """
        if other.benchmark_name != self.benchmark_name:
            raise ValueError(
                f"Cannot merge benchmark results with different names: {self.benchmark_name} vs {other.benchmark_name}"
            )

        for model_name, other_model_results in other.models.items():
            if model_name in self.models:
                self.models[model_name].runs.extend(other_model_results.runs)
            else:
                self.models[model_name] = other_model_results


class RunResults:
    """Container for all benchmark results in a run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.benchmarks: Dict[str, BenchmarkResults] = {}

    def get_benchmark_results(self, benchmark_name: str) -> Optional[BenchmarkResults]:
        """Get results for a specific benchmark."""
        return self.benchmarks.get(benchmark_name)

    def merge(self, other: "RunResults") -> None:
        """
        Merge another RunResults object into this one.

        Parameters
        ----------
        other : RunResults
            The other RunResults object to merge
        """
        # Merge benchmarks
        for benchmark_name, other_benchmark in other.benchmarks.items():
            if benchmark_name in self.benchmarks:
                # Merge the benchmark results
                self.benchmarks[benchmark_name].merge(other_benchmark)
            else:
                # Add the new benchmark
                self.benchmarks[benchmark_name] = other_benchmark


def safely_parse_json_like_string(s: Any) -> List[Any]:
    """
    Parse validation info, reconstructing class references.
    """
    if not isinstance(s, str) or not s:
        return []

    # NOTE (jmatecz) the validation_info is not loaded properly as
    # argument that require only certain type is stored like this in results:
    #   {'timeout_sec': <class 'int'>}},
    # which can't be parsed correctly. Probably better approach would be
    # storing it differently in results, but for now parsing replaces it
    def replace_class_ref(match: re.Match[str]):
        class_name = match.group(1)
        builtins = {
            "int": "'int'",
            "str": "'str'",
            "float": "'float'",
            "bool": "'bool'",
            "list": "'list'",
            "dict": "'dict'",
        }
        if class_name in builtins:
            return builtins[class_name]
        return f"'{class_name}'"  # Fallback to string representation

    modified_str = re.sub(r"<class '([^']+)'>", replace_class_ref, s)

    return ast.literal_eval(modified_str)


def convert_row_to_benchmark_summary(row: pd.Series) -> RunSummary:
    """
    Convert a DataFrame row to a BenchmarkSummary object.
    Creates either a base BenchmarkSummary or a ToolCallingAgentSummary
    depending on the fields present in the row.

    Parameters
    ----------
    row : pd.Series
        A row from the summary results DataFrame

    Returns
    -------
    BenchmarkSummary
        A BenchmarkSummary or ToolCallingAgentSummary object
    """
    base_fields: Dict[str, Any] = {
        "model_name": row["model_name"],
        "success_rate": float(row["success_rate"]),
        "avg_time": float(row["avg_time"]),
        "total_tasks": row["total_tasks"],
    }

    if "total_extra_tool_calls_used" in row:
        return ToolCallingAgentRunSummary(
            **base_fields,
            total_extra_tool_calls_used=int(row["total_extra_tool_calls_used"]),
        )

    return RunSummary(**base_fields)


def load_detailed_data(file_path: str, benchmark: str) -> List[BaseModel]:
    df = pd.read_csv(file_path)  # type: ignore
    task_results: List[BaseModel] = []

    converter = BENCHMARKS_CONVERTERS[benchmark]
    for _, row in df.iterrows():  # type: ignore
        task_result = converter(row)
        task_results.append(task_result)

    return task_results


def load_summary_data(file_path: str) -> List[RunSummary]:
    """
    Load summary results data from a file path.

    Parameters
    ----------
    file_path : str
        Path to the summary results CSV file

    Returns
    -------
    List[BenchmarkSummary]
        List of BenchmarkSummary objects
    """
    df = pd.read_csv(file_path)  # type: ignore
    summaries: List[RunSummary] = []
    for _, row in df.iterrows():  # type: ignore
        summary = convert_row_to_benchmark_summary(row)
        summaries.append(summary)

    return summaries


def get_available_runs(experiment_dir: str) -> List[str]:
    """
    Get a list of available run folders.

    Parameters
    ----------
    experiment_dir : str
        Path to the experiments directory

    Returns
    -------
    List[str]
        List of run folder names
    """
    run_folders = [
        d
        for d in os.listdir(experiment_dir)
        if os.path.isdir(os.path.join(experiment_dir, d)) and d.startswith("run_")
    ]
    # sort by date
    return sorted(run_folders, key=lambda x: x.split("run_")[1])


def load_single_run(
    path: str, benchmark: str
) -> Tuple[List[BaseModel], List[RunSummary]]:
    """
    Load task results and benchmark summaries from a single run directory.
    Returns
    -------
    Optional[Tuple[List[TaskResult], List[BenchmarkSummary]]]
        Tuple of task results and benchmark summaries, or None if loading fails
    """

    detailed_path = os.path.join(path, DETAILED_FILE_NAME)
    summary_path = os.path.join(path, SUMMARY_FILE_NAME)

    if not os.path.exists(detailed_path):
        raise RuntimeError(f"Detailed {detailed_path} doesn't exist, cannot load data.")
    if not os.path.exists(summary_path):
        raise RuntimeError(
            f"Summary dir {summary_path} doesn't exist, cannot load data."
        )

    detailed_data = load_detailed_data(detailed_path, benchmark=benchmark)
    benchmark_summaries = load_summary_data(summary_path)

    # Verify data consistency
    task_model_name = detailed_data[0].model_name
    summary_model_name = benchmark_summaries[0].model_name

    if task_model_name != summary_model_name:
        st.warning(f"Data mismatch in run {path} - model names don't match")
        raise ValueError("Model name between detailed and summary data doesn't match.")

    return detailed_data, benchmark_summaries


def load_single_repeat(path: str, benchmark: str) -> Tuple[List[BaseModel], RunSummary]:
    """
    Load task results and benchmark summary from a single repeat directory.

    Parameters
    ----------
    path : str
        Path to the repeat directory
    benchmark : str
        Benchmark name

    Returns
    -------
    Tuple[List[BaseModel], RunSummary]
        Tuple of task results and the benchmark summary
    """
    detailed_path = os.path.join(path, DETAILED_FILE_NAME)
    summary_path = os.path.join(path, SUMMARY_FILE_NAME)

    if not os.path.exists(detailed_path):
        raise RuntimeError(
            f"Detailed file {detailed_path} doesn't exist, cannot load data."
        )
    if not os.path.exists(summary_path):
        raise RuntimeError(
            f"Summary file {summary_path} doesn't exist, cannot load data."
        )

    detailed_data = load_detailed_data(detailed_path, benchmark=benchmark)
    benchmark_summaries = load_summary_data(summary_path)

    if not detailed_data or not benchmark_summaries:
        raise RuntimeError(f"No data found in {path}")

    # Verify data consistency
    task_model_name = detailed_data[0].model_name
    summary_model_name = benchmark_summaries[0].model_name

    if task_model_name != summary_model_name:
        st.warning(f"Data mismatch in run {path} - model names don't match")
        raise ValueError("Model name between detailed and summary data doesn't match.")

    # We expect only one summary per repeat
    if len(benchmark_summaries) != 1:
        st.warning(
            f"Expected 1 summary in {path}, but found {len(benchmark_summaries)}"
        )

    return detailed_data, benchmark_summaries[0]


def load_run_results(run_dir: str) -> RunResults:
    """
    Load all benchmark results from a run directory.
    """
    run_name = os.path.basename(run_dir)
    run_results = RunResults(run_id=run_name)

    bench_name = os.listdir(run_dir)[0]
    bench_dir = os.path.join(run_dir, bench_name)

    if not os.path.isdir(bench_dir):
        raise RuntimeError(f"Bench dir {bench_dir} is not dir, cannot load data.")

    benchmark_results = BenchmarkResults(benchmark_name=bench_name)

    # List all model dirs in benchmark folder
    for bench_name in os.listdir(run_dir):
        bench_dir = os.path.join(run_dir, bench_name)
        if not os.path.isdir(bench_dir):
            continue

        benchmark_results = BenchmarkResults(benchmark_name=bench_name)

        # List all model dirs in benchmark folder
        for model_name in os.listdir(bench_dir):
            model_dir = os.path.join(bench_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            model_results = ModelResults(model_name=model_name)
            model_run = ModelRunResults(run_id=run_name)
            # List all repeats in model dir
            for repeat in os.listdir(model_dir):
                repeat_dir = os.path.join(model_dir, repeat)
                if not os.path.isdir(repeat_dir):
                    continue

                task_results, summary = load_single_repeat(
                    path=repeat_dir, benchmark=bench_name
                )

                repeat_results = ModelRepeatResults(repeat_num=int(repeat))
                repeat_results.task_results = task_results
                repeat_results.summary = summary

                model_run.repeats.append(repeat_results)

            if model_run.repeats:
                model_results.runs.append(model_run)
                benchmark_results.models[model_name] = model_results

        if benchmark_results.models:
            run_results.benchmarks[bench_name] = benchmark_results

    return run_results


def load_multiple_runs(run_dirs: List[str]) -> RunResults:
    """
    Load and merge results from multiple run directories.

    Parameters
    ----------
    run_dirs : List[str]
        List of paths to run directories to load

    Returns
    -------
    Optional[RunResults]
        A merged RunResults object containing data from all runs, or None if loading fails
    """

    # Load the first run to initialize the merged results
    merged_results = load_run_results(run_dirs[0])

    # Merge additional runs
    for run_dir in run_dirs[1:]:
        run_results = load_run_results(run_dir)
        merged_results.merge(run_results)

    return merged_results


def get_unique_benchmarks(run_results: RunResults) -> List[str]:
    """
    Get a list of unique benchmark names from a RunResults object.

    Parameters
    ----------
    run_results : RunResults
        The RunResults object to extract benchmark names from

    Returns
    -------
    List[str]
        A list of unique benchmark names
    """
    return list(run_results.benchmarks.keys())


def get_models_for_benchmark(bench_results: BenchmarkResults) -> List[str]:
    """
    Get a list of model names for a specific benchmark.

    Parameters
    ----------
    bench_results : BenchmarkResults
        The BenchmarkResults object to extract model names from

    Returns
    -------
    List[str]
        A list of model names
    """
    return list(bench_results.models.keys())
