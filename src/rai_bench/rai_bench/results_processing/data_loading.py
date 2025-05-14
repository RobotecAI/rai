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
import uuid
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import streamlit as st
from pydantic import BaseModel

from rai_bench.base_benchmark import BenchmarkSummary
from rai_bench.manipulation_o3de.results_tracking import ScenarioResult
from rai_bench.tool_calling_agent.results_tracking import (
    SubTaskResult,
    TaskResult,
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
        total_time=float(row["total_time"]),
        number_of_tool_calls=int(row["number_of_tool_calls"]),
    )


BENCHMARKS_CONVERTERS: Dict[str, Any] = {
    "tool_calling_agent": convert_row_to_task_result,
    "manipulation_o3de": convert_row_to_scenario_result,
}


class ModelRunResults:
    """Container for results from a single model run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.task_results: List[TaskResult] = []
        self.benchmark_summaries: List[BenchmarkSummary] = []


class ModelResults:
    """Results for a specific model across multiple runs."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.runs: List[ModelRunResults] = []

    def check_if_summaries_present(self) -> bool:
        if not self.runs or not any(run.benchmark_summaries for run in self.runs):
            return False
        return True

    @property
    def count(self):
        return sum(len(run.benchmark_summaries) for run in self.runs)

    @property
    def avg_success_rate(self) -> float:
        """Calculate the average success rate across all runs."""
        if not self.check_if_summaries_present():
            return 0.0

        total = sum(
            summary.success_rate
            for run in self.runs
            for summary in run.benchmark_summaries
        )
        return total / self.count

    @property
    def avg_time(self) -> float:
        """Calculate the average completion time across all runs."""
        if not self.check_if_summaries_present():
            return 0.0

        total = sum(
            summary.avg_time for run in self.runs for summary in run.benchmark_summaries
        )
        return total / self.count

    @property
    def avg_extra_tool_calls(self) -> float:
        """Calculate the average extra tool calls used across all runs."""
        if not self.check_if_summaries_present():
            return 0.0

        total = sum(
            summary.total_extra_tool_calls_used
            for run in self.runs
            for summary in run.benchmark_summaries
        )
        return total / self.count


class BenchmarkResults:
    """Results for a specific benchmark across multiple models."""

    def __init__(self, benchmark_name: str):
        self.benchmark_name = benchmark_name
        self.models: Dict[str, ModelResults] = {}

    def get_model_results(self, model_name: str) -> Optional[ModelResults]:
        """Get results for a specific model."""
        return self.models.get(model_name)


class RunResults:
    """Container for all benchmark results in a run."""

    def __init__(self, run_id: str):
        self.run_id = run_id
        self.benchmarks: Dict[str, BenchmarkResults] = {}

    def get_benchmark_results(self, benchmark_name: str) -> Optional[BenchmarkResults]:
        """Get results for a specific benchmark."""
        return self.benchmarks.get(benchmark_name)


def safely_parse_json_like_string(s: Any) -> List[Any]:
    """Parse string representation of Python objects like lists and dicts more safely"""
    if pd.isna(s) or not isinstance(s, str):
        return []
    try:
        return ast.literal_eval(s)
    except (ValueError, SyntaxError):
        return []


def convert_row_to_benchmark_summary(row: pd.Series) -> BenchmarkSummary:
    """
    Convert a DataFrame row to a BenchmarkSummary object.

    Parameters
    ----------
    row : pd.Series
        A row from the summary results DataFrame

    Returns
    -------
    BenchmarkSummary
        A BenchmarkSummary object
    """
    return BenchmarkSummary(
        model_name=row["model_name"],
        success_rate=float(row["success_rate"]),
        avg_time=float(row["avg_time"]),
        total_extra_tool_calls_used=int(row["total_extra_tool_calls_used"]),
        total_tasks=row["total_tasks"],
    )


def load_detailed_data(file_path: str, benchmark: str) -> List[BaseModel]:
    df = pd.read_csv(file_path)  # type: ignore
    task_results: List[TaskResult] = []

    converter = BENCHMARKS_CONVERTERS[benchmark]
    for _, row in df.iterrows():  # type: ignore
        task_result = converter(row)
        task_results.append(task_result)

    return task_results


def load_summary_data(file_path: str) -> List[BenchmarkSummary]:
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
    summaries: List[BenchmarkSummary] = []
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
) -> Optional[Tuple[List[TaskResult], List[BenchmarkSummary]]]:
    """
    Load task results and benchmark summaries from a single run directory.
    Returns
    -------
    Optional[Tuple[List[TaskResult], List[BenchmarkSummary]]]
        Tuple of task results and benchmark summaries, or None if loading fails
    """
    detailed_path = os.path.join(path, DETAILED_FILE_NAME)
    summary_path = os.path.join(path, SUMMARY_FILE_NAME)

    if not os.path.exists(detailed_path) or not os.path.exists(summary_path):
        st.warning(f"Missing files in run directory: {path}")
        return None

    task_results = load_detailed_data(detailed_path, benchmark=benchmark)
    benchmark_summaries = load_summary_data(summary_path)

    if not task_results or not benchmark_summaries:
        st.warning(f"Results empty for run: {path}, skipping...")
        return None

    # Verify data consistency
    if task_results and benchmark_summaries:
        task_model_name = task_results[0].model_name
        summary_model_name = benchmark_summaries[0].model_name

        if task_model_name != summary_model_name:
            st.warning(f"Data mismatch in run {path} - model names don't match")
            return None

    return task_results, benchmark_summaries


def load_run_results(parent_dir: str) -> Optional[RunResults]:
    """
    Load all benchmark results from a run directory.

    Returns
    -------
    Optional[RunResults]
        RunResults object containing all benchmark data, or None if loading fails
    """
    run_id = os.path.basename(parent_dir)
    run_results = RunResults(run_id=run_id)

    # List all benchmarks dirs
    for bench_name in os.listdir(parent_dir):
        bench_dir = os.path.join(parent_dir, bench_name)
        if not os.path.isdir(bench_dir):
            continue

        benchmark_results = BenchmarkResults(benchmark_name=bench_name)

        # List all model dirs in benchmark folder
        for model_name in os.listdir(bench_dir):
            model_dir = os.path.join(bench_dir, model_name)
            if not os.path.isdir(model_dir):
                continue

            model_results = ModelResults(model_name=model_name)

            # List all repeats in model dir
            # NOTE (jm) these repeats counts also different extra calls number
            # now the folder names of the repeats is <number_of_available_extra_tool_call>_<repeat>
            for repeat in os.listdir(model_dir):
                repeat_dir = os.path.join(model_dir, repeat)
                if not os.path.isdir(repeat_dir):
                    continue

                run_data = load_single_run(path=repeat_dir, benchmark=bench_name)
                if not run_data:
                    continue

                # Unpack task results and benchmark summaries
                task_results, benchmark_summaries = run_data

                # Create model run results
                model_run = ModelRunResults(run_id=repeat)
                model_run.task_results = task_results
                model_run.benchmark_summaries = benchmark_summaries

                model_results.runs.append(model_run)

            if model_results.runs:
                benchmark_results.models[model_name] = model_results

        if benchmark_results.models:
            run_results.benchmarks[bench_name] = benchmark_results

    if not run_results.benchmarks:
        return None

    return run_results
