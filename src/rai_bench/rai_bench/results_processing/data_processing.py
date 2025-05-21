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
import json
from typing import Any, Dict, List, Optional, Set

import pandas as pd

from rai_bench.results_processing.data_loading import (
    BenchmarkResults,
    ModelResults,
    RunResults,
)
from rai_bench.tool_calling_agent.results_tracking import (
    TaskResult,
)


def create_model_summary_dataframe(benchmark: BenchmarkResults) -> pd.DataFrame:
    """
    Create a summary DataFrame for model performance visualization.

    Parameters
    ----------
    benchmark : BenchmarkResults
        The benchmark results object

    Returns
    -------
    pd.DataFrame
        DataFrame with model performance metrics
    """
    rows: List[Dict[str, Any]] = []

    for model_name, model_results in benchmark.models.items():
        metrics: Dict[str, Any] = {
            "model_name": model_name,
            "avg_success_rate": model_results.avg_success_rate,
            "avg_time": model_results.avg_time,
            "total_tasks": model_results.total_tasks,
        }

        benchmark_name = benchmark.benchmark_name
        if benchmark_name == "tool_calling_agent" and hasattr(
            model_results, "avg_extra_tool_calls"
        ):
            metrics["total_extra_tool_calls_used"] = model_results.avg_extra_tool_calls

        rows.append(metrics)

    return pd.DataFrame(rows)


def create_extra_calls_dataframe(
    benchmark: BenchmarkResults, extra_calls: int
) -> pd.DataFrame:
    """
    Create a DataFrame for extra tool calls analysis.

    Parameters
    ----------
    benchmark : BenchmarkResults
        The benchmark results object
    extra_calls : int
        The number of extra tool calls to filter by

    Returns
    -------
    pd.DataFrame
        DataFrame with model performance metrics for the specified extra call count
    """
    rows: List[Dict[str, Any]] = []

    for model_name, model_results in benchmark.models.items():
        all_detailed_results: List[TaskResult] = []
        for run in model_results.runs:
            all_detailed_results.extend(run.task_results)

        # Filter by extra tool calls
        filtered_results = [
            r for r in all_detailed_results if r.extra_tool_calls == extra_calls
        ]

        if not filtered_results:
            continue

        avg_score = sum(r.score for r in filtered_results) / len(filtered_results)
        avg_time = sum(r.total_time for r in filtered_results) / len(filtered_results)
        avg_extra_tool_calls = sum(
            r.extra_tool_calls_used for r in filtered_results
        ) / len(filtered_results)

        rows.append(
            {
                "model_name": model_name,
                "extra_tool_calls": extra_calls,
                "avg_success_rate": avg_score * 100,
                "avg_time": avg_time,
                "avg_extra_tool_calls_used": avg_extra_tool_calls,
                "total_tasks": len(filtered_results),
            }
        )

    return pd.DataFrame(rows)


def get_all_detailed_results_from_model_results(
    model_results: ModelResults,
) -> List[TaskResult]:
    all_detailed_results: List[TaskResult] = []
    for run in model_results.runs:
        all_detailed_results.extend(run.task_results)
    return all_detailed_results


def create_task_metrics_dataframe(
    model_results: ModelResults, group_by: str
) -> pd.DataFrame:
    """
    Create a DataFrame with task metrics grouped by a specific attribute.

    Parameters
    ----------
    model_results : ModelResults
        The model results object
    group_by : str
        Attribute to group by (e.g., 'complexity', 'type')

    Returns
    -------
    pd.DataFrame
        DataFrame with grouped task metrics
    """
    all_detailed_results = get_all_detailed_results_from_model_results(
        model_results=model_results
    )

    if not all_detailed_results:
        return pd.DataFrame()

    temp_df = pd.DataFrame(
        [
            {
                group_by: getattr(result, group_by),
                "score": result.score,
                "total_time": result.total_time,
                "extra_tool_calls_used": result.extra_tool_calls_used,
            }
            for result in all_detailed_results
        ]
    )
    total_tasks = temp_df.groupby(group_by).size().reset_index(name="total_tasks")  # type: ignore
    agg_dict = {
        "score": "mean",
        "total_time": "mean",
        "extra_tool_calls_used": "mean",
    }

    grouped = temp_df.groupby(group_by).agg(agg_dict).reset_index()  # type: ignore

    grouped = grouped.rename(
        columns={
            "score": "avg_score",
            "total_time": "avg_time",
            "extra_tool_calls_used": "avg_extra_tool_calls",
        }
    )
    grouped = pd.merge(grouped, total_tasks, on=group_by, how="left")  # type: ignore

    return grouped


def create_task_details_dataframe(
    model_results: ModelResults, task_type: Optional[str] = None
) -> pd.DataFrame:
    """
    Create a DataFrame with task details, optionally filtered by task type.

    Parameters
    ----------
    model_results : ModelResults
        The model results object
    task_type : Optional[str]
        Task type to filter by

    Returns
    -------
    pd.DataFrame
        DataFrame with task details
    """
    all_detailed_results = get_all_detailed_results_from_model_results(
        model_results=model_results
    )

    if not all_detailed_results:
        return pd.DataFrame()

    # filter by task type
    if task_type:
        all_detailed_results = [r for r in all_detailed_results if r.type == task_type]

    rows: List[Dict[str, Any]] = [
        {
            "task_prompt": result.task_prompt,
            "type": result.type,
            "complexity": result.complexity,
            "score": result.score,
            "total_time": result.total_time,
            "extra_tool_calls_used": result.extra_tool_calls_used,
        }
        for result in all_detailed_results
    ]

    return pd.DataFrame(rows)


def get_unique_values_from_results(
    model_results: ModelResults, attribute: str
) -> List[Any]:
    """
    Get unique values for a specific attribute from model results.

    Parameters
    ----------
    model_results : ModelResults
        The model results object
    attribute : str
        The attribute name to extract unique values from

    Returns
    -------
    List[Any]
        List of unique values for the specified attribute
    """
    all_detailed_results = get_all_detailed_results_from_model_results(
        model_results=model_results
    )

    if not all_detailed_results:
        return []

    # extract unique values
    values: Set[Any] = set()
    for result in all_detailed_results:
        if hasattr(result, attribute):
            values.add(getattr(result, attribute))

    return sorted(list(values))


def get_available_extra_tool_calls(benchmark: BenchmarkResults) -> List[int]:
    """
    Get a list of available extra tool calls values in the benchmark.

    Parameters
    ----------
    benchmark : BenchmarkResults
        The benchmark results object

    Returns
    -------
    List[int]
        List of unique extra tool calls values
    """
    extra_calls_values: Set[int] = set()

    for model_results in benchmark.models.values():
        for run in model_results.runs:
            for result in run.task_results:
                extra_calls_values.add(result.extra_tool_calls)

    return sorted(list(extra_calls_values))


def analyze_subtasks(
    benchmark: BenchmarkResults, model_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Analyze subtasks for a specific model in a benchmark.

    Parameters
    ----------
    benchmark : BenchmarkResults
        The benchmark results object
    model_name : str
        Name of the model to analyze

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]
        DataFrames with subtask summaries, tool summaries, and error summaries
    """
    model_results = benchmark.models.get(model_name)
    if not model_results:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    subtask_stats: Dict[str, Dict[str, Any]] = {}
    error_stats: Dict[str, Dict[str, Any]] = {}

    for run in model_results.runs:
        for task_result in run.task_results:
            for validation_info in task_result.validation_info:
                for subtask in validation_info.subtasks:
                    # Get key parameters
                    args = subtask.args
                    expected_tool_name = args.get("expected_tool_name", "")

                    # Create a unique key for this subtask
                    rest_of_args = args.copy()
                    if "expected_tool_name" in rest_of_args:
                        del rest_of_args["expected_tool_name"]
                    key = json.dumps(args, sort_keys=True)

                    if key not in subtask_stats:
                        subtask_stats[key] = {
                            "expected_tool_name": expected_tool_name,
                            "args": rest_of_args,
                            "count": 0,
                            "pass": 0,
                        }
                    subtask_stats[key]["count"] += 1
                    if subtask.passed:
                        subtask_stats[key]["pass"] += 1

                    for error in subtask.errors:
                        if error not in error_stats:
                            error_stats[error] = {
                                "count": 1,
                                "expected_tool_name": expected_tool_name,
                                "args": rest_of_args,
                            }
                        else:
                            error_stats[error]["count"] += 1

    # by subtask summary df
    subtask_rows: List[Dict[str, Any]] = []
    for key, stats in subtask_stats.items():
        count = stats["count"]
        passed = stats["pass"]
        subtask_rows.append(
            {
                "expected_tool_name": stats["expected_tool_name"],
                "subtask_args": json.dumps(stats["args"], ensure_ascii=False),
                "total_runs": count,
                "passed": passed,
                "pass_rate (%)": round(passed / count * 100, 1) if count else 0.0,
            }
        )

    subtask_df = pd.DataFrame(subtask_rows)
    if not subtask_df.empty:
        subtask_df = subtask_df.sort_values("total_runs", ascending=False).reset_index(  # type: ignore
            drop=True
        )

    # by tool name summary df
    if not subtask_df.empty:
        tool_name_df = (
            subtask_df.groupby("expected_tool_name")  # type: ignore
            .agg(
                total_runs=("total_runs", "sum"),
                passed=("passed", "sum"),
            )
            .reset_index()
        )

        tool_name_df["pass_rate (%)"] = (
            tool_name_df["passed"] / tool_name_df["total_runs"] * 100
        ).round(1)

        tool_name_df = tool_name_df.sort_values(  # type: ignore
            "total_runs", ascending=False
        ).reset_index(drop=True)
    else:
        tool_name_df = pd.DataFrame()

    # by error summary df
    error_rows: List[Dict[str, Any]] = []
    for error, stats in error_stats.items():
        error_rows.append(
            {
                "error": error,
                "count": stats["count"],
                "expected_tool_name": stats["expected_tool_name"],
                "args": json.dumps(stats["args"], ensure_ascii=False),
            }
        )

    error_df = pd.DataFrame(error_rows)
    if not error_df.empty:
        error_df = error_df.sort_values("count", ascending=False).reset_index(drop=True)  # type: ignore

    return subtask_df, tool_name_df, error_df


def create_aggregate_model_metrics(run_results: RunResults) -> pd.DataFrame:
    """
    Create a dataframe with aggregated metrics for each model across all benchmarks.

    Parameters
    ----------
    run_results : RunResults
        The RunResults object containing all benchmark data

    Returns
    -------
    pd.DataFrame
        A dataframe with aggregated metrics for each model
    """
    # Dictionary to store aggregated metrics for each model
    model_metrics: Dict[str, Dict[str, Any]] = {}

    for benchmark_name, benchmark in run_results.benchmarks.items():
        for model_name, model_results in benchmark.models.items():
            if not model_results.check_if_summaries_present():
                continue

            if model_name not in model_metrics:
                model_metrics[model_name] = {
                    "success_rate_sum": 0.0,
                    "time_sum": 0.0,
                    "total_tasks": 0,
                    "count": 0,
                }

            # Sum up metrics for this model from this benchmark
            metrics = model_metrics[model_name]
            metrics["success_rate_sum"] += (
                model_results.avg_success_rate * model_results.count
            )
            metrics["time_sum"] += model_results.avg_time * model_results.count
            metrics["count"] += model_results.count
            metrics["total_tasks"] += model_results.total_tasks

            # Benchmark specific metrics
            if benchmark_name == "tool_calling_agent":
                if hasattr(model_results, "extra_tool_calls_sum"):
                    metrics["extra_tool_calls_sum"] += (
                        model_results.avg_extra_tool_calls * model_results.count
                    )
                else:
                    metrics["extra_tool_calls_sum"] = (
                        model_results.avg_extra_tool_calls * model_results.count
                    )
    # Convert to dataframe and calculate averages
    result_data: List[Dict[str, Any]] = []
    for model_name, metrics in model_metrics.items():
        count = metrics["count"]
        if count > 0:
            avgs: Dict[str, Any] = {
                "model_name": model_name,
                "avg_success_rate": (metrics["success_rate_sum"] / count),
                "avg_time": metrics["time_sum"] / count,
                "count": count,
                "total_tasks": metrics["total_tasks"],
            }
            if "extra_tool_calls_sum" in metrics:
                avgs["avg_extra_tool_calls"] = (
                    metrics["extra_tool_calls_sum"] / count,
                )

            result_data.append(avgs)

    return pd.DataFrame(result_data)
