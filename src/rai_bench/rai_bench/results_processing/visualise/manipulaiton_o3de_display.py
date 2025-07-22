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

from typing import List

import pandas as pd
import streamlit as st

from rai_bench.manipulation_o3de.results_tracking import ScenarioResult
from rai_bench.results_processing.data_loading import BenchmarkResults, ModelResults
from rai_bench.results_processing.data_processing import (
    create_model_summary_dataframe,
)
from rai_bench.results_processing.visualise.charts import create_bar_chart
from rai_bench.results_processing.visualise.display import display_models_summ_data


def get_all_detailed_results_from_model_results(
    model_results: ModelResults,
) -> List[ScenarioResult]:
    all_detailed_results: List[ScenarioResult] = []
    for run in model_results.runs:
        all_detailed_results.extend(run.task_results)
    return all_detailed_results


def create_scenario_metrics_dataframe(
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
                "number_of_tool_calls": result.number_of_tool_calls,
            }
            for result in all_detailed_results
        ]
    )
    total_tasks = temp_df.groupby(group_by).size().reset_index(name="total_tasks")  # type: ignore
    agg_dict = {
        "score": "mean",
        "total_time": "mean",
        "number_of_tool_calls": "mean",
    }

    grouped = temp_df.groupby(group_by).agg(agg_dict).reset_index()  # type: ignore
    grouped = pd.merge(grouped, total_tasks, on=group_by, how="left")  # type: ignore

    return grouped


def display_scenario_performance_by_level(model_results: ModelResults):
    scenario_level_df = create_scenario_metrics_dataframe(
        model_results=model_results, group_by="level"
    )
    fig_type_score = create_bar_chart(
        df=scenario_level_df,
        x_column="level",
        y_column="score",
        title="Success Rate per Scenario Level",
        x_label="Scenario Level",
        y_label="Avg Score",
        y_range=(0.0, 1.0),
        count_column="total_tasks",
    )
    st.plotly_chart(fig_type_score, use_container_width=True)  # type: ignore

    fig_type_calls = create_bar_chart(
        df=scenario_level_df,
        x_column="level",
        y_column="number_of_tool_calls",
        title="Avg Tool Calls Used per Scenario Level",
        x_label="Scenario Level",
        y_label="Avg Tool Calls Used",
        count_column="total_tasks",
    )
    st.plotly_chart(fig_type_calls, use_container_width=True)  # type: ignore

    fig_type_calls = create_bar_chart(
        df=scenario_level_df,
        x_column="level",
        y_column="total_time",
        title="Avg Time taken per Scenario Level",
        x_label="Scenario Level",
        y_label="Avg Time taken Used",
        count_column="total_tasks",
    )
    st.plotly_chart(fig_type_calls, use_container_width=True)  # type: ignore


def display_scenario_performance_by_task(model_results: ModelResults):
    scenario_level_df = create_scenario_metrics_dataframe(
        model_results=model_results, group_by="task_prompt"
    )
    fig_type_score = create_bar_chart(
        df=scenario_level_df,
        x_column="task_prompt",
        y_column="score",
        title="Success Rate per Task",
        x_label="Task",
        y_label="Avg Score",
        y_range=(0.0, 1.0),
        count_column="total_tasks",
    )
    st.plotly_chart(fig_type_score, use_container_width=True)  # type: ignore

    fig_type_calls = create_bar_chart(
        df=scenario_level_df,
        x_column="task_prompt",
        y_column="number_of_tool_calls",
        title="Avg Tool Calls Used per Task",
        x_label="Task",
        y_label="Avg Tool Calls Used",
        count_column="total_tasks",
    )
    st.plotly_chart(fig_type_calls, use_container_width=True)  # type: ignore

    fig_type_calls = create_bar_chart(
        df=scenario_level_df,
        x_column="level",
        y_column="total_time",
        title="Avg Time taken per Task",
        x_label="Task",
        y_label="Avg Time taken Used",
        count_column="total_tasks",
    )
    st.plotly_chart(fig_type_calls, use_container_width=True)  # type: ignore


def render_model_performance_tab(bench_results: BenchmarkResults):
    st.header("Model Performance")

    summ_df = create_model_summary_dataframe(bench_results)
    display_models_summ_data(summ_df)


def render_scenario_performance_tab(bench_results: BenchmarkResults):
    """Render the task performance across models tab."""
    st.header("Performance Across Tasks")

    model_names = list(bench_results.models.keys())
    if not model_names:
        st.warning(f"No models available for benchmark: {bench_results.benchmark_name}")
        return

    model_selected = st.radio("Model:", model_names, key="task_model")
    model_results = bench_results.models[model_selected]

    st.subheader(f"Model: {model_selected}")
    st.write(f"Data aggregated across {len(model_results.runs)} runs")

    st.subheader("Performance by Scenario Level")
    display_scenario_performance_by_level(model_results)

    # st.subheader("Performance by Task Complexity")
    # display_task_complexity_performance(model_results)

    # st.subheader("Detailed Task Type Analysis")
    # task_types = get_unique_values_from_results(model_results, "type")

    # if not task_types:
    #     st.warning("No task types available.")
    #     return

    # selected_type = st.selectbox(
    #     "Select Task Type", sorted(task_types), key="task_type"
    # )
    # display_detailed_task_type_analysis(model_results, selected_type)


def render_manipulation_o3de(bench_results: BenchmarkResults):
    tabs = st.tabs(["Model Performance", "Scenarios Analysis"])
    with tabs[0]:
        render_model_performance_tab(bench_results)
    with tabs[1]:
        render_scenario_performance_tab(bench_results)
