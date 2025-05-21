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

from rai_bench.results_processing.data_loading import (
    BenchmarkResults,
    ModelResults,
)
from rai_bench.results_processing.data_processing import (
    analyze_subtasks,
    create_extra_calls_dataframe,
    create_model_summary_dataframe,
    create_task_details_dataframe,
    create_task_metrics_dataframe,
    get_available_extra_tool_calls,
    get_unique_values_from_results,
)
from rai_bench.results_processing.visualise.charts import create_bar_chart, wrap_text
from rai_bench.results_processing.visualise.display import display_models_summ_data
from rai_bench.tool_calling_agent.results_tracking import (
    TaskResult,
)


def display_models_extra_calls_data(df: pd.DataFrame, extra_calls: int):
    """Display data for models with specific extra tool calls."""
    if df.empty:
        st.warning(f"No data available for {extra_calls} extra tool calls.")
        return

    fig1 = create_bar_chart(
        df=df,
        x_column="model_name",
        y_column="avg_success_rate",
        title=f"Success Rate by Model (Extra Tool Calls: {extra_calls})",
        x_label="Model Name",
        y_label="Success Rate (%)",
        color_column="model_name",
        y_range=(0.0, 100.0),
        count_column="total_tasks",
    )
    st.plotly_chart(fig1, use_container_width=True)  # type: ignore

    fig2 = create_bar_chart(
        df=df,
        x_column="model_name",
        y_column="avg_time",
        title=f"Avg Completion Time by Model (Extra Tool Calls: {extra_calls})",
        x_label="Model Name",
        y_label="Avg Time (s)",
        color_column="model_name",
        count_column="total_tasks",
    )
    st.plotly_chart(fig2, use_container_width=True)  # type: ignore


def display_task_type_performance(model_results: ModelResults):
    """Display performance charts by task type."""
    task_type_df = create_task_metrics_dataframe(model_results, "type")

    if task_type_df.empty:
        st.warning("No task type data available.")
        return

    fig_type_score = create_bar_chart(
        df=task_type_df,
        x_column="type",
        y_column="avg_score",
        title="Success Rate by Task Type",
        x_label="Task Type",
        y_label="Avg Score",
        y_range=(0.0, 1.0),
        count_column="total_tasks",
    )
    st.plotly_chart(fig_type_score, use_container_width=True)  # type: ignore

    fig_type_calls = create_bar_chart(
        df=task_type_df,
        x_column="type",
        y_column="avg_extra_tool_calls",
        title="Avg Extra Tool Calls Used by Task Type",
        x_label="Task Type",
        y_label="Avg Extra Tool Calls Used",
        count_column="total_tasks",
    )
    st.plotly_chart(fig_type_calls, use_container_width=True)  # type: ignore


def display_task_complexity_performance(model_results: ModelResults):
    """Display performance charts by task complexity."""
    complexity_df = create_task_metrics_dataframe(model_results, "complexity")

    if complexity_df.empty:
        st.warning("No complexity data available.")
        return

    fig_complexity_score = create_bar_chart(
        df=complexity_df,
        x_column="complexity",
        y_column="avg_score",
        title="Success Rate by Task Complexity",
        x_label="Task Complexity",
        y_label="Avg Score",
        y_range=(0.0, 1.0),
        count_column="total_tasks",
    )
    st.plotly_chart(fig_complexity_score, use_container_width=True)  # type: ignore

    fig_complexity_calls = create_bar_chart(
        df=complexity_df,
        x_column="complexity",
        y_column="avg_extra_tool_calls",
        title="Avg Extra Tool Calls Used by Task Complexity",
        x_label="Task Complexity",
        y_label="Avg Extra Tool Calls Used",
        count_column="total_tasks",
    )
    st.plotly_chart(fig_complexity_calls, use_container_width=True)  # type: ignore


def display_detailed_task_type_analysis(
    model_results: ModelResults, selected_type: str
):
    """Display detailed analysis for a specific task type."""
    # first, get only the tasks of the selected type
    tasks_for_type_df = create_task_details_dataframe(model_results, selected_type)
    if tasks_for_type_df.empty:
        st.warning(f"No tasks of type {selected_type} found.")
        return
    # Now aggregate by complexity for that type
    filtered_by_complexity = (
        tasks_for_type_df.groupby("complexity")  # type: ignore
        .agg(
            avg_score=("score", "mean"),
            avg_time=("total_time", "mean"),
            avg_extra_tool_calls=("extra_tool_calls_used", "mean"),
        )
        .reset_index()
    )
    filtered_by_complexity = filtered_by_complexity[
        filtered_by_complexity["complexity"].notna()
    ]

    # Display success rate by complexity for the selected task type
    if not filtered_by_complexity.empty:
        fig_complexity_score = create_bar_chart(
            df=filtered_by_complexity,
            x_column="complexity",
            y_column="avg_score",
            title=f"Success Rate by Task Complexity for '{selected_type}' Tasks",
            x_label="Task Complexity",
            y_label="Avg Score",
            y_range=(0.0, 1.0),
            count_column="total_tasks",
        )
        st.plotly_chart(fig_complexity_score, use_container_width=True)  # type: ignore

    # Display success rate by individual task
    task_details_df = create_task_details_dataframe(model_results, selected_type)

    if task_details_df.empty:
        st.warning(f"No task details available for type: {selected_type}")
        return

    task_stats = (
        task_details_df.groupby("task_prompt")  # type: ignore
        .agg({"score": "mean", "total_time": "mean"})
        .reset_index()
    )

    # Replace zero scores with a small value to enable hover
    task_stats = task_stats.replace(0, 0.01)  # type: ignore

    # wrapped text column for hover, so that text is multiline
    task_stats["wrapped_prompt"] = task_stats["task_prompt"].apply(wrap_text)  # type: ignore
    task_stats["score"] = task_stats["score"].round(2)
    task_stats["total_time"] = task_stats["total_time"].round(2)
    # Create short labels for x-axis
    short_labels: List[str] = [
        (t[:30] + "...") if len(t) > 30 else t for t in task_stats["task_prompt"]
    ]
    st.warning(
        "The 0.01 values corresponds to 0. They were set to 0.01 to enable hovering over them so you can see the whole task prompt"
    )
    fig_task = create_bar_chart(
        df=task_stats,
        x_column="task_prompt",
        y_column="score",
        title=f"Avg Score for '{selected_type}' Tasks",
        x_label="Task",
        y_label="Avg Score",
        custom_data=["wrapped_prompt", "score"],
        hover_template="<b>Task:</b> %{customdata[0]}\n <b>Score:</b> %{customdata[1]}",
        y_range=(0.0, 1.0),
        x_tickvals=task_stats["task_prompt"].tolist(),
        x_ticktext=short_labels,
        count_column="total_tasks",
    )
    st.plotly_chart(fig_task, use_container_width=True)  # type: ignore

    fig_time = create_bar_chart(
        df=task_stats,
        x_column="task_prompt",
        y_column="total_time",
        title=f"Avg Time for '{selected_type}' Tasks",
        x_label="Task",
        y_label="Avg Time (s)",
        custom_data=["wrapped_prompt", "total_time"],
        hover_template="<b>Task:</b> %{customdata[0]}\n <b>Time:</b> %{customdata[1]}",
        x_tickvals=task_stats["task_prompt"].tolist(),
        x_ticktext=short_labels,
        count_column="total_tasks",
    )
    st.plotly_chart(fig_time, use_container_width=True)  # type: ignore


def display_subtask_analysis(
    subtask_df: pd.DataFrame, tool_name_df: pd.DataFrame, error_df: pd.DataFrame
):
    """Display subtask analysis tables."""
    # Subtasks table
    st.subheader("Subtasks")
    st.dataframe(subtask_df, use_container_width=True)  # type: ignore

    # Tool name summary
    st.subheader("Subtasks grouped by tool name")
    st.dataframe(tool_name_df, use_container_width=True)  # type: ignore

    # Error summary
    st.subheader("Errors by subtask")
    st.dataframe(error_df, use_container_width=True)  # type: ignore

    # Additional analysis: group errors by tool name
    if not error_df.empty:
        error_by_tool_df = (
            error_df.groupby("expected_tool_name")  # type: ignore
            .agg(
                errors_count=("count", "sum"),
            )
            .reset_index()
            .sort_values("errors_count", ascending=False)
        )

        st.subheader("Errors grouped by tool name")
        st.dataframe(error_by_tool_df, use_container_width=True)  # type: ignore


def display_validator_results(task_result: TaskResult):
    """Display validation results for a single task."""
    col1, col2 = st.columns(2)

    # Task overview in the first column
    col1.markdown("#### Task Overview")
    col1.markdown(f"**Type:** {task_result.type}")
    col1.markdown(f"**Complexity:** {task_result.complexity}")
    col1.markdown(f"**Score:** {task_result.score}")
    col1.markdown(f"**Time:** {task_result.total_time:.2f}s")
    col1.markdown(f"**Extra Tool Calls Used:** {task_result.extra_tool_calls_used}")

    # Validation details in the second column
    col2.markdown("#### Validation Results")
    if not task_result.validation_info:
        col2.warning("No validation information available.")
    else:
        for idx, validator in enumerate(task_result.validation_info):
            col2.markdown(f"**Validator {idx + 1}:** {validator.type}")
            col2.markdown(f"**Passed:** {'✅' if validator.passed else '❌'}")

            # Subtasks table
            if validator.subtasks:
                subtask_data = []
                for subtask_idx, subtask in enumerate(validator.subtasks):
                    expected_tool = subtask.args.get("expected_tool_name", "Unknown")
                    subtask_data.append(
                        {
                            "Subtask": subtask_idx + 1,
                            "Expected Tool": expected_tool,
                            "Passed": "✅" if subtask.passed else "❌",
                            "Errors": (
                                ", ".join(subtask.errors) if subtask.errors else "None"
                            ),
                        }
                    )

                col2.dataframe(pd.DataFrame(subtask_data), use_container_width=True)
            else:
                col2.info("No subtasks in this validator.")


def render_tool_calling_model_performance_tab(bench_results: BenchmarkResults):
    st.header("Model Performance")

    summ_df = create_model_summary_dataframe(bench_results)
    display_models_summ_data(summ_df)

    # Extra tool calls analysis
    st.subheader("Performance by Extra Tool Calls")
    extra_calls_values = get_available_extra_tool_calls(bench_results)

    if not extra_calls_values:
        st.warning("No extra tool calls data available.")
        return

    extra_tool_calls_selected = st.radio(
        "Maximum extra tool calls:", extra_calls_values, key="extra_tool_calls"
    )

    extra_calls_df = create_extra_calls_dataframe(
        benchmark=bench_results, extra_calls=extra_tool_calls_selected
    )

    display_models_extra_calls_data(
        df=extra_calls_df, extra_calls=extra_tool_calls_selected
    )


def render_task_performance_tab(bench_results: BenchmarkResults):
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

    # Display performance by task type
    st.subheader("Performance by Task Type")
    display_task_type_performance(model_results)

    # Display performance by complexity
    st.subheader("Performance by Task Complexity")
    display_task_complexity_performance(model_results)

    # Per Task Type Analysis
    st.subheader("Detailed Task Type Analysis")
    task_types = get_unique_values_from_results(model_results, "type")

    if not task_types:
        st.warning("No task types available.")
        return

    selected_type = st.selectbox(
        "Select Task Type", sorted(task_types), key="task_type"
    )
    display_detailed_task_type_analysis(model_results, selected_type)


def render_validator_analysis_tab(bench_results: BenchmarkResults):
    """Render the validator analysis tab."""
    st.header("Validator Analysis")

    model_names = list(bench_results.models.keys())
    if not model_names:
        st.warning(f"No models available for benchmark: {bench_results.benchmark_name}")
        return

    model_selected = st.selectbox("Model", model_names, key="val_model")
    model_results = bench_results.models[model_selected]

    # Run selection (if there are multiple runs)
    if len(model_results.runs) > 1:
        run_ids = [f"{run.run_id}" for run in model_results.runs]
        run_selected_idx = st.selectbox(
            "Run", range(len(run_ids)), format_func=lambda x: run_ids[x]
        )
        run = model_results.runs[run_selected_idx]
    else:
        run = model_results.runs[0]

    if len(run.repeats) > 1:
        repeat_ids = [f"{repeat.repeat_num}" for repeat in run.repeats]
        repeat_selected_idx = st.selectbox(
            "Repeat",
            range(len(repeat_ids)),
            format_func=lambda x: repeat_ids[x],
            key="val_repeat",
        )
        repeat = run.repeats[repeat_selected_idx]
    else:
        repeat = run.repeats[0]

    st.subheader("Per-Task Validators & Subtasks")

    # Display detailed validation results for each task
    for task_result in repeat.task_results:
        with st.expander(f"Task: {task_result.task_prompt}"):
            display_validator_results(task_result)


def render_subtask_analysis_tab(bench_results: BenchmarkResults):
    """Render the subtask analysis tab."""
    st.header("Subtask Analysis")

    # Model selection
    model_names = list(bench_results.models.keys())
    if not model_names:
        st.warning(f"No models available for benchmark: {bench_results.benchmark_name}")
        return

    model_selected = st.selectbox("Model", model_names, key="sub_model")

    # Analyze subtasks
    with st.spinner("Analyzing subtasks..."):
        subtask_df, tool_name_df, error_df = analyze_subtasks(
            bench_results, model_selected
        )

    # Display results
    if subtask_df.empty:
        st.warning("No subtask data available for analysis.")
        return

    # Subtasks table
    st.subheader("Subtasks")
    st.dataframe(subtask_df, use_container_width=True)  # type:ignore

    # Tool name summary
    st.subheader("Subtasks grouped by tool name")
    st.dataframe(tool_name_df, use_container_width=True)  # type:ignore

    # Error summary
    st.subheader("Errors by subtask")
    st.dataframe(error_df, use_container_width=True)  # type:ignore

    # Additional analysis: group errors by tool name
    if not error_df.empty:
        error_by_tool_df = (
            error_df.groupby("expected_tool_name")  # type:ignore
            .agg(
                errors_count=("count", "sum"),
            )
            .reset_index()
            .sort_values("errors_count", ascending=False)
        )

        st.subheader("Errors grouped by tool name")
        st.dataframe(error_by_tool_df, use_container_width=True)  # type:ignore


def render_tool_calling_agent(bench_results: BenchmarkResults):
    tabs = st.tabs(
        [
            "Model Performance",
            "Task Performance",
            "Validator Analysis",
            "Subtask Analysis",
        ]
    )

    with tabs[0]:
        render_tool_calling_model_performance_tab(bench_results)

    with tabs[1]:
        render_task_performance_tab(bench_results)

    # BUG (jmatecz) the validation_info is not loaded properly as
    # argument that require only type is stored like this in results:
    #   {'timeout_sec': <class 'int'>}},
    # which can't be parsed correctly
    with tabs[2]:
        render_validator_analysis_tab(bench_results)

    with tabs[3]:
        render_subtask_analysis_tab(bench_results)
