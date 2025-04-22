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
import json
import os
from typing import Any, Dict, List

import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(layout="wide", page_title="LLM Task Results Visualizer")

EXPERIMENT_DIR = "./src/rai_bench/rai_bench/experiments"
DETAILED_FILE_NAME: str = "results.csv"
SUMMARY_FILE_NAME: str = "results_summary.csv"


def safely_parse_json_like_string(s: Any) -> List[Any]:
    """Parse string representation of Python objects like lists and dicts more safely"""
    if pd.isna(s) or not isinstance(s, str):
        return []
    return ast.literal_eval(s)


def load_detailed_data(file_path: str) -> pd.DataFrame:
    """Load detailed task results data from a file path."""
    df: pd.DataFrame = pd.read_csv(file_path)  # type: ignore

    for col in ["validation_info", "passed", "errors"]:
        if col in df.columns:
            df[col] = df[col].apply(safely_parse_json_like_string)  # type: ignore

    return df


def load_summary_data(file_path: str) -> pd.DataFrame:
    """Load summary results data from a file path."""
    df: pd.DataFrame = pd.read_csv(file_path)  # type: ignore
    return df


def load_all_run_data(
    parent_dir: str,
) -> Dict[str, Dict[str, List[Dict[str, pd.DataFrame]]]] | None:
    """Load data from run folder

    Return: returns a dict structured like:
        bench_name1
            - model_name1
                - run1
                    - detailed_df
                    - summary_df
            - model_name2
                - run1
                ...
        bench_name2
                ...
    """
    run: Dict[str, Dict[str, List[Dict[str, pd.DataFrame]]]] = {}

    # list all benchmarks dirs
    for bench_name in os.listdir(parent_dir):
        bench_dir = os.path.join(parent_dir, bench_name)
        if os.path.isdir(bench_dir):
            run[bench_name] = {}
            # list all model dirs in benchmark folder
            for model_name in os.listdir(bench_dir):
                model_dir = os.path.join(bench_dir, model_name)
                if os.path.isdir(model_dir):
                    run[bench_name][model_name] = []
                    # list all repeats in mode dir
                    for repeat in os.listdir(model_dir):
                        repeat_dir = os.path.join(model_dir, repeat)
                        if os.path.isdir(repeat_dir):
                            run_data = load_single_run(path=repeat_dir)
                            if run_data:
                                run[bench_name][model_name].append(run_data)

                    if not run[bench_name][model_name]:
                        # empty data for model
                        return

    return run


def load_single_run(path: str) -> None | Dict[str, pd.DataFrame]:
    detailed_df = load_detailed_data(os.path.join(path, DETAILED_FILE_NAME))
    summary_df = load_summary_data(os.path.join(path, SUMMARY_FILE_NAME))

    if detailed_df.empty or summary_df.empty:
        print(f"Results empty for run: {path}, skipping...")
        return

    summ_model_name = summary_df["model_name"].iloc[0]  # type: ignore
    if summ_model_name != detailed_df["model_name"].iloc[0]:  # type: ignore
        print(f"Warning: Data mismatch in run {path}")
        return

    run_data = {
        "detailed_df": detailed_df,
        "summary_df": summary_df,
    }
    return run_data


def compute_avg_summary_stats(
    run: Dict[str, Dict[str, List[Dict[str, pd.DataFrame]]]],
) -> Dict[str, Dict[str, Any]]:
    """
    Compute summary statistics from detailed data if summary data is not available
    Return: For every benchmark calculted averages fo every model
    """
    # Group by model_name and calculate stats
    models_stats: Dict[str, Dict[str, Any]] = {}
    for benchmark, models_dict in run.items():
        models_stats[benchmark] = {}
        for model, repeats in models_dict.items():
            combined_df = pd.concat([r["summary_df"] for r in repeats])
            models_stats[benchmark][model] = {
                "avg_success_rate": combined_df["success_rate"].mean(),
                "avg_time": combined_df["avg_time"].mean(),
            }

    return models_stats


def display_model_performance_tab(
    models_stats: Dict[str, Dict[str, Any]],
) -> None:
    """Display overall model performance visualizations"""
    st.header("Model Performance")

    bench_selected: str = st.radio("Benchmark:", list(models_stats.keys()), key=0)
    viz_data: List[Dict[str, Any]] = []
    data = models_stats[bench_selected]
    for model_name, stats in data.items():
        viz_data.append(
            {
                "model_name": model_name,
                "avg_success_rate": stats["avg_success_rate"],
                "avg_time": stats["avg_time"],
            }
        )

    viz_df = pd.DataFrame(viz_data)

    if not viz_df.empty:
        fig1 = px.bar(  # type: ignore
            viz_df,
            x="model_name",
            y="avg_success_rate",
            title="Success Rate by Model",
            labels={
                "avg_success_rate": "Success Rate (%)",
                "model_name": "Model Name",
            },
            color="model_name",
            barmode="group",
        )
        fig1.update_layout(xaxis_tickangle=-45)  # type: ignore
        st.plotly_chart(fig1, use_container_width=True)  # type: ignore

        fig2 = px.bar(  # type: ignore
            viz_df,
            x="model_name",
            y="avg_time",
            title="Average Completion Time by Model",
            labels={"avg_time": "Avg Time (seconds)", "model_name": "Model Name"},
            color="model_name",
            barmode="group",
        )
        fig2.update_layout(xaxis_tickangle=-45)  # type: ignore
        st.plotly_chart(fig2, use_container_width=True)  # type: ignore


def display_performance_across_tasks(
    run: Dict[str, Dict[str, List[Dict[str, pd.DataFrame]]]],
):
    bench_selected: str = st.radio("Benchmark:", list(run.keys()), key=1)
    if bench_selected:
        model_selected: str = st.radio("Model:", list(run[bench_selected].keys()))

        repeats = run[bench_selected][model_selected]
        st.subheader(f"Model: {model_selected}")

        # Combine detailed data from all runs for this model
        all_detailed_dfs = [r["detailed_df"] for r in repeats]
        combined_detailed_df = pd.concat(all_detailed_dfs)

        task_success = (
            combined_detailed_df.groupby("task_prompt")["score"]  # type: ignore
            .mean()
            .reset_index(name="avg_score")
        )

        task_time = (
            combined_detailed_df.groupby("task_prompt")["total_time"]  # type: ignore
            .mean()
            .reset_index(name="avg_time")
        )

        # # TODO (jm) when total tool calls will be available count them here
        # # Count number of times each task was run
        # task_count = (
        #     combined_detailed_df.groupby("task_prompt")
        #     .size()
        #     .reset_index(name="run_count")
        # )

        # TODO (jm) extra calls used
        task_stats = pd.merge(task_success, task_time, on="task_prompt").replace(
            0, 0.01
        )
        fig_task = px.bar(
            task_stats,
            x="task_prompt",
            y="avg_score",
            title=f"Success Rate by Task - Aggregated across {len(repeats)} runs",
            labels={
                "avg_score": "Avg Score",
                "task_prompt": "Task",
            },
            custom_data=["task_prompt"],
        )
        short_labels = [
            text[:30] + "..." if len(text) > 30 else text
            for text in task_stats["task_prompt"]
        ]

        fig_task.update_xaxes(ticktext=short_labels, tickvals=task_stats["task_prompt"])
        fig_task.update_traces(hovertemplate="<b>Task:</b> %{customdata[0]}")
        fig_task.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_task, use_container_width=True)

        fig_time = px.bar(
            task_stats,
            x="task_prompt",
            y="avg_time",
            title=f"Average Time by Task - Aggregated across {len(repeats)} runs",
            labels={
                "avg_time": "Avg Time (seconds)",
                "task_prompt": "Task",
            },
            custom_data=["task_prompt"],
        )
        fig_time.update_xaxes(ticktext=short_labels, tickvals=task_stats["task_prompt"])
        fig_time.update_traces(hovertemplate="<b>Task:</b> %{customdata[0]}")
        fig_time.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig_time, use_container_width=True)


def validators_analysis(run: Dict[str, Dict[str, List[Dict[str, pd.DataFrame]]]]):
    # bench = st.selectbox("Benchmark", list(run.keys()), key="val_bench")
    # model = st.selectbox("Model", list(run[bench].keys()), key="val_model")
    # repeat_df = run[bench][model][0]["detailed_df"]

    # st.markdown("### Per-Task Validators & Subtasks")
    # for idx, values in repeat_df.iterrows():
    #     with st.expander(label=values["task_prompt"]):
    #         container = st.empty()
    #         col1, col2, col3 = container.columns(3)
    #         for val in values["validation_info"]:
    #             # col1.write()
    #             pass
    # st.dataframe(values["validation_info"])
    ...


def subtasks_analysis(run: Dict[str, Dict[str, List[Dict[str, pd.DataFrame]]]]):
    bench = st.selectbox("Benchmark", list(run.keys()), key="sub_bench")
    model = st.selectbox("Model", list(run[bench].keys()), key="sub_model")

    subtask_stats: Dict[str, Dict[str, Any]] = {}
    sorted_by_errors: Dict[str, Dict[str, Any]] = {}
    for repeat in run[bench][model]:
        detailed_df = repeat["detailed_df"]
        for _, row in detailed_df.iterrows():
            validators = row["validation_info"]
            for val in validators:
                subtasks = val["subtasks"]
                for idx, sub in enumerate(subtasks):
                    args = sub["args"]
                    # canonical key so identical dicts collapse
                    key = json.dumps(args, sort_keys=True)
                    passed = sub["passed"]
                    errors: List[str] = sub["errors"]
                    expected_tool_name = sub["args"]["expected_tool_name"]
                    rest_of_args = args.copy()
                    del rest_of_args["expected_tool_name"]

                    if key not in subtask_stats:
                        subtask_stats[key] = {
                            "expected_tool_name": expected_tool_name,
                            "args": rest_of_args,
                            "count": 0,
                            "pass": 0,
                        }
                    subtask_stats[key]["count"] += 1
                    if passed:
                        subtask_stats[key]["pass"] += 1

                    for err in errors:
                        if err not in sorted_by_errors:
                            sorted_by_errors[err] = {
                                "count": 1,
                                "expected_tool_name": expected_tool_name,
                                "args": rest_of_args,
                            }
                        else:
                            sorted_by_errors[err]["count"] += 1

    rows: List[Dict[str, Any]] = []
    for key, stats in subtask_stats.items():
        cnt = stats["count"]
        p = stats["pass"]
        rows.append(
            {
                "expected_tool_name": stats["expected_tool_name"],
                "subtask_args": json.dumps(stats["args"], ensure_ascii=False),
                "total_runs": cnt,
                "passed": p,
                "pass_rate (%)": round(p / cnt * 100, 1) if cnt else 0.0,
            }
        )
    summary_df = (
        pd.DataFrame(rows)
        .sort_values("total_runs", ascending=False)
        .reset_index(drop=True)
    )
    grouped_by_tool_name_df = summary_df.groupby("expected_tool_name").agg(
        total_runs=("total_runs", "sum"),
        passed=("passed", "sum"),
    )
    grouped_by_tool_name_df["pass_rate (%)"] = (
        grouped_by_tool_name_df["passed"] / grouped_by_tool_name_df["total_runs"] * 100
    ).round(1)
    grouped_by_tool_name_df = grouped_by_tool_name_df.sort_values(
        "total_runs", ascending=False
    ).reset_index()

    rows: List[Dict[str, Any]] = []
    for err, stats in sorted_by_errors.items():
        rows.append(
            {
                "error": err,
                "count": stats["count"],
                "expected_tool_name": stats["expected_tool_name"],
                "args": stats["args"],
            }
        )
    errors_df = (
        pd.DataFrame(rows).sort_values("count", ascending=False).reset_index(drop=True)
    )

    grouped_by_subtask_errors_df = errors_df.groupby("expected_tool_name").agg(
        errors_count=("count", "sum"),
    )
    grouped_by_subtask_errors_df = grouped_by_subtask_errors_df.sort_values(
        "errors_count", ascending=False
    ).reset_index()
    st.markdown("### Subtasks")
    st.dataframe(summary_df, use_container_width=True)
    st.markdown("### Subtasks grouped dby tool name")
    st.dataframe(grouped_by_tool_name_df, use_container_width=True)
    st.markdown("### Errors by subtask")
    st.dataframe(errors_df, use_container_width=True)


if __name__ == "__main__":
    st.title("RAI BENCHMARK RESULTS")

    run_folders: List[str] = []
    for d in os.listdir(EXPERIMENT_DIR):
        if os.path.isdir(os.path.join(EXPERIMENT_DIR, d)) and d.startswith("run_"):
            run_folders.append(d)

    selected = st.selectbox("Select run folder", run_folders)
    results_dir = os.path.join(EXPERIMENT_DIR, selected)
    with st.spinner("Loading run data..."):
        run = load_all_run_data(results_dir)

    if run:
        models_stats = compute_avg_summary_stats(run)

        tab1, tab2, tab3, tab4 = st.tabs(
            [
                "Model Performance",
                "Performance Across Tasks",
                "Validator Analysis",
                "Subtask Analysis",
            ]
        )

        with tab1:
            display_model_performance_tab(models_stats)

        with tab2:
            display_performance_across_tasks(run)

        with tab3:
            validators_analysis(run)

        with tab4:
            subtasks_analysis(run)
