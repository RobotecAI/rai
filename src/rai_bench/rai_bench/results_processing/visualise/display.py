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

import pandas as pd
import streamlit as st

from rai_bench.results_processing.data_loading import (
    RunResults,
)
from rai_bench.results_processing.data_processing import (
    create_aggregate_model_metrics,
)
from rai_bench.results_processing.visualise.charts import create_bar_chart


def display_models_summ_data(df: pd.DataFrame):
    """Display summary data charts for models."""
    if df.empty:
        st.warning("No summary data available for visualization.")
        return

    fig1 = create_bar_chart(
        df=df,
        x_column="model_name",
        y_column="avg_success_rate",
        title="Success Rate by Model",
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
        title="Average Completion Time by Model",
        x_label="Model Name",
        y_label="Avg Time (seconds)",
        color_column="model_name",
        count_column="total_tasks",
    )
    st.plotly_chart(fig2, use_container_width=True)  # type: ignore

    if "total_extra_tool_calls_used" in df.columns:
        fig3 = create_bar_chart(
            df=df,
            x_column="model_name",
            y_column="total_extra_tool_calls_used",
            title="Total Extra Tool Calls Used by Model",
            x_label="Model Name",
            y_label="Total Extra Tool Calls",
            color_column="model_name",
            count_column="total_tasks",
        )
        st.plotly_chart(fig3, use_container_width=True)  # type: ignore


def display_aggregate_model_metrics(df: pd.DataFrame):
    """Display aggregate model metrics across all benchmarks."""
    if df.empty:
        st.warning("No aggregate data available for visualization.")
        return
    fig1 = create_bar_chart(
        df=df,
        x_column="model_name",
        y_column="avg_success_rate",
        title="Average Success Rate by Model (Across All Benchmarks)",
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
        title="Average Completion Time by Model (Across All Benchmarks)",
        x_label="Model Name",
        y_label="Avg Time (seconds)",
        color_column="model_name",
        count_column="total_tasks",
    )
    st.plotly_chart(fig2, use_container_width=True)  # type: ignore


def render_overall_performance(run_results: RunResults):
    """Render the overall performance summary across all benchmarks."""
    aggregate_df = create_aggregate_model_metrics(run_results)
    if aggregate_df.empty:
        st.warning("No aggregate data available.")
    else:
        st.subheader("Aggregate Performance Metrics")
        st.write(
            "These charts show average metrics across all benchmarks for each model"
        )
        display_aggregate_model_metrics(aggregate_df)
