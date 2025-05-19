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
import os
from typing import Any, Dict, List, Optional

import streamlit as st

from rai_bench.results_processing.data_loading import (
    RunResults,
    get_available_runs,
    load_multiple_runs,
)
from rai_bench.results_processing.visualise.display import (
    render_overall_performance,
)
from rai_bench.results_processing.visualise.manipulaiton_o3de_display import (
    render_manipulation_o3de,
)
from rai_bench.results_processing.visualise.tool_calling_agent_display import (
    render_tool_calling_agent,
)

EXPERIMENT_DIR = "./src/rai_bench/rai_bench/experiments"
DETAILED_FILE_NAME: str = "results.csv"
SUMMARY_FILE_NAME: str = "results_summary.csv"


BENCHMARK_SECTIONS: Dict[str, Any] = {
    "manipulation_o3de": render_manipulation_o3de,
    "tool_calling_agent": render_tool_calling_agent,
}


def main():
    st.set_page_config(layout="wide", page_title="LLM Task Results Visualizer")
    st.title("RAI BENCHMARK RESULTS")

    run_folders = get_available_runs(EXPERIMENT_DIR)

    if not run_folders:
        st.warning("No benchmark runs found in the experiments directory.")
        return

    if "run_results" not in st.session_state:
        st.session_state.run_results = None
    if "selected_benchmark" not in st.session_state:
        st.session_state.selected_benchmark = None
    if "runs_loaded" not in st.session_state:
        st.session_state.runs_loaded = False
    if "selected_runs" not in st.session_state:
        st.session_state.selected_runs = []

    def update_selected_runs():
        st.session_state.selected_runs = st.session_state.run_selection

    # Function to load data
    def load_run_data():
        selected_dirs: List[str] = []
        for folder in st.session_state.selected_runs:
            selected_dirs.append(os.path.join(EXPERIMENT_DIR, folder))
        with st.spinner("Loading run data..."):
            st.session_state.run_results = load_multiple_runs(selected_dirs)
        st.session_state.runs_loaded = True

    selected = st.multiselect(
        "Select run folder",
        run_folders,
        default=(
            st.session_state.selected_runs if st.session_state.selected_runs else []
        ),
        key="run_selection",
        on_change=update_selected_runs,
    )

    if st.session_state.selected_runs and (
        not st.session_state.runs_loaded or st.button("Load Run Data")
    ):
        load_run_data()

    if st.session_state.runs_loaded and st.session_state.run_results:
        selected_dirs: List[str] = []
        for folder in selected:
            selected_dirs.append(os.path.join(EXPERIMENT_DIR, folder))
        with st.spinner("Loading run data..."):
            st.session_state.run_results = load_multiple_runs(selected_dirs)

        run_results: Optional[RunResults] = st.session_state.run_results

        # Create tabs
        st.markdown("---")
        st.header("Overview")

        # Display overall performance summary across all benchmarks
        render_overall_performance(run_results)

        # Benchmark selection
        st.markdown("---")
        st.header("Benchmark-Specific Analysis")

        benchmark_names = list(run_results.benchmarks.keys())

        selected_benchmark_name = st.selectbox(
            "Select a benchmark for detailed analysis:",
            benchmark_names,
            index=(
                benchmark_names.index(st.session_state.selected_benchmark)
                if st.session_state.selected_benchmark in benchmark_names
                else 0
            ),
        )

        # Store the selected benchmark in session state
        st.session_state.selected_benchmark = selected_benchmark_name
        selected_benchmark = run_results.benchmarks[selected_benchmark_name]

        BENCHMARK_SECTIONS[selected_benchmark_name](selected_benchmark)


if __name__ == "__main__":
    main()
