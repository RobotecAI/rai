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

import streamlit as st

from rai_bench.results_processing.data_loading import (
    BenchmarkResults,
)
from rai_bench.results_processing.data_processing import (
    create_model_summary_dataframe,
)
from rai_bench.results_processing.visualise.display import display_models_summ_data


def render_manipulation_model_performance_tab(bench_results: BenchmarkResults):
    st.header("Model Performance")

    summ_df = create_model_summary_dataframe(bench_results)
    display_models_summ_data(summ_df)


def render_manipulation_o3de(bench_results: BenchmarkResults):
    tabs = st.tabs(
        [
            "Model Performance",
        ]
    )
    with tabs[0]:
        render_manipulation_model_performance_tab(bench_results)
