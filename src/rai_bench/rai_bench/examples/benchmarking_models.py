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

from rai_bench import (
    ManipulationO3DEBenchmarkConfig,
    ToolCallingAgentBenchmarkConfig,
    test_models,
)

if __name__ == "__main__":
    # Define models you want to benchmark
    model_names = ["qwen3:4b", "llama3.2:3b"]
    vendors = ["ollama", "ollama"]

    # Define benchmarks that will be used
    mani_conf = ManipulationO3DEBenchmarkConfig(
        o3de_config_path="src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml",
        levels=[  # define what difficulty of tasks to include in benchmark
            "trivial",
            "easy",
        ],
        repeats=1,  # how many times to repeat
    )
    tool_conf = ToolCallingAgentBenchmarkConfig(
        extra_tool_calls=[0, 5],  # how many extra tool calls allowed to still pass
        task_types=[  # what types of tasks to include
            "basic",
            "custom_interfaces",
        ],
        N_shots=[0, 2],  # examples in system prompt
        prompt_detail=["brief", "descriptive"],  # how descriptive should task prompt be
        repeats=1,
    )

    out_dir = "src/rai_bench/rai_bench/experiments"
    test_models(
        model_names=model_names,
        vendors=vendors,
        benchmark_configs=[mani_conf, tool_conf],
        out_dir=out_dir,
        # if you want to pass any additinal args to model
        additional_model_args=[
            {"reasoning": False},
            {},
        ],
    )
