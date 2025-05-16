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
    model_names = ["qwen2.5:7b", "llama3.2:3b"]
    vendors = ["ollama", "ollama"]

    # Define benchmarks that will be used
    man_conf = ManipulationO3DEBenchmarkConfig(
        o3de_config_path="src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml",  # path to your o3de config
        levels=[  # define what difficulty of tasks to include in benchmark
            "trivial",
        ],
        repeats=1,  # how many times to repeat
    )
    tool_conf = ToolCallingAgentBenchmarkConfig(
        extra_tool_calls=5,  # how many extra tool calls allowed to still pass
        task_types=[  # what types of tasks to include
            "basic",
            "spatial_reasoning",
            "manipulation",
        ],
        repeats=1,
    )
    test_models(
        model_names=model_names,
        vendors=vendors,
        benchmark_configs=[man_conf, tool_conf],
    )
