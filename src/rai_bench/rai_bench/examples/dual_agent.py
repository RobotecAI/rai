# # Copyright (C) 2025 Robotec.AI
# #
# # Licensed under the Apache License, Version 2.0 (the "License");
# # you may not use this file except in compliance with the License.
# # You may obtain a copy of the License at
# #
# #         http://www.apache.org/licenses/LICENSE-2.0
# #
# # Unless required by applicable law or agreed to in writing, software
# # distributed under the License is distributed on an "AS IS" BASIS,
# # WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# # See the License for the specific language governing permissions and
# # limitations under the License.
from langchain_community.chat_models import ChatOllama
from langchain_openai import ChatOpenAI

from rai_bench import (
    ManipulationO3DEBenchmarkConfig,
    ToolCallingAgentBenchmarkConfig,
    test_dual_agents,
)

if __name__ == "__main__":
    # Define models you want to benchmark
    model_name = "gemma3:4b"
    m_llm = ChatOllama(
        model=model_name, base_url="http://localhost:11434", keep_alive=30
    )

    tool_llm = ChatOpenAI(model="gpt-4o-mini", base_url="https://api.openai.com/v1/")
    # Define benchmarks that will be used
    tool_conf = ToolCallingAgentBenchmarkConfig(
        extra_tool_calls=0,  # how many extra tool calls allowed to still pass
        task_types=["spatial_reasoning"],
        repeats=15,
    )

    man_conf = ManipulationO3DEBenchmarkConfig(
        o3de_config_path="src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml",  # path to your o3de config
        levels=[  # define what difficulty of tasks to include in benchmark
            "trivial",
        ],
        repeats=1,  # how many times to repeat
    )

    out_dir = "src/rai_bench/rai_bench/experiments/dual_agents/"

    test_dual_agents(
        multimodal_llms=[m_llm],
        tool_calling_models=[tool_llm],
        benchmark_configs=[man_conf, tool_conf],
        out_dir=out_dir,
    )
