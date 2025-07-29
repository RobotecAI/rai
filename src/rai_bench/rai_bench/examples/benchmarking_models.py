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

from rai_bench import (
    ManipulationO3DEBenchmarkConfig,
    ToolCallingAgentBenchmarkConfig,
    get_llm_for_benchmark,
    test_agents,
)
from rai_bench.agents import (
    AgentFactory,
    TaskVerificationAgentFactory,
)

if __name__ == "__main__":
    # Define benchmarks that will be used
    mani_conf = ManipulationO3DEBenchmarkConfig(
        o3de_config_path="src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml",  # path to your o3de config
        levels=[  # define what difficulty of tasks to include in benchmark
            "trivial",
        ],
        repeats=1,  # how many times to repeat
    )
    tool_conf = ToolCallingAgentBenchmarkConfig(
        extra_tool_calls=[0],  # how many extra tool calls allowed to still pass
        task_types=[  # what types of tasks to include
            "basic",
            "spatial_reasoning",
            "custom_interfaces",
            "manipulation",
        ],
        N_shots=[2],  # examples in system prompt
        prompt_detail=["descriptive"],  # how descriptive should task prompt be
        repeats=1,
    )
    gpt_llm = get_llm_for_benchmark(model_name="gpt-4o-mini", vendor="openai")
    qwen2_5_7b = get_llm_for_benchmark(model_name="qwen2.5:7b", vendor="ollama")
    agent_factories: List[AgentFactory] = [
        # PlanExecuteAgentFactory(
        #     planner_llm=gpt_llm, executor_llm=qwen2_5_7b, replanner_llm=gpt_llm
        # ),
        TaskVerificationAgentFactory(worker_llm=qwen2_5_7b, verification_llm=gpt_llm),
    ]
    out_dir = "src/rai_bench/rai_bench/experiments"
    test_agents(
        agent_factories=agent_factories,
        benchmark_configs=[mani_conf],
        out_dir=out_dir,
    )
