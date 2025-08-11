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
import logging
import uuid
from datetime import datetime
from pathlib import Path

from rai.agents.langchain.megamind import State, StepSuccess, create_megamind
from rai.messages.multimodal import HumanMultimodalMessage

from rai_bench import (
    define_benchmark_logger,
)
from rai_bench.tool_calling_agent.benchmark import ToolCallingAgentBenchmark
from rai_bench.tool_calling_agent.tasks.demo import SortTask
from rai_bench.utils import get_llm_for_benchmark

if __name__ == "__main__":
    # args = parse_tool_calling_benchmark_args()
    now = datetime.now()
    out_dir = f"src/rai_bench/rai_bench/experiments/tool_calling/{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_dir = Path(out_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    bench_logger = define_benchmark_logger(out_dir=experiment_dir, level=logging.DEBUG)

    task = SortTask()
    task.set_logger(bench_logger)

    supervisor_name = "qwen3:30b-a3b-instruct-2507-q8_0"
    # supervisor_name = "gpt-4o-mini"

    executor_name = "qwen3:8b"
    model_name = f"supervisor-{supervisor_name}_executor-{executor_name}"
    supervisor_llm = get_llm_for_benchmark(model_name=supervisor_name, vendor="ollama")
    executor_llm = get_llm_for_benchmark(
        model_name=executor_name, vendor="ollama", reasoning=False
    )

    benchmark = ToolCallingAgentBenchmark(
        tasks=[task],
        logger=bench_logger,
        model_name=model_name,
        results_dir=experiment_dir,
    )

    agent = create_megamind(
        manipulation_tools=task.manipulation_tools(),
        navigation_tools=task.navigation_tools(),
        megamind_llm=supervisor_llm,
        executor_llm=executor_llm,
        system_prompt=task.get_system_prompt(),
    )
    initial_state = State(
        {
            "original_task": task.get_prompt(),
            "messages": [HumanMultimodalMessage(content=task.get_prompt())],
            "step": "",
            "steps_done": [],
            "step_success": StepSuccess(success=False, explanation=""),
            "step_messages": [],
        }
    )
    experiment_id = uuid.uuid4()
    benchmark.run_next(
        agent=agent, initial_state=initial_state, experiment_id=experiment_id
    )

    bench_logger.info("===============================================================")
    bench_logger.info("ALL SCENARIOS DONE. BENCHMARK COMPLETED!")
    bench_logger.info("===============================================================")
