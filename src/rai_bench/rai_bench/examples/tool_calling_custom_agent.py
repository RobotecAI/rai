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

from rai.agents.langchain.core import (
    Executor,
    create_megamind,
    get_initial_megamind_state,
)

from rai_bench import (
    define_benchmark_logger,
)
from rai_bench.tool_calling_agent.benchmark import ToolCallingAgentBenchmark
from rai_bench.tool_calling_agent.interfaces import TaskArgs
from rai_bench.tool_calling_agent.tasks.warehouse import SortingTask
from rai_bench.utils import get_llm_for_benchmark

if __name__ == "__main__":
    now = datetime.now()
    out_dir = f"src/rai_bench/rai_bench/experiments/tool_calling/{now.strftime('%Y-%m-%d_%H-%M-%S')}"
    experiment_dir = Path(out_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    bench_logger = define_benchmark_logger(out_dir=experiment_dir, level=logging.DEBUG)

    task = SortingTask(task_args=TaskArgs(extra_tool_calls=50))
    task.set_logger(bench_logger)

    supervisor_name = "gpt-4o"

    executor_name = "gpt-4o-mini"
    model_name = f"supervisor-{supervisor_name}_executor-{executor_name}"
    supervisor_llm = get_llm_for_benchmark(model_name=supervisor_name, vendor="openai")
    executor_llm = get_llm_for_benchmark(
        model_name=executor_name,
        vendor="openai",
    )

    benchmark = ToolCallingAgentBenchmark(
        tasks=[task],
        logger=bench_logger,
        model_name=model_name,
        results_dir=experiment_dir,
    )
    manipulation_system_prompt = """You are a manipulation specialist robot agent.
Your role is to handle object manipulation tasks including picking up and droping objects using provided tools.

Ask the VLM for objects detection and positions before perfomring any manipulation action.
If VLM doesn't see objects that are objectives of the task, return this information, without proceeding"""

    navigation_system_prompt = """You are a navigation specialist robot agent.
Your role is to handle navigation tasks in space using provided tools.

After performing navigation action, always check your current position to ensure success"""

    executors = [
        Executor(
            name="manipulation",
            llm=executor_llm,
            tools=task.manipulation_tools(),
            system_prompt=manipulation_system_prompt,
        ),
        Executor(
            name="navigation",
            llm=executor_llm,
            tools=task.navigation_tools(),
            system_prompt=navigation_system_prompt,
        ),
    ]
    agent = create_megamind(
        megamind_llm=supervisor_llm,
        megamind_system_prompt=task.get_system_prompt(),
        executors=executors,
        task_planning_prompt=task.get_planning_prompt(),
    )

    experiment_id = uuid.uuid4()
    benchmark.run_next(
        agent=agent,
        initial_state=get_initial_megamind_state(task=task.get_prompt()),
        experiment_id=experiment_id,
    )

    bench_logger.info("===============================================================")
    bench_logger.info("ALL SCENARIOS DONE. BENCHMARK COMPLETED!")
    bench_logger.info("===============================================================")
