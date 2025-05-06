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

from pathlib import Path

from rai.agents.conversational_agent import create_conversational_agent

from rai_bench.examples.tool_calling_agent.tasks import get_all_tasks
from rai_bench.tool_calling_agent.benchmark import ToolCallingAgentBenchmark
from rai_bench.utils import (
    define_benchmark_loggers,
    get_llm_for_benchmark,
    parse_benchmark_args,
)


def run_benchmark(model_name: str, vendor: str, out_dir: str, extra_tool_calls: int):
    experiment_dir = Path(out_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    bench_logger, agent_logger = define_benchmark_loggers(out_dir=experiment_dir)

    all_tasks = get_all_tasks(extra_tool_calls=extra_tool_calls)
    for task in all_tasks:
        task.set_logger(bench_logger)

    benchmark = ToolCallingAgentBenchmark(
        tasks=all_tasks,
        logger=bench_logger,
        model_name=model_name,
        results_dir=experiment_dir,
    )

    llm = get_llm_for_benchmark(model_name=model_name, vendor=vendor)
    for task in all_tasks:
        agent = create_conversational_agent(
            llm=llm,
            tools=task.available_tools,
            system_prompt=task.get_system_prompt(),
            logger=agent_logger,
        )
        benchmark.run_next(agent=agent)

    bench_logger.info("===============================================================")
    bench_logger.info("ALL SCENARIOS DONE. BENCHMARK COMPLETED!")
    bench_logger.info("===============================================================")


if __name__ == "__main__":
    args = parse_benchmark_args()
    run_benchmark(
        model_name=args.model_name,
        vendor=args.vendor,
        out_dir=args.out_dir,
        extra_tool_calls=args.extra_tool_calls,
    )
