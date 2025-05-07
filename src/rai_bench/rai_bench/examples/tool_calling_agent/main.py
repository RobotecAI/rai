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

import argparse
import logging
from datetime import datetime
from pathlib import Path

from rai import get_llm_model_direct
from rai.agents.conversational_agent import create_conversational_agent

from rai_bench.examples.tool_calling_agent.tasks import get_all_tasks
from rai_bench.tool_calling_agent.benchmark import ToolCallingAgentBenchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Tool Calling Agent Benchmark")
    parser.add_argument(
        "--model-name",
        type=str,
        help="Model name to use for benchmarking",
        required=True,
    )
    parser.add_argument("--vendor", type=str, help="Vendor of the model", required=True)
    parser.add_argument(
        "--extra-tool-calls",
        type=int,
        help="Number of extra tools calls agent can make and still pass the task",
        default=0,
    )
    now = datetime.now()
    parser.add_argument(
        "--out_dir",
        type=str,
        default=f"src/rai_bench/rai_bench/experiments/o3de_manipulation/{now.strftime('%Y-%m-%d_%H-%M-%S')}",
        help="Output directory for results and logs",
    )

    return parser.parse_args()


def run_benchmark(model_name: str, vendor: str, out_dir: str, extra_tool_calls: int):
    experiment_dir = Path(out_dir)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_filename = experiment_dir / "benchmark.log"

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    bench_logger = logging.getLogger("Benchmark logger")
    bench_logger.setLevel(logging.INFO)
    bench_logger.addHandler(file_handler)

    agent_logger = logging.getLogger("Agent logger")
    agent_logger.setLevel(logging.INFO)
    agent_logger.addHandler(file_handler)

    all_tasks = get_all_tasks(extra_tool_calls=extra_tool_calls)
    for task in all_tasks:
        task.set_logger(bench_logger)

    benchmark = ToolCallingAgentBenchmark(
        tasks=all_tasks,
        logger=bench_logger,
        model_name=model_name,
        results_dir=experiment_dir,
    )

    llm = get_llm_model_direct(model_name=model_name, vendor=vendor)
    for task in all_tasks:
        agent = create_conversational_agent(
            llm=llm,
            tools=task.available_tools,
            system_prompt=task.get_system_prompt(),
            logger=agent_logger,
        )
        benchmark.run_next(agent=agent)


if __name__ == "__main__":
    args = parse_args()
    run_benchmark(
        model_name=args.model_name,
        vendor=args.vendor,
        out_dir=args.out_dir,
        extra_tool_calls=args.extra_tool_calls,
    )
