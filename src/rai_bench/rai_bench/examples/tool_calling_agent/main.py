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

from rai import (
    get_llm_model,
    get_llm_model_config_and_vendor,
)
from rai.agents.conversational_agent import create_conversational_agent

from rai_bench.examples.tool_calling_agent.tasks import all_tasks
from rai_bench.tool_calling_agent.benchmark import ToolCallingAgentBenchmark


def parse_args():
    parser = argparse.ArgumentParser(description="Run the Tool Calling Agent Benchmark")
    parser.add_argument(
        "--model-type",
        type=str,
        default="complex_model",
        help="Model type to use for benchmarking",
    )
    parser.add_argument(
        "--vendor", type=str, default=None, help="Vendor of the model (optional)"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    now = datetime.now()
    experiment_dir = Path(
        "src/rai_bench/rai_bench/experiments/tool_calling_agent"
    ) / now.strftime("%Y-%m-%d_%H-%M-%S")
    experiment_dir.mkdir(parents=True, exist_ok=True)
    log_filename = experiment_dir / "benchmark.log"
    results_filename = experiment_dir / "results.csv"

    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)

    bench_logger = logging.getLogger("Benchmark logger")
    bench_logger.setLevel(logging.DEBUG)
    bench_logger.addHandler(file_handler)

    agent_logger = logging.getLogger("Agent logger")
    agent_logger.setLevel(logging.INFO)
    agent_logger.addHandler(file_handler)

    for task in all_tasks:
        task.set_logger(bench_logger)

    benchmark = ToolCallingAgentBenchmark(
        tasks=all_tasks, logger=bench_logger, results_filename=results_filename
    )

    # model_type = "complex_model"
    model_config = get_llm_model_config_and_vendor(
        model_type=args.model_type, vendor=args.vendor
    )[0]
    model_name = getattr(model_config, args.model_type)
    for task in all_tasks:
        agent = create_conversational_agent(
            llm=get_llm_model(model_type=args.model_type, vendor=args.vendor),
            tools=task.available_tools,
            system_prompt=task.get_system_prompt(),
            logger=agent_logger,
        )
        benchmark.run_next(agent=agent, model_name=model_name)
