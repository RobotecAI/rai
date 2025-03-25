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
from datetime import datetime
from pathlib import Path

from rai.agents.conversational_agent import create_conversational_agent
from rai.utils.model_initialization import get_llm_model, get_llm_model_config

from rai_bench.examples.tool_calling_agent_bench_tasks import tasks
from rai_bench.tool_calling_agent_bench.agent_bench import ToolCallingAgentBenchmark

if __name__ == "__main__":
    current_test_name = Path(__file__).stem

    now = datetime.now()
    experiment_dir = (
        Path("src/rai_bench/rai_bench/experiments")
        / current_test_name
        / now.strftime("%Y-%m-%d_%H-%M-%S")
    )
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
    bench_logger.setLevel(logging.INFO)
    bench_logger.addHandler(file_handler)

    agent_logger = logging.getLogger("Agent logger")
    agent_logger.setLevel(logging.INFO)
    agent_logger.addHandler(file_handler)

    for task in tasks:
        task.logger = bench_logger

    benchmark = ToolCallingAgentBenchmark(
        tasks=tasks, logger=bench_logger, results_filename=results_filename
    )

    model_type = "simple_model"
    model_config = get_llm_model_config(model_type=model_type)
    model_name = getattr(model_config, model_type)

    for task in tasks:
        agent = create_conversational_agent(
            llm=get_llm_model(model_type=model_type),
            tools=task.expected_tools,
            system_prompt=task.get_system_prompt(),
            logger=agent_logger,
        )
        benchmark.run_next(agent=agent, model_name=model_name)
