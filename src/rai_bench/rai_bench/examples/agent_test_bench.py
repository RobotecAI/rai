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
import os
from datetime import datetime
from pathlib import Path
from typing import List

from rai.agents.conversational_agent import create_conversational_agent
from rai.utils.model_initialization import get_llm_model

from rai_bench.agent_bench.agent_bench import AgentBenchmark
from rai_bench.agent_bench.agent_tasks import (
    AgentTask,
    GetROS2CameraTask,
    GetROS2TopicsTask,
)

# define loggers
now = datetime.now()
current_test_name = os.path.splitext(os.path.basename(__file__))[0]

# Define loggers
now = datetime.now()
experiment_dir = os.path.join(
    "src/rai_bench/rai_bench/experiments",
    current_test_name,
    now.strftime("%Y-%m-%d_%H-%M-%S"),
)
Path(experiment_dir).mkdir(parents=True, exist_ok=True)
log_filename = f"{experiment_dir}/benchmark.log"
results_filename = f"{experiment_dir}/results.csv"

file_handler = logging.FileHandler(log_filename)
file_handler.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
file_handler.setFormatter(formatter)

bench_logger = logging.getLogger("Benchmark logger")
bench_logger.setLevel(logging.INFO)
bench_logger.addHandler(file_handler)

agent_logger = logging.getLogger("Agent logger")
agent_logger.setLevel(logging.INFO)
agent_logger.addHandler(file_handler)

tasks: List[AgentTask] = [
    GetROS2CameraTask(logger=bench_logger),
    GetROS2TopicsTask(logger=bench_logger),
]
benchmark = AgentBenchmark(
    tasks=tasks, logger=logging.getLogger(__name__), results_filename=results_filename
)

for _, task in enumerate(tasks):
    agent = create_conversational_agent(
        llm=get_llm_model(model_type="complex_model"),
        tools=task.expected_tools,
        system_prompt="""
    You are the helpful assistant.
    """,
        logger=agent_logger,
    )
    benchmark.run_next(agent=agent)
