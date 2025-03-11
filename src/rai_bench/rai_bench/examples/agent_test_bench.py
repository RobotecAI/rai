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
# Get the current file name without extension
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
    GetROS2CameraTask(),
    GetROS2TopicsTask(),
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
    )
    benchmark.run_next(agent=agent)
