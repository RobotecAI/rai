import csv
import logging
import time
from typing import Any, Dict, List

from rai.agents.conversational_agent import create_conversational_agent
from rai.messages.multimodal import HumanMultimodalMessage
from rai.utils.model_initialization import get_llm_model

from rai_bench.agent_bench.agent_tasks import (
    AgentTask,
    GetROS2CameraTask,
    GetROS2TopicsTask,
)

loggers_type = logging.Logger


class AgentBenchmark:
    def __init__(
        self, tasks: List[AgentTask], logger: loggers_type | None = None
    ) -> None:
        self.tasks = enumerate(iter(tasks))
        self.num_tasks = len(tasks)
        self.tasks_results: List[Dict[str, Any]] = []
        self.results_filename = "agent_benchmark_results.csv"
        self.fieldnames = [
            "task",
            "result",
            "total_time",
        ]
        self._initialize_results_file()
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def run_next(self, agent) -> None:
        try:
            i, task = next(self.tasks)
            self.logger.info(
                f"RUNNING TASK NUMBER {i + 1} / {self.num_tasks}, TASK {task.get_prompt()}"
            )
            ts = time.perf_counter()
            response = agent.invoke(
                {"messages": [HumanMultimodalMessage(content=task.get_prompt())]}
            )
            result = task.verify_tool_calls(response=response)
            te = time.perf_counter()
            total_time = te - ts
            self.logger.info(f"TASK SUCCESS: {result}, TOTAL TIME: {total_time:.3f}")
            task_result: dict[str, Any] = {
                "task": task.get_prompt(),
                "result": result,
                "total_time": total_time,
            }
            self.tasks_results.append(task_result)
            self._save_task_result_to_csv(task_result)
        except StopIteration:
            print("No more scenarios left to run.")

    def _save_task_result_to_csv(self, result: Dict[str, Any]) -> None:
        """Save a single task result to the CSV file."""
        with open(
            self.results_filename, mode="a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(result)

    def _initialize_results_file(self):
        """Initialize the CSV file with headers."""
        with open(
            self.results_filename, mode="w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()


tasks: List[AgentTask] = [
    GetROS2TopicsTask(),
    GetROS2CameraTask(),
]

benchmark = AgentBenchmark(
    tasks=tasks,
    logger=logging.getLogger(__name__),
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
