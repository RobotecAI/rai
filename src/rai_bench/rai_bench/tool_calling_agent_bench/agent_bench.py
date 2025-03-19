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

import csv
import logging
import time
import uuid
from typing import Iterator, List, Sequence, Tuple
from uuid import UUID

from langchain_core.runnables.config import RunnableConfig
from pydantic import BaseModel, Field
from rai.messages.multimodal import HumanMultimodalMessage

from rai_bench.tool_calling_agent_bench.agent_tasks_interfaces import (
    ToolCallingAgentTask,
)
from rai_bench.tool_calling_agent_bench.scores_tracing import ScoreTracingHandler

loggers_type = logging.Logger


class TaskResult(BaseModel):
    task: str = Field(..., description="The task to be executed.")
    model: str = Field(..., description="AI model used.")
    success: bool = Field(
        ..., description="Whether the task was successfully completed."
    )
    errors: List[str] = Field(
        ..., description="List of errors that occurred during the task execution."
    )
    total_time: float = Field(..., description="Total time taken to complete the task.")
    run_id: UUID = Field(..., description="UUID of the task run.")


class ToolCallingAgentBenchmark:
    """
    Benchmark for LangChain tool calling agents.
    """

    def __init__(
        self,
        tasks: Sequence[ToolCallingAgentTask],
        logger: loggers_type | None = None,
        results_filename: str = "agent_benchmark_results.csv",
    ) -> None:
        self._tasks: Iterator[Tuple[int, ToolCallingAgentTask]] = enumerate(iter(tasks))
        self.num_tasks = len(tasks)
        self.task_results: List[TaskResult] = []
        self.results_filename = results_filename
        self.fieldnames = [field for field in TaskResult.__annotations__.keys()]
        self._initialize_results_file()
        self.score_tracing_handler = ScoreTracingHandler()
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def run_next(self, agent, model_name: str) -> None:
        try:
            i, task = next(self._tasks)
            self.logger.info(
                f"RUNNING TASK NUMBER {i + 1} / {self.num_tasks}, TASK {task.get_prompt()}"
            )
            callbacks = self.score_tracing_handler.get_callbacks()
            ts = time.perf_counter()
            run_id = uuid.uuid4()
            config: RunnableConfig = {
                "run_id": run_id,
                "callbacks": callbacks,
                "tags": [task.complexity, model_name],
            }
            response = agent.invoke(
                {"messages": [HumanMultimodalMessage(content=task.get_prompt())]},
                config=config,
            )
            te = time.perf_counter()
            total_time = te - ts

            task.verify_tool_calls(response=response)
            result = task.result
            for callback in callbacks:
                self.score_tracing_handler.send_score(
                    callback=callback,
                    run_id=run_id,
                    success=result.success,
                    errors=result.errors,
                )

            self.logger.info(
                f"TASK SUCCESS: {result.success}, TOTAL TIME: {total_time:.3f}"
            )

            task_result = TaskResult(
                task=task.get_prompt(),
                model=model_name,
                success=result.success,
                errors=result.errors if result.errors else [],
                total_time=total_time,
                run_id=run_id,
            )
            self.task_results.append(task_result)
            self._save_task_result_to_csv(task_result)
        except StopIteration:
            print("No more scenarios left to run.")

    def _save_task_result_to_csv(self, result: TaskResult) -> None:
        """Save a single task result to the CSV file."""
        with open(
            self.results_filename, mode="a", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writerow(result.model_dump())

    def _initialize_results_file(self):
        """Initialize the CSV file with headers."""
        with open(
            self.results_filename, mode="w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.fieldnames)
            writer.writeheader()
