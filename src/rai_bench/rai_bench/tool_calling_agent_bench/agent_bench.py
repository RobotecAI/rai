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
import statistics
import time
import uuid
from typing import Dict, Iterator, List, Sequence, Tuple
from uuid import UUID

from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphRecursionError
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


class BenchmarkSummary(BaseModel):
    model: str = Field(..., description="AI model used.")
    success_rate: float = Field(
        ..., description="Percentage of successfully completed tasks."
    )
    avg_time: float = Field(..., description="Average time taken across all tasks.")
    total_tasks: int = Field(..., description="Total number of tasks executed.")
    run_id: UUID = Field(..., description="UUID of the average results entry.")


class ToolCallingAgentBenchmark:
    """
    Benchmark for LangChain tool calling agents.
    """

    def __init__(
        self,
        tasks: Sequence[ToolCallingAgentTask],
        logger: loggers_type | None = None,
        results_filename: str = "agent_benchmark_results.csv",
        summary_filename: str | None = None,
    ) -> None:
        self._tasks: Iterator[Tuple[int, ToolCallingAgentTask]] = enumerate(iter(tasks))
        self.num_tasks = len(tasks)
        self.task_results: List[TaskResult] = []
        self.results_filename = results_filename
        self.summary_filename = summary_filename or results_filename.replace(
            ".csv", "_summary.csv"
        )
        self.fieldnames = [field for field in TaskResult.__annotations__.keys()]
        self.summary_fieldnames = [
            field for field in BenchmarkSummary.__annotations__.keys()
        ]
        self._initialize_results_file()
        self._initialize_summary_file()
        self.score_tracing_handler = ScoreTracingHandler()
        self.model_results: Dict[str, List[TaskResult]] = {}
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

            run_id = uuid.uuid4()
            config: RunnableConfig = {
                "run_id": run_id,
                "callbacks": callbacks,
                "tags": [task.complexity, model_name],
                "recursion_limit": task.recursion_limit,
            }
            ts = time.perf_counter()
            try:
                response = agent.invoke(
                    {"messages": [HumanMultimodalMessage(content=task.get_prompt())]},
                    config=config,
                )
                task.verify_tool_calls(response=response)
            except GraphRecursionError as e:
                self.logger.error(f"Graph Recursion Error: {e}")
                task.result.errors.append(f"Graph Recursion Error: {e}")
            te = time.perf_counter()
            total_time = te - ts
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

            if model_name not in self.model_results:
                self.model_results[model_name] = []
            self.model_results[model_name].append(task_result)

            self._save_task_result_to_csv(task_result)

            completed_tasks = sum(
                len(results) for results in self.model_results.values()
            )
            if completed_tasks == self.num_tasks:
                self._compute_and_save_summary()

        except StopIteration:
            if self.task_results:
                self._compute_and_save_summary()
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

    def _initialize_summary_file(self):
        """Initialize the summary CSV file with headers."""
        with open(
            self.summary_filename, mode="w", newline="", encoding="utf-8"
        ) as file:
            writer = csv.DictWriter(file, fieldnames=self.summary_fieldnames)
            writer.writeheader()

    def _compute_and_save_summary(self):
        """Save the average results to the summary CSV file."""
        self.logger.info("Computing and saving average results...")

        for model_name, results in self.model_results.items():
            if not results:
                continue

            success_count = sum(1 for r in results if r.success)
            success_rate = success_count / len(results) * 100
            avg_time = statistics.mean(r.total_time for r in results)

            summary = BenchmarkSummary(
                model=model_name,
                success_rate=round(success_rate, 2),
                avg_time=round(avg_time, 3),
                total_tasks=len(results),
                run_id=uuid.uuid4(),
            )

            with open(
                self.summary_filename, mode="a", newline="", encoding="utf-8"
            ) as file:
                writer = csv.DictWriter(file, fieldnames=self.summary_fieldnames)
                writer.writerow(summary.model_dump())

            self.logger.info(
                f"Summary for model {model_name}: Success rate {success_rate:.2f}%, "
                f"Average time {avg_time:.3f}s, Total tasks: {len(results)}"
            )
