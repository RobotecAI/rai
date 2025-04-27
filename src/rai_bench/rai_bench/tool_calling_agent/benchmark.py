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
from pathlib import Path
from typing import Dict, Iterator, List, Sequence, Tuple

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel, Field
from rai.messages import HumanMultimodalMessage

from rai_bench.tool_calling_agent.interfaces import (
    Task,
)
from rai_bench.tool_calling_agent.scores_tracing import (
    ScoreTracingHandler,
    TaskResult,
)
from rai_bench.tool_calling_agent.tasks.spatial import (
    SpatialReasoningAgentTask,
)

loggers_type = logging.Logger


class BenchmarkSummary(BaseModel):
    model_name: str = Field(..., description="Name of the LLM.")
    success_rate: float = Field(
        ..., description="Percentage of successfully completed tasks."
    )
    avg_time: float = Field(..., description="Average time taken across all tasks.")
    total_tasks: int = Field(..., description="Total number of executed tasks.")


class ToolCallingAgentBenchmark:
    """Benchmark for LangChain tool calling agents.

    Parameters
    ----------
    tasks : Sequence[Task]
        Sequence of tasks to be passed to the agent.
    logger : loggers_type | None, optional
        Logger, by default None
    results_filename : Path, optional
        Filename of the CSV file to store the results, by default Path("agent_benchmark_results.csv")
    summary_filename : str | None, optional
        Filename of the CSV file to store the summary of the results, by default None
    """

    def __init__(
        self,
        tasks: Sequence[Task],
        logger: loggers_type | None = None,
        results_filename: Path = Path("agent_benchmark_results.csv"),
        summary_filename: str | None = None,
    ) -> None:
        self._tasks: Iterator[Tuple[int, Task]] = enumerate(iter(tasks))
        self.num_tasks = len(tasks)
        self.task_results: List[TaskResult] = []
        self.results_filename = results_filename
        self.summary_filename = (
            Path(summary_filename)
            if summary_filename
            else results_filename.with_name(
                results_filename.stem + "_summary" + results_filename.suffix
            )
        )
        self.csv_initialize(self.results_filename, TaskResult)
        self.csv_initialize(self.summary_filename, BenchmarkSummary)

        self.score_tracing_handler = ScoreTracingHandler()
        self.model_results: Dict[str, List[TaskResult]] = {}

        self.logger = logger if logger else logging.getLogger(__name__)

    @staticmethod
    def csv_initialize(filename: Path, base_model_cls: type[BaseModel]):
        """Initialize a CSV file based on a Pydantic model class.

        Parameters
        ----------
        filename : Path
            Filename of the CSV file.
        base_model_cls : type[BaseModel]
            Pydantic model class to be used for creating the columns in the CSV file.
        """
        with open(filename, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=base_model_cls.__annotations__.keys()
            )
            writer.writeheader()

    @staticmethod
    def csv_writerow(filename: Path, base_model_instance: BaseModel):
        """Write a single row to a CSV file based on a Pydantic model instance contents,
        ensuring that multiline strings are converted to one-line strings by replacing newlines.

        Parameters
        ----------
        filename : Path
            Filename of the CSV file.
        base_model_instance : BaseModel
            Pydantic model instance which contains the data to be written to the CSV file.
        """
        row = base_model_instance.model_dump()

        for key, value in row.items():
            if isinstance(value, str):
                # Replace newline characters with a single space so they dont break csv
                row[key] = " ".join(value.split())

        with open(filename, mode="a", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(
                file, fieldnames=base_model_instance.__annotations__.keys()
            )
            writer.writerow(row)

    def run_next(self, agent: CompiledStateGraph, model_name: str) -> None:
        """Runs the next task of the benchmark.

        Parameters
        ----------
        agent : CompiledStateGraph
            LangChain tool calling agent.
        model_name : str
            Name of the LLM model.
        """
        # try:
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
            "recursion_limit": 4 * task.max_tool_calls_number,
        }

        ts = time.perf_counter()
        messages: List[BaseMessage] = []
        prev_count: int = 0
        try:
            if isinstance(task, SpatialReasoningAgentTask):
                for state in agent.stream(
                    {
                        "messages": [
                            HumanMultimodalMessage(
                                content=task.get_prompt(), images=task.get_images()
                            )
                        ]
                    },
                    config=config,
                ):
                    node = next(iter(state))
                    all_messages = state[node]["messages"]
                    for new_msg in all_messages[prev_count:]:
                        messages.append(new_msg)
                    prev_count = len(messages)
            else:
                for state in agent.stream(
                    {"messages": [HumanMultimodalMessage(content=task.get_prompt())]},
                    config=config,
                ):
                    node = next(iter(state))
                    all_messages = state[node]["messages"]
                    for new_msg in all_messages[prev_count:]:
                        messages.append(new_msg)
                    prev_count = len(messages)

        except GraphRecursionError as e:
            self.logger.error(msg=f"Reached recursion limit {e}")

        self.logger.debug(messages)
        tool_calls = task.get_tool_calls_from_messages(messages=messages)
        score = task.validate(tool_calls=tool_calls)
        te = time.perf_counter()
        total_time = te - ts

        validation_info = task.dump_validators()
        errors = [
            s.errors
            for validator_info in validation_info
            for s in validator_info.subtasks
        ]

        for callback in callbacks:
            self.score_tracing_handler.send_score(
                callback=callback,
                run_id=run_id,
                score=score,
                errors=errors,
            )

        self.logger.info(f"TASK SCORE: {score}, TOTAL TIME: {total_time:.3f}")

        task_result = TaskResult(
            task_prompt=task.get_prompt(),
            system_prompt=task.get_system_prompt(),
            type=task.type,
            extra_tool_calls=task.extra_tool_calls,
            complexity=task.complexity,
            model_name=model_name,
            validation_info=validation_info,
            score=score,
            total_time=total_time,
            run_id=run_id,
        )

        self.task_results.append(task_result)

        if model_name not in self.model_results:
            self.model_results[model_name] = []
        self.model_results[model_name].append(task_result)

        self.csv_writerow(self.results_filename, task_result)

        completed_tasks = sum(len(results) for results in self.model_results.values())
        if completed_tasks == self.num_tasks:
            self._compute_and_save_summary()

    def _compute_and_save_summary(self):
        self.logger.info("Computing and saving average results...")
        for model_name, results in self.model_results.items():
            if not results:
                continue

            success_count = sum(1 for r in results if r.score)
            success_rate = success_count / len(results) * 100
            avg_time = statistics.mean(r.total_time for r in results)

            summary = BenchmarkSummary(
                model_name=model_name,
                success_rate=round(success_rate, 2),
                avg_time=round(avg_time, 3),
                total_tasks=len(results),
            )

            self.csv_writerow(self.summary_filename, summary)

            self.logger.info(
                f"Summary for model {model_name}: Success rate {success_rate:.2f}%, "
                f"Average time {avg_time:.3f}s, Total tasks: {len(results)}"
            )
