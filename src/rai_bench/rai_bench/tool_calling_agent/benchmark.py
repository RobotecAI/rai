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
import statistics
import time
import uuid
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from rai.messages import HumanMultimodalMessage

from rai_bench.base_benchmark import BaseBenchmark, BenchmarkSummary
from rai_bench.tool_calling_agent.interfaces import (
    Task,
)
from rai_bench.tool_calling_agent.results_tracking import (
    ScoreTracingHandler,
    TaskResult,
)
from rai_bench.tool_calling_agent.tasks.spatial import (
    SpatialReasoningAgentTask,
)

loggers_type = logging.Logger


class ToolCallingAgentBenchmark(BaseBenchmark):
    """Benchmark for LangChain tool calling agents."""

    def __init__(
        self,
        tasks: Sequence[Task],
        model_name: str,
        results_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            results_dir=results_dir,
            logger=logger,
        )
        self._tasks: Iterator[Tuple[int, Task]] = enumerate(iter(tasks))
        self.num_tasks = len(tasks)
        self.task_results: List[TaskResult] = []

        self.score_tracing_handler = ScoreTracingHandler()
        self.tasks_results: List[TaskResult] = []
        self.csv_initialize(self.results_filename, TaskResult)

    def run_next(
        self,
        agent: CompiledStateGraph,
    ) -> None:
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
            "======================================================================================"
        )
        self.logger.info(
            f"RUNNING TASK NUMBER {i + 1} / {self.num_tasks}, TASK {task.get_prompt()}"
        )
        callbacks = self.score_tracing_handler.get_callbacks()
        run_id = uuid.uuid4()
        config: RunnableConfig = {
            "run_id": run_id,
            "callbacks": callbacks,
            "tags": [task.complexity, self.model_name],
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
        total_extra_calls: int = 0
        for validator_info in validation_info:
            total_extra_calls += validator_info.extra_tool_calls_used
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
            extra_tool_calls_used=total_extra_calls,
            complexity=task.complexity,
            model_name=self.model_name,
            validation_info=validation_info,
            score=score,
            total_time=total_time,
            run_id=run_id,
        )

        self.task_results.append(task_result)

        self.csv_writerow(self.results_filename, task_result)
        # computing after every iteration in case of early stopping
        self.compute_and_save_summary()

    def compute_and_save_summary(self):
        self.logger.info("Computing and saving average results...")

        success_count = sum(1 for r in self.task_results if r.score == 1.0)
        success_rate = success_count / len(self.task_results) * 100
        avg_time = statistics.mean(r.total_time for r in self.task_results)
        total_extra_calls = sum(r.extra_tool_calls_used for r in self.task_results)

        summary = BenchmarkSummary(
            model_name=self.model_name,
            success_rate=round(success_rate, 2),
            avg_time=round(avg_time, 3),
            total_extra_tool_calls_used=total_extra_calls,
            total_tasks=len(self.task_results),
        )
        self.csv_initialize(self.summary_filename, BenchmarkSummary)
        self.csv_writerow(self.summary_filename, summary)
