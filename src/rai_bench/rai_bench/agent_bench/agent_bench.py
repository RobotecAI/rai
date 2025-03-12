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
from typing import Any, Dict, List

from langfuse.callback import CallbackHandler
from rai.messages.multimodal import HumanMultimodalMessage
from rai.utils.model_initialization import get_tracing_callbacks

from rai_bench.agent_bench.agent_tasks import (
    AgentTask,
    Result,
)

loggers_type = logging.Logger


class AgentBenchmark:
    def __init__(
        self,
        tasks: List[AgentTask],
        logger: loggers_type | None = None,
        results_filename: str = "agent_benchmark_results.csv",
    ) -> None:
        self.tasks = enumerate(iter(tasks))
        self.num_tasks = len(tasks)
        self.tasks_results: List[Dict[str, Any]] = []
        self.results_filename = results_filename
        self.fieldnames = [
            "task",
            "success",
            "errors",
            "total_time",
            "callback_trace_id",
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
            callbacks = get_tracing_callbacks()
            ts = time.perf_counter()
            response = agent.invoke(
                {"messages": [HumanMultimodalMessage(content=task.get_prompt())]},
                config={"callbacks": callbacks},
            )
            te = time.perf_counter()
            total_time = te - ts

            result: Result = task.verify_tool_calls(response=response)
            trace_id = None
            # TODO (mkotynia) currently only for langfuse, to handle langsmith tracing as well and then refactor
            if callbacks:
                callback: CallbackHandler = callbacks[0]
                trace_id = callback.get_trace_id()
                callback.langfuse.score(
                    trace_id=trace_id,
                    name="tool calls result",
                    value=float(result.success),
                    comment="; ".join(result.errors),
                )

            self.logger.info(
                f"TASK SUCCESS: {result.success}, TOTAL TIME: {total_time:.3f}"
            )

            task_result: Dict[str, Any] = {
                "task": task.get_prompt(),
                "success": result.success,
                "errors": "; ".join(result.errors) if result.errors else "",
                "total_time": total_time,
                "callback_trace_id": trace_id,
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
