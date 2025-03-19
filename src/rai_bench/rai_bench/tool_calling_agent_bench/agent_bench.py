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
from typing import Any, Dict, List, Sequence

from rai.messages.multimodal import HumanMultimodalMessage

from rai_bench.tool_calling_agent_bench.agent_tasks_interfaces import (
    ToolCallingAgentTask,
)
from rai_bench.tool_calling_agent_bench.scores_tracing import ScoreTracingHandler

loggers_type = logging.Logger


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
        self._tasks = enumerate(iter(tasks))
        self.num_tasks = len(tasks)
        self.tasks_results: List[Dict[str, Any]] = []
        self.results_filename = results_filename
        self.fieldnames = [
            "task",
            "success",
            "errors",
            "total_time",
            "callback_trace_ids",
        ]
        self._initialize_results_file()
        self.score_tracing_handler = ScoreTracingHandler()
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)

    def run_next(self, agent) -> None:
        try:
            i, task = next(self._tasks)
            self.logger.info(
                f"RUNNING TASK NUMBER {i + 1} / {self.num_tasks}, TASK {task.get_prompt()}"
            )
            callbacks = self.score_tracing_handler.get_callbacks()
            ts = time.perf_counter()
            response = agent.invoke(
                {"messages": [HumanMultimodalMessage(content=task.get_prompt())]},
                config={"callbacks": callbacks, "tags": [task.complexity]},
            )
            te = time.perf_counter()
            total_time = te - ts

            task.verify_tool_calls(response=response)
            result = task.result
            trace_ids: List[str] = []
            for callback in callbacks:
                trace_id = self.score_tracing_handler.get_trace_id(callback)
                if trace_id:
                    trace_ids.append(trace_id)
                    self.score_tracing_handler.send_score(
                        callback=callback,
                        trace_id=trace_id,
                        success=result.success,
                        errors=result.errors,
                    )

            self.logger.info(
                f"TASK SUCCESS: {result.success}, TOTAL TIME: {total_time:.3f}"
            )

            task_result: Dict[str, Any] = {
                "task": task.get_prompt(),
                "success": result.success,
                "errors": "; ".join(result.errors) if result.errors else "",
                "total_time": total_time,
                "callback_trace_ids": trace_ids,
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
