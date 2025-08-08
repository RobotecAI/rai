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

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from pydantic import BaseModel
from rai.agents.langchain.core import (
    create_structured_output_runnable,
)
from rai.messages import HumanMultimodalMessage

from rai_bench.base_benchmark import BaseBenchmark, RunSummary, TimeoutException
from rai_bench.results_processing.langfuse_scores_tracing import ScoreTracingHandler
from rai_bench.utils import get_llm_model_name
from rai_bench.vlm_benchmark.interfaces import ImageReasoningTask, TaskValidationError
from rai_bench.vlm_benchmark.results_tracking import (
    TaskResult,
)


class VLMBenchmark(BaseBenchmark):
    """Benchmark for VLMs."""

    def __init__(
        self,
        tasks: Sequence[ImageReasoningTask[BaseModel]],
        model_name: str,
        results_dir: Path,
        logger: logging.Logger | None = None,
    ) -> None:
        super().__init__(
            model_name=model_name,
            results_dir=results_dir,
            logger=logger,
        )
        self._tasks: Iterator[Tuple[int, ImageReasoningTask[BaseModel]]] = enumerate(
            iter(tasks)
        )
        self.num_tasks = len(tasks)
        self.task_results: List[TaskResult] = []

        self.score_tracing_handler = ScoreTracingHandler()
        self.tasks_results: List[TaskResult] = []
        self.csv_initialize(self.results_filename, TaskResult)

    def run_next(self, agent: CompiledStateGraph, experiment_id: uuid.UUID) -> None:
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
            "tags": [
                f"experiment-id:{experiment_id}",
                "benchmark:vlm-benchmark",
                self.model_name,
                f"task-complexity:{task.complexity}",
            ],
            "recursion_limit": len(agent.get_graph().nodes),
        }

        ts = time.perf_counter()
        messages: List[BaseMessage] = []
        prev_count: int = 0
        errors: List[str] = []
        try:
            with self.time_limit(60):
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
        except TimeoutException as e:
            self.logger.error(msg=f"Task timeout: {e}")
        except GraphRecursionError as e:
            self.logger.error(msg=f"Reached recursion limit {e}")
        except Exception as e:
            self.logger.error(msg=f"Unexpected error occured: {e}")
        structured_output = None
        try:
            structured_output = task.get_structured_output_from_messages(
                messages=messages
            )
        except TaskValidationError as e:
            errors.append(str(e))

        if structured_output is not None:
            score = task.validate(output=structured_output)
        else:
            errors.append(f"Not valid structured output: {type(structured_output)}")
            score = False

        te = time.perf_counter()
        total_time = te - ts

        self.logger.info(f"TASK SCORE: {score}, TOTAL TIME: {total_time:.3f}")

        task_result = TaskResult(
            task_prompt=task.get_prompt(),
            system_prompt=task.get_system_prompt(),
            type=task.type,
            complexity=task.complexity,
            model_name=self.model_name,
            score=score,
            total_time=total_time,
            run_id=run_id,
        )

        self.task_results.append(task_result)

        self.csv_writerow(self.results_filename, task_result)
        # computing after every iteration in case of early stopping
        self.compute_and_save_summary()

        for callback in callbacks:
            self.score_tracing_handler.send_score(
                callback=callback,
                run_id=run_id,
                score=score,
                errors=[errors],
            )

    def compute_and_save_summary(self):
        self.logger.info("Computing and saving average results...")

        success_count = sum(1 for r in self.task_results if r.score == 1.0)
        success_rate = success_count / len(self.task_results) * 100
        avg_time = statistics.mean(r.total_time for r in self.task_results)

        summary = RunSummary(
            model_name=self.model_name,
            success_rate=round(success_rate, 2),
            avg_time=round(avg_time, 3),
            total_tasks=len(self.task_results),
        )
        self.csv_initialize(self.summary_filename, RunSummary)
        self.csv_writerow(self.summary_filename, summary)


def run_benchmark(
    llm: BaseChatModel,
    out_dir: Path,
    tasks: List[ImageReasoningTask[BaseModel]],
    bench_logger: logging.Logger,
    experiment_id: uuid.UUID = uuid.uuid4(),
):
    benchmark = VLMBenchmark(
        tasks=tasks,
        logger=bench_logger,
        model_name=get_llm_model_name(llm),
        results_dir=out_dir,
    )

    for task in tasks:
        agent = create_structured_output_runnable(
            llm=llm,
            structured_output=task.structured_output,
            system_prompt=task.get_system_prompt(),
            logger=bench_logger,
        )

        benchmark.run_next(agent=agent, experiment_id=experiment_id)

    bench_logger.info("===============================================================")
    bench_logger.info("ALL SCENARIOS DONE. BENCHMARK COMPLETED!")
    bench_logger.info("===============================================================")
