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
import io
import logging
import statistics
import time
import uuid
from pathlib import Path
from typing import Iterator, List, Sequence, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from rai.agents.langchain.core import (
    create_conversational_agent,
)
from rai.messages import HumanMultimodalMessage
from tqdm import tqdm

from rai_bench.base_benchmark import BaseBenchmark, TimeoutException
from rai_bench.results_processing.langfuse_scores_tracing import ScoreTracingHandler
from rai_bench.tool_calling_agent.interfaces import (
    Task,
)
from rai_bench.tool_calling_agent.results_tracking import (
    TaskResult,
    ToolCallingAgentRunSummary,
)
from rai_bench.utils import get_llm_model_name


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
        initial_state: dict,
        experiment_id: uuid.UUID,
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
        # NOTE (jmatejcz) recursion limit calculated as (all_nodes_num - 2) * required tool calls
        # -2 because we don't want to include START and END node
        # then we add numer of additional calls that can be made
        # and +2 as we have to pass once though START and END

        recurssion_limit = (
            (len(agent.get_graph().nodes) - 2) * task.required_calls
            + task.additional_calls
            + 2
        )
        config: RunnableConfig = {
            "run_id": run_id,
            "callbacks": callbacks,
            "tags": [
                f"experiment-id:{experiment_id}",
                "benchmark:tool-calling-agent",
                self.model_name,
                f"task-complexity:{task.complexity}",
                f"extra-tool-calls:{task.extra_tool_calls}",
            ],
            "recursion_limit": recurssion_limit,
        }
        self.logger.debug(f"recurssion limit: {recurssion_limit}")

        ts = time.perf_counter()
        messages: List[BaseMessage] = []
        prev_count: int = 0
        try:
            with self.time_limit(200 * task.max_tool_calls_number):
                for state in agent.stream(
                    initial_state,
                    config=config,
                ):
                    node = next(iter(state))
                    if "messages" in state[node]:
                        all_messages = state[node]["messages"]
                        for new_msg in all_messages[prev_count:]:
                            messages.append(new_msg)
                            if isinstance(new_msg, AIMessage):
                                self.logger.debug(
                                    f"Message from node '{node}': {new_msg.content}, tool_calls: {new_msg.tool_calls}"
                                )
                        prev_count = len(messages)
        except TimeoutException as e:
            self.logger.error(msg=f"Task timeout: {e}")
        except GraphRecursionError as e:
            tool_calls = task.get_tool_calls_from_messages(messages=messages)
            score = task.validate(tool_calls=tool_calls)
            score = 0.0
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

        self.logger.info(f"TASK SCORE: {score}, TOTAL TIME: {total_time:.3f}")

        task_result = TaskResult(
            task_prompt=task.get_base_prompt(),
            system_prompt=task.get_system_prompt(),
            examples_in_system_prompt=task.n_shots,
            prompt_detail=task.prompt_detail,
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

        for callback in callbacks:
            self.score_tracing_handler.send_score(
                callback=callback,
                run_id=run_id,
                score=score,
                errors=errors,
            )

    def compute_and_save_summary(self):
        self.logger.info("Computing and saving average results...")

        success_count = sum(1 for r in self.task_results if r.score == 1.0)
        success_rate = success_count / len(self.task_results) * 100
        avg_time = statistics.mean(r.total_time for r in self.task_results)
        total_extra_calls = sum(r.extra_tool_calls_used for r in self.task_results)

        summary = ToolCallingAgentRunSummary(
            model_name=self.model_name,
            success_rate=round(success_rate, 2),
            avg_time=round(avg_time, 3),
            total_extra_tool_calls_used=total_extra_calls,
            total_tasks=len(self.task_results),
        )
        self.csv_initialize(self.summary_filename, ToolCallingAgentRunSummary)
        self.csv_writerow(self.summary_filename, summary)


def _print_results_table(task_results: List[TaskResult]) -> None:
    rows = [
        {
            "task_prompt": r.task_prompt,
            "type": r.type,
            "complexity": r.complexity,
            "n_shots": r.examples_in_system_prompt,
            "prompt_detail": r.prompt_detail,
            "score": f"{r.score:.2f}",
        }
        for r in task_results
    ]
    avg = statistics.mean(r.score for r in task_results)

    buf = io.StringIO()
    writer = csv.DictWriter(
        buf,
        fieldnames=[
            "task_prompt",
            "type",
            "complexity",
            "n_shots",
            "prompt_detail",
            "score",
        ],
    )
    writer.writeheader()
    writer.writerows(rows)
    writer.writerow(
        {
            "task_prompt": "AVERAGE",
            "type": "",
            "complexity": "",
            "n_shots": "",
            "prompt_detail": "",
            "score": f"{avg:.2f}",
        }
    )

    col_widths = {
        "task_prompt": 45,
        "type": 18,
        "complexity": 10,
        "n_shots": 6,
        "prompt_detail": 13,
        "score": 5,
    }
    header = {k: k for k in col_widths}

    def fmt_row(d: dict) -> str:
        return "  ".join(
            str(d[k]).ljust(col_widths[k])[: col_widths[k]] for k in col_widths
        )

    separator = "  ".join("-" * w for w in col_widths.values())
    print("\n" + fmt_row(header))
    print(separator)
    for r in rows:
        print(fmt_row(r))
    print(separator)
    print(
        fmt_row(
            {
                "task_prompt": "AVERAGE",
                "type": "",
                "complexity": "",
                "n_shots": "",
                "prompt_detail": "",
                "score": f"{avg:.2f}",
            }
        )
    )
    print()


def run_benchmark(
    llm: BaseChatModel,
    out_dir: Path,
    tasks: List[Task],
    bench_logger: logging.Logger,
    experiment_id: uuid.UUID = uuid.uuid4(),
):
    benchmark = ToolCallingAgentBenchmark(
        tasks=tasks,
        logger=bench_logger,
        model_name=get_llm_model_name(llm),
        results_dir=out_dir,
    )

    with tqdm(total=len(tasks), unit="task", dynamic_ncols=True) as pbar:
        for task in tasks:
            short_prompt = task.get_base_prompt()[:50].replace("\n", " ")
            pbar.set_description(f"{short_prompt!r}")
            agent = create_conversational_agent(
                llm=llm,
                tools=task.available_tools,
                system_prompt=task.get_system_prompt(),
                logger=bench_logger,
            )
            benchmark.run_next(
                agent=agent,
                initial_state={
                    "messages": [HumanMultimodalMessage(content=task.get_prompt())]
                },
                experiment_id=experiment_id,
            )
            avg_score = statistics.mean(r.score for r in benchmark.task_results)
            last_score = benchmark.task_results[-1].score
            pbar.set_postfix(avg=f"{avg_score:.2f}", last=f"{last_score:.2f}")
            pbar.update(1)

    bench_logger.info("ALL SCENARIOS DONE. BENCHMARK COMPLETED!")
    _print_results_table(benchmark.task_results)
