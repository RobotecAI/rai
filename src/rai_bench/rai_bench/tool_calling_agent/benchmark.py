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
from typing import Iterator, List, Optional, Sequence, Tuple

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.runnables.config import RunnableConfig
from langgraph.errors import GraphRecursionError
from langgraph.graph.state import CompiledStateGraph
from rai.agents.langchain.core import (
    create_conversational_agent,
)
from rai.agents.langchain.core.react_agent import ReActAgentState
from rai.messages import HumanMultimodalMessage

from rai_bench.agents import create_multimodal_to_tool_agent
from rai_bench.base_benchmark import BaseBenchmark, TimeoutException
from rai_bench.results_processing.langfuse_scores_tracing import ScoreTracingHandler
from rai_bench.tool_calling_agent.interfaces import (
    Task,
)
from rai_bench.tool_calling_agent.results_tracking import (
    TaskResult,
    ToolCallingAgentRunSummary,
)
from rai_bench.tool_calling_agent.tasks.spatial import (
    SpatialReasoningAgentTask,
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
        initial_state: ReActAgentState,
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
        # NOTE (jmatejcz) recursion limit calculated as all_nodes_num -> one pass though whole node
        # plus (task.max_tool_calls_number-1 because the first pass is already added in)
        # times number of nodes - 2 because we dont cout start and end node
        # this can be to much for larger graphs that dont use all nodes on extra calls
        # in such ase adjust this value
        # recurssion_limit = len(agent.get_graph().nodes) + (
        #     task.max_tool_calls_number - 1
        # ) * (len(agent.get_graph().nodes) - 2)
        recurssion_limit = 200
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

    for task in tasks:
        agent = create_conversational_agent(
            llm=llm,
            tools=task.available_tools,
            system_prompt=task.get_system_prompt(),
            logger=bench_logger,
        )
        benchmark.run_next(agent=agent, experiment_id=experiment_id)

    bench_logger.info("===============================================================")
    bench_logger.info("ALL SCENARIOS DONE. BENCHMARK COMPLETED!")
    bench_logger.info("===============================================================")


def run_benchmark_dual_agent(
    multimodal_llm: BaseChatModel,
    tool_calling_llm: BaseChatModel,
    out_dir: Path,
    tasks: List[Task],
    bench_logger: logging.Logger,
    experiment_id: uuid.UUID = uuid.uuid4(),
    m_system_prompt: Optional[str] = None,
    tool_system_prompt: Optional[str] = None,
):
    benchmark = ToolCallingAgentBenchmark(
        tasks=tasks,
        logger=bench_logger,
        model_name=get_llm_model_name(multimodal_llm),
        results_dir=out_dir,
    )

    basic_tool_system_prompt = (
        "Based on the conversation call the tools with appropriate arguments"
    )
    for task in tasks:
        agent = create_multimodal_to_tool_agent(
            multimodal_llm=multimodal_llm,
            tool_llm=tool_calling_llm,
            tools=task.available_tools,
            multimodal_system_prompt=(
                m_system_prompt if m_system_prompt else task.get_system_prompt()
            ),
            tool_system_prompt=(
                tool_system_prompt if tool_system_prompt else basic_tool_system_prompt
            ),
            logger=bench_logger,
            debug=False,
        )

        benchmark.run_next(agent=agent, experiment_id=experiment_id)

    bench_logger.info("===============================================================")
    bench_logger.info("ALL SCENARIOS DONE. BENCHMARK COMPLETED!")
    bench_logger.info("===============================================================")
