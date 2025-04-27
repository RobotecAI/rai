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
from typing import Any, Dict, List
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_core.tracers.langchain import LangChainTracer
from langfuse.callback import CallbackHandler
from pydantic import BaseModel, Field
from rai.initialization import get_tracing_callbacks

loggers_type = logging.Logger


class SubTaskResult(BaseModel):
    args: Dict[str, Any]
    errors: List[str]
    passed: bool


class ValidatorResult(BaseModel):
    type: str
    subtasks: List[SubTaskResult]
    extra_tool_calls_used: int
    passed: bool


class TaskResult(BaseModel):
    task_prompt: str = Field(..., description="The task prompt.")
    system_prompt: str = Field(..., description="The system prompt.")
    complexity: str = Field(..., description="Complexity of the task.")
    type: str = Field(..., description="Type of task, for example: manipulation")
    model_name: str = Field(..., description="Name of the LLM.")
    validation_info: List[ValidatorResult] = Field(
        ..., description="Validation structure, errors, etc."
    )
    extra_tool_calls: int = Field(
        ...,
        description="Maximum number of extra tool calls agent can make and still pass a task",
    )
    score: float = Field(
        ...,
        description="Value between 0 and 1, describing how many validation setps passed",
    )
    total_time: float = Field(..., description="Total time taken to complete the task.")
    run_id: UUID = Field(..., description="UUID of the task run.")


class BenchmarkSummary(BaseModel):
    model_name: str = Field(..., description="Name of the LLM.")
    success_rate: float = Field(
        ..., description="Percentage of successfully completed tasks."
    )
    avg_time: float = Field(..., description="Average time taken across all tasks.")
    total_tasks: int = Field(..., description="Total number of executed tasks.")


class ScoreTracingHandler:
    """
    Class to handle sending scores to tracing backends.
    """

    # TODO (mkotynia) handle grouping single benchmark scores to sessions
    # TODO (mkotynia) trace and send more metadata?
    @staticmethod
    def get_callbacks() -> List[BaseCallbackHandler]:
        return get_tracing_callbacks()

    @staticmethod
    def send_score(
        callback: BaseCallbackHandler,
        run_id: UUID,
        score: float,
        errors: List[List[str]],
    ) -> None:
        comment = (
            "; ".join(", ".join(error_group) for error_group in errors)
            if errors
            else ""
        )
        if isinstance(callback, CallbackHandler):
            callback.langfuse.score(
                trace_id=str(run_id),
                name="tool calls result",
                value=score,
                comment=comment,
            )
            return None
        if isinstance(callback, LangChainTracer):
            callback.client.create_feedback(
                run_id=run_id,
                key="tool calls result",
                score=score,
                comment=comment,
            )
            return None
        raise NotImplementedError(
            f"Callback {callback} of type {callback.__class__.__name__} not supported"
        )
