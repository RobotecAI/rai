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
from typing import Any, List, Optional, Sequence

from langchain_core.messages import AIMessage
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai.messages import preprocess_image

from rai_bench.tool_calling_agent_bench.agent_tasks_interfaces import (
    SpatialReasoningAgentTask,
)


class TaskParametrizationError(Exception):
    """Exception raised when the task parameters are not valid."""

    pass


SPATIAL_REASONING_SYSTEM_PROMPT = "You are a helpful and knowledgeable AI assistant that specializes in interpreting and analyzing visual content. Your task is to answer questions based on the images provided to you. Please response with the use of the provided tools."


class ReturnBoolResponseToolInput(BaseModel):
    response: bool = Field(..., description="The response to the question.")


class ReturnBoolResponseTool(BaseTool):
    """Tool that returns a boolean response."""

    name: str = "return_bool_response"
    description: str = "Return a bool response to the question."
    args_schema = ReturnBoolResponseToolInput

    def _run(self, response: bool) -> bool:
        if type(response) is bool:
            return response
        raise ValueError("Invalid response type. Response must be a boolean.")


class BoolImageTaskInput(BaseModel):
    question: str = Field(..., description="The question to be answered.")
    images_paths: List[str] = Field(
        ...,
        description="List of image file paths to be used for answering the question.",
    )
    expected_response: bool = Field(
        ..., description="The expected answer to the question."
    )


class BoolImageTask(SpatialReasoningAgentTask):
    complexity = "easy"

    def __init__(
        self,
        task_input: BoolImageTaskInput,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        super().__init__(logger)
        self.expected_tools = [ReturnBoolResponseTool()]
        self.question = task_input.question
        self.images_paths = task_input.images_paths
        self.expected_response = task_input.expected_response

    def get_system_prompt(self) -> str:
        return SPATIAL_REASONING_SYSTEM_PROMPT

    def get_prompt(self):
        return self.question

    def get_images(self):
        images = [preprocess_image(image_path) for image_path in self.images_paths]
        return images

    def verify_tool_calls(self, response: dict[str, Any]):
        messages = response["messages"]
        ai_messages: Sequence[AIMessage] = [
            message for message in messages if isinstance(message, AIMessage)
        ]

        if ai_messages:
            if self._check_tool_calls_num_in_ai_message(ai_messages[0], expected_num=1):
                self._check_tool_call(
                    tool_call=ai_messages[0].tool_calls[0],
                    expected_name="return_bool_response",
                    expected_args={"response": self.expected_response},
                )
        if not self.result.errors:
            self.result.success = True
