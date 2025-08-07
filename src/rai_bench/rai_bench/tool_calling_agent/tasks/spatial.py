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
from abc import abstractmethod
from typing import List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai.messages import preprocess_image

from rai_bench.tool_calling_agent.interfaces import Task, Validator

loggers_type = logging.Logger

SPATIAL_REASONING_SYSTEM_PROMPT = "You are a helpful and knowledgeable AI assistant that specializes in interpreting and analyzing visual content. Your task is to answer questions based on the images provided to you. Please response with the use of the provided tools."


class TaskParametrizationError(Exception):
    """Exception raised when the task parameters are not valid."""

    pass


class SpatialReasoningAgentTask(Task):
    """Abstract class for spatial reasoning tasks for tool calling agent."""

    def __init__(
        self,
        validators: List[Validator],
        extra_tool_calls: int = 0,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            validators=validators,
            extra_tool_calls=extra_tool_calls,
            logger=logger,
        )
        self.expected_tools: List[BaseTool]
        self.question: str
        self.images_paths: List[str]

    @property
    def type(self) -> str:
        return "spatial_reasoning"

    @abstractmethod
    def get_images(self) -> List[str]:
        """Get the images related to the task.

        Returns
        -------
        List[str]
            List of image paths
        """
        pass

    def get_system_prompt(self) -> str:
        return SPATIAL_REASONING_SYSTEM_PROMPT


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


class BoolImageTask(SpatialReasoningAgentTask):
    complexity = "easy"

    def __init__(
        self,
        task_input: BoolImageTaskInput,
        validators: List[Validator],
        extra_tool_calls: int = 0,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            validators=validators,
            extra_tool_calls=extra_tool_calls,
            logger=logger,
        )
        self.question = task_input.question
        self.images_paths = task_input.images_paths

    @property
    def available_tools(self) -> List[BaseTool]:
        return [ReturnBoolResponseTool()]

    def get_prompt(self):
        return self.question

    def get_images(self):
        images = [preprocess_image(image_path) for image_path in self.images_paths]
        return images
