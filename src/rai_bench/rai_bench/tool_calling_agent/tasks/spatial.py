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
from abc import ABC, abstractmethod
from typing import List

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai.messages import preprocess_image

from rai_bench.tool_calling_agent.interfaces import Task, TaskArgs, Validator

loggers_type = logging.Logger

SPATIAL_REASONING_SYSTEM_PROMPT_0_SHOT = """You are a helpful and knowledgeable AI assistant that specializes in interpreting and analyzing visual content. Your task is to answer questions based on the images provided to you. Please response with the use of the provided tools."""

SPATIAL_REASONING_SYSTEM_PROMPT_2_SHOT = (
    SPATIAL_REASONING_SYSTEM_PROMPT_0_SHOT
    + """

Example of tool calls:
- return_bool_response, args: {'response': True}
- return_bool_response, args: {'response': False}"""
)

# NOTE (jmatejcz) In this case we are using only one tool so there is no difference bettween 2 and 5 shot
SPATIAL_REASONING_SYSTEM_PROMPT_5_SHOT = (
    SPATIAL_REASONING_SYSTEM_PROMPT_2_SHOT
    + """
- return_bool_response, args: {'response': True}  # When object is clearly visible
- return_bool_response, args: {'response': False}  # When object is not present
- return_bool_response, args: {'response': True}  # When spatial relationship is correct"""
)


class TaskParametrizationError(Exception):
    """Exception raised when the task parameters are not valid."""

    pass


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


class SpatialReasoningAgentTask(Task):
    """Abstract class for spatial reasoning tasks for tool calling agent."""

    type = "spatial_reasoning"

    def __init__(
        self,
        validators: List[Validator],
        task_args: TaskArgs,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            validators=validators,
            task_args=task_args,
            logger=logger,
        )
        self.expected_tools: List[BaseTool]
        self.question: str
        self.images_paths: List[str]

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
        if self.n_shots == 0:
            return SPATIAL_REASONING_SYSTEM_PROMPT_0_SHOT
        elif self.n_shots == 2:
            return SPATIAL_REASONING_SYSTEM_PROMPT_2_SHOT
        else:
            return SPATIAL_REASONING_SYSTEM_PROMPT_5_SHOT


class BoolImageTask(SpatialReasoningAgentTask, ABC):
    def __init__(
        self,
        task_input: BoolImageTaskInput,
        validators: List[Validator],
        task_args: TaskArgs,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            validators=validators,
            task_args=task_args,
            logger=logger,
        )
        self.question = task_input.question
        self.images_paths = task_input.images_paths

    @property
    def available_tools(self) -> List[BaseTool]:
        return [ReturnBoolResponseTool()]

    @property
    def optional_tool_calls_number(self) -> int:
        return 0

    def get_base_prompt(self) -> str:
        return self.question

    def get_prompt(self):
        if self.prompt_detail == "brief":
            return self.get_base_prompt()
        elif self.prompt_detail == "moderate":
            return f"{self.get_base_prompt()} using visual analysis"
        else:
            return (
                f"{self.get_base_prompt()} using the visual analysis system. "
                "You can examine the provided image(s) carefully to identify relevant features, "
                "analyze the visual content, and provide a boolean response based on your observations."
            )

    def get_images(self):
        images = [preprocess_image(image_path) for image_path in self.images_paths]
        return images


# NOTE (jmatejcz) spatial reasoning task's deiffculty is based soly on prompt and image
# so in this case when declaring task, please subjectivly decide how hard is the task
# examples:
# easy -> locating single object, tell if it is present
# medium -> tell in what state is the object (is door open?) or locating multiple objects
# hard -> locating multiple objects and resoning about their relative positions (is X on the right side of Y?)
class BoolImageTaskEasy(BoolImageTask):
    complexity = "easy"


class BoolImageTaskMedium(BoolImageTask):
    complexity = "medium"


class BoolImageTaskHard(BoolImageTask):
    complexity = "hard"
