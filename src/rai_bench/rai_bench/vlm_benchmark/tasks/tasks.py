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
from typing import List

from pydantic import BaseModel, Field
from rai.messages import preprocess_image

from rai_bench.vlm_benchmark.interfaces import ImageReasoningTask

loggers_type = logging.Logger


class BoolAnswerWithJustification(BaseModel):
    """A boolean answer to the user question along with justification for the answer."""

    answer: bool
    justification: str


class BoolImageTaskInput(BaseModel):
    question: str = Field(..., description="The question to be answered.")
    images_paths: List[str] = Field(
        ...,
        description="List of image file paths to be used for answering the question.",
    )
    expected_answer: bool = Field(
        ..., description="The expected answer to the question."
    )


class BoolImageTask(ImageReasoningTask[BoolAnswerWithJustification]):
    complexity = "easy"

    def __init__(
        self,
        task_input: BoolImageTaskInput,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(
            logger=logger,
        )
        self.question = task_input.question
        self.images_paths = task_input.images_paths
        self.expected_answer = task_input.expected_answer

    @property
    def structured_output(self) -> type[BoolAnswerWithJustification]:
        return BoolAnswerWithJustification

    @property
    def type(self) -> str:
        return "bool_response_image_task"

    def get_prompt(self):
        return self.question

    def get_images(self):
        images = [preprocess_image(image_path) for image_path in self.images_paths]
        return images

    def validate(self, output: BoolAnswerWithJustification) -> bool:
        return output.answer == self.expected_answer
