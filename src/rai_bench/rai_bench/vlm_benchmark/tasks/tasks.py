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
from typing import List, Type

from pydantic import Field
from rai.messages import preprocess_image

from rai_bench.vlm_benchmark.interfaces import (
    ImageReasoningAnswer,
    ImageReasoningTask,
    ImageReasoningTaskInput,
)

loggers_type = logging.Logger


class BoolAnswerWithJustification(ImageReasoningAnswer[bool]):
    """A boolean answer to the user question along with justification for the answer."""


class QuantityAnswerWithJustification(ImageReasoningAnswer[int]):
    """A quantity answer telling the number of objects to the user question along with justification for the answer."""


class MultipleChoiceAnswerWithJustification(ImageReasoningAnswer[List[str]]):
    """A multiple choice answer to the user question along with justification for the answer."""


class BoolImageTaskInput(ImageReasoningTaskInput[bool]):
    """Input for a task that requires a boolean answer to a question about an image."""


class QuantityImageTaskInput(ImageReasoningTaskInput[int]):
    """Input for a task that requires counting objects in an image."""


class MultipleChoiceImageTaskInput(ImageReasoningTaskInput[List[str]]):
    """Input for a task that requires selecting one or more answers from a list of options."""

    options: List[str] = Field(
        ...,
        description="List of possible answers to the question.",
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

    def validate(self, output: BoolAnswerWithJustification) -> float:
        return float(output.answer == self.expected_answer)


class QuantityImageTask(ImageReasoningTask[QuantityAnswerWithJustification]):
    """A task that requires counting objects in an image."""

    complexity = "medium"

    def __init__(
        self,
        task_input: QuantityImageTaskInput,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.question = task_input.question
        self.images_paths = task_input.images_paths
        self.expected_answer = task_input.expected_answer

    @property
    def type(self) -> str:
        return "quantity_response_image_task"

    @property
    def structured_output(self) -> Type[QuantityAnswerWithJustification]:
        return QuantityAnswerWithJustification

    def validate(self, output: QuantityAnswerWithJustification) -> float:
        return float(output.answer == self.expected_answer)

    def get_prompt(self) -> str:
        return self.question

    def get_images(self):
        images = [preprocess_image(image_path) for image_path in self.images_paths]
        return images


class MultipleChoiceImageTask(
    ImageReasoningTask[MultipleChoiceAnswerWithJustification]
):
    """A task that requires selecting one or more answers from a set of options."""

    complexity = "hard"

    def __init__(
        self,
        task_input: MultipleChoiceImageTaskInput,
        logger: loggers_type | None = None,
    ) -> None:
        super().__init__(logger=logger)
        self.question = task_input.question
        self.images_paths = task_input.images_paths
        self.options = task_input.options
        self.expected_answer = task_input.expected_answer

    @property
    def type(self) -> str:
        return "multiple_choice_response_image_task"

    @property
    def structured_output(self) -> Type[MultipleChoiceAnswerWithJustification]:
        return MultipleChoiceAnswerWithJustification

    def validate(self, output: MultipleChoiceAnswerWithJustification) -> float:
        answers_processed = set([answer.casefold() for answer in output.answer])
        expected_processed = set([answer.casefold() for answer in self.expected_answer])

        if not answers_processed.issubset(expected_processed):
            return 0.0

        correct_count = len(answers_processed.intersection(expected_processed))
        total_expected = len(expected_processed)

        return float(correct_count / total_expected) if total_expected > 0 else 0.0

    def get_prompt(self) -> str:
        return (
            self.question
            + " Choose one or more answers from the options: "
            + ", ".join(self.options)
        )

    def get_images(self):
        images = [preprocess_image(image_path) for image_path in self.images_paths]
        return images
