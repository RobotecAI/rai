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
from typing import Generic, List, Literal, TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import DEFAULT_RECURSION_LIMIT
from pydantic import BaseModel

loggers_type = logging.Logger

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


IMAGE_REASONING_SYSTEM_PROMPT = "You are a helpful and knowledgeable AI assistant that specializes in interpreting and analyzing visual content. Your task is to answer questions based on the images provided to you. Please response in requested structured output format."


class ImageReasoningTask(ABC, Generic[BaseModelT]):
    complexity: Literal["easy", "medium", "hard"]
    recursion_limit: int = DEFAULT_RECURSION_LIMIT

    def __init__(
        self,
        logger: loggers_type | None = None,
    ) -> None:
        """
        Abstract base class representing a complete task to be validated.

        A Task consists of multiple Validators, where each Validator can be treated as a single
        step that is scored atomically. Each Task has a consistent prompt and available tools,
        with validation methods that can be parameterized.

        Attributes
        ----------
        logger : logging.Logger
            Logger for recording task validation results and errors.
        result : Result
            Object tracking the validation results across all validators.
        """
        if logger:
            self.logger = logger
        else:
            self.logger = logging.getLogger(__name__)
        self.question: str
        self.images_paths: List[str]

    def set_logger(self, logger: loggers_type):
        self.logger = logger

    @property
    @abstractmethod
    def structured_output(self) -> type[BaseModelT]:
        """Structured output that agent should return."""
        pass

    @property
    @abstractmethod
    def type(self) -> str:
        """Type of task, for example: image_reasoning"""
        pass

    def get_system_prompt(self) -> str:
        """Get the system prompt that will be passed to agent

        Returns
        -------
        str
            System prompt
        """
        return IMAGE_REASONING_SYSTEM_PROMPT

    @abstractmethod
    def get_prompt(self) -> str:
        """Get the task instruction - the prompt that will be passed to agent.

        Returns
        -------
        str
            Prompt
        """
        pass

    @abstractmethod
    def validate(self, output: BaseModelT) -> bool:
        """Validate result of the task."""
        pass

    @abstractmethod
    def get_images(self) -> List[str]:
        """Get the images related to the task.

        Returns
        -------
        List[str]
            List of image paths
        """
        pass

    def get_structured_output_from_messages(
        self, messages: List[BaseMessage]
    ) -> BaseModelT:
        """Get structured output from messages and validate it matches the expected type."""
        for message in reversed(messages):
            if isinstance(message, self.structured_output):
                return message
        raise ValueError(f"No {self.structured_output.__name__} found in messages")
