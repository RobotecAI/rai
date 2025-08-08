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
from typing import Generic, List, Literal, Optional, TypeVar

from langchain_core.messages import BaseMessage
from langchain_core.runnables.config import DEFAULT_RECURSION_LIMIT
from pydantic import BaseModel, ConfigDict, ValidationError

loggers_type = logging.Logger

BaseModelT = TypeVar("BaseModelT", bound=BaseModel)


IMAGE_REASONING_SYSTEM_PROMPT = "You are a helpful and knowledgeable AI assistant that specializes in interpreting and analyzing visual content. Your task is to answer questions based on the images provided to you. Please response in requested structured output format."


class LangchainRawOutputModel(BaseModel):
    """
    A Pydantic model for wrapping Langchain message parsing results from a structured output agent. See documentation for more details:
    https://github.com/langchain-ai/langchain/blob/02001212b0a2b37d90451d8493089389ea220cab/libs/core/langchain_core/language_models/chat_models.py#L1430-L1432


    Attributes
    ----------
    raw : BaseMessage
        The original raw message object from Langchain before parsing.
    parsed : BaseModel
        The parsed and validated Pydantic model instance derived from the raw message.
    parsing_error : Optional[BaseException]
        Any exception that occurred during the parsing process, None if parsing
        was successful.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)
    raw: BaseMessage
    parsed: BaseModel
    parsing_error: Optional[BaseException]


class TaskValidationError(Exception):
    pass


class ImageReasoningTask(ABC, Generic[BaseModelT]):
    complexity: Literal["easy", "medium", "hard"]
    recursion_limit: int = DEFAULT_RECURSION_LIMIT

    def __init__(
        self,
        logger: loggers_type | None = None,
    ) -> None:
        """
        Abstract base class representing a complete image reasoning task to be validated.

        Each Task has a consistent prompt and structured output schema, along
        with validation methods that check the output against the expected result.

        Attributes
        ----------
        logger : logging.Logger
            Logger for recording task validation results and errors.
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
    ) -> BaseModelT | None:
        """Get structured output from the messages."""
        for message in reversed(messages):
            if isinstance(message, dict):
                try:
                    validated_message = LangchainRawOutputModel.model_validate(message)
                    if validated_message.parsing_error is not None:
                        raise TaskValidationError(
                            f"Parsing error: {validated_message.parsing_error}"
                        )

                    parsed = validated_message.parsed
                    if isinstance(parsed, self.structured_output):
                        return parsed
                except ValidationError:
                    continue
        return None
