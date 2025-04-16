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
from typing import List, Tuple

from langchain_core.messages import ToolCall

from rai_bench.tool_calling_agent.interfaces import (
    SubTask,
    SubTaskValidationError,
    Validator,
)

loggers_type = logging.Logger


class OrderedCallsValidator(Validator):
    """
    Validator that requires a strict order of subtaks
    The next subtask will be validate only when the previous one was completed
    """

    def __init__(
        self, subtasks: List[SubTask], logger: loggers_type | None = None
    ) -> None:
        super().__init__(subtasks=subtasks, logger=logger)
        if len(self.subtasks) < 1:
            raise ValueError("Validator must have at least 1 subtask.")

    def validate(self, tool_calls: List[ToolCall]) -> Tuple[bool, List[ToolCall]]:
        # Before validation create new iterator, in case validator
        # was used before in other task
        subtask_iter = iter(self.subtasks)
        if len(tool_calls) < 1:
            self.validation_error("Not a single tool call to validate")
            return False, tool_calls

        else:
            subtask = next(subtask_iter)
            for i, tool_call in enumerate(tool_calls):
                try:
                    if subtask.validate(tool_call=tool_call):
                        subtask = next(subtask_iter)
                except SubTaskValidationError as e:
                    # TODO (jm) should we log every error to results, even if its not causing validator failure?
                    # for now we log subtask errors to log file, but not to results.csv
                    self.logger.error(str(e))
                    # self.log_error(msg=str(e))
                except StopIteration:
                    return True, tool_calls[i + 1 :]

            self.validation_error(
                "Not all subtasks were completed in given tool calls."
            )
            return False, []


class NotOrderedCallsValidator(Validator):
    """
    Validator that don't enforce order of subtaks
    Every subtask will be validated against every tool call
    """

    def __init__(
        self, subtasks: List[SubTask], logger: loggers_type | None = None
    ) -> None:
        super().__init__(subtasks=subtasks, logger=logger)

    def validate(self, tool_calls: List[ToolCall]) -> Tuple[bool, List[ToolCall]]:
        if len(tool_calls) < 1:
            self.validation_error("Not a single tool call to validate")
            return False, tool_calls
        to_be_done = self.subtasks.copy()
        for i, tool_call in enumerate(tool_calls):
            if not to_be_done:
                # all subtask completed
                return True, tool_calls[i + 1 :]
            for u, subtask in enumerate(to_be_done):
                try:
                    subtask.validate(tool_call=tool_call)
                    to_be_done.pop(u)
                    break
                except SubTaskValidationError as e:
                    self.logger.error(str(e))

        if not to_be_done:
            # all tool calls iterated
            # all subtask completed
            return True, []
        self.validation_error("Not all subtasks were completed in given tool calls.")
        return False, []
