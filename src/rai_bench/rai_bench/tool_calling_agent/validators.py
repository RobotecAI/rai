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
from typing import Dict, List, Tuple

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
    The next subtask will be validated only when the previous one was completed
    """

    def __init__(
        self, subtasks: List[SubTask], logger: loggers_type | None = None
    ) -> None:
        super().__init__(subtasks=subtasks, logger=logger)
        if len(self.subtasks) < 1:
            raise ValueError("Validator must have at least 1 subtask.")

    @property
    def type(self) -> str:
        return "ordered"

    def validate(self, tool_calls: List[ToolCall]) -> Tuple[bool, List[ToolCall]]:
        self.reset()
        # Before validation create new iterator, in case validator
        # was used before in other task
        subtask_iter = iter(enumerate(self.subtasks))
        if len(tool_calls) < 1:
            self.logger.debug("Not a single tool call to validate")
            self.passed = False
            return False, tool_calls

        else:
            u, subtask = next(subtask_iter)
            for i, tool_call in enumerate(tool_calls):
                try:
                    if subtask.validate(tool_call=tool_call):
                        self.subtasks_passed[u] = True
                        # go to next subtask
                        u, subtask = next(subtask_iter)
                except SubTaskValidationError as e:
                    self.add_subtask_errors(idx=u, msgs=[str(e)])

                except StopIteration:
                    self.passed = True
                    self.extra_calls_used = i + 1 - self.required_calls
                    return True, tool_calls[i + 1 :]

            self.logger.debug(f"Validation failed for task {u + 1}")
            self.passed = False
            if len(tool_calls) > self.required_calls:
                self.extra_calls_used += len(tool_calls) - self.required_calls
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
        if len(self.subtasks) < 1:
            raise ValueError("Validator must have at least 1 subtask.")

    @property
    def type(self) -> str:
        return "not ordered"

    def validate(self, tool_calls: List[ToolCall]) -> Tuple[bool, List[ToolCall]]:
        self.reset()
        if len(tool_calls) < 1:
            self.logger.debug("Not a single tool call to validate")
            self.passed = False
            return False, tool_calls

        # for saving to result which tasks where not done
        to_be_done_idx = list(range(len(self.subtasks)))

        for i, tool_call in enumerate(tool_calls):
            if not to_be_done_idx:
                # all subtask completed
                self.passed = True
                self.extra_calls_used = i - self.required_calls
                return True, tool_calls[i:]

            matched = False
            possible_errors: Dict[int, str] = {}
            for u in to_be_done_idx:
                try:
                    if self.subtasks[u].validate(tool_call=tool_call):
                        to_be_done_idx.remove(u)
                        self.subtasks_passed[u] = True
                        matched = True
                        break
                except SubTaskValidationError as e:
                    possible_errors[u] = str(e)

            if not matched:
                # tool call did not match any subtask
                # so add recent error from every subtask
                # NOTE (jm) this can make multiple errors from 1 tool call in results

                for idx, error in possible_errors.items():
                    self.add_subtask_errors(idx=idx, msgs=[error])

        if not to_be_done_idx:
            # all tool calls iterated
            # all subtask completed
            self.passed = True
            self.extra_calls_used = len(tool_calls) - self.required_calls
            return True, []

        self.logger.debug(
            f"Validation failed for tasks: {[idx + 1 for idx in to_be_done_idx]}"
        )
        self.passed = False
        if len(tool_calls) > self.required_calls:
            self.extra_calls_used = len(tool_calls) - self.required_calls
        return False, []
