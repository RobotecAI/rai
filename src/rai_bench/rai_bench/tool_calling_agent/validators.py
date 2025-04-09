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
    def __init__(
        self, subtasks: List[SubTask], logger: loggers_type | None = None
    ) -> None:
        super().__init__(subtasks=subtasks, logger=logger)
        self.subtask_iter = iter(subtasks)

    def validate(self, tool_calls: List[ToolCall]) -> Tuple[bool, List[ToolCall]]:
        for i, tool_call in enumerate(tool_calls):
            try:
                subtask = next(self.subtask_iter)
                subtask.validate(tool_call=tool_call)
            except SubTaskValidationError as e:
                self.log_error(msg=str(e))
                continue
            except StopIteration:
                return True, tool_calls[i:]

        self.log_error(msg="Failed to validate")
        return False, []


class NotOrderedCallsValidator(Validator):
    def __init__(
        self, subtasks: List[SubTask], logger: loggers_type | None = None
    ) -> None:
        super().__init__(subtasks=subtasks, logger=logger)

    def validate(self, tool_calls: List[ToolCall]) -> Tuple[bool, List[ToolCall]]:
        to_be_done = self.subtasks.copy()
        for i, tool_call in enumerate(tool_calls):
            if not to_be_done:
                # all subtask completed
                return True, tool_calls[i:]
            for subtask in to_be_done:
                try:
                    subtask.validate(tool_call=tool_call)
                    to_be_done.pop(i)
                    break
                except SubTaskValidationError:
                    continue
        return False, []
