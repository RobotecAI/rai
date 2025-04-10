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

from typing import Any, Dict

from langchain_core.messages import ToolCall

from rai_bench.tool_calling_agent.interfaces import SubTask


class CheckToolCallSubTask(SubTask):
    def __init__(
        self,
        expected_tool_name: str,
        expected_args: Dict[str, Any] = {},
        expected_optional_args: Dict[str, Any] = {},
    ):
        self.expected_tool_name = expected_tool_name
        self.expected_args = expected_args
        self.expected_optional_args = expected_optional_args

    def validate(
        self,
        tool_call: ToolCall,
    ) -> bool:
        return self._check_tool_call(
            tool_call=tool_call,
            expected_name=self.expected_tool_name,
            expected_args=self.expected_args,
            expected_optional_args=self.expected_optional_args,
        )
