# Copyright (C) 2024 Robotec.AI
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

from typing import Any, Dict, List, Type

from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

from rai.tools.utils import wrap_tool_input


def test_wrap_tool_input():
    class TestToolInput(BaseModel):
        a: int = Field(..., description="The number")
        b: str = Field(..., description="The string")
        c: Dict[str, Any] = Field(..., description="The dictionary")
        d: List[int] = Field(..., description="The list of numbers")
        e: bytes = Field(..., description="The bytes")

    class TestTool(BaseTool):
        name: str = "test_tool"
        description: str = "This is a test tool"
        args_schema: Type[TestToolInput] = TestToolInput

        @wrap_tool_input
        def _run(self, tool_input: TestToolInput) -> str:
            assert tool_input.a == 1
            assert tool_input.b == "test"
            assert tool_input.c == {"a": 1, "b": 2}
            assert tool_input.d == [1, 2, 3]
            assert tool_input.e == b"test"
            return "done"

    tool = TestTool()
    result = tool.invoke(
        {
            "a": 1,
            "b": "test",
            "c": {"a": 1, "b": 2},
            "d": [1, 2, 3],
            "e": b"test",
        }
    )
    assert result == "done"
