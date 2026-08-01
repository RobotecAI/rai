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

import pytest
from pydantic import ValidationError

from rai_bench.tool_calling_agent.interfaces import TaskArgs


def test_task_args_extra_tool_calls_default_and_positive() -> None:
    assert TaskArgs().extra_tool_calls == 0
    assert TaskArgs(extra_tool_calls=0).extra_tool_calls == 0
    assert TaskArgs(extra_tool_calls=3).extra_tool_calls == 3


def test_task_args_extra_tool_calls_rejects_negative() -> None:
    with pytest.raises(ValidationError):
        TaskArgs(extra_tool_calls=-1)


def test_task_args_extra_tool_calls_rejects_bool() -> None:
    with pytest.raises(ValidationError):
        TaskArgs(extra_tool_calls=True)  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        TaskArgs(extra_tool_calls=False)  # type: ignore[arg-type]
