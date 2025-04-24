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

from collections import deque
from typing import List

import pytest
from rai.agents.langchain.agent import LangChainAgent, newMessageBehaviorType


@pytest.mark.parametrize(
    "new_message_behavior,in_buffer,out_buffer,output",
    [
        ("take_all", [1, 2, 3], [], [1, 2, 3]),
        ("keep_last", [1, 2, 3], [], [3]),
        ("queue", [1, 2, 3], [2, 3], [1]),
        ("interuppt_take_all", [1, 2, 3], [], [1, 2, 3]),
        ("interuppt_keep_last", [1, 2, 3], [], [3]),
    ],
)
def test_reduce_messages(
    new_message_behavior: newMessageBehaviorType,
    in_buffer: List,
    out_buffer: List,
    output: List,
):
    buffer = deque(in_buffer)
    output = LangChainAgent._apply_reduction_behavior(new_message_behavior, buffer)
    assert output == output
    assert buffer == deque(out_buffer)
