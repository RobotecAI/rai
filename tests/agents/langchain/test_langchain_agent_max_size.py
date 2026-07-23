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

from unittest.mock import MagicMock

import pytest
from rai.agents.langchain.agent import LangChainAgent


def _make_agent(max_size: int) -> LangChainAgent:
    return LangChainAgent(
        target_connectors={},
        runnable=MagicMock(),
        max_size=max_size,
    )


def test_max_size_must_be_positive():
    for bad in (0, -1):
        with pytest.raises(ValueError, match="max_size must be positive"):
            _make_agent(bad)


def test_max_size_rejects_bool_and_non_int():
    for bad in (True, False, 1.5, "10", None):
        with pytest.raises(TypeError, match="max_size must be a positive int"):
            _make_agent(bad)  # type: ignore[arg-type]


def test_max_size_accepts_positive():
    agent = _make_agent(1)
    assert agent.max_size == 1
