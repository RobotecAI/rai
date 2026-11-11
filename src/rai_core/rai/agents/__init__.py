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

"""Agents. The LLM agents load LangChain on first access."""

import importlib

from rai.agents.base import AgentState, BaseAgent, InvalidTransitionError, SyncAgent
from rai.agents.runner import ThreadedAgent, run

_LAZY = {
    "ReActAgent": "rai.agents.react",
    "StateBasedAgent": "rai.agents.react",
    "StateContext": "rai.agents.react",
    "StateSource": "rai.agents.react",
    "HRIChunk": "rai.agents.hri",
    "HRIOutput": "rai.agents.hri",
    "hri_connector_sender": "rai.agents.hri",
}

__all__ = [
    "AgentState",
    "BaseAgent",
    "InvalidTransitionError",
    "SyncAgent",
    "ThreadedAgent",
    "run",
    *_LAZY,
]


def __getattr__(name: str):
    if name in _LAZY:
        return getattr(importlib.import_module(_LAZY[name]), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
