# Copyright (C) 2026 Robotec.AI
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

"""ReplayChatModel: a chat model that answers with recorded responses."""

import threading
from pathlib import Path
from typing import Any, Optional, Union

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, messages_from_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import ConfigDict, PrivateAttr

from rai.recorder import LLM, Recording


class ReplayDivergence(AssertionError):
    """A replayed session departed from its recording."""

    def __init__(self, message: str, divergences: Any = ()) -> None:
        super().__init__(message)
        self.divergences = list(divergences)


class ReplayExhausted(ReplayDivergence):
    """The agent asked the model more often than the recording has answers."""


def recorded_agent(recording: Recording, agent: Optional[str]) -> str:
    """agent, or the only agent with model calls in the recording."""
    names = {c["agent"] for c in recording.values(LLM)}
    if agent is not None:
        if agent not in names:
            raise ValueError(
                f"No model calls of agent {agent!r} in {recording.path}"
                f" (agents: {', '.join(sorted(names)) or 'none'})"
            )
        return agent
    if len(names) != 1:
        raise ValueError(
            f"{recording.path} has model calls of {len(names)} agents"
            f" ({', '.join(sorted(names))}): pass agent="
        )
    return names.pop()


class ReplayChatModel(BaseChatModel):
    """Answers each model call with the next recorded response of one agent
    (``/rai/agent/llm``), in order, whatever the prompt.

    Tools are accepted (``bind_tools``) and ignored: the recorded responses
    carry the tool calls. When the recording runs out it raises
    :class:`ReplayExhausted`. Prompts are compared with the recording by
    :class:`~rai.replay.AgentReplay`, which can make this model fail its next
    call (:meth:`fail_next`) so a diverged run stops at once.
    """

    responses: list[Any]
    model_name: str = "replay"

    model_config = ConfigDict(arbitrary_types_allowed=True)

    _index: int = PrivateAttr(default=0)
    _error: Optional[BaseException] = PrivateAttr(default=None)
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    @classmethod
    def from_recording(
        cls, recording: Union[str, Path, Recording], agent: Optional[str] = None
    ) -> "ReplayChatModel":
        if not isinstance(recording, Recording):
            recording = Recording(recording)
        name = recorded_agent(recording, agent)
        calls = [c for c in recording.values(LLM) if c["agent"] == name]
        responses = [
            recording.restore(c["message"]) for c in calls if c["phase"] == "response"
        ]
        models = [c["model"] for c in calls if c["phase"] == "request" and c["model"]]
        return cls(responses=responses, model_name=models[0] if models else "replay")

    @property
    def _llm_type(self) -> str:
        return "replay"

    @property
    def calls(self) -> int:
        """Responses served so far."""
        return self._index

    @property
    def remaining(self) -> int:
        return len(self.responses) - self._index

    def fail_next(self, error: BaseException) -> None:
        """Raise error on the next call instead of answering."""
        self._error = error

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ReplayChatModel":
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        with self._lock:
            if self._error is not None:
                error, self._error = self._error, None
                raise error
            if self._index >= len(self.responses):
                raise ReplayExhausted(
                    f"model call {self._index + 1}: the recording has only"
                    f" {len(self.responses)} responses"
                )
            data = self.responses[self._index]
            self._index += 1
        (message,) = messages_from_dict([data])
        if not isinstance(message, AIMessage):
            message = AIMessage(content=message.content)
        return ChatResult(generations=[ChatGeneration(message=message)])
