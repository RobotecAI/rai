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

"""A Streamlit chat for a :class:`~rai.agents.ReActAgent`, rendered from its
:mod:`event stream <rai.agents.events>`.

::

    # app.py, run with `streamlit run app.py` (needs rai_core[ui])
    import streamlit as st
    from rai.agents import ReActAgent
    from rai.ui.streamlit import chat

    @st.cache_resource
    def make_agent() -> ReActAgent:
        return ReActAgent("complex", tools)

    chat(make_agent(), title="My robot", greeting="Hi! What can I do for you?")

Each prompt runs one :meth:`~rai.agents.ReActAgent.ainvoke` in the script
thread. Model text appears after each model call, every tool call is a status
block showing its effect and result (denials stand out), and images from tool
artifacts (``{"images": [base64, ...]}``) are shown inline. The conversation
the model sees is ``agent.history``; the page keeps its own :class:`Turn` list
per browser session. Don't also run the agent with :func:`rai.run`.

:class:`Turn` folds events into a view model with no Streamlit dependency;
only :func:`chat` and :func:`reset` import streamlit.
"""

import asyncio
import base64
import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from rai.agents.events import (
    AgentError,
    AgentEvent,
    FinalMessage,
    Interrupted,
    ModelEnd,
    ModelStart,
    ToolEnd,
    ToolStart,
)

if TYPE_CHECKING:
    from rai.agents import ReActAgent

_KEY = "rai_chat_turns"
_STATES = {"running": "running", "success": "complete"}  # others: "error"
_SUFFIX = {
    "error": " :red[failed]",
    "denied": " :orange[denied]",
    "cancelled": " :gray[cancelled]",
}


@dataclass
class ToolCall:
    call_id: str
    name: str
    args: dict[str, Any]
    effect: str
    status: str = "running"  # success, error, denied or cancelled
    content: str = ""
    images: list[bytes] = field(default_factory=list)

    @property
    def label(self) -> str:
        icon = "🚫 " if self.status == "denied" else ""
        return (
            f"{icon}**{self.name}** :gray[{self.effect}]{_SUFFIX.get(self.status, '')}"
        )

    @property
    def state(self) -> str:
        return _STATES.get(self.status, "error")

    @property
    def expanded(self) -> bool:
        return bool(self.images) or self.status in ("error", "denied")


@dataclass
class Turn:
    """What the page shows for one prompt: the reply as text and tool calls."""

    prompt: str
    parts: list[str | ToolCall] = field(default_factory=list)
    thinking: bool = False
    done: bool = False
    error: str | None = None

    def apply(self, event: AgentEvent) -> bool:
        """Fold ``event`` into the turn. Returns whether anything changed."""
        match event:
            case ModelStart():
                self.thinking = True
            case ModelEnd(message=m):
                self.thinking = False
                self._say(m.text)
            case ToolStart():
                self.parts.append(
                    ToolCall(event.call_id, event.name, event.args, str(event.effect))
                )
            case ToolEnd() if (call := self._call(event.call_id)) is not None:
                call.status = event.status
                call.content = _text(event.content)
                call.images = _images(event.artifact)
            case FinalMessage(message=m):
                self._say(m.text)
                self._finish()
            case Interrupted():
                self._finish("Interrupted.")
            case AgentError(error=e):
                self._finish(f"{type(e).__name__}: {e}")
            case _:
                return False
        return True

    def _say(self, text: str) -> None:
        if text and not (self.parts and self.parts[-1] == text):
            self.parts.append(text)

    def _call(self, call_id: str) -> ToolCall | None:
        calls = (p for p in self.parts if isinstance(p, ToolCall))
        return next((c for c in calls if c.call_id == call_id), None)

    def _finish(self, error: str | None = None) -> None:
        self.thinking, self.done, self.error = False, True, error
        for part in self.parts:
            if isinstance(part, ToolCall) and part.status == "running":
                part.status = "cancelled"


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, default=str)


def _images(artifact: Any) -> list[bytes]:
    images = artifact.get("images") if isinstance(artifact, dict) else None
    return [base64.b64decode(i) for i in images or []]


def _render_reply(st: Any, turn: Turn) -> None:
    for part in turn.parts:
        if isinstance(part, str):
            st.markdown(part)
            continue
        with st.status(part.label, state=part.state, expanded=part.expanded):
            st.code(json.dumps(part.args, default=str), language="json")
            if part.content:
                st.code(part.content, language=None, wrap_lines=True)
            for image in part.images:
                st.image(image)
    if turn.thinking:
        st.caption("Thinking…")
    if turn.error:
        st.error(turn.error)


def reset(agent: "ReActAgent") -> None:
    """Forget the conversation: the page's turns and ``agent.history``."""
    import streamlit as st

    st.session_state[_KEY] = []
    agent.history = []


def chat(
    agent: "ReActAgent",
    *,
    title: str | None = None,
    greeting: str | None = None,
    prepare: Callable[[str], Any] | None = None,
) -> None:
    """Render the chat page and run the agent on a new prompt.

    ``prepare`` turns the prompt into what :meth:`~rai.agents.ReActAgent.ainvoke`
    gets, e.g. a message with a camera image attached.
    """
    import streamlit as st

    if title:
        st.title(title)
    if st.sidebar.button("Clear conversation"):
        reset(agent)
    turns: list[Turn] = st.session_state.setdefault(_KEY, [])
    if greeting:
        st.chat_message("assistant").markdown(greeting)
    for turn in turns:
        st.chat_message("user").markdown(turn.prompt)
        with st.chat_message("assistant"):
            _render_reply(st, turn)

    prompt = st.chat_input()
    if not prompt:
        return
    turn = Turn(prompt)
    turns.append(turn)
    st.chat_message("user").markdown(prompt)
    reply = st.chat_message("assistant").empty()

    def show(event: AgentEvent) -> None:
        if turn.apply(event):
            with reply.container():
                _render_reply(st, turn)

    message = prepare(prompt) if prepare else prompt
    unsubscribe = agent.subscribe(show)
    try:
        asyncio.run(agent.ainvoke(message))
    except Exception as e:
        if not turn.done:  # failed before the turn started, so no AgentError
            show(AgentError(agent=agent.name, turn="", error=e))
    finally:
        unsubscribe()
