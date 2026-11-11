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

"""LLM agents on LangChain's ``create_agent``: :class:`ReActAgent` and
:class:`StateBasedAgent`."""

import asyncio
import logging
import uuid
from collections import deque
from collections.abc import Awaitable, Callable, Iterable, Mapping, Sequence
from contextvars import ContextVar
from typing import Any, Literal

from langchain.agents import create_agent
from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
    ModelRequest,
    ModelResponse,
)
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Checkpointer

from rai.agents.base import BaseAgent
from rai.agents.events import (
    AgentError,
    FinalMessage,
    Interrupted,
    ModelEnd,
    ModelStart,
    ToolEnd,
    ToolStart,
)
from rai.communication.hri_connector import HRIMessage
from rai.llm import get_llm, get_tracing_callbacks
from rai.messages import human, with_tool_images
from rai.tools import Effect, RaiTool, from_langchain

logger = logging.getLogger(__name__)

StateContent = str | list[str | dict[str, Any]]
"""Text, or LangChain content blocks (text and images)."""

StateSource = Callable[[], Awaitable[StateContent | None]]
"""Reads one piece of robot state. ``None`` means nothing new: the last value stays."""

_TURN: ContextVar[str] = ContextVar("rai_agent_turn", default="")


class _AgentHooks(AgentMiddleware[Any, Any]):
    """Emits the agent's model and tool events, and shows tool images to the model."""

    def __init__(self, agent: BaseAgent) -> None:
        super().__init__()
        self.agent = agent

    def _before_model(self, request: ModelRequest[Any]) -> ModelRequest[Any]:
        self.agent.emit(ModelStart(agent=self.agent.name, turn=_TURN.get()))
        return request.override(messages=with_tool_images(request.messages))

    def _after_model(self, response: Any) -> None:
        result = response.result if isinstance(response, ModelResponse) else [response]
        for message in result:
            if isinstance(message, AIMessage):
                self.agent.emit(
                    ModelEnd(agent=self.agent.name, turn=_TURN.get(), message=message)
                )

    def _before_tool(self, request: Any) -> Effect:
        call = request.tool_call
        meta = getattr(request.tool, "metadata", None) or {}
        effect = Effect(meta.get("effect", Effect.ACTUATE))
        self.agent.emit(
            ToolStart(
                agent=self.agent.name,
                turn=_TURN.get(),
                call_id=call["id"],
                name=call["name"],
                args=call["args"],
                effect=effect,
            )
        )
        return effect

    def _after_tool(self, request: Any, effect: Effect, result: Any) -> None:
        if not isinstance(result, ToolMessage):
            return
        status = result.status
        if status == "error" and str(result.content).startswith("Denied"):
            status = "denied"  # the RaiTool hook chain refused the call
        self.agent.emit(
            ToolEnd(
                agent=self.agent.name,
                turn=_TURN.get(),
                call_id=result.tool_call_id,
                name=request.tool_call["name"],
                effect=effect,
                status=status,
                content=result.content,
                artifact=result.artifact,
            )
        )

    def wrap_model_call(self, request, handler):  # type: ignore[override]
        response = handler(self._before_model(request))
        self._after_model(response)
        return response

    async def awrap_model_call(self, request, handler):  # type: ignore[override]
        response = await handler(self._before_model(request))
        self._after_model(response)
        return response

    def wrap_tool_call(self, request, handler):  # type: ignore[override]
        effect = self._before_tool(request)
        result = handler(request)
        self._after_tool(request, effect, result)
        return result

    async def awrap_tool_call(self, request, handler):  # type: ignore[override]
        effect = self._before_tool(request)
        result = await handler(request)
        self._after_tool(request, effect, result)
        return result


class StateContext(AgentMiddleware[Any, Any]):
    """Appends the latest robot state to every model request, never to the history.

    All sources are read concurrently before each model call, each under its own
    ``timeout_s``. A source that fails, times out or returns ``None`` keeps its
    last good value.
    """

    def __init__(self, sources: Mapping[str, StateSource], timeout_s: float = 5.0):
        super().__init__()
        self.sources = dict(sources)
        self.timeout_s = timeout_s
        self.last: dict[str, StateContent] = {}

    async def _read(self, name: str, source: StateSource) -> None:
        try:
            async with asyncio.timeout(self.timeout_s):
                value = await source()
        except Exception as e:
            logger.warning(
                "State source %s failed, keeping its last value: %r", name, e
            )
            return
        if value is not None:
            self.last[name] = value

    async def observe(self) -> HumanMessage | None:
        """Refresh all sources and render them as one message."""
        await asyncio.gather(*(self._read(n, s) for n, s in self.sources.items()))
        blocks: list[str | dict[str, Any]] = []
        for name in self.sources:
            value = self.last.get(name)
            if isinstance(value, str):
                blocks.append({"type": "text", "text": f"{name}: {value}"})
            elif value is not None:
                blocks += [{"type": "text", "text": f"{name}:"}, *value]
        if not blocks:
            return None
        return HumanMessage([{"type": "text", "text": "Current robot state:"}, *blocks])

    async def awrap_model_call(self, request, handler):  # type: ignore[override]
        if observation := await self.observe():
            request = request.override(messages=[*request.messages, observation])
        return await handler(request)


def _to_message(message: str | BaseMessage | HRIMessage) -> BaseMessage:
    if isinstance(message, str):
        return HumanMessage(message)
    if isinstance(message, HRIMessage):
        return human(message.text, message.images, message.audios)
    return message


def _unanswered(messages: Sequence[BaseMessage]) -> list[ToolMessage]:
    """Error results for tool calls of the last AI message that never finished,
    so the history stays valid for the next model call."""
    answered = {m.tool_call_id for m in messages if isinstance(m, ToolMessage)}
    last = next((m for m in reversed(messages) if isinstance(m, AIMessage)), None)
    return [
        ToolMessage(
            "Interrupted before the tool finished.",
            tool_call_id=call["id"],
            name=call["name"],
            status="error",
        )
        for call in (last.tool_calls if last else [])
        if call["id"] not in answered
    ]


class ReActAgent(BaseAgent):
    """A tool-calling LLM agent (``langchain.agents.create_agent``).

    Use it directly with :meth:`ainvoke` (no start needed), or run it with
    :func:`rai.run` / :class:`ThreadedAgent` and feed it with :meth:`submit`.
    Stopping or interrupting cancels the running turn, including the tool that
    is executing. Tools that are not :class:`RaiTool` are wrapped with
    :func:`rai.tools.from_langchain`, so the hook chain always runs.

    Without a ``checkpointer`` the agent keeps the conversation in
    :attr:`history`. One is created only for approval middleware.
    """

    def __init__(
        self,
        model: BaseChatModel | str = "complex",
        tools: Iterable[BaseTool] = (),
        *,
        system_prompt: str | SystemMessage | None = None,
        middleware: Sequence[AgentMiddleware[Any, Any]] = (),
        checkpointer: Checkpointer | None = None,
        recursion_limit: int = 50,
        on_busy: Literal["interrupt", "queue"] = "interrupt",
        name: str | None = None,
    ):
        super().__init__(name=name)
        self.tools = [t if isinstance(t, RaiTool) else from_langchain(t) for t in tools]
        if checkpointer is None and any(
            isinstance(m, HumanInTheLoopMiddleware) for m in middleware
        ):
            checkpointer = InMemorySaver()
        self.graph = create_agent(
            get_llm(model) if isinstance(model, str) else model,
            self.tools,
            system_prompt=system_prompt,
            middleware=[_AgentHooks(self), *middleware],
            checkpointer=checkpointer,
        )
        self.config: RunnableConfig = {
            "callbacks": get_tracing_callbacks(),
            "recursion_limit": recursion_limit,
            "configurable": {"thread_id": self.name},
        }
        self.history: list[BaseMessage] = []
        self.on_busy = on_busy
        self._inbox: deque[Any] = deque()
        self._wake = asyncio.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._turn: asyncio.Task[AIMessage] | None = None

    async def ainvoke(self, *messages: str | BaseMessage | HRIMessage) -> AIMessage:
        """Run one turn on ``messages`` and return the final answer.

        Turns of one agent must not overlap. Cancelling the call cancels the turn.
        """
        turn = uuid.uuid4().hex[:12]
        token = _TURN.set(turn)
        new = [_to_message(m) for m in messages]
        checkpointed = self.graph.checkpointer is not None
        state = {"messages": [*self.history, *new]}
        try:
            async for state in self.graph.astream(
                {"messages": new if checkpointed else state["messages"]},
                self.config,
                stream_mode="values",
            ):
                pass
        except (Exception, asyncio.CancelledError) as e:
            if isinstance(e, asyncio.CancelledError):
                self.emit(Interrupted(agent=self.name, turn=turn))
            else:
                self.emit(AgentError(agent=self.name, turn=turn, error=e))
            if patch := _unanswered(state["messages"]):
                if checkpointed:
                    await self.graph.aupdate_state(self.config, {"messages": patch})
                state = {"messages": [*state["messages"], *patch]}
            raise
        finally:
            _TURN.reset(token)
            self.history = state["messages"]
        final = state["messages"][-1]
        self.emit(FinalMessage(agent=self.name, turn=turn, message=final))
        return final

    def submit(self, *messages: Any) -> None:
        """Queue messages (``str``, LangChain or HRI messages) for the next turn.

        Thread-safe. With ``on_busy="interrupt"`` the running turn is cancelled
        and the next one sees every pending message.
        """
        self._inbox.extend(messages)
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._on_submit)

    def _on_submit(self) -> None:
        self._wake.set()
        if self.on_busy == "interrupt" and self._turn is not None:
            self._turn.cancel()

    def interrupt(self) -> None:
        """Cancel the running turn and its current tool call. Thread-safe."""
        if self._loop is not None:
            self._loop.call_soon_threadsafe(
                lambda: self._turn is not None and self._turn.cancel()
            )

    async def on_start(self) -> None:
        self._loop = asyncio.get_running_loop()
        if self._inbox:
            self._wake.set()

    async def run(self) -> None:
        while True:
            await self._wake.wait()
            self._wake.clear()
            messages = [self._inbox.popleft() for _ in range(len(self._inbox))]
            if not messages:
                continue
            self._turn = turn = asyncio.create_task(self.ainvoke(*messages))
            try:
                await asyncio.wait({turn})
            finally:  # stop() cancels run(): cancel the turn and let tools clean up
                turn.cancel()
                await asyncio.wait({turn})
                self._turn = None
            if not turn.cancelled() and (error := turn.exception()) is not None:
                self.logger.error("Turn failed: %r", error)


class StateBasedAgent(ReActAgent):
    """A :class:`ReActAgent` that sees the current robot state on every model call.

    ``state_sources`` maps a name to an async callable, e.g. an aggregator's
    :meth:`~rai.aggregators.BaseAggregator.observe`. The state is added to each
    model request only; it is never written to the history.
    """

    def __init__(
        self,
        model: BaseChatModel | str = "complex",
        tools: Iterable[BaseTool] = (),
        *,
        state_sources: Mapping[str, StateSource],
        state_timeout_s: float = 5.0,
        middleware: Sequence[AgentMiddleware[Any, Any]] = (),
        **kwargs: Any,
    ):
        self.context = StateContext(state_sources, state_timeout_s)
        super().__init__(model, tools, middleware=[self.context, *middleware], **kwargs)
