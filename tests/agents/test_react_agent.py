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

"""ReActAgent and StateBasedAgent against a scripted chat model (offline)."""

import asyncio
import time
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
import rai
from langchain_core.callbacks import BaseCallbackHandler
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool
from rai.agents import (
    AgentState,
    HRIChunk,
    HRIOutput,
    ReActAgent,
    StateBasedAgent,
    hri_connector_sender,
)
from rai.agents.events import (
    AgentError,
    FinalMessage,
    Interrupted,
    ModelEnd,
    ModelStart,
    StateChanged,
    ToolEnd,
    ToolStart,
)
from rai.communication import HRIMessage
from rai.content import Image
from rai.tools import (
    Allow,
    Deny,
    Effect,
    RaiTool,
    ToolCallContext,
    global_tool_hooks,
    rai_tool,
)


@pytest.fixture(autouse=True)
def _clean_global_hooks():
    global_tool_hooks.clear()
    yield
    global_tool_hooks.clear()


class ScriptedChatModel(BaseChatModel):
    """Plays `steps` in order: an AIMessage, a final text, or a list of
    (tool name, args) calls. Records the messages of every call in `seen`."""

    steps: list[Any]
    seen: list[list[Any]]

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        step = self.steps.pop(0) if self.steps else "done"
        if isinstance(step, str):
            step = AIMessage(step)
        elif not isinstance(step, AIMessage):
            calls = [
                {"name": n, "args": a, "id": f"call_{len(self.seen)}_{i}"}
                for i, (n, a) in enumerate(step)
            ]
            step = AIMessage("", tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=step)])


def model(*steps: Any) -> ScriptedChatModel:
    return ScriptedChatModel(steps=list(steps), seen=[])


@rai_tool(Effect.READ)
async def get_status(topic: str) -> str:
    """Return the latest status on a topic."""
    return f"ok {topic}"


@rai_tool(Effect.ACTUATE, targets=["/cmd_vel"])
async def drive(speed: float) -> str:
    """Drive the base."""
    return f"driving {speed}"


@rai_tool(Effect.READ)
async def broken() -> str:
    """Always fails."""
    raise RuntimeError("sensor offline")


def recorded(agent: ReActAgent) -> list[Any]:
    events: list[Any] = []
    agent.subscribe(events.append)
    return events


def kinds(events: list[Any]) -> list[str]:
    return [type(e).__name__ for e in events if not isinstance(e, StateChanged)]


# ---- ReAct loop ----


def test_react_loop_runs_tools_and_keeps_history():
    agent = ReActAgent(
        model([("get_status", {"topic": "/battery"})], "battery ok"), [get_status]
    )
    answer = asyncio.run(agent.ainvoke("check the battery"))
    assert answer.text == "battery ok"
    assert [type(m).__name__ for m in agent.history] == [
        "HumanMessage",
        "AIMessage",
        "ToolMessage",
        "AIMessage",
    ]
    assert agent.history[2].content == "ok /battery"

    agent.graph.get_graph()  # the graph stays a plain compiled LangGraph
    asyncio.run(agent.ainvoke("again"))  # second turn sees the first one
    assert len(agent.history) == 6


def test_hri_messages_reach_the_model_as_content_blocks():
    agent = ReActAgent(model("a cat"))
    image = Image(np.zeros((4, 4, 3), np.uint8))
    asyncio.run(agent.ainvoke(HRIMessage("what is it?", [image])))
    blocks = agent.history[0].content
    assert [b["type"] for b in blocks] == ["text", "image"]


def test_tool_error_and_denial_are_tool_messages_and_events():
    async def deny_actuate(call: ToolCallContext) -> Allow | Deny:
        return Deny("e-stop") if call.effect is Effect.ACTUATE else Allow()

    global_tool_hooks.add(deny_actuate)
    agent = ReActAgent(
        model(
            [
                ("broken", {}),
                ("drive", {"speed": 1.0}),
                ("get_status", {"topic": "/a"}),
            ],
            "done",
        ),
        [broken, drive, get_status],
    )
    events = recorded(agent)
    assert asyncio.run(agent.ainvoke("go")).text == "done"  # the run continues

    results = {m.name: m for m in agent.history if isinstance(m, ToolMessage)}
    assert results["broken"].status == "error"
    assert "sensor offline" in results["broken"].content
    assert results["drive"].status == "error"
    assert "Denied by deny_actuate: e-stop" in results["drive"].content
    assert results["get_status"].status == "success"

    ends = {e.name: e for e in events if isinstance(e, ToolEnd)}
    assert (ends["broken"].status, ends["broken"].effect) == ("error", Effect.READ)
    assert (ends["drive"].status, ends["drive"].effect) == ("denied", Effect.ACTUATE)
    assert ends["get_status"].status == "success"


def test_event_stream_contents():
    agent = ReActAgent(
        model([("get_status", {"topic": "/a"})], "all good"), [get_status], name="bot"
    )
    events = recorded(agent)
    asyncio.run(agent.ainvoke("hi"))
    assert kinds(events) == [
        "ModelStart",
        "ModelEnd",
        "ToolStart",
        "ToolEnd",
        "ModelStart",
        "ModelEnd",
        "FinalMessage",
    ]
    assert {e.agent for e in events} == {"bot"}
    assert len({e.turn for e in events}) == 1 and events[0].turn
    start = next(e for e in events if isinstance(e, ToolStart))
    assert (start.name, start.args, start.effect) == (
        "get_status",
        {"topic": "/a"},
        Effect.READ,
    )
    assert events[1].message.tool_calls and isinstance(events[1], ModelEnd)
    assert events[-1].message.text == "all good"


def test_model_error_emits_agent_error():
    class Failing(ScriptedChatModel):
        def _generate(self, *args, **kwargs):
            raise RuntimeError("provider down")

    agent = ReActAgent(Failing(steps=[], seen=[]))
    events = recorded(agent)
    with pytest.raises(RuntimeError, match="provider down"):
        asyncio.run(agent.ainvoke("hi"))
    assert isinstance(events[-1], AgentError)
    assert [type(m).__name__ for m in agent.history] == ["HumanMessage"]


def test_legacy_tools_are_wrapped_so_hooks_run():
    class Legacy(BaseTool):
        name: str = "legacy"
        description: str = "A tool without an effect."

        def _run(self, text: str) -> str:
            return text

    async def deny_actuate(call: ToolCallContext) -> Allow | Deny:
        return Deny("no") if call.effect is Effect.ACTUATE else Allow()

    global_tool_hooks.add(deny_actuate)
    agent = ReActAgent(model([("legacy", {"text": "x"})], "ok"), [Legacy()])
    assert isinstance(agent.tools[0], RaiTool)
    assert agent.tools[0].effect is Effect.ACTUATE  # unknown effect
    asyncio.run(agent.ainvoke("go"))
    assert "Denied by deny_actuate" in agent.history[2].content


def test_tool_images_reach_the_model_but_not_the_history():
    @rai_tool(Effect.READ)
    async def camera() -> tuple[str, dict]:
        """Take a picture."""
        return "picture taken", {"images": ["aGVsbG8="]}

    camera.response_format = "content_and_artifact"
    llm = model([("camera", {})], "I see a robot")
    agent = ReActAgent(llm, [camera])
    asyncio.run(agent.ainvoke("look"))
    second_call = llm.seen[1]
    assert isinstance(second_call[-2], ToolMessage)
    assert "aGVsbG8=" in str(second_call[-1].content)
    assert not any("aGVsbG8=" in str(m.content) for m in agent.history)


def test_tracing_callbacks_are_attached_once_via_config(monkeypatch):
    class Tracer(BaseCallbackHandler):
        def __init__(self):
            self.calls = 0

        def on_chat_model_start(self, *args, **kwargs):
            self.calls += 1

    tracer = Tracer()
    get = MagicMock(return_value=[tracer])
    monkeypatch.setattr("rai.agents.react.get_tracing_callbacks", get)
    agent = ReActAgent(model([("get_status", {"topic": "/a"})], "ok"), [get_status])
    asyncio.run(agent.ainvoke("hi"))
    asyncio.run(agent.ainvoke("hi"))
    assert get.call_count == 1
    assert tracer.calls == 3  # 2 model calls, then 1


# ---- cancellation ----


def slow_drive(started: asyncio.Event, cleanup: list[str]) -> RaiTool:
    @rai_tool(Effect.ACTUATE)
    async def drive_far(seconds: float) -> str:
        """Drive for a while."""
        started.set()
        try:
            await asyncio.sleep(seconds)
            return "arrived"
        finally:
            cleanup.append("stopped motors")

    return drive_far


@pytest.mark.timeout(10)
def test_stop_cancels_the_running_tool():
    async def main():
        started, cleanup = asyncio.Event(), []
        agent = ReActAgent(
            model([("drive_far", {"seconds": 30})], "never"),
            [slow_drive(started, cleanup)],
        )
        events = recorded(agent)
        await agent.start()
        agent.submit("drive")
        await asyncio.wait_for(started.wait(), 5)
        t0 = time.perf_counter()
        await agent.stop()
        assert time.perf_counter() - t0 < 0.5
        assert cleanup == ["stopped motors"]
        assert agent.state is AgentState.STOPPED
        assert "Interrupted" in kinds(events) and "FinalMessage" not in kinds(events)
        # the history stays valid: every tool call has a result
        assert isinstance(agent.history[-1], ToolMessage)
        assert agent.history[-1].status == "error"

    asyncio.run(main())


@pytest.mark.timeout(10)
def test_new_message_interrupts_the_turn_and_runs_the_next():
    async def main():
        started, cleanup = asyncio.Event(), []
        agent = ReActAgent(
            model([("drive_far", {"seconds": 30})], "stopped as asked"),
            [slow_drive(started, cleanup)],
        )
        events = recorded(agent)
        finals: asyncio.Queue[Any] = asyncio.Queue()
        agent.subscribe(lambda e: isinstance(e, FinalMessage) and finals.put_nowait(e))
        await agent.start()
        agent.submit("drive")
        await asyncio.wait_for(started.wait(), 5)
        agent.submit("stop!")
        final = await asyncio.wait_for(finals.get(), 5)
        assert final.message.text == "stopped as asked"
        assert cleanup == ["stopped motors"]
        assert kinds(events).count("Interrupted") == 1
        texts = [m.text for m in agent.history if isinstance(m, HumanMessage)]
        assert texts == ["drive", "stop!"]
        await agent.stop()

    asyncio.run(main())


@pytest.mark.timeout(10)
def test_interrupt_cancels_direct_turn_from_another_thread():
    async def main():
        started, cleanup = asyncio.Event(), []
        agent = ReActAgent(
            model([("drive_far", {"seconds": 30})]), [slow_drive(started, cleanup)]
        )
        await agent.start()
        agent.submit("drive")
        await asyncio.wait_for(started.wait(), 5)
        await asyncio.to_thread(agent.interrupt)
        while not cleanup:
            await asyncio.sleep(0.01)
        assert agent.state is AgentState.RUNNING  # only the turn ended
        await agent.stop()

    asyncio.run(main())


# ---- StateBased ----


def test_state_is_in_every_model_call_and_never_in_history():
    """Regression: the old StateBased agent piled up '/rosout: /rosout: ...'."""
    calls = 0

    async def logs() -> str:
        nonlocal calls
        calls += 1
        return f"logs: ok {calls}"

    llm = model(
        [("get_status", {"topic": "/a"})], [("get_status", {"topic": "/b"})], "done"
    )
    agent = StateBasedAgent(llm, [get_status], state_sources={"/rosout": logs})
    asyncio.run(agent.ainvoke("hi"))
    asyncio.run(agent.ainvoke("again"))

    assert len(llm.seen) == 4
    for i, messages in enumerate(llm.seen, start=1):
        observations = [m for m in messages if "Current robot state" in str(m.content)]
        assert observations == [messages[-1]]  # exactly once, always last
        assert f"/rosout: logs: ok {i}" in str(messages[-1].content)
        assert "/rosout: /rosout" not in str(messages)
    assert not any("Current robot state" in str(m.content) for m in agent.history)
    assert len(agent.history) == 8


def test_state_sources_run_concurrently_with_own_timeouts_and_keep_last_good():
    state = {"n": 0}

    async def flaky() -> str:
        state["n"] += 1
        if state["n"] == 2:
            await asyncio.sleep(10)  # hangs: timed out, last value kept
        if state["n"] == 3:
            raise RuntimeError("sensor glitch")
        return f"v{state['n']}"

    async def slow() -> str:
        await asyncio.sleep(0.15)
        return "slow ok"

    async def empty() -> None:
        return None

    llm = model(
        [("get_status", {"topic": "/a"})],
        [("get_status", {"topic": "/a"})],
        [("get_status", {"topic": "/a"})],
        "done",
    )
    agent = StateBasedAgent(
        llm,
        [get_status],
        state_sources={"flaky": flaky, "slow": slow, "empty": empty},
        state_timeout_s=0.3,
    )
    t0 = time.perf_counter()
    asyncio.run(agent.ainvoke("hi"))
    elapsed = time.perf_counter() - t0
    seen = [str(messages[-1].content) for messages in llm.seen]
    assert ["flaky: v" + x in s for s, x in zip(seen, "1114")] == [True] * 4
    assert all("slow: slow ok" in s and "empty" not in s for s in seen)
    # 4 calls; sequential sources would take >= 4 * (0.15 + 0.3)
    assert elapsed < 4 * 0.45


# ---- HRI output ----


def test_hri_output_sends_text_per_turn():
    chunks: list[HRIChunk] = []
    call = {"name": "drive", "args": {"speed": 1}, "id": "1"}
    llm = model(AIMessage("Let me check.", tool_calls=[call]), "Done driving.")
    agent = ReActAgent(llm, [drive])
    agent.subscribe(HRIOutput(chunks.append, tool_status=True))
    asyncio.run(agent.ainvoke("drive"))
    turn = chunks[0].turn
    assert [(c.text, c.seq, c.end) for c in chunks] == [
        ("Let me check.", 0, False),
        ("Running drive.", 1, False),
        ("Done driving.", 2, True),
    ]
    assert {c.turn for c in chunks} == {turn}


def test_hri_output_closes_interrupted_turn():
    chunks: list[HRIChunk] = []
    out = HRIOutput(chunks.append, tool_status=True)
    out(
        ToolStart(
            agent="a", turn="t", call_id="1", name="x", args={}, effect=Effect.READ
        )
    )
    out(Interrupted(agent="a", turn="t"))
    out(Interrupted(agent="a", turn="other"))  # nothing was said: nothing to close
    out(ModelStart(agent="a", turn="t"))
    assert [(c.text, c.seq, c.end) for c in chunks] == [
        ("Running x.", 0, False),
        ("", 1, True),
    ]


def test_hri_connector_sender_sends_hri_messages():
    connector = MagicMock()
    send = hri_connector_sender(connector, "/to_human")
    send(HRIChunk("hello", "turn-1", 2, True))
    connector.sync.send.assert_called_once_with(
        "/to_human",
        HRIMessage("hello", communication_id="turn-1", seq_no=2, seq_end=True),
    )


# ---- runtime ----


@pytest.mark.timeout(10)
def test_rai_run_with_an_agent():
    agent = ReActAgent(model([("get_status", {"topic": "/a"})], "bye"), [get_status])

    def stop_when_done(event: Any) -> None:
        if isinstance(event, FinalMessage):
            asyncio.get_running_loop().create_task(agent.stop())

    agent.subscribe(stop_when_done)
    agent.submit("hello")  # before start: kept for the first turn
    rai.run(agent)
    assert agent.state is AgentState.STOPPED
    assert agent.history[-1].text == "bye"


@pytest.mark.timeout(10)
def test_events_iterator_ends_after_stop():
    async def main():
        agent = ReActAgent(model("hello"))
        stream = agent.events()
        await agent.start()
        await agent.ainvoke("hi")
        await agent.stop()
        return [e async for e in stream]

    events = asyncio.run(main())
    states = [e.new for e in events if isinstance(e, StateChanged)]
    assert states == ["starting", "running", "stopping", "stopped"]
    assert "FinalMessage" in kinds(events)


def test_stop_before_start_does_not_hang():
    from rai.agents import ThreadedAgent

    agent = ReActAgent(model())
    ThreadedAgent(agent).stop()
    assert agent.state is AgentState.STOPPED


@pytest.mark.timeout(10)
def test_user_checkpointer_keeps_history_and_valid_after_cancel():
    from langgraph.checkpoint.memory import InMemorySaver

    async def main():
        started, cleanup = asyncio.Event(), []
        llm = model("hi there", [("drive_far", {"seconds": 30})], "next")
        agent = ReActAgent(
            llm, [slow_drive(started, cleanup)], checkpointer=InMemorySaver()
        )
        await agent.ainvoke("hello")
        turn = asyncio.create_task(agent.ainvoke("drive"))
        await asyncio.wait_for(started.wait(), 5)
        turn.cancel()
        with pytest.raises(asyncio.CancelledError):
            await turn
        await agent.ainvoke("status?")
        # the third model call sees both earlier turns and the closed tool call
        kinds_seen = [type(m).__name__ for m in llm.seen[2]]
        assert kinds_seen == [
            "HumanMessage",
            "AIMessage",
            "HumanMessage",
            "AIMessage",
            "ToolMessage",
            "HumanMessage",
        ]
        assert agent.history[-1].text == "next"

    asyncio.run(main())
