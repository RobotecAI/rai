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

import asyncio
import time
from typing import Any

import pytest
from langchain.agents import create_agent
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool, StructuredTool
from rai.tools import (
    Allow,
    Deny,
    Effect,
    RaiTool,
    ToolCallContext,
    effect_from_mcp_annotations,
    from_langchain,
    global_tool_hooks,
    rai_tool,
)


@pytest.fixture(autouse=True)
def _clean_global_hooks():
    global_tool_hooks.clear()
    yield
    global_tool_hooks.clear()


class ScriptedChatModel(BaseChatModel):
    """Returns `steps` in order: a list of (name, args) tool calls, or a final text."""

    steps: list[Any]
    seen: list[list[Any]] = []

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.seen.append(list(messages))
        step = self.steps.pop(0) if self.steps else "done"
        if isinstance(step, str):
            msg = AIMessage(step)
        else:
            calls = [
                {"name": n, "args": a, "id": f"call_{len(self.seen)}_{i}"}
                for i, (n, a) in enumerate(step)
            ]
            msg = AIMessage("", tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=msg)])


def run_agent(tools: list[BaseTool], steps: list[Any]) -> list[Any]:
    agent = create_agent(ScriptedChatModel(steps=steps), tools)
    return asyncio.run(agent.ainvoke({"messages": [HumanMessage("go")]}))["messages"]


def tool_messages(messages: list[Any]) -> list[ToolMessage]:
    return [m for m in messages if isinstance(m, ToolMessage)]


@rai_tool(Effect.READ)
async def get_status(topic: str) -> str:
    """Return the latest status on a topic."""
    return f"ok {topic}"


@rai_tool(Effect.ACTUATE, targets=["/cmd_vel"])
async def drive(speed: float) -> str:
    """Drive the base."""
    return f"driving {speed}"


async def allow(call: ToolCallContext) -> Allow | Deny:
    return Allow()


async def deny_actuate(call: ToolCallContext) -> Allow | Deny:
    return Deny("actuation disabled") if call.effect is Effect.ACTUATE else Allow()


# ---- declaring tools ----


def test_decorator_builds_typed_tool():
    assert isinstance(get_status, RaiTool)
    assert (get_status.name, get_status.description) == (
        "get_status",
        "Return the latest status on a topic.",
    )
    assert set(get_status.args) == {"topic"}
    assert (get_status.effect, get_status.targets) == (Effect.READ, ())
    assert (drive.effect, drive.targets) == (Effect.ACTUATE, ("/cmd_vel",))
    assert asyncio.run(get_status.ainvoke({"topic": "/a"})) == "ok /a"
    assert get_status.invoke({"topic": "/a"}) == "ok /a"  # sync path wraps async


def test_decorator_on_sync_function():
    @rai_tool(Effect.WRITE, name="set_mode")
    def set_mode_impl(mode: str) -> str:
        """Set the mode."""
        return mode.upper()

    assert set_mode_impl.name == "set_mode"
    assert asyncio.run(set_mode_impl.ainvoke({"mode": "x"})) == "X"


def test_sync_invoke_from_running_loop():
    async def main():
        return get_status.invoke({"topic": "/b"})

    assert asyncio.run(main()) == "ok /b"


def test_base_class():
    class Echo(RaiTool):
        name: str = "echo"
        description: str = "Echo text."
        default_effect = Effect.WRITE

        async def _acall(self, text: str) -> str:
            return text

    tool = Echo(targets=["/chat"])
    assert (tool.effect, tool.targets) == (Effect.WRITE, ("/chat",))
    assert set(tool.args) == {"text"}  # schema inferred from _acall
    assert asyncio.run(tool.ainvoke({"text": "hi"})) == "hi"
    assert Echo(effect=Effect.READ).effect is Effect.READ
    assert tool.metadata == {"effect": Effect.WRITE, "targets": ("/chat",)}


def test_base_class_defaults_to_actuate_and_forbids_run_override():
    class Bare(RaiTool):
        name: str = "bare"
        description: str = "Bare."

        async def _acall(self) -> str:
            return "x"

    assert Bare().effect is Effect.ACTUATE
    with pytest.raises(TypeError, match="_acall"):

        class Bypass(RaiTool):  # noqa: F841 - would skip the hooks
            async def _arun(self, *args, **kwargs):
                return "x"


# ---- wrapping third-party and MCP tools ----


@pytest.mark.parametrize(
    "annotations, effect",
    [
        ({"readOnlyHint": True}, Effect.READ),
        ({"readOnlyHint": False, "destructiveHint": False}, Effect.WRITE),
        ({"destructiveHint": True}, Effect.ACTUATE),
        ({"readOnlyHint": None, "destructiveHint": None}, Effect.ACTUATE),
        (None, Effect.ACTUATE),
    ],
)
def test_effect_from_mcp_annotations(annotations, effect):
    assert effect_from_mcp_annotations(annotations) is effect


def test_from_langchain_mcp_tool_keeps_artifact_and_runs_hooks():
    async def call_tool(x: int) -> tuple[str, dict]:
        return f"got {x}", {"raw": x}

    mcp_tool = StructuredTool(
        name="remote",
        description="Remote MCP tool.",
        args_schema={"type": "object", "properties": {"x": {"type": "integer"}}},
        coroutine=call_tool,
        response_format="content_and_artifact",
        metadata={"readOnlyHint": None, "destructiveHint": False},
    )
    tool = from_langchain(mcp_tool, targets=["/remote"])
    assert (tool.effect, tool.targets) == (Effect.WRITE, ("/remote",))
    call = {"name": "remote", "args": {"x": 3}, "id": "1", "type": "tool_call"}
    msg = asyncio.run(tool.ainvoke(call))
    assert (msg.content, msg.artifact, msg.status) == ("got 3", {"raw": 3}, "success")

    async def deny_all(call: ToolCallContext) -> Allow | Deny:
        return Deny("no")

    tool.hooks.add(deny_all)
    msg = asyncio.run(tool.ainvoke(call))
    assert msg.status == "error" and "Denied by deny_all: no" in msg.content


def test_from_langchain_plain_tools():
    class Legacy(BaseTool):  # an existing tool without effect metadata
        name: str = "legacy"
        description: str = "Legacy tool."

        def _run(self, text: str) -> str:
            return text * 2

    tool = from_langchain(Legacy())
    assert tool.effect is Effect.ACTUATE  # missing effect: most restrictive
    assert asyncio.run(tool.ainvoke({"text": "a"})) == "aa"

    def add(a: int, b: int) -> int:
        """Add."""
        return a + b

    tool = from_langchain(StructuredTool.from_function(add), effect=Effect.READ)
    assert tool.effect is Effect.READ
    assert asyncio.run(tool.ainvoke({"a": 1, "b": 2})) == 3


# ---- errors and timeouts become tool messages ----


def test_errors_are_tool_messages_in_agent_loop():
    @rai_tool(Effect.READ)
    async def broken(x: int) -> str:
        """Always fails."""
        raise RuntimeError("sensor offline")

    messages = run_agent(
        [broken, get_status],
        [
            [("broken", {"x": 1})],
            [("broken", {"x": "not a number"})],
            [("get_status", {"topic": "/s"})],
            "finished",
        ],
    )
    err, invalid, ok = tool_messages(messages)
    assert (err.status, err.content) == ("error", "RuntimeError: sensor offline")
    assert invalid.status == "error"
    assert (ok.status, ok.content) == ("success", "ok /s")
    assert messages[-1].content == "finished"


def test_tool_timeout():
    @rai_tool(Effect.READ, timeout_s=0.05)
    async def slow() -> str:
        """Slow."""
        await asyncio.sleep(5)
        return "late"

    start = time.monotonic()
    msg = asyncio.run(
        slow.ainvoke({"name": "slow", "args": {}, "id": "1", "type": "tool_call"})
    )
    assert time.monotonic() - start < 1
    assert (msg.status, msg.content) == ("error", "Timed out after 0.05s")


def test_cancel_reaches_running_tool():
    state: dict[str, Any] = {"cleaned": False}

    @rai_tool(Effect.ACTUATE)
    async def long_drive() -> str:
        """Drive for a long time."""
        state["started"].set()
        try:
            await asyncio.sleep(10)
        finally:
            state["cleaned"] = True  # e.g. publish a zero Twist
        return "arrived"

    async def main():
        state["started"] = asyncio.Event()
        model = ScriptedChatModel(steps=[[("long_drive", {})]])
        agent = create_agent(model, [long_drive])
        task = asyncio.create_task(agent.ainvoke({"messages": [HumanMessage("go")]}))
        await asyncio.wait_for(state["started"].wait(), 5)
        start = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - start

    assert asyncio.run(main()) < 0.3
    assert state["cleaned"]


# ---- fail-closed hooks ----


def test_hook_denial_in_agent_loop_continues_run():
    global_tool_hooks.add(deny_actuate)
    messages = run_agent(
        [get_status, drive],
        [[("drive", {"speed": 1.0})], [("get_status", {"topic": "/s"})], "ok"],
    )
    denied, read = tool_messages(messages)
    assert denied.status == "error"
    assert denied.content == "Denied by deny_actuate: actuation disabled"
    assert (read.status, read.content) == ("success", "ok /s")
    assert messages[-1].content == "ok"


async def raising(call: ToolCallContext) -> Allow | Deny:
    raise RuntimeError("hook bug")


async def hanging(call: ToolCallContext) -> Allow | Deny:
    await asyncio.sleep(30)
    return Allow()


async def returns_none(call: ToolCallContext) -> Allow | Deny:
    return None  # type: ignore[return-value]


def sync_hook(call: ToolCallContext) -> Allow:  # sync, not awaitable
    return Allow()


@pytest.mark.parametrize(
    "hook, reason",
    [
        (raising, "hook raising failed (RuntimeError: hook bug)"),
        (hanging, "hook hanging timed out after 0.05s"),
        (returns_none, "hook returns_none returned None, expected Allow or Deny"),
        (sync_hook, "hook sync_hook failed (TypeError"),
    ],
)
def test_hook_failures_deny(hook, reason):
    drive.hooks.add(hook, timeout_s=0.05)
    try:
        start = time.monotonic()
        msg = asyncio.run(
            drive.ainvoke(
                {"name": "drive", "args": {"speed": 1}, "id": "1", "type": "tool_call"}
            )
        )
    finally:
        drive.hooks.clear()
    assert time.monotonic() - start < 1
    assert msg.status == "error" and msg.content.startswith(f"Denied: {reason}")


def test_hook_order_context_and_first_denial_wins():
    seen: list[tuple[str, ToolCallContext]] = []

    def recorder(label: str, decision: Allow | Deny):
        async def hook(call: ToolCallContext) -> Allow | Deny:
            seen.append((label, call))
            return decision

        hook.__name__ = label
        return hook

    global_tool_hooks.add(recorder("global1", Allow()))
    global_tool_hooks.add(recorder("global2", Allow()))
    drive.hooks.add(recorder("tool1", Deny("stop")))
    drive.hooks.add(recorder("tool2", Allow()))
    try:
        with pytest.raises(Exception, match="Denied by tool1: stop"):
            asyncio.run(
                drive.model_copy(update={"handle_tool_error": False}).ainvoke(
                    {"speed": 2}
                )
            )
    finally:
        drive.hooks.clear()
    assert [label for label, _ in seen] == ["global1", "global2", "tool1"]
    call = seen[0][1]
    assert (call.name, call.effect, call.targets) == (
        "drive",
        Effect.ACTUATE,
        ("/cmd_vel",),
    )
    assert dict(call.args) == {"speed": 2.0}
    with pytest.raises(TypeError):
        call.args["speed"] = 9  # type: ignore[index]


def test_remove_hook():
    global_tool_hooks.add(deny_actuate)
    global_tool_hooks.add(allow)
    global_tool_hooks.remove(deny_actuate)
    assert len(global_tool_hooks) == 1
    assert asyncio.run(drive.ainvoke({"speed": 1})) == "driving 1.0"


def test_no_hooks_skips_the_chain(monkeypatch):
    async def must_not_run(*args):
        raise AssertionError("hook chain ran with no hooks registered")

    monkeypatch.setattr("rai.tools.base.run_tool_hooks", must_not_run)
    assert asyncio.run(drive.ainvoke({"speed": 1})) == "driving 1.0"
