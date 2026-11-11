import asyncio
import time

import pytest
from langchain_core.messages import HumanMessage, ToolMessage
from langgraph.types import Command

from ..common import guard
from ..common import robot as rb
from ..common.mcp_e2e import EXT_SERVER, drive_server
from ..common.robot import Effect
from ..common.script import Call, Script, Text
from ..conftest import wait_for
from .fake import ScriptedChatModel
from .mcp import load_client_tools
from .slice import build_agent, effect_of, guarded, make_tools

CFG = {"configurable": {"thread_id": "t"}}
DRIVE = Call("publish_twist", {"linear_x": 0.5, "angular_z": 0.1, "duration_s": 0.3})


def run(agent, payload):
    return asyncio.run(agent.ainvoke(payload, CFG))


def ask(text):
    return {"messages": [HumanMessage(text)]}


def tool_msgs(state):
    return [m for m in state["messages"] if isinstance(m, ToolMessage)]


def test_tools(robot):
    io, published = robot
    read, drive = make_tools(io)
    assert (effect_of(read), effect_of(drive)) == (Effect.READ, Effect.ACTUATE)
    assert "battery=87%" in asyncio.run(read.ainvoke({"topic": "/d1/status"}))
    asyncio.run(drive.ainvoke(DRIVE.args))
    assert wait_for(lambda: published and published[-1] == (0.0, 0.0))
    assert (0.5, 0.1) in published


def test_react_loop(robot):
    io, _ = robot
    script = Script(
        [[Call("get_latest_message", {"topic": "/d1/status"})], Text("done")]
    )
    out = run(
        build_agent(ScriptedChatModel(script=script), make_tools(io)), ask("status?")
    )
    assert out["messages"][-1].text == "done"
    assert "battery=87%" in tool_msgs(out)[0].text


def test_state_context_per_call_not_in_history(robot):
    io, _ = robot
    read = [Call("get_latest_message", {"topic": "/d1/status"})]
    script = Script([read, read, Text("done")])
    agent = build_agent(
        ScriptedChatModel(script=script),
        make_tools(io),
        context=lambda: rb.observe_status(io),
    )
    out = run(agent, ask("status?"))
    assert len(script.seen) == 3
    for seen in script.seen:  # exactly one fresh observation, always last
        assert sum("[robot state]" in t for _, t in seen) == 1
        assert "[robot state]" in seen[-1][1]
    assert not any("[robot state]" in m.text for m in out["messages"])


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_approval_gate(robot, decision):
    io, published = robot
    script = Script([[DRIVE], Text("done")])
    agent = build_agent(ScriptedChatModel(script=script), make_tools(io))
    out = run(agent, ask("drive"))
    assert out.get("__interrupt__") and not published
    answer = (
        {"type": decision}
        if decision == "approve"
        else {"type": "reject", "message": "no"}
    )
    out = run(agent, Command(resume={"decisions": [answer]}))
    assert out["messages"][-1].text == "done"
    if decision == "approve":
        assert wait_for(lambda: published and published[-1] == (0.0, 0.0))
    else:
        time.sleep(0.2)
        assert not published and tool_msgs(out)[-1].status == "error"


@pytest.mark.parametrize(
    "hook", [guard.crashing_hook, guard.hanging_hook, guard.deny_actuate]
)
def test_fail_closed(robot, hook):
    io, published = robot
    read = Call("get_latest_message", {"topic": "/d1/status"})
    script = Script([[DRIVE], [read], Text("done")])
    agent = build_agent(
        ScriptedChatModel(script=script),
        make_tools(io),
        hooks=[hook],
        approve_actuation=False,
        hook_timeout_s=0.2,
    )
    t0 = time.monotonic()
    out = run(agent, ask("drive"))
    assert time.monotonic() - t0 < 2
    denied, read_result = tool_msgs(out)
    assert denied.status == "error" and denied.text.startswith("Denied")
    if hook is guard.deny_actuate:  # a working hook still lets READ through
        assert "battery" in read_result.text
    assert out["messages"][-1].text == "done" and not published


def test_cancellation_stops_slow_tool(robot):
    io, published = robot
    slow = Call("publish_twist", {"linear_x": 0.3, "angular_z": 0.0, "duration_s": 10})
    agent = build_agent(
        ScriptedChatModel(script=Script([[slow], Text("done")])),
        make_tools(io),
        approve_actuation=False,
    )

    async def main():
        task = asyncio.create_task(agent.ainvoke(ask("drive"), CFG))
        while not published:
            await asyncio.sleep(0.01)
        await asyncio.sleep(0.2)
        t0 = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - t0

    assert asyncio.run(main()) < 0.3
    assert wait_for(lambda: published[-1] == (0.0, 0.0))
    n = len(published)
    time.sleep(0.3)
    assert len(published) == n


def test_mcp_client():
    tools = asyncio.run(load_client_tools(EXT_SERVER))
    assert [t.name for t in tools] == ["robot_name"] and effect_of(
        tools[0]
    ) is Effect.WRITE
    rec = guard.Recorder()
    script = Script([[Call("robot_name", {})], Text("done")])
    out = run(
        build_agent(ScriptedChatModel(script=script), tools, hooks=[rec]), ask("name?")
    )
    assert "rosbot-xl-7" in tool_msgs(out)[0].text and rec.calls[0].name == "robot_name"


@pytest.mark.parametrize("robot_kind", ["fake", "ros"])
def test_mcp_server_stdio_e2e(request, robot_kind):
    sim = request.getfixturevalue("ros") if robot_kind == "ros" else None
    r = asyncio.run(drive_server("langchain", robot_kind))
    assert r["tools"]["get_latest_message"].annotations.readOnlyHint
    assert r["tools"]["publish_twist"].annotations.destructiveHint
    assert "battery=87%" in r["read"].content[0].text and not r["drive"].isError
    if sim:
        assert wait_for(lambda: (0.2, 0.0) in sim.published)


@pytest.mark.parametrize("hook", ["deny", "crash"])
def test_mcp_server_fail_closed(hook):
    r = asyncio.run(drive_server("langchain", "fake", hook))
    assert r["drive"].isError and "Denied" in r["drive"].content[0].text
    assert "battery" in r["read"].content[0].text or hook == "crash"


def test_denied_call_still_reaches_human(robot):
    """HITL runs in after_model, before any wrap_tool_call: the human is asked about a call
    the safety hook will deny anyway. (PydanticAI denies first; see its twin test.)"""
    io, published = robot
    agent = build_agent(
        ScriptedChatModel(script=Script([[DRIVE], Text("done")])),
        make_tools(io),
        hooks=[guard.deny_actuate],
    )
    out = run(agent, ask("drive"))
    assert out.get("__interrupt__")
    out = run(agent, Command(resume={"decisions": [{"type": "approve"}]}))
    assert tool_msgs(out)[-1].text.startswith("Denied") and not published


@pytest.mark.parametrize(
    "hook", [guard.crashing_hook, guard.hanging_hook, guard.deny_actuate]
)
def test_tool_level_guard_on_agent_path(robot, hook):
    """The A3 design: the fail-closed adapter inside the tool, no FailClosed middleware."""
    io, published = robot
    tools = [guarded(t, [hook], timeout_s=0.2) for t in make_tools(io)]
    script = Script([[DRIVE], Text("done")])
    agent = build_agent(
        ScriptedChatModel(script=script), tools, approve_actuation=False
    )
    out = run(agent, ask("drive"))
    denied = tool_msgs(out)[0]
    assert denied.status == "error" and denied.text.startswith("Denied")
    assert out["messages"][-1].text == "done" and not published
