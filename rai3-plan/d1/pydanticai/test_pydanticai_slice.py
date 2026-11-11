import asyncio
import time

import pytest
from pydantic_ai import DeferredToolRequests, DeferredToolResults, ToolDenied
from pydantic_ai.messages import RetryPromptPart, ToolReturnPart

from ..common import guard
from ..common import robot as rb
from ..common.mcp_e2e import EXT_SERVER, drive_server
from ..common.robot import Effect
from ..common.script import Call, Script, Text
from ..conftest import wait_for
from .fake import scripted_model
from .mcp import client_toolset
from .slice import build_agent, make_tools

DRIVE = Call("publish_twist", {"linear_x": 0.5, "angular_z": 0.1, "duration_s": 0.3})


def run(agent, prompt=None, **kw):
    return asyncio.run(agent.run(prompt, **kw))


def tool_results(result):
    """(tool name, text, failed) for every tool result in the run's history."""
    return [
        (
            p.tool_name,
            str(p.content),
            isinstance(p, RetryPromptPart) or p.outcome != "success",
        )
        for m in result.all_messages()
        for p in m.parts
        if isinstance(p, ToolReturnPart | RetryPromptPart)
    ]


def test_tools(robot):
    io, published = robot
    ts = make_tools(io)
    read, drive = ts.tools["get_latest_message"], ts.tools["publish_twist"]
    assert (read.metadata["effect"], drive.metadata["effect"]) == (
        Effect.READ,
        Effect.ACTUATE,
    )
    assert "battery=87%" in asyncio.run(read.function(topic="/d1/status"))
    asyncio.run(drive.function(**DRIVE.args))
    assert wait_for(lambda: published and published[-1] == (0.0, 0.0))
    assert (0.5, 0.1) in published


def test_react_loop(robot):
    io, _ = robot
    script = Script(
        [[Call("get_latest_message", {"topic": "/d1/status"})], Text("done")]
    )
    out = run(build_agent(scripted_model(script), [make_tools(io)]), "status?")
    assert out.output == "done"
    assert "battery=87%" in tool_results(out)[0][1]


def test_state_context_per_call_not_in_history(robot):
    io, _ = robot
    read = [Call("get_latest_message", {"topic": "/d1/status"})]
    script = Script([read, read, Text("done")])
    agent = build_agent(
        scripted_model(script), [make_tools(io)], context=lambda: rb.observe_status(io)
    )
    out = run(agent, "status?")
    assert len(script.seen) == 3
    for seen in script.seen:  # exactly one fresh observation, always last
        assert sum("[robot state]" in t for _, t in seen) == 1
        assert "[robot state]" in seen[-1][1]
    assert "[robot state]" not in str(out.all_messages())


@pytest.mark.parametrize("decision", ["approve", "reject"])
def test_approval_gate(robot, decision):
    io, published = robot
    script = Script([[DRIVE], Text("done")])
    agent = build_agent(scripted_model(script), [make_tools(io)])
    out = run(agent, "drive")
    assert isinstance(out.output, DeferredToolRequests) and not published
    answer = True if decision == "approve" else ToolDenied("no")
    results = DeferredToolResults(
        approvals={c.tool_call_id: answer for c in out.output.approvals}
    )
    out = run(agent, message_history=out.all_messages(), deferred_tool_results=results)
    assert out.output == "done"
    if decision == "approve":
        assert wait_for(lambda: published and published[-1] == (0.0, 0.0))
    else:
        time.sleep(0.2)
        assert not published and tool_results(out)[-1][2]


@pytest.mark.parametrize(
    "hook", [guard.crashing_hook, guard.hanging_hook, guard.deny_actuate]
)
def test_fail_closed(robot, hook):
    io, published = robot
    read = Call("get_latest_message", {"topic": "/d1/status"})
    script = Script([[DRIVE], [read], Text("done")])
    agent = build_agent(
        scripted_model(script),
        [make_tools(io)],
        hooks=[hook],
        approve_actuation=False,
        hook_timeout_s=0.2,
    )
    t0 = time.monotonic()
    out = run(agent, "drive")
    assert time.monotonic() - t0 < 2
    denied, read_result = tool_results(out)
    assert denied[2] and denied[1].startswith("Denied")
    if hook is guard.deny_actuate:  # a working hook still lets READ through
        assert "battery" in read_result[1]
    assert out.output == "done" and not published


def test_cancellation_stops_slow_tool(robot):
    io, published = robot
    slow = Call("publish_twist", {"linear_x": 0.3, "angular_z": 0.0, "duration_s": 10})
    agent = build_agent(
        scripted_model(Script([[slow], Text("done")])),
        [make_tools(io)],
        approve_actuation=False,
    )

    async def main():
        task = asyncio.create_task(agent.run("drive"))
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
    rec = guard.Recorder()
    script = Script([[Call("robot_name", {})], Text("done")])
    out = run(
        build_agent(scripted_model(script), [client_toolset(EXT_SERVER)], hooks=[rec]),
        "name?",
    )
    assert "rosbot-xl-7" in tool_results(out)[0][1]
    assert rec.calls[0].name == "robot_name" and rec.calls[0].effect is Effect.WRITE


@pytest.mark.parametrize("robot_kind", ["fake", "ros"])
def test_mcp_server_stdio_e2e(request, robot_kind):
    sim = request.getfixturevalue("ros") if robot_kind == "ros" else None
    r = asyncio.run(drive_server("pydanticai", robot_kind))
    assert r["tools"]["get_latest_message"].annotations.readOnlyHint
    assert r["tools"]["publish_twist"].annotations.destructiveHint
    assert "battery=87%" in r["read"].content[0].text and not r["drive"].isError
    if sim:
        assert wait_for(lambda: (0.2, 0.0) in sim.published)


@pytest.mark.parametrize("hook", ["deny", "crash"])
def test_mcp_server_fail_closed(hook):
    r = asyncio.run(drive_server("pydanticai", "fake", hook))
    assert r["drive"].isError and "Denied" in r["drive"].content[0].text
    assert "battery" in r["read"].content[0].text or hook == "crash"


def test_denied_call_never_reaches_human(robot):
    """FailClosed wraps ApprovalRequiredToolset, so a denied call is never put to a human."""
    io, published = robot
    agent = build_agent(
        scripted_model(Script([[DRIVE], Text("done")])),
        [make_tools(io)],
        hooks=[guard.deny_actuate],
    )
    out = run(agent, "drive")
    assert (
        out.output == "done"
        and tool_results(out)[0][1].startswith("Denied")
        and not published
    )
