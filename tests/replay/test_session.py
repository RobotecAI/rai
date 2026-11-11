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

"""F3 end to end: record a session with the flight recorder, then replay its
topics live (ReplayPlayer) and re-run the agent offline (AgentReplay)."""

import asyncio
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from geometry_msgs.msg import Twist
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from rai.agents import ReActAgent
from rai.agents.events import AgentEvent
from rai.communication.ros2 import ROS2Connector
from rai.communication.ros2.egress import global_egress
from rai.recorder import FlightRecorder, Recording
from rai.replay import (
    AgentReplay,
    Bag,
    ReplayChatModel,
    ReplayExhausted,
    ReplayPlayer,
    expect_published,
    expect_tool_called,
)
from rai.tools import Effect, RaiTool, global_tool_hooks, rai_tool
from std_msgs.msg import String

from tests.communication.ros2.helpers import ros_setup

_ = ros_setup
pytestmark = pytest.mark.timeout(90)

SYSTEM = "You drive a robot."
STATUS = "battery 80%"


@pytest.fixture(autouse=True)
def _clean_hooks() -> Iterator[None]:
    yield
    global_egress.clear()
    global_egress._observers = ()
    global_tool_hooks.clear()
    global_tool_hooks._observers = ()


class ScriptedChatModel(BaseChatModel):
    """Plays steps in order: a final text, or a list of (tool, args) calls."""

    steps: list[Any]
    model_name: str = "scripted-1"

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools: Any, **kwargs: Any) -> "ScriptedChatModel":
        return self

    def _generate(
        self, messages: Any, stop: Any = None, run_manager: Any = None, **kwargs: Any
    ) -> ChatResult:
        step = self.steps.pop(0) if self.steps else "done"
        if isinstance(step, str):
            message = AIMessage(step)
        else:
            calls = [
                {"name": n, "args": a, "id": f"call_{uuid.uuid4().hex[:6]}"}
                for n, a in step
            ]
            message = AIMessage("", tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


def make_tools(
    connector: Any, status: str, cmd_vel: str, *, gain: float = 1.0, say: str = ""
) -> list[RaiTool]:
    """The tools under test. gain and say change their behaviour."""

    @rai_tool(Effect.READ, name="read_status")
    async def read_status() -> str:
        """Read the robot status."""
        msg = await connector.receive(status, msg_type=String, timeout=5.0)
        return msg.data

    @rai_tool(Effect.ACTUATE, name="drive")
    async def drive(speed: float) -> str:
        """Drive the base."""
        msg = Twist()
        msg.linear.x = speed * gain
        await connector.publish(cmd_vel, msg)
        return say or f"driving {speed}"

    return [read_status, drive]


def wait_for(condition: Callable[[], bool], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.05)


@dataclass
class Session:
    path: Path
    status: str
    cmd_vel: str
    answer: str


@pytest.fixture(scope="module")
def session(
    ros_setup: None, tmp_path_factory: pytest.TempPathFactory
) -> Iterator[Session]:
    """A recorded session: the agent reads a status topic, then drives."""
    suffix = uuid.uuid4().hex[:8]
    status, cmd_vel = f"/rpl_{suffix}/status", f"/rpl_{suffix}/cmd_vel"
    client = ROS2Connector(f"rpl_client_{suffix}")
    robot = ROS2Connector(f"rpl_robot_{suffix}")
    agent = ReActAgent(
        ScriptedChatModel(
            steps=[[("read_status", {})], [("drive", {"speed": 0.5})], "all done"]
        ),
        make_tools(client, status, cmd_vel),
        system_prompt=SYSTEM,
        name="pilot",
    )
    stop = threading.Event()

    def publish_status() -> None:
        while not stop.is_set():
            robot.sync.publish(status, String(data=STATUS))
            time.sleep(0.05)

    path = tmp_path_factory.mktemp("replay") / "session.mcap"
    publisher = threading.Thread(target=publish_status, daemon=True)
    publisher.start()
    try:
        recorder = FlightRecorder(
            path,
            agent=agent,
            topics={status: String, cmd_vel: Twist},
            connector=client,
        )
        with recorder:
            probe = client.subscribe(cmd_vel, msg_type=Twist, maxsize=100)

            def recorded_both() -> bool:  # discovery, before the agent runs
                client.sync.publish(cmd_vel, Twist())
                recorder.flush()
                return {status, cmd_vel} <= set(Recording(path).topics)

            wait_for(recorded_both)
            probe.close()
            client.sync.receive(status, msg_type=String, timeout=10.0)
            answer = asyncio.run(agent.ainvoke("go"))
            time.sleep(0.3)  # more status after the run, for the player
            recorder.flush()
    finally:
        stop.set()
        publisher.join()
        robot.close()
        client.close()
    assert recorder.dropped == 0 and recorder.errors == 0
    yield Session(path, status, cmd_vel, str(answer.content))


def test_the_recording(session: Session) -> None:
    assert session.answer == "all done"
    expect_tool_called(session.path, "read_status", {}, status="success", count=1)
    (drive,) = expect_tool_called(session.path, "drive", {"speed": 0.5})
    assert drive.content == "driving 0.5"
    bag = Bag(session.path)
    assert bag.topics == {
        session.status: "std_msgs/msg/String",
        session.cmd_vel: "geometry_msgs/msg/Twist",
    }
    assert bag.clock == "wall"


# ------------------------------------------------------------ (a) live


def test_player_feeds_a_fresh_agent_on_the_real_connector(session: Session) -> None:
    connector = ROS2Connector(f"rpl_live_{uuid.uuid4().hex[:8]}")
    agent = ReActAgent(
        ScriptedChatModel(steps=[[("read_status", {})], "ok"]),
        make_tools(connector, session.status, session.cmd_vel),
        name="fresh",
    )
    events: list[AgentEvent] = []
    agent.subscribe(events.append)
    try:
        with ReplayPlayer(session.path, loop=True, rate=2.0) as player:
            assert player.skipped == [session.cmd_vel]  # motion is never replayed
            assert list(player.topics) == [session.status]
            sub = connector.subscribe(session.status, msg_type=String)
            assert player.wait_for_subscribers(timeout=15.0)
            player.start()
            first = connector.sync.receive(session.status, timeout=10.0)
            assert first.data == STATUS
            sub.close()
            asyncio.run(agent.ainvoke("status?"))
            assert player.published > 0
    finally:
        connector.close()
    (call,) = expect_tool_called(events, "read_status", status="success")
    assert call.content == STATUS


def test_player_seek_and_end(session: Session) -> None:
    first, last = Bag(session.path).bounds()
    with ReplayPlayer(session.path, topics=[session.status], rate=50.0) as player:
        player.start()
        assert player.wait(timeout=30.0)
        assert player.loops == 1 and player.stamp == last
        count = player.published
        player.seek((last - first) / 1e9 - 0.2)  # replay the last 0.2 s
        wait_for(lambda: player.done)
        assert count < player.published < 2 * count


# ------------------------------------------------------------ (d) clock


def test_clock_follows_the_recording(session: Session) -> None:
    first, last = Bag(session.path).bounds()
    sim = ROS2Connector(f"rpl_sim_{uuid.uuid4().hex[:8]}", use_sim_time=True)
    try:
        with ReplayPlayer(
            session.path, topics=[session.status], clock=True, loop=True
        ) as player:
            player.start()
            wait_for(lambda: sim.now().nanoseconds >= first)
            seen = []
            for _ in range(5):
                seen.append(sim.now().nanoseconds)
                time.sleep(0.05)
    finally:
        sim.close()
    assert all(first <= t <= last for t in seen), (first, seen, last)
    assert seen[-1] > seen[0]  # it moves (a loop restart would go back: unlikely)


# ------------------------------------------------------------ (b), (c) offline


def _agent(replay: AgentReplay, session: Session, **behaviour: Any) -> ReActAgent:
    tools = make_tools(replay.connector, session.status, session.cmd_vel, **behaviour)
    return ReActAgent(replay.model, tools, system_prompt=SYSTEM, name="pilot")


def test_offline_rerun_matches_the_recording(session: Session) -> None:
    replay = AgentReplay(session.path)
    assert replay.agent_name == "pilot"
    assert [[m.content for m in turn] for turn in replay.turns()] == [["go"]]
    started = time.monotonic()
    report = asyncio.run(replay.run(_agent(replay, session)))
    assert report.ok, str(report)
    assert time.monotonic() - started < 10  # no real waiting
    assert (
        report.tool_calls
        == report.recorded_tool_calls
        == [
            ("read_status", {}),
            ("drive", {"speed": 0.5}),
        ]
    )
    assert report.model_calls == report.recorded_model_calls == 3
    expect_tool_called(replay, "read_status", status="success")
    assert replay.calls[0].content == STATUS
    (twist,) = expect_published(replay, session.cmd_vel, {"linear.x": 0.5}, count=1)
    assert [(r.kind, r.name, r.origin) for r in report.sent] == [
        ("publish", session.cmd_vel, "drive")
    ]
    # the virtual clock followed the session: drive ran at its recorded time
    tools = Recording(session.path).messages(["/rai/agent/tools"])
    drive_start = [m.log_time for m in tools if m.value["name"] == "drive"][0]
    assert report.sent[0].stamp == drive_start


def test_divergent_egress_is_reported(session: Session) -> None:
    replay = AgentReplay(session.path)
    report = asyncio.run(replay.run(_agent(replay, session, gain=1.4)))
    assert not report.ok
    (divergence,) = report.divergences
    assert divergence.kind == "egress" and "drive" in divergence.where
    assert "0.5" in divergence.detail and "0.7" in divergence.detail
    assert "DIVERGED" in str(report)


def test_divergent_tool_result_stops_a_strict_run(session: Session) -> None:
    replay = AgentReplay(session.path)
    report = asyncio.run(replay.run(_agent(replay, session, say="refusing")))
    first = report.divergences[0]
    assert first.kind == "tool_result"
    assert 'expected "driving 0.5", got "refusing"' in first.detail
    assert report.model_calls == 3  # the third call failed instead of answering
    with pytest.raises(AssertionError, match="tool_result at tool drive"):
        report.raise_on_divergence()

    lenient = AgentReplay(session.path, strict=False)
    report = asyncio.run(lenient.run(_agent(lenient, session, say="refusing")))
    kinds = [d.kind for d in report.divergences]
    assert kinds == ["tool_result", "prompt"]  # the next prompt has the new result
    assert "message 5 (tool)" in report.divergences[1].detail


def test_a_different_prompt_diverges(session: Session) -> None:
    replay = AgentReplay(session.path)
    tools = make_tools(replay.connector, session.status, session.cmd_vel)
    agent = ReActAgent(replay.model, tools, system_prompt="Be careful.", name="pilot")
    report = asyncio.run(replay.run(agent))
    (divergence,) = report.divergences
    assert divergence.kind == "prompt" and divergence.where == "model call 1"
    assert "Be careful." in divergence.detail
    assert report.tool_calls == []


def test_replay_model_runs_out(session: Session) -> None:
    model = ReplayChatModel.from_recording(session.path)
    assert model.model_name == "scripted-1" and model.remaining == 3
    for _ in range(3):
        model.invoke("anything")
    with pytest.raises(ReplayExhausted, match="only 3 responses"):
        model.invoke("more")
