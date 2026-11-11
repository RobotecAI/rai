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

"""F2: the agent flight recorder (rai.recorder) writes one MCAP per session."""

import asyncio
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from builtin_interfaces.msg import Time
from geometry_msgs.msg import Twist
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, messages_from_dict
from langchain_core.outputs import ChatGeneration, ChatResult
from rai.agents import ReActAgent, StateBasedAgent
from rai.agents.events import StateChanged, ToolStart
from rai.communication import HRIMessage
from rai.communication.ros2 import ROS2Connector
from rai.communication.ros2 import runtime as rt
from rai.communication.ros2.egress import global_egress
from rai.content import Image
from rai.messages import to_blocks
from rai.recorder import (
    DECISIONS,
    EVENTS,
    IMAGES,
    LLM,
    LOG,
    OBSERVATIONS,
    TOOLS,
    FlightRecorder,
    Recording,
    observation_images,
)
from rai.safety import EStop
from rai.tools import Allow, Deny, Effect, ToolCallContext, global_tool_hooks, rai_tool
from rosgraph_msgs.msg import Clock
from std_msgs.msg import String

from tests.communication.ros2.helpers import ros_setup

_ = ros_setup
pytestmark = pytest.mark.timeout(60)

USAGE = {"input_tokens": 12, "output_tokens": 3, "total_tokens": 15}


@pytest.fixture(autouse=True)
def _clean_hooks() -> Iterator[None]:
    yield
    global_egress.clear()
    global_egress._observers = ()
    global_tool_hooks.clear()
    global_tool_hooks._observers = ()
    rt._goal_watchers = ()


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
            message = AIMessage(step, usage_metadata=USAGE)
        else:
            calls = [
                {"name": n, "args": a, "id": f"call_{uuid.uuid4().hex[:6]}"}
                for n, a in step
            ]
            message = AIMessage("", tool_calls=calls, usage_metadata=USAGE)
        return ChatResult(generations=[ChatGeneration(message=message)])


def twist(x: float) -> Twist:
    msg = Twist()
    msg.linear.x = x
    return msg


def wait_for(condition: Callable[[], bool], timeout: float = 15.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.05)


def in_file_order_monotonic(recording: Recording) -> None:
    """log_time never goes back within a channel, and equals publish_time.
    The log merges agent events with hook decisions, which reach the recorder
    later but carry the time they were taken: it is ordered on read."""
    by_topic: dict[str, list[Any]] = {}
    for message in recording.messages(decode=False):
        by_topic.setdefault(message.topic, []).append(message)
    for topic, messages in by_topic.items():
        messages.sort(key=lambda m: m.sequence)
        stamps = [m.log_time for m in messages]
        assert topic == LOG or stamps == sorted(stamps), topic
        assert all(m.log_time == m.publish_time for m in messages), topic


# ------------------------------------------------------------ a full session


def test_records_a_scripted_react_run(ros_setup: None, tmp_path: Path) -> None:
    suffix = uuid.uuid4().hex[:8]
    cmd_vel, status = f"/rec_{suffix}/cmd_vel", f"/rec_{suffix}/status"
    client = ROS2Connector(f"rec_client_{suffix}")
    other = ROS2Connector(f"rec_other_{suffix}")

    @rai_tool(Effect.ACTUATE, name="drive")
    async def drive(speed: float) -> str:
        """Drive the base."""
        await asyncio.sleep(0.05)
        await client.publish(cmd_vel, twist(speed))
        await asyncio.sleep(0.05)
        return f"driving {speed}"

    @rai_tool(Effect.ACTUATE, name="launch")
    async def launch() -> str:
        """Never runs: denied."""
        return "launched"

    async def no_launch(call: ToolCallContext) -> Allow | Deny:
        return Deny("launch is forbidden") if call.name == "launch" else Allow()

    EStop().install()  # egress and tool decisions to record (all allow)
    global_tool_hooks.add(no_launch)
    agent = ReActAgent(
        ScriptedChatModel(
            steps=[[("drive", {"speed": 0.5})], [("launch", {})], "all done"]
        ),
        [drive, launch],
        system_prompt="You drive a robot.",
        name="pilot",
    )
    seen: list[Any] = []
    agent.subscribe(seen.append)
    stop_status = threading.Event()

    def publish_status() -> None:
        i = 0
        while not stop_status.is_set():
            other.sync.publish(status, String(data=f"s{i}"))
            i += 1
            time.sleep(0.02)

    path = tmp_path / "session.mcap"
    publisher = threading.Thread(target=publish_status, daemon=True)
    publisher.start()
    try:
        # status: type from the graph, once discovered; cmd_vel: type given
        recorder = FlightRecorder(
            path, agent=agent, topics={status: None, cmd_vel: Twist}, connector=client
        )
        with recorder:
            # wait for discovery, reading the unfinished file
            probe = client.subscribe(cmd_vel, msg_type=Twist, maxsize=100)

            def recorded_both() -> bool:
                client.sync.publish(cmd_vel, twist(-1.0))
                recorder.flush()
                return {status, cmd_vel} <= set(Recording(path).topics)

            wait_for(recorded_both)
            probe.close()
            answer = asyncio.run(agent.ainvoke("go"))
            recorder.flush()
            time.sleep(0.1)
    finally:
        stop_status.set()
        publisher.join()
        other.close()
        client.close()
    assert answer.text == "all done"
    assert recorder.dropped == 0 and recorder.errors == 0

    rec = Recording(path)
    assert rec.complete
    assert rec.topics == {
        EVENTS: "rai.AgentEvent",
        LLM: "rai.LLMCall",
        TOOLS: "rai.ToolCall",
        LOG: "foxglove.Log",
        DECISIONS: "rai.HookDecision",
        cmd_vel: "geometry_msgs/msg/Twist",
        status: "std_msgs/msg/String",
    }
    schemas = {s.name: s.encoding for s in rec.schemas.values()}
    assert schemas["geometry_msgs/msg/Twist"] == "ros2msg"
    assert schemas["rai.LLMCall"] == "jsonschema"
    assert rec.metadata["rai.session"]["clock"] == "wall"
    assert rec.metadata["rai.recorder.stats"]["dropped"] == "0"
    in_file_order_monotonic(rec)

    # agent events: one per event the agent emitted while recording
    events = [e for e in rec.values(EVENTS) if e["agent"] == "pilot"]
    assert [e["event"] for e in events] == [type(e).__name__ for e in seen]

    # LLM turns: model, prompt delta, response, usage
    llm = rec.values(LLM)
    assert [c["phase"] for c in llm] == ["request", "response"] * 3
    requests, responses = llm[::2], llm[1::2]
    assert {c["model"] for c in requests} == {"scripted-1"}
    assert [c["prefix"] for c in requests] == [0, 2, 4]
    assert [m["type"] for m in requests[0]["messages"]] == ["system", "human"]
    assert responses[0]["tool_calls"][0]["name"] == "drive"
    assert responses[-1]["text"] == "all done"
    assert all(c["usage"] == USAGE and c["latency_ms"] >= 0 for c in responses)
    prompts = rec.prompts("pilot")
    full = messages_from_dict(prompts[-1]["messages"])
    assert [m.type for m in full] == ["system", "human", "ai", "tool", "ai", "tool"]
    assert full[1].content == "go" and full[3].content == "driving 0.5"

    # tool calls with timing, the denial with its reason
    tools = rec.values(TOOLS)
    assert [(t["name"], t["phase"]) for t in tools] == [
        ("drive", "start"),
        ("drive", "end"),
        ("launch", "start"),
        ("launch", "end"),
    ]
    assert tools[0]["args"] == {"speed": 0.5} and tools[0]["effect"] == "actuate"
    assert tools[1]["status"] == "success" and tools[1]["duration_ms"] >= 100
    assert tools[3]["status"] == "denied"
    assert "launch is forbidden" in tools[3]["content"]

    # hook decisions of both hook points
    decisions = rec.values(DECISIONS)
    tool_side = [(d["name"], d["hook"], d["decision"]) for d in decisions if d["side"] == "tool"]  # fmt: skip
    assert ("launch", "no_launch", "deny") in tool_side
    assert ("drive", "no_launch", "allow") in tool_side
    egress = [
        m for m in rec.messages([DECISIONS])
        if m.value["side"] == "egress" and m.value["origin"] == "drive"
    ]  # fmt: skip
    assert egress and egress[0].value["name"] == cmd_vel
    assert any(e["level"] == 3 and "no_launch denied" in e["message"] for e in rec.values(LOG))  # fmt: skip

    # raw topics decode with the recorded ros2msg schema
    statuses = rec.values(status)
    assert statuses and all(s.data.startswith("s") for s in statuses)
    driven = [m for m in rec.messages([cmd_vel]) if m.value.linear.x == 0.5]
    assert len(driven) == 1

    # alignment: the tool's publish lands inside the tool call, after its egress check
    start, end = [
        m.log_time for m in rec.messages([TOOLS]) if m.value["name"] == "drive"
    ]
    assert start < egress[0].log_time <= driven[0].log_time < end
    assert driven[0].log_time - egress[0].log_time < 50_000_000  # 50 ms

    if shutil.which("mcap"):
        info = subprocess.run(
            ["mcap", "info", str(path)], capture_output=True, text=True
        )
        assert info.returncode == 0 and cmd_vel in info.stdout


# ------------------------------------------------------------ content


def test_observations_and_images_are_stored_once(tmp_path: Path) -> None:
    camera = Image(np.full((8, 8, 3), 200, np.uint8))
    frames = iter([to_blocks(["front", camera])] * 2 + [None])

    async def front() -> Any:
        return next(frames, None)

    async def battery() -> str:
        return "80%"

    photo = Image(np.zeros((6, 6, 3), np.uint8))
    agent = StateBasedAgent(
        ScriptedChatModel(steps=["one", "two", "three"]),
        state_sources={"front camera": front, "battery": battery},
        name="watcher",
    )
    path = tmp_path / "obs.mcap"
    with FlightRecorder(path, agent=agent, audit=False):
        for _ in range(3):
            asyncio.run(agent.ainvoke(HRIMessage("look", [photo])))

    rec = Recording(path)
    camera_topic = observation_images("front camera")
    assert camera_topic == "/rai/agent/observations/front_camera/image"
    assert rec.topics[camera_topic] == "foxglove.CompressedImage"
    assert len(rec.values(camera_topic)) == 1  # the same frame: stored once
    assert len(rec.values(IMAGES)) == 1  # the user's photo, in every prompt
    observations = rec.values(OBSERVATIONS)
    assert [o["fresh"] for o in observations] == [
        ["front camera", "battery"],
        ["front camera", "battery"],
        ["battery"],
    ]
    blocks = observations[0]["values"]["front camera"]
    assert blocks[1]["base64"].startswith("rai-image:")
    restored = rec.restore(blocks)
    assert restored[1]["base64"] == to_blocks([camera])[0]["base64"]
    image = rec.values(camera_topic)[0]
    assert image["format"] == "jpeg" and image["frame_id"] in blocks[1]["base64"]
    prompt = rec.prompts()[-1]["messages"]
    assert prompt[0]["data"]["content"][1]["base64"] == to_blocks([photo])[0]["base64"]


def test_flood_drops_instead_of_blocking(tmp_path: Path) -> None:
    agent = ReActAgent(ScriptedChatModel(steps=[]), name="flood")
    path = tmp_path / "flood.mcap"
    n = 20_000
    with FlightRecorder(path, agent=agent, audit=False, maxsize=64) as recorder:
        start = time.perf_counter()
        for i in range(n):
            agent.emit(
                ToolStart(
                    agent="flood",
                    turn="t",
                    call_id=str(i),
                    name="x",
                    args={"i": i, "pad": "p" * 200},
                    effect=Effect.READ,
                )
            )
        elapsed = time.perf_counter() - start
    assert elapsed < 2.0, f"emitting {n} events took {elapsed:.2f}s"
    assert recorder.dropped > 0
    assert recorder.dropped_by_input == {"agent": recorder.dropped}
    rec = Recording(path)
    events = rec.values(EVENTS)
    assert len(events) + recorder.dropped == n
    assert rec.metadata["rai.recorder.stats"]["dropped"] == str(recorder.dropped)


def test_an_unfinished_file_is_readable(tmp_path: Path) -> None:
    agent = ReActAgent(ScriptedChatModel(steps=["hi"]), name="crashy")
    path = tmp_path / "crash.mcap"
    recorder = FlightRecorder(path, agent=agent, audit=False, flush_s=0.05).start()
    try:
        asyncio.run(agent.ainvoke("hello"))
        recorder.flush()
        cut = tmp_path / "cut.mcap"
        data = path.read_bytes()
        cut.write_bytes(data[:-7])  # a crash in the middle of a write
        live, broken = Recording(path), Recording(cut)
        assert not live.complete and not broken.complete
        assert [c["phase"] for c in live.values(LLM)] == ["request", "response"]
        assert len(broken.values(LLM)) <= 2  # the cut chunk may be lost
        assert live.metadata["rai.session"]["agents"] == '["crashy"]'
    finally:
        recorder.stop()
    assert Recording(path).complete


def test_sim_time_stamps(ros_setup: None, tmp_path: Path) -> None:
    suffix = uuid.uuid4().hex[:8]
    sim = ROS2Connector(f"rec_sim_{suffix}", use_sim_time=True)
    clock = ROS2Connector(f"rec_clock_{suffix}")
    agent = ReActAgent(ScriptedChatModel(steps=[]), name="simmer")
    try:
        sim_s = 1_000_000

        def at_sim_time() -> bool:
            clock.sync.publish("/clock", Clock(clock=Time(sec=sim_s)))
            return sim.now().nanoseconds >= sim_s * 10**9

        wait_for(at_sim_time)
        path = tmp_path / "sim.mcap"
        with FlightRecorder(path, agent=agent, audit=False, connector=sim):
            agent.emit(StateChanged(agent="simmer", old="a", new="b"))  # type: ignore[arg-type]
    finally:
        clock.close()
        sim.close()
    rec = Recording(path)
    assert rec.metadata["rai.session"]["clock"] == "ros"
    (event,) = rec.messages([EVENTS])
    assert sim_s * 10**9 <= event.log_time < (sim_s + 60) * 10**9


def test_missing_extra_is_explained(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "mcap.writer", None)
    with pytest.raises(ImportError, match=r"rai_core\[recorder\]"):
        FlightRecorder(tmp_path / "x.mcap")
