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


"""RunPolicyTool on real ROS topics against the fake openpi server: observation
assembly, actions at rate, safe stop on every exit, prompt cancel, egress deny,
timeout, success signal and a server that is down."""

import asyncio
import statistics
import threading
import time
import uuid
from collections.abc import Iterator
from typing import Any

import numpy as np
import pytest
from geometry_msgs.msg import Twist
from rai.communication.ros2 import ROS2Connector
from rai.communication.ros2.egress import MOTION, Allow, Deny, Egress
from rai.communication.ros2.images import numpy_to_imgmsg
from rai.tools.policies import (
    FakePolicyServer,
    OpenPIClient,
    PolicyProgress,
    RunPolicyTool,
)
from sensor_msgs.msg import JointState
from std_msgs.msg import Bool, Float64MultiArray
from trajectory_msgs.msg import JointTrajectory

from tests.communication.ros2.helpers import ros_setup

_ = ros_setup
pytestmark = pytest.mark.timeout(60)

JOINTS = ["j1", "j2", "j3"]


def unique(prefix: str) -> str:
    return f"/policy_{prefix}_{uuid.uuid4().hex[:8]}"


class Robot:
    """Publishes a camera, joint states and an EEF array at 50 Hz on its own
    connector, and records every command it receives with its arrival time."""

    def __init__(self) -> None:
        self.connector = ROS2Connector(f"policy_robot_{uuid.uuid4().hex[:8]}")
        self.camera, self.joints = unique("camera"), unique("joint_states")
        self.eef, self.arm = unique("eef"), unique("arm_cmd")
        self.base, self.done = unique("cmd_vel"), unique("done")
        self.commands: list[tuple[float, str, Any]] = []
        self.publish_camera = True
        image = np.zeros((48, 64, 3), np.uint8)
        image[..., 0] = 10  # bgr8: blue 10, green 20, red 30
        image[..., 1] = 20
        image[..., 2] = 30
        self.image = numpy_to_imgmsg(image, "bgr8")
        # in message order j3, j1, j2: the tool must reorder
        self.state = JointState(name=["j3", "j1", "j2"], position=[3.0, 1.0, 2.0])
        for topic in (self.arm, self.base):
            self.connector.subscribe(
                topic,
                lambda m, t=topic: self.commands.append((time.monotonic(), t, m)),
                msg_type=JointTrajectory if topic == self.arm else Twist,
                maxsize=None,
            )
        self._running = True
        self._thread = threading.Thread(target=self._spin, daemon=True)
        self._thread.start()

    def _spin(self) -> None:
        while self._running:
            if self.publish_camera:
                self.connector.sync.publish(self.camera, self.image)
            self.connector.sync.publish(self.joints, self.state)
            self.connector.sync.publish(self.eef, Float64MultiArray(data=[9.0]))
            time.sleep(0.02)

    def arm_commands(self) -> list[tuple[float, JointTrajectory]]:
        return [(t, m) for t, topic, m in self.commands if topic == self.arm]

    def close(self) -> None:
        self._running = False
        self._thread.join(2.0)
        self.connector.close()


@pytest.fixture
def robot(ros_setup: None) -> Iterator[Robot]:
    robot = Robot()
    yield robot
    robot.close()


@pytest.fixture
def agent(ros_setup: None) -> Iterator[ROS2Connector]:
    connector = ROS2Connector(f"policy_agent_{uuid.uuid4().hex[:8]}")
    yield connector
    connector.close()


def chunk(step: float, rows: int = 5) -> np.ndarray:
    """Rows i: [step + i/10, ..] over 7 dims: 3 arm joints, 1 gripper, 3 base."""
    base = step + np.arange(rows)[:, None] / 10.0
    return np.repeat(base, 7, axis=1)


def make_tool(
    robot: Robot, agent: ROS2Connector, url: str, **kwargs: Any
) -> RunPolicyTool:
    config: dict[str, Any] = dict(
        connector=agent,
        policy=OpenPIClient(url, timeout_s=2.0, connect_timeout_s=1.0),
        inputs=[
            {
                "kind": "image",
                "key": "observation/image",
                "topic": robot.camera,
                "size": (32, 32),
            },
            {
                "kind": "joint_state",
                "key": "observation/state",
                "topic": robot.joints,
                "joints": JOINTS,
            },
            {"kind": "array", "key": "observation/state", "topic": robot.eef},
        ],
        outputs=[
            {
                "kind": "joint_trajectory",
                "topic": robot.arm,
                "joints": JOINTS,
                "dims": [0, 1, 2],
            },
            {
                "kind": "twist",
                "topic": robot.base,
                "dims": [4, None, None, None, None, 5],
                "clip": (-1.0, 1.0),
            },
        ],
        rate_hz=20.0,
        max_duration_s=5.0,
    )
    config.update(kwargs)
    return RunPolicyTool(**config)


def wait_for_sensors(agent: ROS2Connector, robot: Robot) -> None:
    for topic in (robot.camera, robot.joints, robot.eef):
        agent.sync.receive(topic, timeout=10.0)


def is_stop(msg: Any) -> bool:
    if isinstance(msg, JointTrajectory):
        return not msg.points
    return msg == Twist()


def settle(robot: Robot, timeout: float = 2.0) -> None:
    """Wait until the last command is a stop on both outputs."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        last = {topic: m for _, topic, m in robot.commands}
        if len(last) == 2 and all(is_stop(m) for m in last.values()):
            return
        time.sleep(0.02)


# ------------------------------------------------------------------ tests


def test_observation_assembly(robot: Robot, agent: ROS2Connector) -> None:
    with FakePolicyServer([chunk(0.0)]) as server:
        tool = make_tool(robot, agent, server.url)
        wait_for_sensors(agent, robot)
        result = tool.invoke({"instruction": "pick the cup", "max_duration_s": 0.3})
    assert "ran for its 0.3 s limit" in result
    obs = server.observations[0]
    assert obs["prompt"] == "pick the cup"
    image = obs["observation/image"]
    assert image.shape == (32, 32, 3) and image.dtype == np.uint8
    # 48x64 padded into 32x32: rows 4..27 hold the picture, in RGB order
    np.testing.assert_array_equal(image[16, 16], [30, 20, 10])
    np.testing.assert_array_equal(image[0, 0], [0, 0, 0])
    # joints reordered to j1, j2, j3, then the EEF array concatenated
    np.testing.assert_array_equal(obs["observation/state"], [1.0, 2.0, 3.0, 9.0])
    assert tool.effect == "actuate"
    assert set(tool.targets) == {robot.arm, robot.base}


def test_actions_published_at_rate_then_stopped(
    robot: Robot, agent: ROS2Connector
) -> None:
    progress: list[PolicyProgress] = []
    with FakePolicyServer([chunk(0.0), chunk(1.0)]) as server:
        tool = make_tool(
            robot, agent, server.url, on_progress=progress.append, progress_period_s=0.5
        )
        wait_for_sensors(agent, robot)
        result = tool.invoke({"instruction": "wave", "max_duration_s": 1.0})
    settle(robot)
    assert "ran for its 1 s limit" in result
    arm = robot.arm_commands()
    moves = [(t, m) for t, m in arm if m.points]
    assert 15 <= len(moves) <= 21, len(moves)  # 20 Hz for 1 s, minus inference waits
    first = [list(m.points[0].positions) for _, m in moves[:6]]
    assert first[:5] == [[i / 10] * 3 for i in range(5)]
    assert first[5] == [1.0] * 3  # the second chunk
    assert moves[0][1].joint_names == JOINTS
    assert moves[0][1].points[0].time_from_start.nanosec == 50_000_000
    within_chunk = [b[0] - a[0] for a, b in zip(moves[:4], moves[1:5])]
    assert 0.03 < statistics.median(within_chunk) < 0.07
    assert len(server.observations) >= 3
    assert is_stop(arm[-1][1])  # the safe stop came last
    twists = [m for _, topic, m in robot.commands if topic == robot.base]
    assert twists[1].linear.x == pytest.approx(0.1) and twists[
        1
    ].angular.z == pytest.approx(0.1)
    assert max(t.linear.x for t in twists) <= 1.0  # clipped
    assert is_stop(twists[-1])
    assert progress and progress[0].instruction == "wave" and progress[-1].actions > 0


def test_cancel_stops_publishing_promptly(robot: Robot, agent: ROS2Connector) -> None:
    with FakePolicyServer([chunk(0.0, rows=50)]) as server:
        tool = make_tool(robot, agent, server.url, rate_hz=50.0)
        wait_for_sensors(agent, robot)

        async def run() -> float:
            task = asyncio.ensure_future(tool.ainvoke({"instruction": "go"}))
            await asyncio.sleep(0.6)
            cancelled = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert time.monotonic() - cancelled < 0.2
            return cancelled

        cancelled = asyncio.run(run())
    settle(robot)
    arm = robot.arm_commands()
    moves = [t for t, m in arm if m.points]
    assert len(moves) > 10
    last_move, stop = moves[-1], arm[-1]
    assert is_stop(stop[1])
    latency = last_move - cancelled
    print(
        f"cancel -> last command {latency * 1e3:.1f} ms, -> stop {(stop[0] - cancelled) * 1e3:.1f} ms"
    )
    assert latency < 0.2
    assert stop[0] - cancelled < 0.2


def test_cancel_during_inference(robot: Robot, agent: ROS2Connector) -> None:
    with FakePolicyServer([chunk(0.0)], delay_s=2.0) as server:
        tool = make_tool(robot, agent, server.url)
        wait_for_sensors(agent, robot)

        async def run() -> float:
            task = asyncio.ensure_future(tool.ainvoke({"instruction": "go"}))
            while not server.observations:
                await asyncio.sleep(0.01)
            start = time.monotonic()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            return time.monotonic() - start

        assert asyncio.run(run()) < 0.2
    settle(robot)
    assert all(is_stop(m) for _, _, m in robot.commands)  # only the safe stop
    assert robot.commands


def test_egress_deny_stops_with_an_error(robot: Robot, agent: ROS2Connector) -> None:
    class EStop:
        __name__ = "estop"
        triggered = False

        def __call__(self, egress: Egress) -> Any:
            return Deny("e-stop is active") if self.triggered else Allow()

    estop = EStop()
    agent.egress.add(estop, types=MOTION)
    with FakePolicyServer([chunk(0.0, rows=100)]) as server:
        tool = make_tool(robot, agent, server.url)
        wait_for_sensors(agent, robot)
        threading.Timer(0.5, lambda: setattr(estop, "triggered", True)).start()
        message = asyncio.run(
            tool.ainvoke(
                {
                    "type": "tool_call",
                    "name": tool.name,
                    "id": "1",
                    "args": {"instruction": "go"},
                }
            )
        )
    assert message.status == "error"
    assert "e-stop is active" in message.content
    assert "robot halted" in message.content
    time.sleep(0.2)
    moves = [t for t, m in robot.arm_commands() if m.points]
    assert moves and moves[-1] < time.monotonic() - 0.15  # nothing after the deny


def test_success_signal_ends_the_run(robot: Robot, agent: ROS2Connector) -> None:
    done = unique("done")
    with FakePolicyServer([chunk(0.0)]) as server:
        tool = make_tool(robot, agent, server.url, success_topic=done)
        wait_for_sensors(agent, robot)

        def signal() -> None:
            robot.connector.sync.publish(done, Bool(data=False))
            time.sleep(0.3)
            robot.connector.sync.publish(done, Bool(data=True))

        threading.Timer(0.3, signal).start()
        start = time.monotonic()
        result = tool.invoke({"instruction": "go", "max_duration_s": 5.0})
    assert result.startswith("Policy succeeded")
    assert time.monotonic() - start < 2.0
    settle(robot)
    assert is_stop(robot.arm_commands()[-1][1])


def test_success_topic_timeout_message(robot: Robot, agent: ROS2Connector) -> None:
    with FakePolicyServer([chunk(0.0)]) as server:
        tool = make_tool(
            robot, agent, server.url, success_topic=unique("done"), max_duration_s=0.4
        )
        wait_for_sensors(agent, robot)
        result = tool.invoke({"instruction": "go", "max_duration_s": 100.0})  # capped
    assert "stopped at its 0.4 s limit with no success signal" in result


def test_server_down_is_a_clean_tool_error(robot: Robot, agent: ROS2Connector) -> None:
    import socket

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    tool = make_tool(robot, agent, f"ws://127.0.0.1:{port}")
    wait_for_sensors(agent, robot)
    result = tool.invoke({"instruction": "go"})
    assert "Cannot connect to the openpi policy server" in result
    assert "robot halted" in result
    settle(robot)
    assert robot.commands and all(is_stop(m) for _, _, m in robot.commands)


def test_server_error_midway(robot: Robot, agent: ROS2Connector) -> None:
    with FakePolicyServer([chunk(0.0), RuntimeError("CUDA out of memory")]) as server:
        tool = make_tool(robot, agent, server.url)
        wait_for_sensors(agent, robot)
        result = tool.invoke({"instruction": "go"})
    assert "CUDA out of memory" in result and "5 actions" in result
    settle(robot)
    assert is_stop(robot.arm_commands()[-1][1])


def test_stale_camera_is_an_error(robot: Robot, agent: ROS2Connector) -> None:
    with FakePolicyServer([chunk(0.0)]) as server:
        tool = make_tool(robot, agent, server.url, observation_timeout_s=0.3)
        wait_for_sensors(agent, robot)
        robot.publish_camera = False
        time.sleep(0.7)  # older than observation_max_age_s
        result = tool.invoke({"instruction": "go"})
    assert robot.camera in result
    assert not server.observations


def test_prefetch_overlaps_inference(robot: Robot, agent: ROS2Connector) -> None:
    with FakePolicyServer([chunk(0.0)], delay_s=0.1) as server:
        tool = make_tool(robot, agent, server.url, prefetch_steps=3)
        wait_for_sensors(agent, robot)
        tool.invoke({"instruction": "go", "max_duration_s": 1.2})
    moves = [t for t, m in robot.arm_commands() if m.points]
    gaps = [b - a for a, b in zip(moves, moves[1:])]
    assert len(moves) >= 19
    assert max(gaps) < 0.09  # no wait for the policy once running


def test_action_width_and_config_checks(robot: Robot, agent: ROS2Connector) -> None:
    with FakePolicyServer([np.zeros((2, 3))]) as server:
        tool = make_tool(robot, agent, server.url)
        wait_for_sensors(agent, robot)
        result = tool.invoke({"instruction": "go"})
    assert "returned 3 action dims" in result
    with pytest.raises(ValueError, match="not in the MOTION set"):
        make_tool(robot, agent, "ws://x", motion_types=frozenset())
    with pytest.raises(ValueError, match="same length"):
        make_tool(
            robot,
            agent,
            "ws://x",
            outputs=[
                {
                    "kind": "joint_trajectory",
                    "topic": "/a",
                    "joints": ["a"],
                    "dims": [0, 1],
                }
            ],
        )
    with pytest.raises(ValueError, match="image input key"):
        make_tool(
            robot,
            agent,
            "ws://x",
            inputs=[
                {"kind": "image", "key": "k", "topic": "/a"},
                {"kind": "array", "key": "k", "topic": "/b"},
            ],
        )


def test_forbidden_output_topic(robot: Robot, agent: ROS2Connector) -> None:
    tool = make_tool(robot, agent, "ws://127.0.0.1:1", forbidden=[robot.base])
    result = tool.invoke({"instruction": "go"})
    assert "is not writable" in result
    assert not robot.commands


def test_progress_reaches_the_langchain_event_stream(
    robot: Robot, agent: ROS2Connector
) -> None:
    with FakePolicyServer([chunk(0.0)]) as server:
        tool = make_tool(robot, agent, server.url, progress_period_s=0.2)
        wait_for_sensors(agent, robot)

        async def run() -> list[Any]:
            events = tool.astream_events(
                {"instruction": "go", "max_duration_s": 0.7}, version="v2"
            )
            return [e async for e in events if e["event"] == "on_custom_event"]

        events = asyncio.run(run())
    assert events and events[0]["name"] == "rai_policy_progress"
    assert events[-1]["data"]["actions"] > 0
