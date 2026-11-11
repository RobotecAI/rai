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

"""F3: recorded bags and the offline ReplayConnector (no ROS graph)."""

import asyncio
import importlib.util
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from geometry_msgs.msg import TransformStamped, Twist
from nav2_msgs.action import NavigateToPose
from rai.communication.ros2 import EgressDenied, ROS2TimeoutError
from rai.communication.ros2.egress import Allow, Deny, Egress, global_egress
from rai.communication.ros2.runtime import message_class
from rai.replay import (
    Bag,
    ReplayConnector,
    ReplayEnded,
    ScriptedGoal,
    expect_called,
    expect_goal,
    expect_no_motion,
    expect_nothing_sent,
    expect_published,
)
from rai.tools.ros2.generic.topics import (
    GetROS2TopicsNamesAndTypesTool,
    GetROS2TransformTool,
    PublishROS2MessageTool,
    ReceiveROS2MessageTool,
)
from rclpy.serialization import deserialize_message
from rclpy.time import Time
from std_msgs.msg import String
from std_srvs.srv import SetBool
from tf2_msgs.msg import TFMessage
from tf2_ros import LookupException

from .bags import T0, Messages, S, write_mcap, write_rosbag2

pytestmark = pytest.mark.timeout(60)


@pytest.fixture(autouse=True)
def _clean_hooks() -> Iterator[None]:
    yield
    global_egress.clear()


def status(i: int) -> String:
    return String(data=f"s{i}")


def tf(parent: str, child: str, x: float, stamp: int) -> TFMessage:
    t = TransformStamped()
    t.header.frame_id, t.child_frame_id = parent, child
    t.header.stamp.sec, t.header.stamp.nanosec = divmod(stamp, S)
    t.transform.translation.x = x
    t.transform.rotation.w = 1.0
    return TFMessage(transforms=[t])


MESSAGES: Messages = [
    *[("/status", status(i), T0 + i * S) for i in range(1, 4)],  # 1001, 1002, 1003 s
    ("/tf_static", tf("map", "odom", 1.0, T0), T0),
    ("/tf", tf("odom", "base_link", 0.5, T0 + 1500_000_000), T0 + 1500_000_000),
    ("/tf", tf("odom", "base_link", 0.7, T0 + 2500_000_000), T0 + 2500_000_000),
]


@pytest.fixture
def bag(tmp_path: Path) -> Path:
    return write_mcap(tmp_path / "bag.mcap", MESSAGES)


# ------------------------------------------------------------ bags


def _read(path: Path) -> list[tuple[str, str, int, Any]]:
    # decoded: CDR padding bytes differ between writers
    return [
        (m.topic, m.type, m.stamp, deserialize_message(m.data, message_class(m.type)))
        for m in Bag(path).messages()
    ]


@pytest.mark.parametrize("storage", ["mcap", "sqlite3"])
def test_rosbag2_bags_read_like_mcap(bag: Path, tmp_path: Path, storage: str) -> None:
    if importlib.util.find_spec("rosbag2_py") is None:
        pytest.skip("rosbag2_py is not installed")
    import rosbag2_py

    if storage not in rosbag2_py.get_registered_writers():
        pytest.skip(f"no rosbag2 {storage} storage plugin")
    directory = write_rosbag2(tmp_path / f"bag_{storage}", MESSAGES, storage)
    expected = _read(bag)
    assert _read(directory) == expected
    files = [p for p in directory.iterdir() if p.suffix in (".mcap", ".db3")]
    assert len(files) == 1 and _read(files[0]) == expected
    replay = ReplayConnector(directory)
    assert asyncio.run(replay.receive("/status")).data == "s1"
    assert Bag(directory).clock == "bag"


def test_bag_topics_bounds_and_start(bag: Path) -> None:
    b = Bag(bag)
    assert b.topics == {
        "/status": "std_msgs/msg/String",
        "/tf": "tf2_msgs/msg/TFMessage",
        "/tf_static": "tf2_msgs/msg/TFMessage",
    }
    assert b.bounds() == (T0, T0 + 3 * S)
    assert [m.stamp for m in b.messages(["/status"], start=T0 + 2 * S)] == [
        T0 + 2 * S,
        T0 + 3 * S,
    ]


# ------------------------------------------------------------ topics and clock


def test_receive_follows_the_virtual_clock(bag: Path) -> None:
    replay = ReplayConnector(bag)
    assert replay.clock_ns == T0  # the first message
    assert replay.now() == Time(nanoseconds=T0, clock_type=replay.now().clock_type)

    async def scenario() -> None:
        # nothing yet on /status: the wait advances to the first message
        assert (await replay.receive("/status", timeout=5.0)).data == "s1"
        assert replay.clock_ns == T0 + S
        replay.advance(1.5)  # 1002.5 s
        assert (await replay.receive("status")).data == "s2"  # latest, any age
        # fresh enough: none at 1002.5 within 0.1 s -> wait for s3 at 1003
        assert (await replay.receive("/status", max_age=0.1)).data == "s3"
        assert replay.clock_ns == T0 + 3 * S
        # the recording has nothing newer: times out and the clock moves on
        replay.advance(0.5)
        with pytest.raises(ROS2TimeoutError):
            await replay.receive("/status", max_age=0.1, timeout=0.5)
        assert replay.clock_ns == T0 + 4 * S
        with pytest.raises(ROS2TimeoutError):
            await replay.receive("/never", timeout=2.0)
        assert replay.clock_ns == T0 + 6 * S
        with pytest.raises(TypeError, match="recorded as std_msgs/msg/String"):
            await replay.receive("/status", msg_type=Twist)

    asyncio.run(scenario())
    replay.seek(T0)  # back to the start
    assert replay.sync.receive("/status", timeout=1.0).data == "s1"


def test_subscriptions_get_messages_as_time_passes(bag: Path) -> None:
    replay = ReplayConnector(bag)
    pushed: list[str] = []
    callback = replay.subscribe("/status", lambda m: pushed.append(m.data))
    awaited: list[str] = []

    async def on_message(msg: String) -> None:
        awaited.append(msg.data)

    async_callback = replay.subscribe("/status", on_message)
    replay.advance(1.0)
    assert pushed == ["s1"] and awaited == ["s1"]
    callback.close()
    replay.advance_to(T0 + 3 * S)
    assert pushed == ["s1"] and awaited == ["s1", "s2", "s3"]
    async_callback.close()

    replay.seek(T0)
    pull = replay.subscribe("/status", maxsize=None)
    raw = replay.subscribe("/status", raw=True, maxsize=None)

    async def drain() -> list[str]:
        return [m.data async for m in pull]  # ends with the recording

    assert asyncio.run(drain()) == ["s1", "s2", "s3"]
    assert replay.clock_ns == T0 + 3 * S
    assert len(raw._state.buffer) == 3 and isinstance(raw._state.buffer[0], bytes)
    with pytest.raises(ReplayEnded):
        asyncio.run(pull.get())


def test_latched_topics_and_transforms(bag: Path) -> None:
    replay = ReplayConnector(bag, start=T0 + 2 * S)
    statics: list[Any] = []
    replay.subscribe("/tf_static", statics.append)
    assert len(statics) == 1  # latched: the recorded one at once
    transform = replay.sync.transform("map", "base_link")
    assert transform.transform.translation.x == pytest.approx(1.5)  # 1.0 + 0.5
    assert set(replay.tf_frames()) == {"odom", "base_link"}
    replay.advance(1.0)
    transform = replay.sync.transform("odom", "base_link")
    assert transform.transform.translation.x == pytest.approx(0.7)
    with pytest.raises(LookupException):
        replay.sync.transform("map", "camera", timeout=0.5)

    early = ReplayConnector(bag)  # at 1000 s: /tf comes at 1001.5 s
    assert early.sync.transform("odom", "base_link", timeout=3.0)
    assert early.clock_ns == T0 + 1500_000_000


# ------------------------------------------------------------ egress


def test_egress_is_recorded_not_sent(bag: Path) -> None:
    replay = ReplayConnector(
        bag,
        services={"/set": SetBool.Response(success=True, message="done")},
        actions={
            "navigate": ScriptedGoal(
                result=NavigateToPose.Result(),
                feedback=[NavigateToPose.Feedback(distance_remaining=1.0)],
                duration=2.0,
            )
        },
    )
    expect_nothing_sent(replay)
    feedback: list[Any] = []
    done: list[Any] = []

    async def scenario() -> None:
        await replay.publish("/cmd_vel", Twist())
        response = await replay.call("/set", SetBool.Request(data=True))
        assert response.message == "done"
        with pytest.raises(ROS2TimeoutError, match="no scripted response"):
            await replay.call("/other", SetBool.Request(), timeout=0.5)
        handle = await replay.send_goal(
            "/navigate",
            NavigateToPose.Goal(),
            on_feedback=feedback.append,
            on_done=done.append,
        )
        assert not handle.done and feedback[0].distance_remaining == 1.0
        assert isinstance(await handle.result(timeout=5.0), NavigateToPose.Result)
        assert handle.done and done == [handle]
        second = await replay.send_goal("/navigate", NavigateToPose.Goal())
        assert await second.cancel() and second.status.name == "CANCELED"

    asyncio.run(scenario())
    assert [(r.kind, r.name) for r in replay.sent] == [
        ("publish", "/cmd_vel"),
        ("service", "/set"),
        ("service", "/other"),
        ("goal", "/navigate"),
        ("goal", "/navigate"),
        ("cancel", "/navigate"),
    ]
    assert replay.sent[1].type == "std_srvs/srv/SetBool"
    assert replay.sent[3].type == "nav2_msgs/action/NavigateToPose"
    expect_published(replay, "cmd_vel", count=1)
    expect_called(replay, "/set", {"data": True})
    expect_goal(replay, "/navigate", count=2)
    with pytest.raises(AssertionError, match="expected no motion"):
        expect_no_motion(replay)
    with pytest.raises(AssertionError, match=r"publish /cmd_vel \(geometry_msgs"):
        expect_published(replay, "/cmd_vel", {"linear.x": 1.0})
    assert ("/set", ["std_srvs/srv/SetBool"]) in replay.get_services_names_and_types()
    actions = replay.get_actions_names_and_types()
    assert actions == [("/navigate", ["nav2_msgs/action/NavigateToPose"])]
    assert ("/cmd_vel", ["geometry_msgs/msg/Twist"]) in (
        replay.get_topics_names_and_types()
    )


def test_egress_hooks_still_run(bag: Path) -> None:
    replay = ReplayConnector(bag)
    seen: list[Egress] = []

    def no_motion(egress: Egress) -> Allow | Deny:
        seen.append(egress)
        return Deny("e-stop") if egress.type.endswith("Twist") else Allow()

    replay.egress.add(no_motion)
    global_egress.add(lambda e: Allow(), names=["/status_out"])
    replay.sync.publish("/status_out", String(data="x"))
    with pytest.raises(EgressDenied, match="e-stop"):
        replay.sync.publish("/cmd_vel", Twist())
    assert [e.name for e in seen] == ["/status_out", "/cmd_vel"]
    assert [r.name for r in replay.sent] == ["/status_out"]
    assert len(replay.denied) == 1
    expect_no_motion(replay)


def test_rai_tools_run_unchanged(bag: Path) -> None:
    replay = ReplayConnector(bag, start=T0 + 2 * S)
    receive = ReceiveROS2MessageTool(connector=replay)
    assert "s2" in receive.invoke({"topic": "/status"})
    publish = PublishROS2MessageTool(connector=replay)
    publish.invoke(
        {
            "topic": "/cmd_vel",
            "message": {"linear": {"x": 0.3}},
            "message_type": "geometry_msgs/msg/Twist",
        }
    )
    expect_published(replay, "/cmd_vel", {"linear.x": 0.3}, within=0.1, count=1)
    topics = GetROS2TopicsNamesAndTypesTool(connector=replay).invoke({})
    assert "/status" in topics and "std_msgs/msg/String" in topics
    transform = GetROS2TransformTool(connector=replay).invoke(
        {"target_frame": "odom", "source_frame": "base_link"}
    )
    assert "0.5" in transform and "WARNING" not in transform  # 0.5 s old
    replay.close()
    with pytest.raises(Exception, match="closed"):
        replay.sync.receive("/status")
