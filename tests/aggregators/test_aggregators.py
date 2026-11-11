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

"""Aggregators v2: bounded timestamped buffers, on-demand cached summaries,
and StateBasedAgent owning their subscriptions."""

import asyncio
import threading
import time
import uuid
from typing import Any

import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import Field
from rai.agents import StateBasedAgent
from rai.aggregators import Aggregator
from rai.aggregators.ros2 import ImageSummary, LastImage, RosoutDigest
from rai.communication.ros2 import ROS2Connector
from rcl_interfaces.msg import Log
from rclpy.executors import SingleThreadedExecutor
from rclpy.node import Node
from std_msgs.msg import Int32

from tests.communication.ros2.helpers import ros_setup

_ = ros_setup
pytestmark = pytest.mark.timeout(60)


class Counting(Aggregator[int]):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(None, "/n", **kwargs)
        self.calls: list[list[int]] = []

    async def summarize(self, msgs):
        self.calls.append(list(msgs))
        return f"sum={sum(msgs)}"


class CountingModel(BaseChatModel):
    """Answers 'summary N'; records every call's messages."""

    calls: list[Any] = Field(default_factory=list)

    @property
    def _llm_type(self) -> str:
        return "counting"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls.append(messages)
        reply = AIMessage(f"summary {len(self.calls)}")
        return ChatResult(generations=[ChatGeneration(message=reply)])


def images_sent(messages: Any) -> int:
    return sum(b.get("type") == "image" for b in messages[-1].content)


def test_window_bounds_buffer_and_max_age_drops_old():
    agg = Counting(window=3, max_age_s=0.2)
    for i in range(10):
        agg.push(i)
    assert [m for _, m in agg.messages()] == [7, 8, 9]
    time.sleep(0.25)
    agg.push(10)
    assert [m for _, m in agg.messages()] == [10]
    with pytest.raises(ValueError, match="window"):
        Counting(window=0)


def test_summary_is_cached_and_recomputed_only_on_new_data():
    agg = Counting(max_age_s=None, cache_s=0.1)
    assert asyncio.run(agg()) == "no messages on /n"
    agg.push(1)
    assert asyncio.run(agg()) == "no messages on /n"  # cache hit
    time.sleep(0.12)
    assert asyncio.run(agg()) == "sum=1"  # expired and new data: miss
    time.sleep(0.12)
    assert asyncio.run(agg()) == "sum=1"  # expired, nothing new: no recompute
    agg.push(2)
    time.sleep(0.12)
    assert asyncio.run(agg()) == "sum=3"
    assert agg.calls == [[1], [1, 2]]


def test_stale_data_is_reported_not_repeated():
    agg = Counting(max_age_s=0.1, cache_s=0.0)
    agg.push(5)
    assert asyncio.run(agg()) == "sum=5"
    time.sleep(0.15)
    assert asyncio.run(agg()) == "no messages on /n in the last 0.1 s"


def test_failed_summary_is_retried():
    agg = Counting(cache_s=0.0)
    agg.summarize = None  # type: ignore[assignment]
    agg.push(1)
    with pytest.raises(TypeError):
        asyncio.run(agg())
    del agg.summarize
    assert asyncio.run(agg()) == "sum=1"


def test_image_summary_calls_the_model_only_when_asked(ros2_image):
    model = CountingModel()
    camera = ImageSummary(None, "/cam", model=model, cache_s=0.1)
    for _ in range(50):
        camera.push(ros2_image)
    assert model.calls == []  # buffering never calls the model
    assert len(camera.messages()) == 1  # frames=1 keeps one frame
    assert asyncio.run(camera()) == "summary 1"
    assert asyncio.run(camera()) == "summary 1"  # cached within cache_s
    time.sleep(0.12)
    assert asyncio.run(camera()) == "summary 1"  # no new frame
    camera.push(ros2_image)
    time.sleep(0.12)
    assert asyncio.run(camera()) == "summary 2"
    assert len(model.calls) == 2 and images_sent(model.calls[0]) == 1


def test_image_summary_compares_keyframes(ros2_image):
    model = CountingModel()
    camera = ImageSummary(None, "/cam", model=model, frames=3, window=10)
    for _ in range(25):
        camera.push(ros2_image)
    assert len(camera.messages()) == 10
    asyncio.run(camera())
    assert images_sent(model.calls[0]) == 3
    assert "3 frames" in model.calls[0][-1].content[0]["text"]


def test_last_image_and_rosout_digest(ros2_image):
    last = LastImage(None, "/cam")
    last.push(ros2_image)
    blocks = asyncio.run(last())
    assert [b["type"] for b in blocks] == ["image"]
    logs = RosoutDigest(None, cache_s=0.0)
    for level, text in [(10, "noise"), (30, "hot"), (30, "hot"), (40, "x"), (25, "y")]:
        logs.push(Log(level=level, name="arm", msg=text))
    assert asyncio.run(logs()) == "[arm] [WARN] hot (x2)\n[arm] [ERROR] x\n[arm] [25] y"


# ---- real topics ----


def unique(prefix: str) -> str:
    return f"/agg_{prefix}_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def connector(ros_setup: None):
    with ROS2Connector() as connector:
        yield connector


@pytest.fixture
def publisher_node(ros_setup: None):
    node = Node(f"agg_pub_{uuid.uuid4().hex[:8]}")
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    thread = threading.Thread(target=executor.spin, daemon=True)
    thread.start()
    yield node
    executor.shutdown(timeout_sec=1.0)
    thread.join(2.0)
    node.destroy_node()


def wait_for(condition: Any, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.02)
    assert condition()


def test_buffer_bounds_and_max_age_on_a_real_topic(connector, publisher_node):
    topic = unique("int")
    publisher = publisher_node.create_publisher(Int32, topic, 10)
    agg = Counting(window=3, max_age_s=0.5, cache_s=0.0)
    agg.connector, agg.topic, agg.msg_type = connector, topic, Int32
    agg.start()
    agg.start()  # idempotent
    wait_for(lambda: publisher.get_subscription_count() > 0)
    for i in range(10):
        publisher.publish(Int32(data=i))
    wait_for(lambda: [m.data for _, m in agg.messages()] == [7, 8, 9])
    time.sleep(0.6)
    assert agg.messages() == []
    subscription = agg._subscription
    agg.stop()
    agg.stop()
    assert subscription.closed and agg._subscription is None
    publisher.publish(Int32(data=11))
    time.sleep(0.2)
    assert agg.messages() == []  # closed: nothing is delivered any more


def test_state_based_agent_owns_aggregator_subscriptions(connector, publisher_node):
    topic = unique("rosout")
    publisher = publisher_node.create_publisher(Log, topic, 10)
    logs = RosoutDigest(connector, topic)
    llm = CountingModel()
    agent = StateBasedAgent(llm, [], state_sources={"logs": logs})
    assert logs._subscription is None
    subscriptions = []

    async def main() -> None:
        async with agent:
            subscriptions.append(logs._subscription)
            await asyncio.to_thread(
                wait_for, lambda: publisher.get_subscription_count() == 1
            )
            publisher.publish(Log(level=40, name="arm", msg="overheated"))
            await asyncio.to_thread(wait_for, lambda: logs.messages() != [])
            await agent.ainvoke("status?")

    asyncio.run(main())
    assert subscriptions[0] is not None and subscriptions[0].closed
    assert logs._subscription is None
    assert "logs: [arm] [ERROR] overheated" in str(llm.calls[0][-1].content)
    assert "overheated" not in str(agent.history)
