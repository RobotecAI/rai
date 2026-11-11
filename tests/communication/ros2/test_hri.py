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

"""ROS2HRIConnector: text, images and audio through a real ROS round trip."""

import asyncio
import time
import uuid
from typing import Any

import numpy as np
import pytest
from rai.communication import HRIConnector, HRIMessage
from rai.communication.ros2 import ROS2HRIConnector
from rai.communication.ros2.connectors.hri_connector import from_ros, to_ros
from rai.content import Audio, Image

from .helpers import ros_setup

_ = ros_setup
pytestmark = pytest.mark.timeout(60)
RNG = np.random.default_rng(0)


def topic() -> str:
    return f"/hri_{uuid.uuid4().hex[:8]}"


@pytest.fixture
def hri(ros_setup: None):
    connector = ROS2HRIConnector()
    yield connector
    connector.close()


def wait_until(predicate: Any, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline and not predicate():
        time.sleep(0.01)
    return bool(predicate())


def images() -> list[Image]:
    return [
        Image(RNG.integers(0, 255, (24, 32, 3), dtype=np.uint8), "rgb8"),
        Image(RNG.integers(0, 255, (24, 32, 3), dtype=np.uint8), "bgr8"),
        Image(RNG.integers(0, 255, (24, 32), dtype=np.uint8), "mono8"),
        Image(RNG.integers(0, 65535, (24, 32), dtype=np.uint16), "mono16"),
        Image(RNG.random((24, 32), dtype=np.float32), "32FC1"),
    ]


def audios() -> list[Audio]:
    return [
        Audio(RNG.integers(-32768, 32767, 800, dtype=np.int16), 16000),
        Audio(RNG.integers(-32768, 32767, (800, 2), dtype=np.int16), 44100),
        Audio(RNG.uniform(-1, 1, (800, 2)).astype(np.float32), 22050),
    ]


def assert_same(sent: HRIMessage, got: HRIMessage) -> None:
    assert (got.text, got.communication_id, got.seq_no, got.seq_end) == (
        sent.text,
        sent.communication_id,
        sent.seq_no,
        sent.seq_end,
    )
    assert [i.encoding for i in got.images] == [i.encoding for i in sent.images]
    for a, b in zip(sent.images, got.images):
        assert np.array_equal(a.data, b.data)
    assert len(got.audios) == len(sent.audios)
    for a, b in zip(sent.audios, got.audios):
        assert b.samples.dtype == np.int16  # HRI audio travels as int16
        assert (b.sample_rate, b.channels) == (a.sample_rate, a.channels)
        assert np.array_equal(a.as_int16().samples, b.samples)


def test_conversion_round_trip_without_ros():
    message = HRIMessage("hi", images(), audios(), "turn", 3, True)
    assert_same(message, from_ros(to_ros(message)))


def test_ros2_hri_connector_is_an_hri_connector(hri):
    assert isinstance(hri, HRIConnector)


def warm_up(hri: ROS2HRIConnector, name: str) -> None:
    """Send until the connector's own subscription to name matched."""
    got: list[HRIMessage] = []
    sub = hri.subscribe(name, got.append)
    assert wait_until(lambda: hri.sync.send(name, "warm-up") or got)
    time.sleep(0.1)
    sub.close()


def test_text_images_and_audio_round_trip(hri):
    name = topic()
    warm_up(hri, name)
    got: list[HRIMessage] = []
    hri.subscribe(name, got.append)
    message = HRIMessage("hello", images(), audios(), HRIMessage.new_id(), 2, True)
    hri.sync.send(name, message)
    assert wait_until(lambda: got)
    assert_same(message, got[0])


def test_two_connectors_round_trip(ros_setup):
    name = topic()
    with ROS2HRIConnector("hri_a") as a, ROS2HRIConnector("hri_b") as b:
        got: list[HRIMessage] = []
        b.subscribe(name, got.append)
        message = HRIMessage("over the wire", images()[:1], audios()[:1])
        assert wait_until(lambda: a.sync.send(name, message) or got)
        assert_same(message, got[0])


def test_send_text_and_receive_latest(hri):
    name = topic()
    warm_up(hri, name)
    hri.sync.send(name, "first")
    hri.sync.send(name, "second")
    assert wait_until(lambda: hri.sync.receive(name).text == "second")


def test_async_send_and_iterate(hri):
    name = topic()
    warm_up(hri, name)

    async def main() -> list[str]:
        sub = hri.subscribe(name)
        await hri.send(name, "a")
        await hri.send(name, HRIMessage("b"))
        texts = [(await anext(sub)).text for _ in range(2)]
        sub.close()
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(sub), 1.0)
        return texts

    assert asyncio.run(main()) == ["a", "b"]


def test_burst_with_images_is_complete_and_ordered(hri):
    name = topic()
    got: list[int] = []

    def slow(message: HRIMessage) -> None:
        time.sleep(0.002)
        got.append(message.seq_no)

    warm_up(hri, name)
    hri.subscribe(name, slow)
    image = Image(np.zeros((480, 640, 3), np.uint8))
    for i in range(300):
        hri.sync.send(name, HRIMessage("x", [image], seq_no=i))
    assert wait_until(lambda: len(got) == 300, 20.0), len(got)
    assert got == list(range(300))
