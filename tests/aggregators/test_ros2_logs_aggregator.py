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

from dataclasses import dataclass
from typing import List
from unittest.mock import patch

from langchain_core.language_models.fake_chat_models import FakeChatModel
from langchain_core.messages import HumanMessage
from pydantic import BaseModel, Field
from rai.aggregators.ros2.aggregators import (
    ROS2GetLastImageAggregator,
    ROS2ImgVLMDescriptionAggregator,
    ROS2ImgVLMDiffAggregator,
    ROS2LogsAggregator,
)
from rai.communication.ros2.api import convert_ros_img_to_base64
from rai.messages import HumanMultimodalMessage
from rcl_interfaces.msg import Log
from sensor_msgs.msg import Image


@dataclass
class DummyLog:
    level: int
    name: str
    function: str
    msg: str


def test_ros2_logs_aggregator_deduplicates_and_clears_buffer():
    aggregator = ROS2LogsAggregator()

    repeated_log = DummyLog(
        level=30, name="demo_node", function="do_work", msg="System warming up"
    )
    unique_log = DummyLog(
        level=40, name="demo_node", function="do_work", msg="System failure detected"
    )

    aggregator(repeated_log)
    aggregator(DummyLog(**repeated_log.__dict__))
    aggregator(unique_log)

    summary = aggregator.get()

    assert isinstance(summary, HumanMessage)
    assert (
        summary.content
        == "Logs summary: ['[demo_node] [WARNING] [do_work] System warming up', 'Log above repeated 1 times']"
    )
    assert aggregator.get_buffer() == []


def test_ros2_logs_aggregator_str():
    aggregator = ROS2LogsAggregator()
    assert str(aggregator) == "ROS2LogsAggregator(len=0)"
    aggregator(
        Log(level=30, name="demo_node", function="do_work", msg="System warming up")
    )
    assert str(aggregator) == "ROS2LogsAggregator(len=1)"


def test_ros2_logs_aggregator_overflow():
    aggregator = ROS2LogsAggregator(max_size=2)
    for i in range(10):
        aggregator(
            Log(
                level=30,
                name="demo_node",
                function="do_work",
                msg=f"System warming up: {i}",
            )
        )
    assert len(aggregator.get_buffer()) == 2
    assert aggregator.get_buffer()[0].msg == "System warming up: 8"
    assert aggregator.get_buffer()[1].msg == "System warming up: 9"


def test_ros2_logs_aggregator_constructor():
    aggregator = ROS2LogsAggregator(max_size=100)
    assert aggregator.max_size == 100


def test_ros2_last_image_aggregator(ros2_image: Image):
    aggregator = ROS2GetLastImageAggregator()
    aggregator(ros2_image)
    assert aggregator.get_buffer()[0] == ros2_image
    b64_image = convert_ros_img_to_base64(ros2_image)
    assert aggregator.get() == HumanMultimodalMessage(content="", images=[b64_image])
    assert aggregator.get_buffer() == []
    assert aggregator.get() is None


def test_ros2_last_image_aggregator_constructor():
    aggregator = ROS2GetLastImageAggregator(max_size=100)
    assert aggregator.max_size == 100


def test_ros2_img_vlm_description_aggregator(ros2_image: Image):
    class ROS2ImgDescription(BaseModel):
        key_elements: List[str] = Field(..., description="Key elements of the image")

    class DummyModel(FakeChatModel):
        def invoke(self, *args, **kwargs):
            return ROS2ImgDescription(key_elements=["test 12345"])

        def with_structured_output(self, *args, **kwargs):
            return self

    with patch(
        "rai.aggregators.ros2.aggregators.get_llm_model", return_value=DummyModel()
    ):
        aggregator = ROS2ImgVLMDescriptionAggregator()
        aggregator(ros2_image)
        assert aggregator.get_buffer()[0] == ros2_image
        assert aggregator.get() == HumanMessage(
            content="These are the key elements of the last camera image frame: key_elements=['test 12345']"
        )
        assert aggregator.get_buffer() == []
        assert aggregator.get() is None


def test_ros2_img_vlm_description_aggregator_constructor():
    aggregator = ROS2ImgVLMDescriptionAggregator(max_size=100)
    assert aggregator.max_size == 100
    assert aggregator.llm is not None
    aggregator = ROS2ImgVLMDescriptionAggregator(llm=FakeChatModel())
    assert isinstance(aggregator.llm, FakeChatModel)


def test_ros2_img_vlm_diff_aggregator_constructor():
    aggregator = ROS2ImgVLMDiffAggregator(max_size=100)
    assert aggregator.max_size == 100
    assert aggregator.llm is not None
    aggregator = ROS2ImgVLMDiffAggregator(llm=FakeChatModel())
    assert isinstance(aggregator.llm, FakeChatModel)


def test_ros2_img_vlm_diff_aggregator(ros2_image: Image):
    class ROS2ImgDiffOutput(BaseModel):
        are_different: bool = Field(..., description="Whether the images are different")
        differences: List[str] = Field(..., description="Description of the difference")

    class DummyModel(FakeChatModel):
        def invoke(self, *args, **kwargs):
            return ROS2ImgDiffOutput(
                are_different=True,
                differences=["object moved out of frame"],
            )

        def with_structured_output(self, *args, **kwargs):
            return self

    with patch(
        "rai.aggregators.ros2.aggregators.get_llm_model", return_value=DummyModel()
    ):
        aggregator = ROS2ImgVLMDiffAggregator()
        for _ in range(3):
            aggregator(ros2_image)
        assert aggregator.get_buffer()[0] == ros2_image
        result = aggregator.get()
        assert result == HumanMessage(
            content=(
                "Result of the analysis of the 3 keyframes selected from 3 last images:\n"
                "are_different=True differences=['object moved out of frame']"
            )
        )
        assert aggregator.get_buffer() == []
        assert aggregator.get() is None
