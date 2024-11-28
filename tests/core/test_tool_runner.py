# Copyright (C) 2024 Robotec.AI
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

import logging

from langchain_core.messages import AIMessage, ToolCall, ToolMessage
from langchain_core.tools import tool

from rai.agents.tool_runner import ToolRunner
from rai.messages import HumanMultimodalMessage, ToolMultimodalMessage
from rai.messages.utils import preprocess_image
from rai.tools.ros.debugging import ros2_topic


@tool(response_format="content_and_artifact")
def get_image():
    """Get an image from the camera"""
    return "Image retrieved", {
        "images": [preprocess_image("docs/imgs/o3deSimulation.png")]
    }


def test_tool_runner():
    runner = ToolRunner(tools=[ros2_topic], logger=logging.getLogger(__name__))

    tool_call = ToolCall(name="ros2_topic", args={"command": "list"}, id="12345")
    state = {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
    output = runner.invoke(state)
    assert isinstance(
        output["messages"][0], AIMessage
    ), "First message is not an AIMessage"
    assert isinstance(
        output["messages"][1], ToolMessage
    ), "Tool output is not a tool message"
    assert (
        len(output["messages"][-1].content) > 0
    ), "Tool output is empty. At least rosout should be visible."


def test_tool_runner_multimodal():
    runner = ToolRunner(
        tools=[ros2_topic, get_image], logger=logging.getLogger(__name__)
    )

    tool_call = ToolCall(name="get_image", args={}, id="12345")
    state = {"messages": [AIMessage(content="", tool_calls=[tool_call])]}
    output = runner.invoke(state)

    assert isinstance(
        output["messages"][0], AIMessage
    ), "First message is not an AIMessage"
    assert isinstance(
        output["messages"][1], ToolMultimodalMessage
    ), "Tool output is not a multimodal message"
    assert isinstance(
        output["messages"][2], HumanMultimodalMessage
    ), "Human output is not a multimodal message"
