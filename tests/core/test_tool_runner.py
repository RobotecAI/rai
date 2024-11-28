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
