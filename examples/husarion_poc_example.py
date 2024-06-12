from typing import Any, List, Sequence, cast

from langchain.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_openai import ChatOpenAI
from rich import print

from rai.tools.hmi_tools import PlayVoiceMessageTool, WaitForSecondsTool
from rai.tools.ros_cli_tools_simple import (
    Ros2InterfaceTool,
    Ros2ServiceTool,
    Ros2TopicTool,
    SetGoalPoseTool,
)
from rai.tools.ros_tools import (
    GetCameraImageTool,
    GetCurrentPositionTool,
    GetOccupancyGridTool,
    SetWaypointTool,
)
from rai.tools.utils import run_requested_tools


def main():
    tools = [
        GetOccupancyGridTool(),
        GetCameraImageTool(),
        PlayVoiceMessageTool(),
        WaitForSecondsTool(),
        Ros2TopicTool(),
        Ros2ServiceTool(),
        Ros2InterfaceTool(),
        SetGoalPoseTool(),
        SetWaypointTool(),
        GetCurrentPositionTool(),
    ]

    messages: List[AnyMessage] = [
        SystemMessage(
            content="You are an autonomous agent. Your main goal is to fulfill the user's requests. "
            "Do not make assumptions about the environment you are currently in. "
            "Use the tooling provided to gather information about the environment."
            "You are always required to send a voice message to the user about your decisions. This is crucial."
            "The voice message should contain a very short information about what is going on and what is the next step. "
        ),
        HumanMessage(
            content="The robot is moving. Use vision to understand the surroundings, and add waypoints based on observations. camera is accesible at topic /camera/camera/color/image_raw ."
        ),
    ]

    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)

    while True:
        print("---- llm call ----")
        ai_msg = cast(AIMessage, llm_with_tools.invoke(messages))
        messages.append(ai_msg)
        messages = run_requested_tools(ai_msg, tools, messages)


if __name__ == "__main__":
    main()
