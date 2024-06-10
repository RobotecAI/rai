from typing import List

from langchain.tools import BaseTool
from langchain_community.tools import WikipediaQueryRun
from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.messages import HumanMessage as _HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage as _ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI

from rai.langchain_extension.tooling import RaiToolMessage
from rai.tools.hmi_tools import PlayVoiceMessageTool, WaitForSecondsTool
from rai.tools.ros_cli_tools_simple import (
    Ros2InterfaceTool,
    Ros2ServiceTool,
    Ros2TopicTool,
)
from rai.tools.ros_tools import GetCameraImageTool, GetCurrentMapTool


class HumanMessage(_HumanMessage):  # handle images
    def __init__(self, message, images=None, **kwargs):
        images = images or []
        content = [
            {"type": "text", "text": message},
        ]
        images_prepared = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                },
            }
            for image in images
        ]
        content.extend(images_prepared)
        super().__init__(content=content, **kwargs)


class ToolMessage(_ToolMessage):
    def __init__(self, message, images=None, **kwargs):
        images = images or []
        content = [
            {"type": "text", "text": message},
        ]
        images_prepared = [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                },
            }
            for image in images
        ]
        content.extend(images_prepared)
        super().__init__(content=content, **kwargs)


def run_requested_tools(
    ai_msg: AIMessage, tools: List[BaseTool], messages: List[AnyMessage]
):
    selected_tools: List[BaseTool] = []
    for tool_call in ai_msg.tool_calls:
        selected_tool = {k.__name__: k for k in tools}[tool_call["name"].lower()]
        selected_tools.append(selected_tool)

        print(f'Running tool {selected_tool.__name__} with args {tool_call["args"]}')
        tool_instance = selected_tool(**tool_call["args"])
        tool_output = tool_instance.run()
        if isinstance(tool_output, dict):
            tool_message = RaiToolMessage(
                content=tool_output.get("content", ""),
                images=tool_output.get("images", []),
                tool_call_id=tool_call["id"],
            )
        else:
            tool_message = [
                ToolMessage(message=str(tool_output), tool_call_id=tool_call["id"])
            ]
        messages.extend(tool_message)

    return messages, selected_tools


def main():
    tools = [
        GetCurrentMapTool(),
        GetCameraImageTool(),
        PlayVoiceMessageTool(),
        WaitForSecondsTool(),
        Ros2TopicTool(),
        Ros2ServiceTool(),
        Ros2InterfaceTool(),
    ]
    messages = [
        SystemMessage(
            content="You are an autonomous agent. Your main goal is to fulfill the user's requests. "
            "Do not make assumptions about the environment you are currently in. "
            "Use the tooling provided to gather information about the environment."
            "You are always required to send a voice message to the user about your decisions. This is crucial."
            "The voice message should contain a very short information about what is going on and what is the next step. "
        ),
        HumanMessage(
            message="Visit every location on the map. Be aware of the obstacles and map limits. Make sure to wait for 5 seconds after each move."
        ),
    ]
    llm = ChatOpenAI(model="gpt-4o")
    llm_with_tools = llm.bind_tools(tools)
    while True:
        print("---- llm call ----")
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        messages, selected_tools = run_requested_tools(ai_msg, tools, messages)


if __name__ == "__main__":
    main()
