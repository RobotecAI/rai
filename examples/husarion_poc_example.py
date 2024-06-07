from typing import List

from langchain_core.messages import AIMessage, AnyMessage
from langchain_core.messages import HumanMessage as _HumanMessage
from langchain_core.messages import SystemMessage
from langchain_core.messages import ToolMessage as _ToolMessage
from langchain_core.pydantic_v1 import BaseModel
from langchain_openai import ChatOpenAI

from rai.tools.hmi_tools import send_voice_message, wait_for_seconds
from rai.tools.ros_cli_tools import set_goal_pose_relative_to_the_map
from rai.tools.ros_cli_tools_simple import ros2_interface, ros2_service, ros2_topic
from rai.tools.ros_tools import (
    get_current_image,
    get_current_map,
    get_current_position_relative_to_the_map,
)


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
    ai_msg: AIMessage, tools: List[BaseModel], messages: List[AnyMessage]
):
    selected_tools: List[BaseModel] = []
    for tool_call in ai_msg.tool_calls:
        selected_tool = {k.__name__: k for k in tools}[tool_call["name"].lower()]
        selected_tools.append(selected_tool)

        print(f'Running tool {selected_tool.__name__} with args {tool_call["args"]}')
        tool_instance = selected_tool(**tool_call["args"])
        tool_output = tool_instance.run()
        if isinstance(tool_output, dict):
            tool_message = ToolMessage(
                message=tool_output.get("content", ""),
                images=tool_output.get("images", []),
                tool_call_id=tool_call["id"],
            )
        else:
            tool_message = ToolMessage(
                message=str(tool_output), tool_call_id=tool_call["id"]
            )
        messages.append(tool_message)

    return messages, selected_tools


def main():
    tools = [
        get_current_map,
        get_current_position_relative_to_the_map,
        get_current_image,
        set_goal_pose_relative_to_the_map,
        send_voice_message,
        ros2_service,
        ros2_interface,
        ros2_topic,
        wait_for_seconds,
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
