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

from rai.langchain_extension.tooling import RaiToolMessage
from rai.tools.hmi_tools import PlayVoiceMessageTool, WaitForSecondsTool
from rai.tools.ros_cli_tools_simple import (
    Ros2InterfaceTool,
    Ros2ServiceTool,
    Ros2TopicTool,
    SetGoalPoseTool,
)
from rai.tools.ros_tools import GetCameraImageTool, GetOccupancyGridTool


def run_tool_call(tool_call: ToolCall, tools: Sequence[BaseTool]) -> Any:
    selected_tool = {k.name: k for k in tools}[tool_call["name"]]
    try:
        args = selected_tool.args_schema(**tool_call["args"])  # type: ignore
    except Exception as e:
        return f"Error in preparing arguments for {selected_tool.name}: {e}"

    print(f"Running tool: {selected_tool.name} with args: {args.dict()}")

    try:
        tool_output = selected_tool.run(args.dict())
    except Exception as e:
        return f"Error running tool {selected_tool.name}: {e}"

    return tool_output


def run_requested_tools(
    ai_msg: AIMessage, tools: Sequence[BaseTool], messages: List[AnyMessage]
):
    for tool_call in ai_msg.tool_calls:
        tool_output = run_tool_call(tool_call, tools)
        if isinstance(tool_output, dict):
            tool_message = RaiToolMessage(
                content=tool_output.get("content", ""),  # type: ignore
                images=tool_output.get("images", []),  # type: ignore
                tool_call_id=tool_call["id"],  # type: ignore
            ).to_openai()
        else:
            tool_message = [
                ToolMessage(
                    content=str(tool_output), tool_call_id=tool_call["id"]  # type: ignore
                )
            ]
        messages.extend(tool_message)

    return messages


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
    ]

    messages: List[AnyMessage] = [
        SystemMessage(
            content="You are an autonomous agent. Your main goal is to fulfill the user's requests. "
            "Do not make assumptions about the environment you are currently in. "
            "Use the tooling provided to gather information about the environment."
            "You are always required to send a voice message to the user about your decisions. This is crucial."
            "The voice message should contain a very short information about what is going on and what is the next step. "
        ),
        HumanMessage(content="Visit every location on the map."),
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
