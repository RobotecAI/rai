from typing import Any, Dict, List, Literal, Optional, Sequence

from langchain.tools import BaseTool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)

from rai.scenario_engine.messages import ToolMultimodalMessage


def images_to_vendor_format(images: List[str], vendor: str) -> List[Dict[str, Any]]:
    if vendor == "openai":
        return [
            {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{image}",
                },
            }
            for image in images
        ]
    else:
        raise ValueError(f"Vendor {vendor} not supported")


def run_tool_call(
    tool_call: ToolCall, tools: Sequence[BaseTool]
) -> Dict[str, Any] | Any:
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
    ai_msg: AIMessage,
    tools: Sequence[BaseTool],
    messages: List[BaseMessage],
    llm_type: Literal["openai", "bedrock"],
):
    internal_messages: List[BaseMessage] = []
    for tool_call in ai_msg.tool_calls:
        tool_output = run_tool_call(tool_call, tools)
        assert isinstance(tool_call["id"], str), "Tool output must have an id."
        if isinstance(tool_output, dict):
            tool_message = ToolMultimodalMessage(
                content=tool_output.get("content", "No response from the tool."),
                images=tool_output.get("images"),
                tool_call_id=tool_call["id"],
            )
            tool_message = tool_message.postprocess(format=llm_type)
        else:
            tool_message = [
                ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
            ]
        internal_messages.extend(tool_message)

    # because we can't answer an aiMessage with an alternating sequence of tool and human messages
    # we sort the messages by type so that the tool messages are sent first
    # for more information see implementation of ToolMultimodalMessage.postprocess

    internal_messages.sort(key=lambda x: x.__class__.__name__, reverse=True)
    messages.extend(internal_messages)
    return messages
