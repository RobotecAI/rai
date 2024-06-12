from typing import Any, List, Sequence

from langchain.tools import BaseTool
from langchain_core.messages import AIMessage, AnyMessage, ToolCall, ToolMessage
from rich import print

from rai.langchain_extension.tooling import ToolMessageWithOptionalImages


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
            tool_message = ToolMessageWithOptionalImages(
                content=tool_output.get("content", "No response from the tool."),  # type: ignore
                images=tool_output.get("images", None),  # type: ignore
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
