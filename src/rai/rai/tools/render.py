from copy import deepcopy

from langchain.tools import BaseTool
from langchain_core.tools import (
    create_schema_from_function,
    render_text_description_and_args,
)


def filter_out_injected_tool_args(tool: BaseTool) -> BaseTool:
    """
    Create a copy of tool with args filtered out.

    The main purpose of this function is to make `langchain_core.tools.render_text_description_with_args`
    usable for `@tool`s with `IntectedToolArg`s.

    It doens't work without this modification, because implementation of StructuredTool
    includes arguments annotated as `InjectedToolArg` in tool's `args_schema`,
    which can cause `pydantic.errors.PydanticInvalidForJsonSchema` for arguments that
    are not parsalbe by pydantic (like `rclpy.node.Node`)
    """
    new_tool = deepcopy(tool)
    new_tool.args_schema = create_schema_from_function(
        tool.name, tool._run, include_injected=False
    )
    return new_tool


def render_text_description_with_args_from_tools(tools: list[BaseTool]) -> str:
    tools = [filter_out_injected_tool_args(t) for t in tools]
    return render_text_description_and_args(tools)
