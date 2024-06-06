import subprocess

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_openai import ChatOpenAI

from rai.tools.ros_cli_tools import ros2_cli_tools
from rai.tools.ros_cli_tools_simple import (
    ros2_generic_cli_call,
    ros2_interface,
    ros2_service,
)


class done(BaseModel):
    """
    Use when the task is done.
    """

    def run(self):
        pass


def run_requested_tools(ai_msg, tools, messages):
    selected_tools = []
    for tool_call in ai_msg.tool_calls:
        selected_tool = {k.__name__: k for k in tools}[tool_call["name"].lower()]
        selected_tools.append(selected_tool)
        tool_class = {k.__name__: k for k in tools}[tool_call["name"].lower()]

        # Initialize the tool instance with the provided arguments
        tool_instance = tool_class(**tool_call["args"])

        print(
            f'Running tool {tool_instance.__class__.__name__} command: {str(tool_call["args"])[:50]}... -> Status: ',
            end="",
        )

        tool_output = tool_instance.run()

        if tool_output is not None:
            print(f'{"Error" if tool_output.returncode else "Success"}')
        if isinstance(tool_output, bytes):
            tool_output = tool_output.decode("utf-8")
        messages.append(
            ToolMessage(content=str(tool_output), tool_call_id=tool_call["id"])
        )
    return messages, selected_tools


def main():
    llm = ChatOpenAI(model="gpt-4o")
    tools = [ros2_interface, ros2_service, done]
    llm_with_tools = llm.bind_tools(tools)
    query = "Spawn 4 boxes"
    messages = [HumanMessage(query)]
    while True:
        print("---- llm call ----")
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        messages, selected_tools = run_requested_tools(ai_msg, tools, messages)
        if done in selected_tools:
            print(messages)
            break


if __name__ == "__main__":
    main()
