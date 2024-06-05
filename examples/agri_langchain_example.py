import logging
import sys

from agri_example import SYSTEM_PROMPT, TASK_PROMPT, TRACTOR_INTRODUCTION
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage as _HumanMessage
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from rai.message import Message
from rai.tools.ros_mock_tools import (
    continue_action,
    finish,
    replan_without_current_path,
    stop,
    use_honk,
    use_lights,
)

logging.getLogger("httpx").setLevel(logging.WARNING)
sys.path.append(".")


class HumanMessage(_HumanMessage):  # handle images
    def __init__(self, message, images=None):
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
        super().__init__(content=content)


def run_requested_tools(ai_msg, tools, messages):
    selected_tools = []
    for tool_call in ai_msg.tool_calls:
        selected_tool = {k.__name__: k for k in tools}[tool_call["name"].lower()]
        selected_tools.append(selected_tool)
        tool_output = selected_tool.run(tool_call["args"])
        logging.info(
            f'Running tool {selected_tool.__name__} with args {tool_call["args"]} -> Status: {tool_output}',
        )
        messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    return messages, selected_tools


def main():
    messages = [
        SystemMessage(
            SYSTEM_PROMPT + "\n Remember to finish the conversation if you're done."
        ),
        HumanMessage(
            TRACTOR_INTRODUCTION,
            images=[Message.preprocess_image("examples/imgs/tractor.png")],
        ),
        HumanMessage(
            TASK_PROMPT,
            images=[Message.preprocess_image("examples/imgs/cat_before.png")],
        ),
    ]
    llm = ChatOpenAI(model="gpt-4o")
    tools = [use_lights, use_honk, replan_without_current_path, continue_action, stop]
    llm_with_tools = llm.bind_tools(tools)

    while True:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        messages, selected_tools = run_requested_tools(ai_msg, tools, messages)

        if use_honk in selected_tools:  # hack, scare the cat away
            logging.info(
                "Scaring the cat away (injecting new cat free image into message history)"
            )
            messages.append(
                HumanMessage(
                    TASK_PROMPT,
                    images=[Message.preprocess_image("examples/imgs/cat_after.png")],
                )
            )

        if {finish, continue_action}.intersection(set(selected_tools)):
            logging.info("Finishing conversation")
            break


if __name__ == "__main__":
    main()
