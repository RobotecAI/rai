import logging
import sys

from agri_example import SYSTEM_PROMPT, TASK_PROMPT, TRACTOR_INTRODUCTION
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage as _HumanMessage
from langchain_core.messages import SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI

from rai.message import Message
from rai.tools.ros_mock_tools import (
    ContinueActionTool,
    ReplanWithoutCurrentPathTool,
    StopTool,
    UseHonkTool,
    UseLightsTool,
)
from rai.tools.utils import run_requested_tools

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
    tools = [
        UseLightsTool(),
        UseHonkTool(),
        ReplanWithoutCurrentPathTool(),
        ContinueActionTool(),
        StopTool(),
    ]
    llm_with_tools = llm.bind_tools(tools)

    while True:
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        messages = run_requested_tools(ai_msg, tools, messages)

        if "UseHonkTool" in str(messages):  # hack, scare the cat away
            logging.info(
                "Scaring the cat away (injecting new cat free image into message history)"
            )
            messages.append(
                HumanMessage(
                    TASK_PROMPT,
                    images=[Message.preprocess_image("examples/imgs/cat_after.png")],
                )
            )


if __name__ == "__main__":
    main()
