import logging
import sys

from agri_example import SYSTEM_PROMPT, TASK_PROMPT, TRACTOR_INTRODUCTION
from langchain.schema import HumanMessage, SystemMessage
from langchain_core.messages import HumanMessage as _HumanMessage
from langchain_core.messages import SystemMessage
from langchain_openai import ChatOpenAI
from rich import print

from rai.langchain_extension.history_saver import HistorySaver
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
    def __init__(self, content, images=None):
        images = images or []
        final_content = [
            {"type": "text", "text": content},
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
        final_content.extend(images_prepared)
        super().__init__(content=final_content)


def main():
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=TRACTOR_INTRODUCTION,
            images=[Message.preprocess_image("examples/imgs/tractor.png")],
        ),
        HumanMessage(
            content=TASK_PROMPT,
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
        print("------ llm call ------")
        ai_msg = llm_with_tools.invoke(messages)
        messages.append(ai_msg)
        messages = run_requested_tools(ai_msg, tools, messages)

        if UseHonkTool().__class__.__name__ in [
            tool_call["name"] for tool_call in ai_msg.tool_calls
        ]:  # hack, scare the cat away
            messages.append(
                HumanMessage(
                    content=TASK_PROMPT,
                    images=[Message.preprocess_image("examples/imgs/cat_after.png")],
                )
            )
        path = HistorySaver(messages).save_to_html()
        print(f"History saved to {path}")
        if ContinueActionTool().__class__.__name__ in [  # awful, find better way
            tool_call["name"] for tool_call in ai_msg.tool_calls
        ]:
            break


if __name__ == "__main__":
    main()
