import base64
from typing import List, Literal, Type, cast

import cv2
import numpy as np
import pytest
from langchain.tools import BaseTool
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage
from langchain_core.pydantic_v1 import BaseModel, Field
from pytest import FixtureRequest

from rai.scenario_engine.messages import HumanMultimodalMessage
from rai.scenario_engine.tool_runner import run_requested_tools


class GetImageToolInput(BaseModel):
    name: str = Field(..., title="Name of the image")


class GetImageTool(BaseTool):

    name: str = "GetImageTool"
    description: str = "Get an image from the user"

    args_schema: Type[GetImageToolInput] = GetImageToolInput  # type: ignore

    def _run(self, name: str):
        image = np.zeros((100, 100, 3), dtype=np.uint8)
        # convert to png
        image = image.astype(np.uint8)
        _, image = cv2.imencode(".png", image)
        base64_image = base64.b64encode(image).decode("utf-8")
        return {"content": f"Here is the image {name}", "images": [base64_image]}


@pytest.mark.parametrize(
    ("llm", "llm_type"),
    [("chat_openai_multimodal", "openai"), ("chat_bedrock_multimodal", "bedrock")],
)
def test_multimodal_openai(
    llm: BaseChatModel, llm_type: Literal["openai", "bedrock"], request: FixtureRequest
):
    llm = request.getfixturevalue(llm)  # type: ignore
    tools = [GetImageTool()]
    llm_with_tools = llm.bind_tools(tools)  # type: ignore

    scenario: List[BaseMessage] = [
        HumanMultimodalMessage(
            content="Can you please describe the contents of test.png image? Remember to use the available tools."
        ),
    ]

    ai_msg = cast(AIMessage, llm_with_tools.invoke(scenario))
    scenario.append(ai_msg)
    scenario = run_requested_tools(ai_msg, tools, scenario, llm_type=llm_type)
    ai_msg = llm_with_tools.invoke(scenario)
