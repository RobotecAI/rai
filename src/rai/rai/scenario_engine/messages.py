import base64
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.base import BaseMessage
from langchain_core.tools import BaseTool


class MultimodalMessage(BaseMessage):
    images: Optional[List[str]] = None
    audios: Optional[Any] = None

    def __init__(
        self,
        **kwargs: Any,
    ):
        super().__init__(**kwargs)  # type: ignore

        if self.audios is not None:
            raise ValueError("Audio is not yet supported")

        _content: List[Union[str, Dict[str, Union[Dict[str, str], str]]]] = []

        if isinstance(self.content, str):
            _content.append({"type": "text", "text": self.content})
        else:
            raise ValueError("Content must be a string")  # for now, to guarantee compat

        if isinstance(self.images, list):
            _image_content = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}",
                    },
                }
                for image in self.images
            ]
            _content.extend(_image_content)
        self.content = _content


class HumanMultimodalMessage(HumanMessage, MultimodalMessage):
    pass


class SystemMultimodalMessage(SystemMessage, MultimodalMessage):
    pass


class ToolMultimodalMessage(ToolMessage, MultimodalMessage):
    def postprocess(self, format: Literal["openai", "bedrock"]):
        if format == "openai":
            return self._postprocess_openai()
        elif format == "bedrock":
            return self._postprocess_bedrock()
        else:
            raise ValueError(f"Unsupported format: {format}")

    def _postprocess_openai(self):
        """OpenAI does not allow images in the tool message.
        Functions dumps the message into human multimodal message and tool message.
        """
        if isinstance(self.images, list):
            human_message = HumanMultimodalMessage(
                content=f"Image returned by a tool call {self.tool_call_id}",
                images=self.images,
            )
            # at this point self.content is a list of dicts
            # we need to extract the text from each dict
            tool_message = ToolMultimodalMessage(
                tool_call_id=self.tool_call_id,
                content=" ".join([part.get("text", "") for part in self.content]),
            )
            return [tool_message, human_message]
        else:
            # TODO(maciejmajek): find out if content can be a list
            return ToolMessage(tool_call_id=self.tool_call_id, content=self.content)

    def _postprocess_bedrock(self):
        return self._postprocess_openai()
        # https://github.com/langchain-ai/langchain-aws/issues/75
        # at this moment im not sure if bedrock supports images in the tool message
        content = self.content
        # bedrock expects image and not image_url
        content[1]["type"] = "image"
        content[1]["image"] = content[1].pop("image_url")
        content[1]["image"]["source"] = content[1]["image"].pop("url")

        return ToolMessage(tool_call_id=self.tool_call_id, content=content)


class AiMultimodalMessage(AIMessage, MultimodalMessage):
    pass


class FutureAiMessage:
    """
    Class to represent a response from the AI that is not yet available.
    """

    def __init__(self, tools: List[BaseTool], max_tokens: int = 4096):
        self.tools = tools
        self.max_tokens = max_tokens


class AgentLoop:
    """
    Class to represent a loop of agent actions.
    """

    def __init__(self, tools: List[BaseTool], stop_tool: str, stop_iters: int = 10):
        self.stop_tool = stop_tool
        self.stop_iters = stop_iters
        if not any([tool.__class__.__name__ == stop_tool for tool in tools]):
            raise ValueError("Stop tool not in tools")
        self.tools: List[BaseTool] = tools


def preprocess_image(
    image: Union[str, bytes, np.ndarray[Any, np.dtype[np.uint8]]],
    encoding_function: Callable[[Any], str] = lambda x: base64.b64encode(x).decode(
        "utf-8"
    ),
) -> str:
    if isinstance(image, str) and image.startswith(("http://", "https://")):
        response = requests.get(image)
        response.raise_for_status()
        image_data = response.content
    elif isinstance(image, str):
        with open(image, "rb") as image_file:
            image_data = image_file.read()
    elif isinstance(image, bytes):
        image_data = image
        encoding_function = lambda x: x.decode("utf-8")
    elif isinstance(image, np.ndarray):  # type: ignore
        image_data = image.tobytes()
        encoding_function = lambda x: base64.b64encode(x).decode("utf-8")
    else:
        image_data = image

    return encoding_function(image_data)
