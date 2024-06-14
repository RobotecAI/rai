import base64
from typing import Any, Callable, Dict, List, Literal, Optional, Union

import numpy as np
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.messages.base import BaseMessage


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
            tool_message = ToolMultimodalMessage(
                tool_call_id=self.tool_call_id, content=self.content
            )
            return [tool_message, human_message]
        else:
            # TODO(maciejmajek): find out if content can be a list
            return ToolMessage(tool_call_id=self.tool_call_id, content=self.content)

    def _postprocess_bedrock(self):
        return ToolMessage(tool_call_id=self.tool_call_id, content=self.content)


class AiMultimodalMessage(AIMessage, MultimodalMessage):
    pass


class FutureAiMessage:
    """
    Class to represent a response from the AI that is not yet available.
    """

    def __init__(self, max_tokens: int = 4096):
        self.max_tokens = max_tokens


class AgentLoop:
    """
    Class to represent a loop of agent actions.
    """

    def __init__(self, stop_action: str, stop_iters: int = 10):
        self.stop_action = stop_action
        self.stop_iters = stop_iters


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
