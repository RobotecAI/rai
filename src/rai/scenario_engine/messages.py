import base64
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import requests
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.pydantic_v1 import BaseModel


class MultimodalMessage(BaseModel):
    images_content: Optional[List[Dict[str, Union[str, Dict[str, str]]]]] = None
    audio_content: Optional[List[Dict[str, Union[str, Dict[str, str]]]]] = None

    def __init__(
        self,
        images: Optional[List[str]] = None,
        audio: Optional[Any] = None,
        **kwargs: Any,
    ):
        _images = None
        if images is not None:
            assert isinstance(
                images, list
            ), "Images must be a list of base64 png strings"
            _images = [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{image}",
                    },
                }
                for image in images
            ]

        if audio is not None:
            raise NotImplementedError("Audio content is not yet supported")

        super().__init__(images_content=_images, audio_content=None, **kwargs)


class HumanMultimodalMessage(MultimodalMessage, HumanMessage):
    pass


class SystemMultimodalMessage(MultimodalMessage, SystemMessage):
    pass


class AiMultimodalMessage(MultimodalMessage, AIMessage):
    pass


class ToolMultimodalMessage(MultimodalMessage, ToolMessage):
    def to_openai(self) -> Tuple[ToolMessage, HumanMultimodalMessage] | ToolMessage:
        pass

    def to_bedrock(self):
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
