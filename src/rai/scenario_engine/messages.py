import base64
from typing import Any, Callable, Type, Union

import numpy as np
import requests
from langchain_core.tools import BaseTool


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
