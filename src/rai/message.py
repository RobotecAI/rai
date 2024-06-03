import base64
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import requests
import xxhash

from rai.requirements import MessageLengthRequirement, Requirement, RequirementSeverity


class Message:
    def __init__(
        self,
        role: str,
        content: str,
        images: Optional[List[Union[Callable[[], str], str]]] = None,
    ):
        if images is not None:
            assert isinstance(images, list), "Images must be a list"
            assert all(
                isinstance(image, (str, Callable)) for image in images
            ), "Images must be byte64 encoded utf8 strings or callables that return them"

        self.content = content
        self.role = role

        # raw images are strings and callables to allow for lazy loading
        self._raw_images = images
        self._images = []

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role,
            "content": self.content,
            "images": self.images,
        }

    @property
    def images(self) -> List[str]:
        if self._raw_images is None:
            return []

        # lazy initialization
        self._images: List[str] = [
            image() if callable(image) else image for image in self._raw_images
        ]
        return self._images

    def __str__(self):
        return f"{self.role}: {self.content}"

    def __repr__(self) -> str:
        return self.__str__()

    @property
    def __hash__(self) -> int:  # type: ignore
        return xxhash.xxh32_intdigest(self.content + self.role + str(self.images))

    @staticmethod
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


class ConstantMessage(Message):  # TODO(boczekbartek): 1:1 copy of Message - remove?
    def __init__(
        self,
        role: str,
        content: str,
        images: Optional[List[Union[Callable[[], str], str]]] = None,
    ):
        super().__init__(role=role, content=content, images=images)


class UserMessage(ConstantMessage):
    def __init__(
        self, content: str, images: Optional[List[Union[Callable[[], str], str]]] = None
    ):
        super().__init__(role="user", content=content, images=images)


class SystemMessage(ConstantMessage):
    def __init__(
        self, content: str, images: Optional[List[Union[Callable[[], str], str]]] = None
    ):
        super().__init__(role="system", content=content, images=images)


class AssistantMessage:
    def __init__(
        self,
        requirements: Optional[List[Requirement]] = None,
        max_retries: int = 3,
        max_tokens: int = 4096,
    ):
        self.requirements = requirements or []
        self.max_retries = max_retries
        self.max_tokens = max_tokens

    def check_requirements(self, message: Message) -> List[Dict[str, Any]]:
        return [
            dict(
                name=requirement.__class__.__name__,
                status=requirement(message),
                severity=requirement.severity,
            )
            for requirement in self.requirements
        ]


class LengthLimitedAssistantMessage(AssistantMessage):
    def __init__(
        self,
        max_length: int,
        severity: RequirementSeverity = RequirementSeverity.OPTIONAL,
    ):
        super().__init__(
            requirements=[
                MessageLengthRequirement(max_length=max_length, severity=severity)
            ]
        )


class ConditionalMessage:
    def __init__(
        self,
        if_true: Message,
        if_false: Message,
        condition: Callable[[List[Message]], bool],
    ):
        self.if_true = if_true
        self.if_false = if_false
        self.condition = condition

    def __call__(self, messages: List[Message]) -> Message:
        response = self.condition(messages)
        if response:
            return self.if_true
        return self.if_false
