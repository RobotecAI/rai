# Copyright (C) 2024 Robotec.AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#         http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from abc import ABC, abstractmethod
from typing import Annotated, List, Literal, Optional, Sequence

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage as LangchainBaseMessage
from langchain_core.messages import HumanMessage

from rai.messages import AiMultimodalMessage, HumanMultimodalMessage
from rai.messages.multimodal import MultimodalMessage as RAIMultimodalMessage


class HRIMessage:
    type: Literal["ai", "human"]
    text: str
    images: Optional[Annotated[List[str], "base64 encoded png images"]]
    audios: Optional[Annotated[List[str], "base64 encoded wav audio"]]

    def __repr__(self):
        return f"HRIMessage(type={self.type}, text={self.text}, images={self.images}, audios={self.audios})"

    def __init__(
        self,
        type: Literal["ai", "human"],
        text: str,
        images: Optional[List[str]] = None,
        audios: Optional[List[str]] = None,
    ):
        self.type = type
        self.text = text
        self.images = images
        self.audios = audios

    def to_langchain(self) -> LangchainBaseMessage:
        match self.type:
            case "human":
                if self.images is None and self.audios is None:
                    return HumanMessage(content=self.text)
                return HumanMultimodalMessage(
                    content=self.text, images=self.images, audios=self.audios
                )
            case "ai":
                if self.images is None and self.audios is None:
                    return AIMessage(content=self.text)
                return AiMultimodalMessage(
                    content=self.text, images=self.images, audios=self.audios
                )
            case _:
                raise ValueError(
                    f"Invalid message type: {self.type} for {self.__class__.__name__}"
                )

    @classmethod
    def from_langchain(
        cls,
        message: LangchainBaseMessage | RAIMultimodalMessage,
    ) -> "HRIMessage":
        if isinstance(message, RAIMultimodalMessage):
            text = str(message.content["text"])
            images = message.images
            audios = message.audios
        else:
            text = str(message.content)
            images = None
            audios = None
        if message.type not in ["ai", "human"]:
            raise ValueError(f"Invalid message type: {message.type} for {cls.__name__}")
        return cls(
            type=message.type,  # type: ignore
            text=text,
            images=images,
            audios=audios,
        )


class HRIConnector(ABC):
    """
    Base class for Human-Robot Interaction (HRI) connectors.
    Used for sending and receiving messages between human and robot from various sources.
    """

    configured_targets: Sequence[str]
    configured_sources: Sequence[str]

    def build_message(
        self,
        message: LangchainBaseMessage | RAIMultimodalMessage,
    ) -> HRIMessage:
        return HRIMessage.from_langchain(message)

    @abstractmethod
    def send_message(self, message: LangchainBaseMessage | RAIMultimodalMessage):
        pass

    @abstractmethod
    def receive_message(self) -> LangchainBaseMessage | RAIMultimodalMessage:
        pass
