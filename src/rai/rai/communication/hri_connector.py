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

from typing import Annotated, List, Literal, Optional, Sequence

from langchain_core.messages import AIMessage
from langchain_core.messages import BaseMessage as LangchainBaseMessage
from langchain_core.messages import HumanMessage
from pydantic import BaseModel

from rai.messages import AiMultimodalMessage, HumanMultimodalMessage
from rai.messages.multimodal import MultimodalMessage as RAIMultimodalMessage

from .base_connector import BaseConnector, BaseMessage


class HRIPayload(BaseModel):
    text: str
    images: Optional[Annotated[List[str], "base64 encoded png images"]]
    audios: Optional[Annotated[List[str], "base64 encoded wav audio"]]


class HRIMessage(BaseMessage):
    def __init__(
        self,
        payload: HRIPayload,
        message_author: Literal["ai", "human"],
    ):
        self.type = message_author
        self.text = payload.text
        self.images = payload.images
        self.audios = payload.audios

    # type: Literal["ai", "human"]

    def __repr__(self):
        return f"HRIMessage(type={self.type}, text={self.text}, images={self.images}, audios={self.audios})"

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
            text = message.text
            images = message.images
            audios = message.audios
        else:
            text = str(message.content)
            images = None
            audios = None
        if message.type not in ["ai", "human"]:
            raise ValueError(f"Invalid message type: {message.type} for {cls.__name__}")
        return cls(
            payload=HRIPayload(
                text=text,
                images=images,
                audios=audios,
            ),
            type=message.type,  # type: ignore
        )


class HRIConnector(BaseConnector[HRIMessage]):
    """
    Base class for Human-Robot Interaction (HRI) connectors.
    Used for sending and receiving messages between human and robot from various sources.
    """

    configured_targets: Sequence[str]
    configured_sources: Sequence[str]

    def _build_message(
        self,
        message: LangchainBaseMessage | RAIMultimodalMessage,
    ) -> HRIMessage:
        return HRIMessage.from_langchain(message)

    def send_all_targets(self, message: LangchainBaseMessage | RAIMultimodalMessage):
        for target in self.configured_targets:
            to_send = self._build_message(message)
            self.send_message(to_send, target)

    def receive_all_sources(self, timeout_sec: float = 1.0) -> dict[str, HRIMessage]:
        ret = {}
        for source in self.configured_sources:
            received = self.receive_message(source, timeout_sec)
            ret[source] = received
        return ret
