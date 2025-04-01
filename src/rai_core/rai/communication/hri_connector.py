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

import base64
import uuid
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, Generic, Literal, Optional, Sequence, TypeVar, get_args

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import BaseMessage as LangchainBaseMessage
from PIL import Image
from PIL.Image import Image as ImageType
from pydub import AudioSegment

from rai.messages import AiMultimodalMessage, HumanMultimodalMessage
from rai.messages.multimodal import MultimodalMessage as RAIMultimodalMessage

from .base_connector import BaseConnector, BaseMessage


class HRIException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


@dataclass
class HRIPayload:
    text: str
    images: list[ImageType] = field(default_factory=list)
    audios: list[AudioSegment] = field(default_factory=list)

    def __post_init__(self):
        if not isinstance(self.text, str):
            raise TypeError(f"Text should be of type str, got {type(self.text)}")
        if not isinstance(self.images, list):
            raise TypeError(f"Images should be of type list, got {type(self.images)}")
        if not isinstance(self.audios, list):
            raise TypeError(f"Audios should be of type list, got {type(self.audios)}")


class HRIMessage(BaseMessage):
    def __init__(
        self,
        payload: HRIPayload,
        metadata: Optional[Dict[str, Any]] = None,
        message_author: Literal["ai", "human"] = "ai",
        communication_id: Optional[str] = None,
        seq_no: int = 0,
        seq_end: bool = False,
        **kwargs,
    ):
        super().__init__(payload, metadata)
        self.message_author = message_author
        self.text = payload.text
        self.images = payload.images
        self.audios = payload.audios
        self.communication_id = communication_id
        self.seq_no = seq_no
        self.seq_end = seq_end

    def __bool__(self) -> bool:
        return bool(self.text or self.images or self.audios)

    def __repr__(self):
        return f"HRIMessage(type={self.message_author}, text={self.text}, images={self.images}, audios={self.audios}, communication_id={self.communication_id}, seq_no={self.seq_no}, seq_end={self.seq_end})"

    def _image_to_base64(self, image: ImageType) -> str:
        buffered = BytesIO()
        image.save(buffered, "PNG")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    def _audio_to_base64(self, audio: AudioSegment) -> str:
        buffered = BytesIO()
        audio.export(buffered, format="wav")
        return base64.b64encode(buffered.getvalue()).decode("utf-8")

    @classmethod
    def _base64_to_image(cls, base64_str: str) -> ImageType:
        img_data = base64.b64decode(base64_str)
        return Image.open(BytesIO(img_data))

    @classmethod
    def _base64_to_audio(cls, base64_str: str) -> AudioSegment:
        audio_data = base64.b64decode(base64_str)
        return AudioSegment.from_file(BytesIO(audio_data), format="wav")

    def to_langchain(self) -> LangchainBaseMessage:
        base64_images = [self._image_to_base64(image) for image in self.images]
        base64_audios = [self._audio_to_base64(audio) for audio in self.audios]
        match self.message_author:
            case "human":
                if self.images == [] and self.audios == []:
                    return HumanMessage(content=self.text)

                return HumanMultimodalMessage(
                    content=self.text, images=base64_images, audios=base64_audios
                )
            case "ai":
                if self.images == [] and self.audios == []:
                    return AIMessage(content=self.text)
                return AiMultimodalMessage(
                    content=self.text, images=base64_images, audios=base64_images
                )
            case _:
                raise ValueError(
                    f"Invalid message type: {self.message_author} for {self.__class__.__name__}"
                )

    @classmethod
    def from_langchain(
        cls,
        message: LangchainBaseMessage | RAIMultimodalMessage,
        communication_id: Optional[str] = None,
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
                images=(
                    [cls._base64_to_image(image) for image in images] if images else []
                ),
                audios=(
                    [cls._base64_to_audio(audio) for audio in audios] if audios else []
                ),
            ),
            message_author=message.type,  # type: ignore
            communication_id=communication_id,
        )

    @classmethod
    def generate_communication_id(cls) -> str:
        """Generate a unique communication ID."""
        return str(uuid.uuid1())


T = TypeVar("T", bound=HRIMessage)


class HRIConnector(Generic[T], BaseConnector[T]):
    """
    Base class for Human-Robot Interaction (HRI) connectors.
    Used for sending and receiving messages between human and robot from various sources.
    """

    configured_targets: Sequence[str]
    configured_sources: Sequence[str]

    def __init__(
        self, configured_targets: Sequence[str], configured_sources: Sequence[str]
    ):
        self.configured_targets = configured_targets
        self.configured_sources = configured_sources
        if not hasattr(self, "__orig_bases__"):
            self.__orig_bases__ = {}
            raise HRIException(
                f"Error while instantiating {str(self.__class__)}: Message type T derived from HRIMessage needs to be provided e.g. Connector[MessageType]()"
            )
        self.T_class = get_args(self.__orig_bases__[-1])[0]

    def _build_message(
        self,
        message: LangchainBaseMessage | RAIMultimodalMessage,
        communication_id: Optional[str] = None,
    ) -> T:
        return self.T_class.from_langchain(message, communication_id)

    def send_all_targets(
        self,
        message: LangchainBaseMessage | RAIMultimodalMessage,
        communication_id: Optional[str] = None,
    ):
        for target in self.configured_targets:
            to_send = self._build_message(message, communication_id)
            self.send_message(to_send, target)

    def receive_all_sources(self, timeout_sec: float = 1.0) -> dict[str, T]:
        ret = {}
        for source in self.configured_sources:
            received = self.receive_message(source, timeout_sec)
            ret[source] = received
        return ret
