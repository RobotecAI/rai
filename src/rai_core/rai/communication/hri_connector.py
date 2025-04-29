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
from io import BytesIO
from typing import Generic, Literal, Optional, TypeVar

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.messages import BaseMessage as LangchainBaseMessage
from PIL import Image
from PIL.Image import Image as ImageType
from pydantic import Field
from pydub import AudioSegment

from rai.communication.base_connector import BaseConnector, BaseMessage
from rai.messages import AIMultimodalMessage, HumanMultimodalMessage
from rai.messages import MultimodalMessage as RAIMultimodalMessage


class HRIException(Exception):
    def __init__(self, msg):
        super().__init__(msg)


class HRIMessage(BaseMessage):
    text: str = Field(default="")
    images: list[ImageType] = Field(default_factory=list)
    audios: list[AudioSegment] = Field(default_factory=list)
    message_author: Optional[Literal["ai", "human", "unspecified"]] = Field(
        default="unspecified"
    )
    communication_id: Optional[str] = Field(default=None)
    seq_no: int = Field(default=0)
    seq_end: bool = Field(default=False)

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
        if self.message_author == "unspecified":
            raise ValueError("Message author is not compatible with Langchain.")
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
                return AIMultimodalMessage(
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
        seq_no: int = 0,
        seq_end: bool = False,
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
            text=text,
            images=(
                [cls._base64_to_image(image) for image in images] if images else []
            ),
            audios=(
                [cls._base64_to_audio(audio) for audio in audios] if audios else []
            ),
            message_author=message.type,  # type: ignore
            communication_id=communication_id,
            seq_no=seq_no,
            seq_end=seq_end,
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

    def build_message(
        self,
        message: LangchainBaseMessage | RAIMultimodalMessage,
        communication_id: Optional[str] = None,
        seq_no: int = 0,
        seq_end: bool = False,
    ) -> T:
        """
        Build a new message object from a given input message.

        Parameters
        ----------
        message : LangchainBaseMessage or RAIMultimodalMessage
            The source message to transform into the target type.
        communication_id : str, optional
            An optional identifier for the communication session. Defaults to `None`.
        seq_no : int, optional
            The sequence number of the message in the communication stream. Defaults to `0`.
        seq_end : bool, optional
            Flag indicating whether this message is the final one in the sequence. Defaults to `False`.

        Returns
        -------
        T
            A message instance of type `T` compatible with the Connector, created from the provided input.

        Notes
        -----
        This method uses `self.T_class.from_langchain` for conversion and assumes compatibility.
        """
        return self.T_class.from_langchain(message, communication_id, seq_no, seq_end)  # type: ignore
