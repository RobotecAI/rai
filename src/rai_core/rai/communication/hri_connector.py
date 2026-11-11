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

import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, Protocol, runtime_checkable

from rai.content import Audio, Image


@dataclass
class HRIMessage:
    """What a human and a robot say to each other: text, images and audio.

    communication_id groups the messages of one utterance or turn, seq_no
    orders them and seq_end marks the last one.
    """

    text: str = ""
    images: list[Image] = field(default_factory=list)
    audios: list[Audio] = field(default_factory=list)
    communication_id: str = ""
    seq_no: int = 0
    seq_end: bool = False

    def __bool__(self) -> bool:
        return bool(self.text or self.images or self.audios)

    @staticmethod
    def new_id() -> str:
        """A new communication_id."""
        return uuid.uuid4().hex


@runtime_checkable
class HRIConnector(Protocol):
    """A transport for HRIMessages (ROS 2, a sound device, a UI)."""

    async def send(self, target: str, message: HRIMessage) -> None: ...

    def subscribe(
        self, source: str, callback: Optional[Callable[[HRIMessage], Any]] = None
    ) -> Any: ...

    def close(self) -> None: ...
