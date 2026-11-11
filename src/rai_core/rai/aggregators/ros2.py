# Copyright (C) 2025 Robotec.AI
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

"""ROS 2 aggregators: ``/rosout`` digest, last camera frame, VLM camera summary."""

import asyncio
from collections import Counter
from collections.abc import Sequence
from typing import Any

from langchain_core.language_models import BaseChatModel
from rcl_interfaces.msg import Log
from sensor_msgs.msg import CompressedImage, Image

from rai.aggregators.base import Aggregator
from rai.communication.ros2.images import encode_image
from rai.llm import get_llm
from rai.messages import human

LEVELS = {10: "DEBUG", 20: "INFO", 30: "WARN", 40: "ERROR", 50: "FATAL"}


class RosoutDigest(Aggregator[Log]):
    """The distinct log lines of the last ``max_age_s`` seconds at
    ``min_level`` or above, oldest first, with repeat counts."""

    def __init__(
        self,
        connector: Any,
        topic: str = "/rosout",
        *,
        min_level: int = 20,
        window: int = 200,
        max_age_s: float | None = 60.0,
        cache_s: float = 1.0,
    ) -> None:
        super().__init__(
            connector,
            topic,
            msg_type="rcl_interfaces/msg/Log",
            window=window,
            max_age_s=max_age_s,
            cache_s=cache_s,
        )
        self.min_level = min_level

    def push(self, msg: Log) -> None:
        if msg.level >= self.min_level:
            super().push(msg)

    async def summarize(self, msgs: Sequence[Log]) -> str:
        counts = Counter(
            f"[{m.name}] [{LEVELS.get(m.level, m.level)}] {m.msg}" for m in msgs
        )
        return "\n".join(f"{x} (x{n})" if n > 1 else x for x, n in counts.items())


class LastImage(Aggregator[Image | CompressedImage]):
    """The newest frame, shown to the model as an image."""

    def __init__(
        self,
        connector: Any,
        topic: str,
        *,
        msg_type: Any = "sensor_msgs/msg/Image",
        max_age_s: float | None = 5.0,
        cache_s: float = 0.0,
    ) -> None:
        super().__init__(
            connector,
            topic,
            msg_type=msg_type,
            window=1,
            max_age_s=max_age_s,
            cache_s=cache_s,
        )

    async def summarize(self, msgs: Sequence[Image | CompressedImage]) -> list[Any]:
        return human(images=[await asyncio.to_thread(encode_image, msgs[-1])]).content


class ImageSummary(Aggregator[Image | CompressedImage]):
    """A vision model's text summary of the camera.

    With ``frames=1`` it describes the newest frame; with more it compares
    that many frames spread evenly over the buffer (``window`` frames of the
    last ``max_age_s`` seconds), oldest first. Only the frames sent are encoded.
    ``model`` is a chat model or a :func:`rai.llm.get_llm` role, created on
    first use.
    """

    DESCRIBE = "Briefly list the key elements in the robot's camera view."
    COMPARE = (
        "These are {n} frames from the robot camera, oldest first. The robot "
        "might be moving. Briefly describe what changed."
    )

    def __init__(
        self,
        connector: Any,
        topic: str,
        *,
        model: BaseChatModel | str = "simple",
        frames: int = 1,
        prompt: str | None = None,
        msg_type: Any = "sensor_msgs/msg/Image",
        window: int = 30,
        max_age_s: float | None = 5.0,
        cache_s: float = 10.0,
    ) -> None:
        super().__init__(
            connector,
            topic,
            msg_type=msg_type,
            window=window if frames > 1 else 1,
            max_age_s=max_age_s,
            cache_s=cache_s,
        )
        self.model, self.frames, self.prompt = model, frames, prompt

    async def summarize(self, msgs: Sequence[Image | CompressedImage]) -> str:
        n = min(self.frames, len(msgs))
        if n > 1:
            msgs = [msgs[round(i * (len(msgs) - 1) / (n - 1))] for i in range(n)]
        else:
            msgs = msgs[-1:]
        images = await asyncio.to_thread(lambda: [encode_image(m) for m in msgs])
        if isinstance(self.model, str):
            self.model = get_llm(self.model)
        prompt = self.prompt or (self.COMPARE.format(n=n) if n > 1 else self.DESCRIBE)
        return (await self.model.ainvoke([human(prompt, images)])).text
