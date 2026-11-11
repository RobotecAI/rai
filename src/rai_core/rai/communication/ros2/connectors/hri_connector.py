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

# The classes of this module use each other's private members.
# pyright: reportPrivateUsage=false

import logging
from array import array
from typing import Any, Callable, Optional, Union

import numpy as np

from rai.communication.hri_connector import HRIMessage
from rai.communication.ros2.connectors.ros2_connector import (
    QoS,
    ROS2Base,
    Subscription,
)
from rai.communication.ros2.images import numpy_to_imgmsg
from rai.communication.ros2.runtime import ExecutorKind, ROS2Runtime
from rai.communication.ros2.views import image as image_view
from rai.content import Audio, Image

HRI_TYPE = "rai_interfaces/msg/HRIMessage"
logger = logging.getLogger(__name__)


def to_ros(message: HRIMessage) -> Any:
    """The rai_interfaces/msg/HRIMessage of an HRIMessage. Audio is sent as int16."""
    from rai_interfaces.msg import AudioMessage
    from rai_interfaces.msg import HRIMessage as RosHRIMessage

    audios = [a.as_int16() for a in message.audios]
    return RosHRIMessage(
        text=message.text,
        images=[numpy_to_imgmsg(i.data, i.encoding) for i in message.images],
        audios=[
            AudioMessage(
                audio=array("h", a.samples.tobytes()),
                sample_rate=a.sample_rate,
                channels=a.channels,
            )
            for a in audios
        ],
        communication_id=message.communication_id,
        seq_no=message.seq_no,
        seq_end=message.seq_end,
    )


def _parts(items: Any, convert: Callable[[Any], Any]) -> list[Any]:
    out: list[Any] = []
    for item in items:
        try:
            out.append(convert(item))
        except ValueError as e:  # keep the rest of what the human said
            logger.warning("Dropped a malformed part of an HRI message: %s", e)
    return out


def from_ros(msg: Any) -> HRIMessage:
    """The HRIMessage of a rai_interfaces/msg/HRIMessage. Images and audio are
    numpy views of the message, not copies."""
    return HRIMessage(
        text=msg.text,
        images=_parts(msg.images, lambda i: Image(image_view(i), i.encoding)),
        audios=_parts(
            msg.audios,
            lambda a: Audio(
                np.frombuffer(a.audio, np.int16).reshape(-1, a.channels or 1),
                a.sample_rate,
            ),
        ),
        communication_id=msg.communication_id,
        seq_no=msg.seq_no,
        seq_end=msg.seq_end,
    )


class ROS2HRIConnector(ROS2Base):
    """Sends and receives :class:`~rai.communication.HRIMessage`s (text, images,
    audio) as ``rai_interfaces/msg/HRIMessage`` on the shared ROS2Runtime.

    HRI QoS is reliable and subscriptions never drop: a message from a human is
    always delivered, in order. Messages are decoded off the executor thread,
    into numpy views of the ROS message.
    """

    def __init__(
        self,
        node_name: Optional[str] = None,
        *,
        runtime: Optional[ROS2Runtime] = None,
        executor: Optional[ExecutorKind] = None,
        use_sim_time: bool = False,
    ) -> None:
        self._attach(node_name, runtime, executor, use_sim_time)
        self.sync = SyncROS2HRIConnector(self)

    async def send(self, target: str, message: Union[HRIMessage, str]) -> None:
        """Publish a message (a str is sent as text)."""
        self._check()
        if isinstance(message, str):
            message = HRIMessage(message)
        await self._runtime.apublish(target, to_ros(message), QoS.HRI)

    def subscribe(
        self,
        source: str,
        callback: Optional[Callable[[HRIMessage], Any]] = None,
        *,
        maxsize: Optional[int] = None,
    ) -> Subscription:
        """HRIMessages from source: to callback (run off the executor thread, one
        at a time, in order), or by iterating the Subscription. maxsize=None
        never drops."""
        return self._subscribe(
            source, callback, HRI_TYPE, QoS.HRI, maxsize, decode=from_ros
        )

    async def receive(
        self, source: str, *, timeout: float = 1.0, max_age: Optional[float] = None
    ) -> HRIMessage:
        """The latest message on source, see ROS2Connector.receive."""
        sample = await self._receive(source, HRI_TYPE, timeout, max_age, QoS.HRI)
        return from_ros(sample[0])


class SyncROS2HRIConnector:
    """Blocking twins of ROS2HRIConnector.send and receive."""

    def __init__(self, connector: ROS2HRIConnector) -> None:
        self._c = connector

    def send(self, target: str, message: Union[HRIMessage, str]) -> None:
        return self._c._runtime.run(self._c.send(target, message))

    def receive(self, source: str, **kwargs: Any) -> HRIMessage:
        return self._c._runtime.run(self._c.receive(source, **kwargs))
