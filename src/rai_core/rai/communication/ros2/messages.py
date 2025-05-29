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

import importlib.util
import logging
from collections import OrderedDict
from typing import Any, List, Literal, cast
from uuid import uuid4

import numpy as np
import rosidl_runtime_py.convert
from cv_bridge import CvBridge
from PIL import Image
from pydub import AudioSegment
from sensor_msgs.msg import Image as ROS2Image

from rai.communication.base_connector import BaseMessage
from rai.communication.hri_connector import HRIMessage

if importlib.util.find_spec("rai_interfaces") is None:
    logging.warning("rai_interfaces is not installed, ROS 2 HRIMessage will not work.")
else:
    import rai_interfaces.msg
    from rai_interfaces.msg import HRIMessage as ROS2HRIMessage_
    from rai_interfaces.msg._audio_message import (
        AudioMessage as ROS2HRIMessage__Audio,
    )


class ROS2Message(BaseMessage):
    pass


class ROS2HRIMessage(HRIMessage, ROS2Message):
    @classmethod
    def from_ros2(
        cls,
        msg: "rai_interfaces.msg.HRIMessage",
        message_author: Literal["ai", "human"],
    ):
        cv_bridge = CvBridge()
        images = [
            cv_bridge.imgmsg_to_cv2(img_msg, "rgb8")
            for img_msg in cast(List[ROS2Image], msg.images)
        ]
        pil_images = [Image.fromarray(img) for img in images]
        audio_segments = [
            AudioSegment(
                data=audio_msg.audio,
                frame_rate=audio_msg.sample_rate,
                sample_width=2,  # bytes, int16
                channels=audio_msg.channels,
            )
            for audio_msg in cast(List[ROS2HRIMessage__Audio], msg.audios)
        ]
        communication_id = msg.communication_id if msg.communication_id != "" else None
        return cls(
            text=msg.text,
            images=pil_images,
            audios=audio_segments,
            message_author=message_author,
            communication_id=communication_id,
            seq_no=msg.seq_no,
            seq_end=msg.seq_end,
        )

    def to_ros2_dict(self) -> OrderedDict[str, Any]:
        cv_bridge = CvBridge()
        img_msgs = [
            cv_bridge.cv2_to_imgmsg(np.array(img), "rgb8") for img in self.images
        ]
        audio_msgs = [
            ROS2HRIMessage__Audio(
                audio=audio.raw_data,
                sample_rate=audio.frame_rate,
                channels=audio.channels,
            )
            for audio in self.audios
        ]

        return cast(
            OrderedDict[str, Any],
            rosidl_runtime_py.convert.message_to_ordereddict(
                ROS2HRIMessage_(
                    text=self.text,
                    images=img_msgs,
                    audios=audio_msgs,
                    communication_id=self.communication_id or "",
                    seq_no=self.seq_no,
                    seq_end=self.seq_end,
                )
            ),
        )

    @staticmethod
    def generate_conversation_id() -> str:
        return str(uuid4())
