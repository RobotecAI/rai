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

import logging
from collections import OrderedDict
from typing import Any, Dict, List, Literal, Optional, cast

import numpy as np
import rosidl_runtime_py.convert
from cv_bridge import CvBridge
from PIL import Image
from pydub import AudioSegment
from sensor_msgs.msg import Image as ROS2Image

from rai.communication.ari_connector import ARIMessage
from rai.communication.hri_connector import HRIMessage, HRIPayload

try:
    import rai_interfaces.msg
    from rai_interfaces.msg import HRIMessage as ROS2HRIMessage_
    from rai_interfaces.msg._audio_message import (
        AudioMessage as ROS2HRIMessage__Audio,
    )
except ImportError:
    logging.warning("rai_interfaces is not installed, ROS 2 HRIMessage will not work.")


class ROS2ARIMessage(ARIMessage):
    def __init__(self, payload: Any, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(payload, metadata)


class ROS2HRIMessage(HRIMessage):
    def __init__(
        self,
        payload: HRIPayload,
        message_author: Literal["ai", "human"],
        conversation_id: Optional[str] = None,
    ):
        super().__init__(payload, {}, message_author, conversation_id)

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
        conversation_id = msg.conversation_id if msg.conversation_id != "" else None
        return ROS2HRIMessage(
            payload=HRIPayload(text=msg.text, images=pil_images, audios=audio_segments),
            message_author=message_author,
            conversation_id=conversation_id,
        )

    def to_ros2_dict(self) -> OrderedDict[str, Any]:
        cv_bridge = CvBridge()
        assert isinstance(self.payload, HRIPayload)
        img_msgs = [
            cv_bridge.cv2_to_imgmsg(np.array(img), "rgb8")
            for img in self.payload.images
        ]
        audio_msgs = [
            ROS2HRIMessage__Audio(
                audio=audio.raw_data,
                sample_rate=audio.frame_rate,
                channels=audio.channels,
            )
            for audio in self.payload.audios
        ]

        return cast(
            OrderedDict[str, Any],
            rosidl_runtime_py.convert.message_to_ordereddict(
                ROS2HRIMessage_(
                    text=self.payload.text,
                    images=img_msgs,
                    audios=audio_msgs,
                    conversation_id=self.conversation_id or "",
                )
            ),
        )
