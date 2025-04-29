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

from typing import List

from .geometry import PoseStamped
from .sensor import Image
from .std import (
    Header,
    Ros2BaseModel,
)
from .vision import Detection2D


class AudioMessage(Ros2BaseModel):
    _prefix: str = "rai_interfaces/msg"
    audio: List[int] = []
    sample_rate: int = 0
    channels: int = 0


class HRIMessage(Ros2BaseModel):
    _prefix: str = "rai_interfaces/msg"
    header: Header = Header()
    text: str = ""
    images: List[Image] = []
    audios: List[AudioMessage] = []
    communication_id: str = ""
    seq_no: int = 0
    seq_end: bool = False


class RAIDetectionArray(Ros2BaseModel):
    _prefix: str = "rai_interfaces/msg"
    header: Header = Header()
    detections: List[Detection2D] = []
    detection_classes: List[str] = []


class ManipulatorMoveToRequest(Ros2BaseModel):
    _prefix: str = "rai_interfaces/srv"
    initial_gripper_state: bool = False
    final_gripper_state: bool = False
    target_pose: PoseStamped = PoseStamped()


class ManipulatorMoveToResponse(Ros2BaseModel):
    _prefix: str = "rai_interfaces/srv"
    success: bool = False


class RAIGroundedSamRequest(Ros2BaseModel):
    _prefix: str = "rai_interfaces/srv"
    detections: RAIDetectionArray = RAIDetectionArray()
    source_img: Image = Image()


class RAIGroundedSamResponse(Ros2BaseModel):
    _prefix: str = "rai_interfaces/srv"
    masks: List[Image] = []


class RAIGroundingDinoRequest(Ros2BaseModel):
    _prefix: str = "rai_interfaces/srv"
    classes: str = ""
    box_threshold: float = 0.0
    text_threshold: float = 0.0
    source_img: Image = Image()


class RAIGroundingDinoResponse(Ros2BaseModel):
    _prefix: str = "rai_interfaces/srv"
    detections: RAIDetectionArray = RAIDetectionArray()
