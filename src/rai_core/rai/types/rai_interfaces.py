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

from rai.types.base import ROS2BaseModel
from rai.types.geometry import PoseStamped
from rai.types.sensor import Image
from rai.types.std import Header
from rai.types.vision import Detection2D


class RAIDetectionArray(ROS2BaseModel):
    _prefix: str = "rai_interfaces/msg"
    header: Header = Header()
    detections: List[Detection2D] = []
    detection_classes: List[str] = []


class BaseRaiSrv(ROS2BaseModel):
    _prefix: str = "rai_interfaces/srv"


class ManipulatorMoveToRequest(BaseRaiSrv):
    initial_gripper_state: bool = False
    final_gripper_state: bool = False
    target_pose: PoseStamped = PoseStamped()


class ManipulatorMoveToResponse(BaseRaiSrv):
    _prefix: str = "rai_interfaces/srv"
    success: bool = False


class RAIGroundedSamRequest(BaseRaiSrv):
    _prefix: str = "rai_interfaces/srv"
    detections: RAIDetectionArray = RAIDetectionArray()
    source_img: Image = Image()


class RAIGroundedSamResponse(BaseRaiSrv):
    _prefix: str = "rai_interfaces/srv"
    masks: List[Image] = []


class RAIGroundingDinoRequest(BaseRaiSrv):
    _prefix: str = "rai_interfaces/srv"
    classes: str = ""
    box_threshold: float = 0.0
    text_threshold: float = 0.0
    source_img: Image = Image()


class RAIGroundingDinoResponse(BaseRaiSrv):
    _prefix: str = "rai_interfaces/srv"
    detections: RAIDetectionArray = RAIDetectionArray()
