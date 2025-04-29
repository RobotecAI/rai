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

from rai.types import (
    Detection2D,
    Header,
    RegionOfInterest,
    Ros2BaseModel,
)


class CameraInfo(Ros2BaseModel):
    _prefix: str = "sensor_msgs/msg"
    header: Header = Header()
    height: int = 0
    width: int = 0
    distortion_model: str = ""
    d: List[float] = []
    k: List[float] = [0.0] * 9
    r: List[float] = [0.0] * 9
    p: List[float] = [0.0] * 12
    binning_x: int = 0
    binning_y: int = 0
    roi: RegionOfInterest = RegionOfInterest()


class Image(Ros2BaseModel):
    _prefix: str = "sensor_msgs/msg"
    header: Header = Header()
    height: int = 0
    width: int = 0
    encoding: str = ""
    is_bigendian: int = 0
    step: int = 0
    data: List[int] = []


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
