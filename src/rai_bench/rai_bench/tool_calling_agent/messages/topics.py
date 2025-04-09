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

from typing import List, Optional

from pydantic import BaseModel

from rai_bench.tool_calling_agent.messages.base import (
    Detection2D,
    Header,
    RegionOfInterest,
)


class CameraInfo(BaseModel):
    header: Optional[Header] = Header()
    height: Optional[int] = 0
    width: Optional[int] = 0
    distortion_model: Optional[str] = ""
    d: Optional[List[float]] = []
    k: Optional[List[float]] = [0.0] * 9
    r: Optional[List[float]] = [0.0] * 9
    p: Optional[List[float]] = [0.0] * 12
    binning_x: Optional[int] = 0
    binning_y: Optional[int] = 0
    roi: Optional[RegionOfInterest] = RegionOfInterest()


class Image(BaseModel):
    header: Optional[Header] = Header()
    height: Optional[int] = 0
    width: Optional[int] = 0
    encoding: Optional[str] = ""
    is_bigendian: Optional[int] = 0
    step: Optional[int] = 0
    data: Optional[List[int]] = []


class AudioMessage(BaseModel):
    audio: Optional[List[int]] = []
    sample_rate: Optional[int] = 0
    channels: Optional[int] = 0


class HRIMessage(BaseModel):
    header: Optional[Header] = Header()
    text: Optional[str] = ""
    images: Optional[List[Image]] = []
    audios: Optional[List[AudioMessage]] = []
    communication_id: Optional[str] = ""
    seq_no: Optional[int] = 0
    seq_end: Optional[bool] = False


class RAIDetectionArray(BaseModel):
    header: Optional[Header] = Header()
    detections: Optional[List[Detection2D]] = []
    detection_classes: Optional[List[str]] = []
