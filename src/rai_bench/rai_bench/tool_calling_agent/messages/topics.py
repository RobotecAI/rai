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

from rai.types import Header, Image, ROS2BaseModel


class AudioMessage(ROS2BaseModel):
    _prefix: str = "rai_interfaces/msg"
    audio: List[int] = []
    sample_rate: int = 0
    channels: int = 0


# NOTE(boczekbartek): this message is duplicated here only for benchmarking purposes.
#                     for communication in rai please use rai.communication.ros2.ROS2HRIMessage
class HRIMessage(ROS2BaseModel):
    _prefix: str = "rai_interfaces/msg"
    header: Header = Header()
    text: str = ""
    images: List[Image] = []
    audios: List[AudioMessage] = []
    communication_id: str = ""
    seq_no: int = 0
    seq_end: bool = False
