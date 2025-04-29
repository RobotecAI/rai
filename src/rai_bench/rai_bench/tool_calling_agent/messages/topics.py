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
