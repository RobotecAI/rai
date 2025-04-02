from typing import List

from pydantic import BaseModel

from rai_bench.tool_calling_agent_bench.messages.base import (
    Detection2D,
    Header,
    RegionOfInterest,
)


class CameraInfo(BaseModel):
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


class Image(BaseModel):
    header: Header = Header()
    height: int = 0
    width: int = 0
    encoding: str = ""
    is_bigendian: int = 0
    step: int = 0
    data: List[int] = []


class AudioMessage(BaseModel):
    audio: List[int] = []
    sample_rate: int = 0
    channels: int = 0


class HRIMessage(BaseModel):
    header: Header = Header()
    text: str = ""
    images: List[Image] = []
    audios: List[AudioMessage] = []


class RAIDetectionArray(BaseModel):
    header: Header = Header()
    detections: List[Detection2D] = []
    detection_classes: List[str] = []
