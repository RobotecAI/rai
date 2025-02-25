from abc import abstractmethod
from typing import Any, Dict, Generic, Optional, TypedDict, TypeVar

from PIL.Image import Image
from pydub import AudioSegment
from pymongo import MongoClient
from pymongo.collection import Collection
from rai.models import PoseStamped

T = TypeVar("T", bound=Dict[str, Any])


class Database(Generic[T]):
    @abstractmethod
    def get_data(self, query: T):
        pass


class SpatioTemporalEntry(TypedDict):
    uuid: str
    pose_stamped: PoseStamped
    image: Optional[Image]
    text: Optional[str]
    audio: Optional[AudioSegment]


class SpatioTemporalDatabase(Database[T]):
    def __init__(self):
        self.client = MongoClient[Any]()
        self.collection: Collection[SpatioTemporalEntry] = self.client["memories"][
            "spatiotemporal"
        ]

    def get_data(self, query: T):
        return self.collection.find(query)
