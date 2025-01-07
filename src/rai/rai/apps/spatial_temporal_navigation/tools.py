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


from typing import List, Type

from langchain_core.tools import BaseTool
from langchain_core.vectorstores.base import VectorStore
from pydantic import BaseModel, Field
from pymongo.collection import Collection

from rai.apps.spatial_temporal_navigation.spatial_temporal_navigation import (
    Observation,
    Pose,
)


def get_memories_near_position(
    observations_collection: Collection,
    pose: Pose,
    k: int = 10,
    max_distance: float = 5.0,
) -> List[Observation]:
    """Get memories near a specific position using regular coordinate comparison"""
    cursor = observations_collection.find()

    observations_with_distances = []
    for doc in cursor:
        obs_pos = doc["position_stamped"]["position"]
        distance = (
            (obs_pos["x"] - pose.x) ** 2
            + (obs_pos["y"] - pose.y) ** 2
            + (obs_pos["z"] - pose.z) ** 2
        ) ** 0.5

        if distance <= max_distance:
            observations_with_distances.append((distance, doc))

    observations_with_distances.sort(key=lambda x: x[0])
    closest_observations = observations_with_distances[:k]

    return [Observation(**doc) for _, doc in closest_observations]


def get_memories_near_timestamp(
    observations_collection: Collection,
    timestamp: float,
    k: int = 10,
    range: float = 300.0,
) -> List[Observation]:
    """Get memories near a specific timestamp"""
    cursor = observations_collection.find(
        {
            "timestamp": {
                "$gte": timestamp - range,
                "$lte": timestamp + range,
            }
        }
    ).limit(k)

    return [Observation(**doc) for doc in cursor]


def get_memories_near_text(
    vectorstore: VectorStore,
    observations_collection: Collection,
    text: str,
    k: int = 10,
) -> List[Observation]:
    """Get memories by text similarity using vectorstore and then fetch from MongoDB"""
    results = vectorstore.similarity_search_with_relevance_scores(text, k=k)

    uuids = [result[0].metadata["uuid"] for result in results]
    uuids = list(map(str, uuids))

    cursor = observations_collection.find({"uuid": {"$in": uuids}})
    return [Observation(**doc) for doc in cursor]


class GetMemoriesNearPositionTool(BaseTool):
    name: str = "get_memories_near_position"
    description: str = (
        "Get memories near a specific position using regular coordinate comparison"
    )
    args_schema: Type[Pose] = Pose

    observations_collection: Collection

    def _run(self, pose: Pose) -> List[Observation]:
        return get_memories_near_position(self.observations_collection, pose)


class GetMemoriesNearTimestampToolInput(BaseModel):
    timestamp: float = Field(
        ..., description="The timestamp to search for memories near"
    )


class GetMemoriesNearTimestampTool(BaseTool):
    name: str = "get_memories_near_timestamp"
    description: str = "Get memories near a specific timestamp"
    args_schema: Type[GetMemoriesNearTimestampToolInput] = (
        GetMemoriesNearTimestampToolInput
    )

    observations_collection: Collection

    def _run(self, timestamp: float) -> List[Observation]:
        return get_memories_near_timestamp(self.observations_collection, timestamp)


class GetMemoriesNearTextToolInput(BaseModel):
    text: str = Field(..., description="The text to search for memories near")


class GetMemoriesNearTextTool(BaseTool):
    name: str = "get_memories_near_text"
    description: str = "Get memories near a specific text"
    args_schema: Type[GetMemoriesNearTextToolInput] = GetMemoriesNearTextToolInput

    vectorstore: VectorStore
    observations_collection: Collection

    def _run(self, text: str) -> List[Observation]:
        return get_memories_near_text(
            self.vectorstore, self.observations_collection, text
        )
