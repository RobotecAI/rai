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

from datetime import datetime
from typing import Any, Dict, List, Type

from langchain_core.tools import BaseTool, BaseToolkit
from langchain_core.vectorstores import VectorStore
from pydantic import BaseModel, ConfigDict, Field
from pymongo.collection import Collection
from pymongo.mongo_client import MongoClient

from rai.agents.spatiotemporal.spatiotemporal_agent import (
    Pose,
    SpatioTemporalRecord,
)
from rai.agents.tool_runner import MultimodalArtifact


class SpatiotemporalToolkit(BaseToolkit):
    name: str = "Spatiotemporal Toolkit"
    description: str = "A toolkit for spatiotemporal data processing"

    mongodb_url: str = Field(default="mongodb://localhost:27017/")
    mongodb_db_name: str = Field(default="rai")
    mongodb_collection_name: str = Field(default="spatiotemporal_collection")
    vectorstore: VectorStore

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="allow")

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.client: MongoClient[Any] = MongoClient(self.mongodb_url)
        self.collection: Collection[Any] = self.client[self.mongodb_db_name][
            self.mongodb_collection_name
        ]

    def get_tools(self) -> list[BaseTool]:
        return [
            GetMemoriesNearPoseTool(
                collection=self.collection,
            ),
            GetMemoriesNearTimestampTool(
                collection=self.collection,
            ),
            GetMemoriesOfObjectTool(
                collection=self.collection, vectorstore=self.vectorstore
            ),
        ]


class GetMemoriesNearPoseToolInput(BaseModel):
    pose: Pose = Field(description="The pose of the robot")
    n_results: int = Field(default=5, description="The number of results to return")
    radius_meters: float = Field(default=1.0, description="The radius in meters")


class GetMemoriesNearPoseTool(BaseTool):
    name: str = "get_memories_near_pose"
    description: str = "Get the past memories of the robot near a specific pose"
    args_schema: Type[GetMemoriesNearPoseToolInput] = GetMemoriesNearPoseToolInput
    collection: Collection[Any]

    def _run(self, pose: Pose, n_results: int = 5, radius_meters: float = 1.0):
        # Get position coordinates
        position = pose.position
        x, y, z = position.x, position.y, position.z

        # Query using Euclidean distance calculation
        results = list(
            self.collection.find(
                {
                    "$expr": {
                        "$lte": [
                            {
                                "$sqrt": {
                                    "$add": [
                                        {
                                            "$pow": [
                                                {
                                                    "$subtract": [
                                                        "$tf.pose.position.x",
                                                        x,
                                                    ]
                                                },
                                                2,
                                            ]
                                        },
                                        {
                                            "$pow": [
                                                {
                                                    "$subtract": [
                                                        "$tf.pose.position.y",
                                                        y,
                                                    ]
                                                },
                                                2,
                                            ]
                                        },
                                        {
                                            "$pow": [
                                                {
                                                    "$subtract": [
                                                        "$tf.pose.position.z",
                                                        z,
                                                    ]
                                                },
                                                2,
                                            ]
                                        },
                                    ]
                                }
                            },
                            radius_meters,
                        ]
                    }
                },
                limit=n_results,
            ),
        )
        images: List[str] = []
        parsed_results = list(map(SpatioTemporalRecord.model_validate, results))
        for result in parsed_results:
            for image in result.images.values():
                images.append(image)
        return "The query returned the following results: " + str(
            [result.temporal_context for result in parsed_results]
        ), MultimodalArtifact(images=images, audios=[])


class GetMemoriesNearTimestampToolInput(BaseModel):
    year: int = Field(description="The year to get memories from")
    month: int = Field(description="The month to get memories from")
    day: int = Field(description="The day to get memories from")
    hour: int = Field(description="The hour to get memories from")
    minute: int = Field(description="The minute to get memories from")
    second: int = Field(description="The second to get memories from")
    time_range: float = Field(
        7200, description="The time range to get memories from in seconds"
    )
    n_results: int = Field(default=5, description="The number of results to return")


class GetMemoriesNearTimestampTool(BaseTool):
    name: str = "get_memories_near_timestamp"
    description: str = "Get the past memories of the robot near a specific timestamp"
    args_schema: Type[GetMemoriesNearTimestampToolInput] = (
        GetMemoriesNearTimestampToolInput
    )
    collection: Collection[Any]

    def _run(
        self,
        year: int,
        month: int,
        day: int,
        hour: int,
        minute: int,
        second: int,
        time_range: float,
        n_results: int = 5,
    ):
        timestamp = datetime(year, month, day, hour, minute, second).timestamp()
        results = list(
            self.collection.find(
                {
                    "timestamp": {
                        "$gte": timestamp - time_range,
                        "$lte": timestamp + time_range,
                    }
                },
                limit=n_results,
            ),
        )
        images: List[str] = []
        parsed_results = list(map(SpatioTemporalRecord.model_validate, results))
        for result in parsed_results:
            for image in result.images.values():
                images.append(image)
        return "The query returned the following results: " + str(
            [result.temporal_context for result in parsed_results]
        ), MultimodalArtifact(images=images, audios=[])


class GetMemoriesOfObjectToolInput(BaseModel):
    object_name: str = Field(description="The name of the object to get memories of")
    n_results: int = Field(default=5, description="The number of results to return")


class GetMemoriesOfObjectTool(BaseTool):
    name: str = "get_memories_of_object"
    description: str = "Get the past memories of the robot of a specific object"
    args_schema: Type[GetMemoriesOfObjectToolInput] = GetMemoriesOfObjectToolInput
    collection: Collection[Any]
    vectorstore: VectorStore

    response_model: str = "content_and_artifact"

    def _run(self, object_name: str, n_results: int = 5):
        documents = self.vectorstore.similarity_search(object_name, k=n_results)
        mongodb_data: List[Dict[str, Any]] = []
        for document in documents:
            id = document.id
            record = self.collection.find_one({"id": id})
            if record is not None:
                mongodb_data.append(record)
        images: List[str] = []
        parsed_results = list(map(SpatioTemporalRecord.model_validate, mongodb_data))
        for result in parsed_results:
            for image in result.images.values():
                images.append(image)

        return "The query returned the following results: " + str(
            [result.temporal_context for result in parsed_results]
        ), MultimodalArtifact(images=images, audios=[])
