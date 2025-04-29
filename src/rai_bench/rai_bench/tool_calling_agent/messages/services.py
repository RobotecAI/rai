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

from rai.types import Image, Pose, ROS2BaseModel


class StringListRequest(ROS2BaseModel):
    pass


class StringListResponse(ROS2BaseModel):
    success: bool = False
    string_list: List[str] = []


class VectorStoreRetrievalRequest(ROS2BaseModel):
    query: str = ""


class VectorStoreRetrievalResponse(ROS2BaseModel):
    success: bool = False
    message: str = ""
    documents: List[str] = []
    scores: List[float] = []


class WhatISeeRequest(ROS2BaseModel):
    pass


class WhatISeeResponse(ROS2BaseModel):
    observations: List[str] = []
    perception_source: str = ""
    image: Image = Image()
    pose: Pose = Pose()


class PlannerInterfaceDescription(ROS2BaseModel):
    name: str = ""
    pipeline_id: str = ""
    planner_ids: List[str] = []


class QueryPlannerInterfaceResponse(ROS2BaseModel):
    planner_interfaces: List[PlannerInterfaceDescription] = []
