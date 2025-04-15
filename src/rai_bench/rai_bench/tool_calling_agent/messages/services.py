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

from rai_bench.tool_calling_agent.messages.base import Pose, PoseStamped, Ros2BaseModel
from rai_bench.tool_calling_agent.messages.topics import Image, RAIDetectionArray


class ManipulatorMoveToRequest(Ros2BaseModel):
    initial_gripper_state: Optional[bool] = False
    final_gripper_state: Optional[bool] = False
    target_pose: Optional[PoseStamped] = PoseStamped()


class ManipulatorMoveToResponse(Ros2BaseModel):
    success: Optional[bool] = False


class RAIGroundedSamRequest(Ros2BaseModel):
    detections: Optional[RAIDetectionArray] = RAIDetectionArray()
    source_img: Optional[Image] = Image()


class RAIGroundedSamResponse(Ros2BaseModel):
    masks: Optional[List[Image]] = []


class RAIGroundingDinoRequest(Ros2BaseModel):
    classes: Optional[str] = ""
    box_threshold: Optional[float] = 0.0
    text_threshold: Optional[float] = 0.0
    source_img: Optional[Image] = Image()


class RAIGroundingDinoResponse(Ros2BaseModel):
    detections: Optional[RAIDetectionArray] = RAIDetectionArray()


class StringListRequest(Ros2BaseModel):
    pass


class StringListResponse(Ros2BaseModel):
    success: Optional[bool] = False
    string_list: Optional[List[str]] = []


class VectorStoreRetrievalRequest(Ros2BaseModel):
    query: Optional[str] = ""


class VectorStoreRetrievalResponse(Ros2BaseModel):
    success: Optional[bool] = False
    message: Optional[str] = ""
    documents: Optional[List[str]] = []
    scores: Optional[List[float]] = []


class WhatISeeRequest(Ros2BaseModel):
    pass


class WhatISeeResponse(Ros2BaseModel):
    observations: Optional[List[str]] = []
    perception_source: Optional[str] = ""
    image: Optional[Image] = Image()
    pose: Optional[Pose] = Pose()


class PlannerInterfaceDescription(Ros2BaseModel):
    name: Optional[str] = ""
    pipeline_id: Optional[str] = ""
    planner_ids: Optional[List[str]] = []


class QueryPlannerInterfaceResponse(Ros2BaseModel):
    planner_interfaces: Optional[List[PlannerInterfaceDescription]] = []
