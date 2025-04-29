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

from rai.types import Pose, PoseStamped, Ros2BaseModel

from rai_bench.tool_calling_agent.messages.topics import Image, RAIDetectionArray


class ManipulatorMoveToRequest(Ros2BaseModel):
    initial_gripper_state: bool = False
    final_gripper_state: bool = False
    target_pose: PoseStamped = PoseStamped()


class ManipulatorMoveToResponse(Ros2BaseModel):
    success: bool = False


class RAIGroundedSamRequest(Ros2BaseModel):
    detections: RAIDetectionArray = RAIDetectionArray()
    source_img: Image = Image()


class RAIGroundedSamResponse(Ros2BaseModel):
    masks: List[Image] = []


class RAIGroundingDinoRequest(Ros2BaseModel):
    classes: str = ""
    box_threshold: float = 0.0
    text_threshold: float = 0.0
    source_img: Image = Image()


class RAIGroundingDinoResponse(Ros2BaseModel):
    detections: RAIDetectionArray = RAIDetectionArray()


class StringListRequest(Ros2BaseModel):
    pass


class StringListResponse(Ros2BaseModel):
    success: bool = False
    string_list: List[str] = []


class VectorStoreRetrievalRequest(Ros2BaseModel):
    query: str = ""


class VectorStoreRetrievalResponse(Ros2BaseModel):
    success: bool = False
    message: str = ""
    documents: List[str] = []
    scores: List[float] = []


class WhatISeeRequest(Ros2BaseModel):
    pass


class WhatISeeResponse(Ros2BaseModel):
    observations: List[str] = []
    perception_source: str = ""
    image: Image = Image()
    pose: Pose = Pose()


class PlannerInterfaceDescription(Ros2BaseModel):
    name: str = ""
    pipeline_id: str = ""
    planner_ids: List[str] = []


class QueryPlannerInterfaceResponse(Ros2BaseModel):
    planner_interfaces: List[PlannerInterfaceDescription] = []
