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

from rai.types import Pose, PoseStamped, RaiBaseModel

from rai_bench.tool_calling_agent.messages.topics import Image, RAIDetectionArray


class ManipulatorMoveToRequest(RaiBaseModel):
    initial_gripper_state: Optional[bool] = False
    final_gripper_state: Optional[bool] = False
    target_pose: Optional[PoseStamped] = PoseStamped()


class ManipulatorMoveToResponse(RaiBaseModel):
    success: Optional[bool] = False


class RAIGroundedSamRequest(RaiBaseModel):
    detections: Optional[RAIDetectionArray] = RAIDetectionArray()
    source_img: Optional[Image] = Image()


class RAIGroundedSamResponse(RaiBaseModel):
    masks: Optional[List[Image]] = []


class RAIGroundingDinoRequest(RaiBaseModel):
    classes: Optional[str] = ""
    box_threshold: Optional[float] = 0.0
    text_threshold: Optional[float] = 0.0
    source_img: Optional[Image] = Image()


class RAIGroundingDinoResponse(RaiBaseModel):
    detections: Optional[RAIDetectionArray] = RAIDetectionArray()


class StringListRequest(RaiBaseModel):
    pass


class StringListResponse(RaiBaseModel):
    success: Optional[bool] = False
    string_list: Optional[List[str]] = []


class VectorStoreRetrievalRequest(RaiBaseModel):
    query: Optional[str] = ""


class VectorStoreRetrievalResponse(RaiBaseModel):
    success: Optional[bool] = False
    message: Optional[str] = ""
    documents: Optional[List[str]] = []
    scores: Optional[List[float]] = []


class WhatISeeRequest(RaiBaseModel):
    pass


class WhatISeeResponse(RaiBaseModel):
    observations: Optional[List[str]] = []
    perception_source: Optional[str] = ""
    image: Optional[Image] = Image()
    pose: Optional[Pose] = Pose()


class PlannerInterfaceDescription(RaiBaseModel):
    name: Optional[str] = ""
    pipeline_id: Optional[str] = ""
    planner_ids: Optional[List[str]] = []


class QueryPlannerInterfaceResponse(RaiBaseModel):
    planner_interfaces: Optional[List[PlannerInterfaceDescription]] = []
