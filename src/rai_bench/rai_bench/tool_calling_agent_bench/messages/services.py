from typing import List

from pydantic import BaseModel

from rai_bench.tool_calling_agent_bench.messages.base import Pose, PoseStamped
from rai_bench.tool_calling_agent_bench.messages.topics import Image, RAIDetectionArray

from typing import List, Optional


class ManipulatorMoveToRequest(BaseModel):
    initial_gripper_state: Optional[bool] = False
    final_gripper_state: Optional[bool] = False
    target_pose: Optional[PoseStamped] = PoseStamped()


class ManipulatorMoveToResponse(BaseModel):
    success: Optional[bool] = False


class RAIGroundedSamRequest(BaseModel):
    detections: Optional[RAIDetectionArray] = RAIDetectionArray()
    source_img: Optional[Image] = Image()


class RAIGroundedSamResponse(BaseModel):
    masks: Optional[List[Image]] = []


class RAIGroundingDinoRequest(BaseModel):
    classes: Optional[str] = ""
    box_threshold: Optional[float] = 0.0
    text_threshold: Optional[float] = 0.0
    source_img: Optional[Image] = Image()


class RAIGroundingDinoResponse(BaseModel):
    detections: Optional[RAIDetectionArray] = RAIDetectionArray()


class StringListRequest(BaseModel):
    pass


class StringListResponse(BaseModel):
    success: Optional[bool] = False
    string_list: Optional[List[str]] = []


class VectorStoreRetrievalRequest(BaseModel):
    query: Optional[str] = ""


class VectorStoreRetrievalResponse(BaseModel):
    success: Optional[bool] = False
    message: Optional[str] = ""
    documents: Optional[List[str]] = []
    scores: Optional[List[float]] = []


class WhatISeeRequest(BaseModel):
    pass


class WhatISeeResponse(BaseModel):
    observations: Optional[List[str]] = []
    perception_source: Optional[str] = ""
    image: Optional[Image] = Image()
    pose: Optional[Pose] = Pose()


class PlannerInterfaceDescription(BaseModel):
    name: Optional[str] = ""
    pipeline_id: Optional[str] = ""
    planner_ids: Optional[List[str]] = []


class QueryPlannerInterfaceResponse(BaseModel):
    planner_interfaces: Optional[List[PlannerInterfaceDescription]] = []
