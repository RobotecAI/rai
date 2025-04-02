from typing import List

from pydantic import BaseModel

from rai_bench.tool_calling_agent_bench.messages.base import Pose, PoseStamped
from rai_bench.tool_calling_agent_bench.messages.topics import Image, RAIDetectionArray


class ManipulatorMoveToRequest(BaseModel):
    initial_gripper_state: bool = False
    final_gripper_state: bool = False
    target_pose: PoseStamped = PoseStamped()


class ManipulatorMoveToResponse(BaseModel):
    success: bool = False


class RAIGroundedSamRequest(BaseModel):
    detections: RAIDetectionArray = RAIDetectionArray()
    source_img: Image = Image()


class RAIGroundedSamResponse(BaseModel):
    masks: List[Image] = []


class RAIGroundingDinoRequest(BaseModel):
    classes: str = ""
    box_threshold: float = 0.0
    text_threshold: float = 0.0
    source_img: Image = Image()


class RAIGroundingDinoResponse(BaseModel):
    detections: RAIDetectionArray = RAIDetectionArray()


class StringListRequest(BaseModel):
    pass


class StringListResponse(BaseModel):
    success: bool = False
    string_list: List[str] = []


class VectorStoreRetrievalRequest(BaseModel):
    query: str = ""


class VectorStoreRetrievalResponse(BaseModel):
    success: bool = False
    message: str = ""
    documents: List[str] = []
    scores: List[float] = []


class WhatISeeRequest(BaseModel):
    pass


class WhatISeeResponse(BaseModel):
    observations: List[str] = []
    perception_source: str = ""
    image: Image = Image()
    pose: Pose = Pose()


class PlannerInterfaceDescription(BaseModel):
    name: str = ""
    pipeline_id: str = ""
    planner_ids: List[str] = []


class QueryPlannerInterfaceResponse(BaseModel):
    planner_interfaces: List[PlannerInterfaceDescription] = []
