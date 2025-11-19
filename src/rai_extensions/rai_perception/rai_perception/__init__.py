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

# Service names for ROS2 - defined here to avoid circular imports
GDINO_SERVICE_NAME = "grounding_dino_classify"
GDINO_NODE_NAME = "grounding_dino_node"
GSAM_SERVICE_NAME = "grounded_sam_segment"
GSAM_NODE_NAME = "grounded_sam_node"

from .agents import GroundedSamAgent, GroundingDinoAgent  # noqa: E402
from .tools import GetDetectionTool, GetDistanceToObjectsTool  # noqa: E402
from .tools.pcl_detection import (  # noqa: E402
    GrippingPointEstimator,
    GrippingPointEstimatorConfig,
    PointCloudFilter,
    PointCloudFilterConfig,
    PointCloudFromSegmentation,
    PointCloudFromSegmentationConfig,
    depth_to_point_cloud,
)
from .tools.pcl_detection_tools import (  # noqa: E402
    GetObjectGrippingPointsTool,
    GetObjectGrippingPointsToolInput,
)

__all__ = [
    "GDINO_NODE_NAME",
    "GDINO_SERVICE_NAME",
    "GSAM_NODE_NAME",
    "GSAM_SERVICE_NAME",
    "GetDetectionTool",
    "GetDistanceToObjectsTool",
    "GetObjectGrippingPointsTool",
    "GetObjectGrippingPointsToolInput",
    "GrippingPointEstimator",
    "GrippingPointEstimatorConfig",
    "GroundedSamAgent",
    "GroundingDinoAgent",
    "PointCloudFilter",
    "PointCloudFilterConfig",
    "PointCloudFromSegmentation",
    "PointCloudFromSegmentationConfig",
    "depth_to_point_cloud",
]
