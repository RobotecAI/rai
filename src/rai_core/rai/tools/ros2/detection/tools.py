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

from typing import Any, Optional, Type

from pydantic import BaseModel, Field

from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.ros2.detection.pcl import (
    GrippingPointEstimator,
    GrippingPointEstimatorConfig,
    PointCloudFilter,
    PointCloudFilterConfig,
    PointCloudFromSegmentation,
    PointCloudFromSegmentationConfig,
)
from rai.tools.timeout import TimeoutError, timeout


class GetGrippingPointToolInput(BaseModel):
    object_name: str = Field(
        ...,
        description="The name of the object to get the gripping point of e.g. 'box', 'apple', 'screwdriver'",
    )


class GetGrippingPointTool(BaseROS2Tool):
    name: str = "get_gripping_point"
    description: str = "Get gripping points for specified object/objects. Returns 3D coordinates where a robot gripper can grasp the object."

    # Configuration for PCL components
    segmentation_config: PointCloudFromSegmentationConfig
    estimator_config: GrippingPointEstimatorConfig
    filter_config: PointCloudFilterConfig

    # Auto-initialized in model_post_init from ROS2 parameters
    target_frame: Optional[str] = None
    source_frame: Optional[str] = None
    camera_topic: Optional[str] = None
    depth_topic: Optional[str] = None
    camera_info_topic: Optional[str] = None
    timeout_sec: Optional[float] = None

    # Components initialized in model_post_init
    gripping_point_estimator: Optional[GrippingPointEstimator] = None
    point_cloud_filter: Optional[PointCloudFilter] = None
    point_cloud_from_segmentation: Optional[PointCloudFromSegmentation] = None

    args_schema: Type[GetGrippingPointToolInput] = GetGrippingPointToolInput

    def model_post_init(self, __context: Any) -> None:
        """Initialize tool with ROS2 parameters and components."""
        self._load_parameters()
        self._initialize_components()

    def _load_parameters(self) -> None:
        """Load configuration from ROS2 parameters."""
        node = self.connector.node
        param_prefix = "detection_tools.gripping_point"

        # Declare required parameters
        required_params = [
            f"{param_prefix}.target_frame",
            f"{param_prefix}.source_frame",
            f"{param_prefix}.camera_topic",
            f"{param_prefix}.depth_topic",
            f"{param_prefix}.camera_info_topic",
        ]

        for param_name in required_params:
            if not node.has_parameter(param_name):
                raise ValueError(
                    f"Required parameter '{param_name}' must be set before initializing GetGrippingPointTool"
                )

        # Optional parameter with default
        node.declare_parameter(f"{param_prefix}.timeout_sec", 10.0)

        # Load parameters
        self.target_frame = node.get_parameter(f"{param_prefix}.target_frame").value
        self.source_frame = node.get_parameter(f"{param_prefix}.source_frame").value
        self.camera_topic = node.get_parameter(f"{param_prefix}.camera_topic").value
        self.depth_topic = node.get_parameter(f"{param_prefix}.depth_topic").value
        self.camera_info_topic = node.get_parameter(
            f"{param_prefix}.camera_info_topic"
        ).value
        self.timeout_sec = node.get_parameter(f"{param_prefix}.timeout_sec").value

        # Validate required parameters are not empty
        if not all(
            [
                self.target_frame,
                self.source_frame,
                self.camera_topic,
                self.depth_topic,
                self.camera_info_topic,
            ]
        ):
            raise ValueError(
                "Required ROS2 parameters for GetGrippingPointTool cannot be empty"
            )

    def _initialize_components(self) -> None:
        """Initialize PCL components with loaded parameters."""
        self.point_cloud_from_segmentation = PointCloudFromSegmentation(
            connector=self.connector,
            camera_topic=self.camera_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            source_frame=self.source_frame,
            target_frame=self.target_frame,
            config=self.segmentation_config,
        )
        self.gripping_point_estimator = GrippingPointEstimator(
            config=self.estimator_config
        )
        self.point_cloud_filter = PointCloudFilter(config=self.filter_config)

    def _run(self, object_name: str) -> str:
        # this will be not work in agent scenario because signal need to be run in main thread, comment out for now
        @timeout(
            self.timeout_sec,
            f"Gripping point detection for object '{object_name}' exceeded {self.timeout_sec} seconds",
        )
        def _run_with_timeout():
            pcl = self.point_cloud_from_segmentation.run(object_name)
            if len(pcl) == 0:
                return f"No {object_name}s detected."

            pcl = self.point_cloud_filter.run(pcl)
            gps = self.gripping_point_estimator.run(pcl)

            message = ""
            if len(gps) == 0:
                message += f"No gripping point found for the object {object_name}\n"
            elif len(gps) == 1:
                message += (
                    f"The gripping point of the object {object_name} is {gps[0]}\n"
                )
            else:
                message += (
                    f"Multiple gripping points found for the object {object_name}\n"
                )

            for i, gp in enumerate(gps):
                message += (
                    f"The gripping point of the object {i + 1} {object_name} is {gp}\n"
                )

            return message

        try:
            return _run_with_timeout()
        except TimeoutError:
            return f"Timeout: Gripping point detection for object '{object_name}' exceeded {self.timeout_sec} seconds"
        except Exception:
            raise
