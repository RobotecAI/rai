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
from rai.tools.timeout import RaiTimeoutError, timeout

from .pcl_detection import (
    GrippingPointEstimator,
    GrippingPointEstimatorConfig,
    PointCloudFilter,
    PointCloudFilterConfig,
    PointCloudFromSegmentation,
    PointCloudFromSegmentationConfig,
)

# Parameter prefix for ROS2 configuration
PCL_DETECTION_PARAM_PREFIX = "pcl.detection.gripping_points"


class GetObjectGrippingPointsToolInput(BaseModel):
    object_name: str = Field(
        ...,
        description="The type of object to get the gripping point for (e.g. 'cube', 'apple', 'screwdriver'). Pass the general object type or class, not a specific instance or ID like 'cube_1' or 'apple_5'.",
    )


class GetObjectGrippingPointsTool(BaseROS2Tool):
    name: str = "get_object_gripping_points"
    description: str = "Get gripping points for all objects of the specified type (e.g. all cubes). Returns 3D coordinates for every object of that class in the scene, where a robot gripper can grasp. Call this tool once with the general object class name, not multiple times for each object instance (e.g., call once with 'cube' to get all cubes)."

    # Configuration for PCL components
    segmentation_config: PointCloudFromSegmentationConfig = Field(
        default_factory=PointCloudFromSegmentationConfig,
        description="Configuration for point cloud segmentation from camera images",
    )
    estimator_config: GrippingPointEstimatorConfig = Field(
        default_factory=GrippingPointEstimatorConfig,
        description="Configuration for gripping point estimation strategies",
    )
    filter_config: PointCloudFilterConfig = Field(
        default_factory=PointCloudFilterConfig,
        description="Configuration for point cloud filtering and outlier removal",
    )

    # Auto-initialized in model_post_init from ROS2 parameters
    target_frame: Optional[str] = Field(
        default=None, description="Target coordinate frame for gripping points"
    )
    source_frame: Optional[str] = Field(
        default=None, description="Source coordinate frame of camera data"
    )
    camera_topic: Optional[str] = Field(
        default=None, description="ROS2 topic for camera RGB images"
    )
    depth_topic: Optional[str] = Field(
        default=None, description="ROS2 topic for camera depth images"
    )
    camera_info_topic: Optional[str] = Field(
        default=None, description="ROS2 topic for camera calibration info"
    )
    timeout_sec: Optional[float] = Field(
        default=None, description="Timeout in seconds for gripping point detection"
    )
    conversion_ratio: Optional[float] = Field(
        default=0.001, description="Conversion ratio from depth units to meters"
    )

    # Components initialized in model_post_init
    gripping_point_estimator: Optional[GrippingPointEstimator] = Field(
        default=None, exclude=True
    )
    point_cloud_filter: Optional[PointCloudFilter] = Field(default=None, exclude=True)
    point_cloud_from_segmentation: Optional[PointCloudFromSegmentation] = Field(
        default=None, exclude=True
    )

    args_schema: Type[GetObjectGrippingPointsToolInput] = (
        GetObjectGrippingPointsToolInput
    )

    def model_post_init(self, __context: Any) -> None:
        """Initialize tool with ROS2 parameters and components."""
        self._load_parameters()
        self._initialize_components()

    def _load_parameters(self) -> None:
        """Load configuration from ROS2 parameters."""
        node = self.connector.node
        param_prefix = PCL_DETECTION_PARAM_PREFIX

        # Declare required parameters
        params = [
            f"{param_prefix}.target_frame",
            f"{param_prefix}.source_frame",
            f"{param_prefix}.camera_topic",
            f"{param_prefix}.depth_topic",
            f"{param_prefix}.camera_info_topic",
        ]

        for param_name in params:
            if not node.has_parameter(param_name):
                raise ValueError(
                    f"Required parameter '{param_name}' must be set before initializing GetObjectGrippingPointsTool"
                )

        # Load parameters
        self.target_frame = node.get_parameter(f"{param_prefix}.target_frame").value
        self.source_frame = node.get_parameter(f"{param_prefix}.source_frame").value
        self.camera_topic = node.get_parameter(f"{param_prefix}.camera_topic").value
        self.depth_topic = node.get_parameter(f"{param_prefix}.depth_topic").value
        self.camera_info_topic = node.get_parameter(
            f"{param_prefix}.camera_info_topic"
        ).value

        # timeout for gripping point detection
        self.timeout_sec = (
            node.get_parameter(f"{param_prefix}.timeout_sec").value
            if node.has_parameter(f"{param_prefix}.timeout_sec")
            else 10.0
        )

        # conversion ratio for point cloud from segmentation
        self.conversion_ratio = (
            node.get_parameter(f"{param_prefix}.conversion_ratio").value
            if node.has_parameter(f"{param_prefix}.conversion_ratio")
            else 0.001
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
            conversion_ratio=self.conversion_ratio,
            config=self.segmentation_config,
        )
        self.gripping_point_estimator = GrippingPointEstimator(
            config=self.estimator_config
        )
        self.point_cloud_filter = PointCloudFilter(config=self.filter_config)

    def _run(self, object_name: str) -> str:
        @timeout(
            self.timeout_sec,
            f"Gripping point detection for object '{object_name}' exceeded {self.timeout_sec} seconds",
        )
        def _run_with_timeout():
            pcl = self.point_cloud_from_segmentation.run(object_name)
            if len(pcl) == 0:
                return f"No {object_name}s detected."

            pcl_filtered = self.point_cloud_filter.run(pcl)
            if len(pcl_filtered) == 0:
                return f"No {object_name}s detected after applying filtering"

            gripping_points = self.gripping_point_estimator.run(pcl_filtered)

            message = ""
            if len(gripping_points) == 0:
                message += f"No gripping point found for the object {object_name}\n"
            elif len(gripping_points) == 1:
                message += f"The gripping point of the object {object_name} is {gripping_points[0]}\n"
            else:
                message += (
                    f"Multiple gripping points found for the object {object_name}\n"
                )

            for i, gp in enumerate(gripping_points):
                message += (
                    f"The gripping point of the object {i + 1} {object_name} is {gp}\n"
                )

            return message

        try:
            return _run_with_timeout()
        except RaiTimeoutError as e:
            self.connector.node.get_logger().warning(f"Timeout: {e}")
            return f"Timeout: Gripping point detection for object '{object_name}' exceeded {self.timeout_sec} seconds"
        except Exception:
            raise
