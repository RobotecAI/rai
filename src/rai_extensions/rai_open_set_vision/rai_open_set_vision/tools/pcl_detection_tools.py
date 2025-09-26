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

from typing import Any, List, Optional, Type

import numpy as np
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


class GetGrippingPointToolInput(BaseModel):
    object_name: str = Field(
        ...,
        description="The name of the object to get the gripping point of e.g. 'box', 'apple', 'screwdriver'",
    )
    timeout_sec: Optional[float] = Field(
        default=None,
        description="Override timeout in seconds. If not provided, uses tool's default timeout.",
    )
    conversion_ratio: Optional[float] = Field(
        default=None,
        description="Override conversion ratio for depth to meters. If not provided, uses tool's default.",
    )


class GetGrippingPointTool(BaseROS2Tool):
    name: str = "get_gripping_point"
    description: str = "Get gripping points for specified object/objects. Returns 3D coordinates where a robot gripper can grasp the object."

    # Configuration for PCL components
    segmentation_config: PointCloudFromSegmentationConfig
    estimator_config: GrippingPointEstimatorConfig
    filter_config: PointCloudFilterConfig

    # Required parameters
    target_frame: str
    source_frame: str
    camera_topic: str
    depth_topic: str
    camera_info_topic: str
    timeout_sec: float = 10.0  # Default timeout
    conversion_ratio: float = 0.001  # Default conversion ratio

    # Components initialized in model_post_init
    gripping_point_estimator: Optional[GrippingPointEstimator] = None
    point_cloud_filter: Optional[PointCloudFilter] = None
    point_cloud_from_segmentation: Optional[PointCloudFromSegmentation] = None

    args_schema: Type[GetGrippingPointToolInput] = GetGrippingPointToolInput

    def model_post_init(self, __context: Any) -> None:
        """Initialize tool components."""
        self._initialize_components()

    def _initialize_components(self) -> None:
        """Initialize PCL components with provided parameters."""
        self.point_cloud_from_segmentation = PointCloudFromSegmentation(
            connector=self.connector,
            camera_topic=self.camera_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            source_frame=self.source_frame,
            target_frame=self.target_frame,
            config=self.segmentation_config,
            conversion_ratio=self.conversion_ratio,
        )
        self.gripping_point_estimator = GrippingPointEstimator(
            config=self.estimator_config
        )
        self.point_cloud_filter = PointCloudFilter(config=self.filter_config)

    def _run(
        self,
        object_name: str,
        timeout_sec: Optional[float] = None,
        conversion_ratio: Optional[float] = None,
    ) -> List[np.ndarray]:
        """Run gripping point detection and return raw gripping points."""

        # Use runtime parameters if provided, otherwise use defaults
        effective_timeout = timeout_sec if timeout_sec is not None else self.timeout_sec
        effective_conversion_ratio = (
            conversion_ratio if conversion_ratio is not None else self.conversion_ratio
        )

        # Update conversion ratio if different from current
        if effective_conversion_ratio != self.conversion_ratio:
            self.point_cloud_from_segmentation.conversion_ratio = (
                effective_conversion_ratio
            )

        @timeout(
            effective_timeout,
            f"Gripping point detection for object '{object_name}' exceeded {effective_timeout} seconds",
        )
        def _run_with_timeout():
            pcl = self.point_cloud_from_segmentation.run(object_name)
            if len(pcl) == 0:
                return []

            pcl = self.point_cloud_filter.run(pcl)
            gps = self.gripping_point_estimator.run(pcl)
            return gps

        try:
            return _run_with_timeout()
        except RaiTimeoutError as e:
            # Log the timeout but still raise it
            self.connector.node.get_logger().warning(f"Timeout: {e}")
            raise  # Let caller decide how to handle
        except Exception:
            raise
