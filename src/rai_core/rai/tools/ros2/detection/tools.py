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
    PointCloudFilter,
    PointCloudFromSegmentation,
)


class GetGrippingPointToolInput(BaseModel):
    object_name: str = Field(
        ...,
        description="The name of the object to get the gripping point of e.g. 'box', 'apple', 'screwdriver'",
    )


# TODO(maciejmajek): Configuration system configurable with namespacing
# TODO(juliajia): Question for Maciej: for comments above on configuration system with namespacing, can you provide an use case for this?
class GetGrippingPointTool(BaseROS2Tool):
    name: str = "get_gripping_point"
    description: str = "Get gripping points for specified object/objects. Returns 3D coordinates where a robot gripper can grasp the object."

    target_frame: str
    source_frame: str
    camera_topic: str  # rgb camera topic
    depth_topic: str
    camera_info_topic: str  # rgb camera info topic

    gripping_point_estimator: GrippingPointEstimator
    point_cloud_filter: PointCloudFilter

    # Auto-initialized in model_post_init
    point_cloud_from_segmentation: Optional[PointCloudFromSegmentation] = None

    timeout_sec: float = Field(
        default=10.0, description="Timeout in seconds to get the gripping point"
    )

    args_schema: Type[GetGrippingPointToolInput] = GetGrippingPointToolInput

    def model_post_init(self, __context: Any) -> None:
        """Initialize PointCloudFromSegmentation with the provided camera parameters."""
        self.point_cloud_from_segmentation = PointCloudFromSegmentation(
            connector=self.connector,
            camera_topic=self.camera_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            source_frame=self.source_frame,
            target_frame=self.target_frame,
        )

    def _run(self, object_name: str) -> str:
        # this will be not work in agent scenario because signal need to be run in main thread, comment out for now
        # @timeout(
        #     self.timeout_sec,
        #     f"Gripping point detection for object '{object_name}' exceeded {self.timeout_sec} seconds",
        # )
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
        except Exception as e:
            if "timed out" in str(e).lower():
                return f"Timeout: Gripping point detection for object '{object_name}' exceeded {self.timeout_sec} seconds"
            raise
