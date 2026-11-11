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

"""Segmentation and grabbing point tools.

Note: Tools are coupled to GroundingDINO and Grounded SAM interfaces
(RAIGroundingDino/RAIGroundedSam service types, box_threshold/text_threshold parameters).

Model support: Currently model-specific (GroundingDINO + Grounded SAM). For future
model-agnostic support, generic RAIDetection/RAISegmentation service interfaces would
be needed. Alternatively, model-specific tools can be created for each model.
"""

from typing import Any, Literal, Sequence, Type

import cv2
import numpy as np
import sensor_msgs.msg
from pydantic import BaseModel, Field
from rai.communication.ros2.api import (
    convert_ros_img_to_base64,
    convert_ros_img_to_ndarray,
)
from sensor_msgs.msg import CameraInfo

from rai_perception.algorithms.point_cloud import depth_to_point_cloud
from rai_perception.components.perception_utils import get_camera_intrinsics
from rai_perception.components.service_utils import SEGMENTATION_SERVICE, segment
from rai_perception.tools.base import PerceptionTool


class GetSegmentationInput(BaseModel):
    camera_topic: str = Field(
        ...,
        description="Ros2 topic for the camera image containing image to run detection on.",
    )
    object_name: str = Field(
        ..., description="Natural language names of the object to grab"
    )


class GetGrabbingPointInput(BaseModel):
    camera_topic: str = Field(
        ...,
        description="Ros2 topic for the camera image containing image to run detection on.",
    )
    depth_topic: str = Field(
        ...,
        description="Ros2 topic for the depth image containing data to run distance calculations on",
    )
    camera_info_topic: str = Field(
        ...,
        description="Ros2 topic for the camera info to get the camera intrinsic from",
    )
    object_name: str = Field(
        ..., description="Natural language names of the object to grab"
    )


class SegmentationBaseTool(PerceptionTool):
    """Detection, then segmentation of the detected boxes."""

    target_fields = {
        "detection_service_name": "service",
        "segmentation_service_name": "service",
    }
    segmentation_service_name: str = Field(default=SEGMENTATION_SERVICE, exclude=True)

    async def _masks(self, image: sensor_msgs.msg.Image, object_name: str) -> list[Any]:
        detections = (await self.detect(image, object_name)).detections
        response = await segment(
            self.connector,
            image,
            detections,
            service=self.segmentation_service_name,
            timeout=self.service_timeout,
        )
        return list(response.masks)


class GetSegmentationTool(SegmentationBaseTool):
    name: str = "GetSegmentationTool"
    description: str = "Get the segmentation masks of an object in a camera image"
    target_args = {"camera_topic": "topic"}
    response_format: Literal["content", "content_and_artifact"] = "content_and_artifact"
    args_schema: Type[GetSegmentationInput] = GetSegmentationInput

    async def _acall(self, camera_topic: str, object_name: str) -> Any:
        masks = await self._masks(await self.receive(camera_topic), object_name)
        return "", {"segmentations": [convert_ros_img_to_base64(m) for m in masks]}


class GetGrabbingPointTool(SegmentationBaseTool):
    """Centroid and gripper rotation of each instance of an object; returns a
    list of (centroid, rotation) tuples in the camera frame."""

    name: str = "GetGrabbingPointTool"
    description: str = "Get the grabbing point of an object"
    target_args = {
        "camera_topic": "topic",
        "depth_topic": "topic",
        "camera_info_topic": "topic",
    }
    args_schema: Type[GetGrabbingPointInput] = GetGrabbingPointInput
    conversion_ratio: float = Field(
        default=0.001, exclude=True, description="Depth units to meters"
    )

    def _process_mask(
        self,
        mask_msg: sensor_msgs.msg.Image,
        depth_msg: sensor_msgs.msg.Image,
        intrinsic: Sequence[float],
        depth_to_meters_ratio: float,
    ):
        mask = convert_ros_img_to_ndarray(mask_msg)
        binary_mask = np.where(mask == 255, 1, 0)
        depth = convert_ros_img_to_ndarray(depth_msg)
        masked_depth_image = np.zeros_like(depth, dtype=np.float32)
        masked_depth_image[binary_mask == 1] = depth[binary_mask == 1]
        masked_depth_image = masked_depth_image * depth_to_meters_ratio

        pcd = depth_to_point_cloud(
            masked_depth_image, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        )

        # TODO: Filter out outliers
        points = pcd

        # The grasp height and rotation below are adapted from tabletop-handybot by Yifei Cheng,
        # Apache License 2.0 (as its package.xml and setup.py declare); see the NOTICE of rai_perception.
        # https://github.com/ycheng517/tabletop-handybot/blob/6d401e577e41ea86529d091b406fbfc936f37a8d/tabletop_handybot/tabletop_handybot/tabletop_handybot_node.py#L413-L424
        grasp_z = points[:, 2].max()
        near_grasp_z_points = points[points[:, 2] > grasp_z - 0.008]
        xy_points = near_grasp_z_points[:, :2]
        xy_points = xy_points.astype(np.float32)
        _, dimensions, theta = cv2.minAreaRect(xy_points)

        gripper_rotation = theta
        # NOTE  - estimated dimentsion from the RGBDCamera5 not very precise, what may cause not desired rotation
        if dimensions[0] > dimensions[1]:
            gripper_rotation -= 90
        if gripper_rotation < -90:
            gripper_rotation += 180
        elif gripper_rotation > 90:
            gripper_rotation -= 180

        # Calculate full 3D centroid for OBJECT
        centroid = np.mean(points, axis=0)
        return centroid, gripper_rotation

    async def _acall(
        self,
        camera_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        object_name: str,
    ) -> list[tuple[Any, float]]:
        image = await self.receive(camera_topic)
        depth = await self.receive(depth_topic)
        camera_info = await self.receive(camera_info_topic, CameraInfo)
        intrinsic = get_camera_intrinsics(camera_info)
        return [
            self._process_mask(mask, depth, intrinsic, self.conversion_ratio)
            for mask in await self._masks(image, object_name)
        ]
