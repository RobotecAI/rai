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

from typing import Any, List, Optional, Sequence, Type

import cv2
import numpy as np
import rclpy
import sensor_msgs.msg
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai.communication.ros2.api import (
    convert_ros_img_to_base64,
    convert_ros_img_to_ndarray,
)
from rai.communication.ros2.connectors import ROS2Connector
from rai.communication.ros2.ros_async import get_future_result
from rclpy import Future
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)

from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino
from rai_perception import GDINO_SERVICE_NAME

# --------------------- Inputs ---------------------


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


# --------------------- Tools ---------------------
class GetSegmentationTool:
    connector: ROS2Connector = Field(..., exclude=True)

    name: str = ""
    description: str = ""

    box_threshold: float = Field(default=0.35, description="Box threshold for GDINO")
    text_threshold: float = Field(default=0.45, description="Text threshold for GDINO")

    args_schema: Type[GetSegmentationInput] = GetSegmentationInput

    def _get_gdino_response(
        self, future: Future
    ) -> Optional[RAIGroundingDino.Response]:
        return get_future_result(future)

    def _get_gsam_response(self, future: Future) -> Optional[RAIGroundedSam.Response]:
        return get_future_result(future)

    def _get_image_message(self, topic: str) -> sensor_msgs.msg.Image:
        msg = self.connector.receive_message(topic).payload
        if type(msg) is sensor_msgs.msg.Image:
            return msg
        else:
            raise Exception("Received wrong message")

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_name: str
    ) -> Future:
        cli = self.connector.node.create_client(RAIGroundingDino, GDINO_SERVICE_NAME)
        while not cli.wait_for_service(timeout_sec=1.0):
            self.connector.node.get_logger().info(
                f"service {GDINO_SERVICE_NAME} not available, waiting again..."
            )
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = object_name
        req.box_threshold = self.box_threshold
        req.text_threshold = self.text_threshold

        future = cli.call_async(req)
        return future

    def _call_gsam_node(
        self, camera_img_message: sensor_msgs.msg.Image, data: RAIGroundingDino.Response
    ):
        cli = self.connector.node.create_client(RAIGroundedSam, "grounded_sam_segment")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.connector.node.get_logger().info(
                "service grounded_sam_segment not available, waiting again..."
            )
        req = RAIGroundedSam.Request()
        req.detections = data.detections
        req.source_img = camera_img_message
        future = cli.call_async(req)

        return future

    def _run(
        self,
        camera_topic: str,
        object_name: str,
    ):
        camera_img_msg = self._get_image_message(camera_topic)

        future = self._call_gdino_node(camera_img_msg, object_name)
        logger = self.connector.node.get_logger()
        try:
            conversion_ratio = self.connector.node.get_parameter(
                "conversion_ratio"
            ).value
            if not isinstance(conversion_ratio, float):
                logger.error(
                    f"Parameter conversion_ratio was set badly: {type(conversion_ratio)}: {conversion_ratio} expected float. Using default value 0.001"
                )
                conversion_ratio = 0.001
        except (ParameterUninitializedException, ParameterNotDeclaredException):
            logger.warning(
                "Parameter conversion_ratio not found in node, using default value: 0.001"
            )
            conversion_ratio = 0.001
        resolved = None
        while rclpy.ok():
            resolved = self._get_gdino_response(future)
            if resolved is not None:
                break

        assert resolved is not None
        future = self._call_gsam_node(camera_img_msg, resolved)

        ret = []
        while rclpy.ok():
            resolved = self._get_gsam_response(future)
            if resolved is not None:
                for img_msg in resolved.masks:
                    ret.append(convert_ros_img_to_base64(img_msg))
                break
        return "", {"segmentations": ret}


def depth_to_point_cloud(
    depth_image: np.ndarray, fx: float, fy: float, cx: float, cy: float
):
    height, width = depth_image.shape

    # Create grid of pixel coordinates
    x = np.arange(width)
    y = np.arange(height)
    x_grid, y_grid = np.meshgrid(x, y)

    # Calculate 3D coordinates
    z = depth_image
    x = (x_grid - cx) * z / fx
    y = (y_grid - cy) * z / fy

    # Stack the coordinates
    points = np.stack((x, y, z), axis=-1)

    # Reshape to a list of points
    points = points.reshape(-1, 3)

    # Remove points with zero depth
    points = points[points[:, 2] > 0]

    return points


class GetGrabbingPointTool(BaseTool):
    connector: ROS2Connector = Field(..., exclude=True)

    name: str = "GetGrabbingPointTool"
    description: str = "Get the grabbing point of an object"
    pcd: List[Any] = []

    args_schema: Type[GetGrabbingPointInput] = GetGrabbingPointInput
    box_threshold: float = Field(default=0.35, description="Box threshold for GDINO")
    text_threshold: float = Field(default=0.45, description="Text threshold for GDINO")

    def _get_gdino_response(
        self, future: Future
    ) -> Optional[RAIGroundingDino.Response]:
        return get_future_result(future)

    def _get_gsam_response(self, future: Future) -> Optional[RAIGroundedSam.Response]:
        return get_future_result(future)

    def _get_image_message(self, topic: str) -> sensor_msgs.msg.Image:
        msg = self.connector.receive_message(topic).payload
        if type(msg) is sensor_msgs.msg.Image:
            return msg
        else:
            raise Exception("Received wrong message")

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_name: str
    ) -> Future:
        cli = self.connector.node.create_client(RAIGroundingDino, GDINO_SERVICE_NAME)
        while not cli.wait_for_service(timeout_sec=1.0):
            self.connector.node.get_logger().info(
                "service not available, waiting again..."
            )
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = object_name
        req.box_threshold = self.box_threshold
        req.text_threshold = self.text_threshold

        future = cli.call_async(req)
        return future

    def _call_gsam_node(
        self, camera_img_message: sensor_msgs.msg.Image, data: RAIGroundingDino.Response
    ):
        cli = self.connector.node.create_client(RAIGroundedSam, "grounded_sam_segment")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.connector.node.get_logger().info(
                "service not available, waiting again..."
            )
        req = RAIGroundedSam.Request()
        req.detections = data.detections
        req.source_img = camera_img_message
        future = cli.call_async(req)

        return future

    def _get_camera_info_message(self, topic: str) -> sensor_msgs.msg.CameraInfo:
        for _ in range(3):
            msg = self.connector.receive_message(topic, timeout_sec=3.0).payload
            if isinstance(msg, sensor_msgs.msg.CameraInfo):
                return msg
            self.connector.node.get_logger().warn(
                "Received wrong message type. Retrying..."
            )

        raise Exception("Failed to receive correct CameraInfo message after 3 attempts")

    def _get_intrinsic_from_camera_info(self, camera_info: sensor_msgs.msg.CameraInfo):
        """Extract camera intrinsic parameters from the CameraInfo message."""

        fx = camera_info.k[0]  # Focal length in x-axis
        fy = camera_info.k[4]  # Focal length in y-axis
        cx = camera_info.k[2]  # Principal point x
        cy = camera_info.k[5]  # Principal point y

        return fx, fy, cx, cy

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

    def _run(
        self,
        camera_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        object_name: str,
    ):
        camera_img_msg = self.connector.receive_message(camera_topic).payload
        depth_msg = self.connector.receive_message(depth_topic).payload
        camera_info = self._get_camera_info_message(camera_info_topic)

        intrinsic = self._get_intrinsic_from_camera_info(camera_info)

        future = self._call_gdino_node(camera_img_msg, object_name)
        logger = self.connector.node.get_logger()
        try:
            conversion_ratio = self.connector.node.get_parameter(
                "conversion_ratio"
            ).value
            if not isinstance(conversion_ratio, float):
                logger.error(
                    f"Parameter conversion_ratio was set badly: {type(conversion_ratio)}: {conversion_ratio} expected float. Using default value 0.001"
                )
                conversion_ratio = 0.001
        except (ParameterUninitializedException, ParameterNotDeclaredException):
            logger.warning(
                "Parameter conversion_ratio not found in node, using default value: 0.001"
            )
            conversion_ratio = 0.001
        resolved = None

        resolved = get_future_result(future)

        assert resolved is not None
        future = self._call_gsam_node(camera_img_msg, resolved)

        ret = []
        resolved = get_future_result(future)
        if resolved is not None:
            for img_msg in resolved.masks:
                ret.append(convert_ros_img_to_base64(img_msg))
        assert resolved is not None
        rets = []
        for mask_msg in resolved.masks:
            rets.append(
                self._process_mask(
                    mask_msg,
                    depth_msg,
                    intrinsic,
                    depth_to_meters_ratio=conversion_ratio,
                )
            )

        return rets
