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

from typing import Any, List, Optional, Type

import cv2
import numpy as np
import rclpy
import rclpy.qos
import sensor_msgs.msg
from pydantic import Field
from rai_open_set_vision import GDINO_SERVICE_NAME
from rclpy import Future
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)

from rai.tools.ros import Ros2BaseInput
from rai.tools.ros.native_actions import Ros2BaseActionTool
from rai.tools.ros.utils import convert_ros_img_to_base64, convert_ros_img_to_ndarray
from rai.tools.utils import wait_for_message
from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino

# from rclpy.wait_for_message import wait_for_message


# --------------------- Inputs ---------------------


class GetSegmentationInput(Ros2BaseInput):
    camera_topic: str = Field(
        ...,
        description="Ros2 topic for the camera image containing image to run detection on.",
    )
    object_name: str = Field(
        ..., description="Natural language names of the object to grab"
    )


class GetGrabbingPointInput(Ros2BaseInput):
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
class GetSegmentationTool(Ros2BaseActionTool):
    name: str = ""
    description: str = ""

    box_threshold: float = Field(default=0.35, description="Box threshold for GDINO")
    text_threshold: float = Field(default=0.45, description="Text threshold for GDINO")

    args_schema: Type[GetSegmentationInput] = GetSegmentationInput

    def _get_gdino_response(
        self, future: Future
    ) -> Optional[RAIGroundingDino.Response]:
        rclpy.spin_once(self.node)
        if future.done():
            try:
                response = future.result()
            except Exception as e:
                self.node.get_logger().info("Service call failed %r" % (e,))
                raise Exception("Service call failed %r" % (e,))
            else:
                assert response is not None
                return response
        return None

    def _get_gsam_response(self, future: Future) -> Optional[RAIGroundedSam.Response]:
        rclpy.spin_once(self.node)
        if future.done():
            try:
                response = future.result()
            except Exception as e:
                self.node.get_logger().info("Service call failed %r" % (e,))
                raise Exception("Service call failed %r" % (e,))
            else:
                assert response is not None
                return response
        return None

    def get_img_from_topic(self, topic: str, timeout_sec: int = 10):
        success, msg = wait_for_message(
            sensor_msgs.msg.Image,
            self.node,
            topic,
            qos_profile=rclpy.qos.qos_profile_sensor_data,
            time_to_wait=timeout_sec,
        )

        if success:
            self.node.get_logger().info(f"Received message of type from topic {topic}")
            return msg
        else:
            error = f"No message received in {timeout_sec} seconds from topic {topic}"
            self.node.get_logger().error(error)
            return error

    def _get_image_message(self, topic: str) -> sensor_msgs.msg.Image:
        msg = self.get_img_from_topic(topic)
        if type(msg) is sensor_msgs.msg.Image:
            return msg
        else:
            raise Exception(f"Received wrong message: {type(msg)}")

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_name: str
    ) -> Future:
        cli = self.node.create_client(RAIGroundingDino, GDINO_SERVICE_NAME)
        while not cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("service not available, waiting again...")
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
        cli = self.node.create_client(RAIGroundedSam, "grounded_sam_segment")
        while not cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("service not available, waiting again...")
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
        logger = self.node.get_logger()
        try:
            conversion_ratio = self.node.get_parameter("conversion_ratio").value
            if not isinstance(conversion_ratio, float):
                logger.error(
                    f"Parametr conversion_ratio was set badly: {type(conversion_ratio)}: {conversion_ratio} expected float. Using default value 0.001"
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


def depth_to_point_cloud(depth_image, fx, fy, cx, cy):
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


class GetGrabbingPointTool(GetSegmentationTool):

    name: str = "GetGrabbingPointTool"
    description: str = "Get the grabbing point of an object"
    pcd: List[Any] = []

    args_schema: Type[GetGrabbingPointInput] = GetGrabbingPointInput

    def _get_camera_info_message(self, topic: str) -> sensor_msgs.msg.CameraInfo:
        self.node.get_logger().info(f"Waiting for CameraInfo from topic {topic}")
        success, msg = wait_for_message(
            sensor_msgs.msg.CameraInfo,
            self.node,
            topic,
            qos_profile=rclpy.qos.qos_profile_sensor_data,
            time_to_wait=3,
        )
        print(msg)

        if success:
            self.node.get_logger().info(f"Received message of type from topic {topic}")
            return msg
        else:
            error = f"No message received in 3 seconds from topic {topic}"
            self.node.get_logger().error(error)
            raise Exception(
                "Failed to receive correct CameraInfo message after 3 attempts"
            )

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
        intrinsic,
    ):
        mask = convert_ros_img_to_ndarray(mask_msg)
        binary_mask = np.where(mask == 255, 1, 0)
        depth = convert_ros_img_to_ndarray(depth_msg)
        masked_depth_image = np.zeros_like(depth, dtype=np.float32)
        masked_depth_image[binary_mask == 1] = depth[binary_mask == 1]

        pcd = depth_to_point_cloud(
            masked_depth_image, intrinsic[0], intrinsic[1], intrinsic[2], intrinsic[3]
        )

        # TODO: Filter out outliers
        points = pcd

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
        # TODO : change offset to be dependant on the height of the object
        centroid[2] += 0.1  # Added a small offset to prevent gripper collision
        return centroid, gripper_rotation

    def _run(
        self,
        camera_topic: str,
        depth_topic: str,
        camera_info_topic: str,
        object_name: str,
    ):
        camera_info = self._get_camera_info_message(camera_info_topic)
        self.logger.info("Received camera info")
        camera_img_msg = self.get_img_from_topic(camera_topic)
        self.logger.info("Received camera image")
        depth_msg = self.get_img_from_topic(depth_topic)
        self.logger.info("Received depth image")
        intrinsic = self._get_intrinsic_from_camera_info(camera_info)
        self.logger.info("Received camera intrinsic")

        future = self._call_gdino_node(camera_img_msg, object_name)
        logger = self.node.get_logger()
        try:
            conversion_ratio = self.node.get_parameter("conversion_ratio").value
            if not isinstance(conversion_ratio, float):
                logger.error(
                    f"Parametr conversion_ratio was set badly: {type(conversion_ratio)}: {conversion_ratio} expected float. Using default value 0.001"
                )
                conversion_ratio = 0.001
        except (ParameterUninitializedException, ParameterNotDeclaredException):
            logger.warning(
                "Parameter conversion_ratio not found in node, using default value: 0.001"
            )
            conversion_ratio = 0.001
        resolved = None
        self.logger.info("Waiting gdino response")
        while rclpy.ok():
            resolved = self._get_gdino_response(future)
            if resolved is not None:
                break

        assert resolved is not None
        self.logger.info("Got gdino response")
        future = self._call_gsam_node(camera_img_msg, resolved)
        self.logger.info("Waiting gsam response")

        ret = []
        while rclpy.ok():
            resolved = self._get_gsam_response(future)
            if resolved is not None:
                for img_msg in resolved.masks:
                    ret.append(convert_ros_img_to_base64(img_msg))
                break
        assert resolved is not None

        self.logger.info("Got gsam response")
        rets = []
        for mask_msg in resolved.masks:
            rets.append(self._process_mask(mask_msg, depth_msg, intrinsic))

        return rets
