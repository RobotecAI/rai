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

from typing import Optional, Type

import rclpy
import sensor_msgs.msg
from pydantic import Field
from rai_open_set_vision import GDINO_SERVICE_NAME
from rclpy import Future
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)

from rai.node import RaiBaseNode
from rai.tools.ros import Ros2BaseInput, Ros2BaseTool
from rai.tools.ros.utils import convert_ros_img_to_base64
from rai_interfaces.srv import RAIGroundedSam, RAIGroundingDino

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
    object_name: str = Field(
        ..., description="Natural language names of the object to grab"
    )


# --------------------- Tools ---------------------
class GetSegmentationTool(Ros2BaseTool):
    node: RaiBaseNode = Field(..., exclude=True)

    name: str = ""
    description: str = ""

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

    def _get_image_message(self, topic: str) -> sensor_msgs.msg.Image:
        msg = self.node.get_raw_message_from_topic(topic)
        if type(msg) is sensor_msgs.msg.Image:
            return msg
        else:
            raise Exception("Received wrong message")

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_name: str
    ) -> Future:
        cli = self.node.create_client(RAIGroundingDino, GDINO_SERVICE_NAME)
        while not cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("service not available, waiting again...")
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = object_name
        req.box_threshold = 0.4  # TODO make this somehow configurable
        req.text_threshold = 0.4

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
