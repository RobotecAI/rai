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

from typing import List, NamedTuple, Optional, Type

import numpy as np
import rclpy
import sensor_msgs.msg
from pydantic import Field
from pydantic import BaseModel
from rai_grounding_dino import GDINO_SERVICE_NAME
from rclpy import Future
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)

from rai.node import RaiBaseNode
from rai.tools.ros import Ros2BaseInput, Ros2BaseTool
from rai.tools.ros.utils import convert_ros_img_to_ndarray
from rai_interfaces.srv import RAIGroundingDino


# --------------------- Inputs ---------------------
class Ros2GetDetectionInput(Ros2BaseInput):
    camera_topic: str = Field(
        ...,
        description="Ros2 topic for the camera image containing image to run detection on.",
    )
    object_names: list[str] = Field(
        ..., description="Natural language names of the objects to detect"
    )


class GetDistanceToObjectsInput(Ros2BaseInput):
    camera_topic: str = Field(
        ...,
        description="Ros2 topic for the camera image containing image to run detection on.",
    )
    depth_topic: str = Field(
        ...,
        description="Ros2 topic for the depth image containing data to run distance calculations on",
    )
    object_names: list[str] = Field(
        ..., description="Natural language names of the objects to detect"
    )


# -------------------- Utils ----------------------


class BoundingBox(BaseModel):
    x_center: float
    y_center: float
    width: float
    height: float


class DetectionData(BaseModel):
    class_name: str
    confidence: float
    bbox: BoundingBox


class DistanceMeasurement(NamedTuple):
    name: str
    distance: float


# --------------------- Tools ---------------------
class GroundingDinoBaseTool(Ros2BaseTool):
    node: RaiBaseNode = Field(..., exclude=True, required=True)

    def _spin(self, future: Future) -> Optional[RAIGroundingDino.Response]:
        rclpy.spin_once(self.node)
        if future.done():
            try:
                response = future.result()
            except Exception as e:
                self.node.get_logger().info("Service call failed %r" % (e,))
                raise Exception("Service call failed %r" % (e,))
            else:
                assert response is not None
                self.node.get_logger().info(f"{response.detections}")
                return response
        return None

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_names: list[str]
    ) -> Future:
        cli = self.node.create_client(RAIGroundingDino, GDINO_SERVICE_NAME)
        while not cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info("service not available, waiting again...")
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = " , ".join(object_names)
        req.box_threshold = 0.4  # TODO make this somehow configurable
        req.text_threshold = 0.4

        future = cli.call_async(req)
        return future

    def _get_image_message(self, topic: str) -> sensor_msgs.msg.Image:
        msg = self.node.get_raw_message_from_topic(topic)
        if type(msg) is sensor_msgs.msg.Image:
            return msg
        else:
            raise Exception("Received wrong message")

    def _parse_detection_array(
        self, detection_response: RAIGroundingDino.Response
    ) -> list[DetectionData]:
        detected = []
        for detection in detection_response.detections.detections:
            class_name = detection.results[0].hypothesis.class_id
            confidence = detection.results[0].hypothesis.score
            bbox = BoundingBox(
                x_center=detection.bbox.center.position.x,
                y_center=detection.bbox.center.position.y,
                width=detection.bbox.size_x,
                height=detection.bbox.size_y,
            )
            detected.append(
                DetectionData(class_name=class_name, confidence=confidence, bbox=bbox)
            )
        return detected


class GetDetectionTool(GroundingDinoBaseTool):
    name: str = "GetDetectionTool"
    description: str = (
        "A tool for detecting specified objects using a ros2 action. The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result."
    )

    args_schema: Type[Ros2GetDetectionInput] = Ros2GetDetectionInput

    def _run(
        self,
        camera_topic: str,
        object_names: list[str],
    ):
        camera_img_msg = self._get_image_message(camera_topic)
        future = self._call_gdino_node(camera_img_msg, object_names)

        while rclpy.ok():
            resolved = self._spin(future)
            if resolved is not None:
                detected = self._parse_detection_array(resolved)
                names = ", ".join([det.class_name for det in detected])
                return f"I have detected the following items in the picture {names or 'None'}"

        return "Failed to get detection"


class GetDistanceToObjectsTool(GroundingDinoBaseTool):
    name: str = "GetDistanceToObjectsTool"
    description: str = (
        "A tool for calculating distance to specified objects using a ros2 action. The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result."
    )

    args_schema: Type[GetDistanceToObjectsInput] = GetDistanceToObjectsInput

    def _get_distance_from_detections(
        self,
        depth_img: sensor_msgs.msg.Image,
        detections: list[DetectionData],
        sigma_threshold: float = 1.0,
        conversion_ratio: float = 0.001,
    ) -> List[DistanceMeasurement]:
        depth_arr = convert_ros_img_to_ndarray(depth_img)
        ret = []
        for detection in detections:
            x_min = int(detection.bbox.x_center - (0.5 * detection.bbox.width))
            x_max = int(detection.bbox.x_center + (0.5 * detection.bbox.width))
            y_min = int(detection.bbox.y_center - (0.5 * detection.bbox.height))
            y_max = int(detection.bbox.y_center + (0.5 * detection.bbox.height))
            roi = depth_arr[x_min:x_max, y_min:y_max]
            mean = np.mean(roi)
            std_dev = np.std(roi)

            if std_dev != 0:
                z_scores = (roi - mean) / std_dev
                mask = np.abs(z_scores) < sigma_threshold
                ret.append(
                    (detection.class_name, np.mean(roi[mask]) * conversion_ratio)
                )
            else:
                ret.append((detection.class_name, mean * conversion_ratio))
        return ret

    def _run(
        self,
        camera_topic: str,
        depth_topic: str,
        object_names: list[str],
    ):
        camera_img_msg = self._get_image_message(camera_topic)
        depth_img_msg = self._get_image_message(depth_topic)
        future = self._call_gdino_node(camera_img_msg, object_names)
        logger = self.node.get_logger()

        try:
            threshold = self.node.get_parameter("outlier_sigma_threshold").value
            if not isinstance(threshold, float):
                logger.error(
                    f"Parametr outlier_sigma_threshold was set badly: {type(threshold)}: {threshold} expected float. Using default value 1.0"
                )
                threshold = 1.0
        except (ParameterUninitializedException, ParameterNotDeclaredException):
            logger.warning(
                "Parameter outlier_sigma_threshold not found in node, using default value: 1.0"
            )
            threshold = 1.0

        try:
            conversion_ratio = self.node.get_parameter("conversion_ratio").value
            if not isinstance(conversion_ratio, float):
                logger.error(
                    f"Parametr conversion_ratio was set badly: {type(threshold)}: {threshold} expected float. Using default value 0.001"
                )
                conversion_ratio = 0.001
        except (ParameterUninitializedException, ParameterNotDeclaredException):
            logger.warning(
                "Parameter conversion_ratio not found in node, using default value: 0.001"
            )
            conversion_ratio = 0.001
        while rclpy.ok():
            resolved = self._spin(future)
            if resolved is not None:
                detected = self._parse_detection_array(resolved)
                measurements = self._get_distance_from_detections(
                    depth_img_msg, detected, threshold, conversion_ratio
                )
                measurement_string = ", ".join(
                    [
                        f"{measurement[0]}: {measurement[1]:.2f}m away"
                        for measurement in measurements
                    ]
                )
                return f"I have detected the following items in the picture {measurement_string or 'no objects'}"
        return "Failed"
