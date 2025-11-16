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

from typing import List, NamedTuple, Type

import numpy as np
import sensor_msgs.msg
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field
from rai.communication.ros2 import ROS2Connector
from rai.communication.ros2.api import convert_ros_img_to_ndarray
from rai.communication.ros2.ros_async import get_future_result
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)
from rclpy.task import Future

from rai_interfaces.srv import RAIGroundingDino
from rai_perception import GDINO_SERVICE_NAME


# --------------------- Inputs ---------------------
class Ros2GetDetectionInput(BaseModel):
    camera_topic: str = Field(
        ...,
        description="Ros2 topic for the camera image containing image to run detection on.",
    )
    object_names: list[str] = Field(
        ..., description="Natural language names of the objects to detect"
    )


class GetDistanceToObjectsInput(BaseModel):
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
class GroundingDinoBaseTool(BaseTool):
    connector: ROS2Connector = Field(..., exclude=True)

    box_threshold: float = Field(default=0.35, description="Box threshold for GDINO")
    text_threshold: float = Field(default=0.45, description="Text threshold for GDINO")

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_names: list[str]
    ) -> Future:
        cli = self.connector.node.create_client(RAIGroundingDino, GDINO_SERVICE_NAME)
        while not cli.wait_for_service(timeout_sec=1.0):
            self.node.get_logger().info(
                f"service {GDINO_SERVICE_NAME} not available, waiting again..."
            )
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = " , ".join(object_names)
        req.box_threshold = self.box_threshold
        req.text_threshold = self.text_threshold

        future = cli.call_async(req)
        return future

    def get_img_from_topic(self, topic: str, timeout_sec: int = 2):
        msg = self.connector.receive_message(topic, timeout_sec=timeout_sec).payload

        if msg is not None:
            self.connector.node.get_logger().info(
                f"Received message of {type(msg)} from topic {topic}"
            )
            return msg
        else:
            error = f"No message received in {timeout_sec} seconds from topic {topic}"
            self.connector.node.get_logger().error(error)
            return error

    def _get_image_message(self, topic: str) -> sensor_msgs.msg.Image:
        msg = self.get_img_from_topic(topic)
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
    description: str = "A tool for detecting specified objects using a ros2 action. The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result."

    args_schema: Type[Ros2GetDetectionInput] = Ros2GetDetectionInput

    def _run(
        self,
        camera_topic: str,
        object_names: list[str],
    ):
        camera_img_msg = self._get_image_message(camera_topic)
        future = self._call_gdino_node(camera_img_msg, object_names)

        resolved = get_future_result(future)

        if resolved is not None:
            detected = self._parse_detection_array(resolved)
            names = ", ".join([det.class_name for det in detected])
            return (
                f"I have detected the following items in the picture {names or 'None'}"
            )
        else:
            return "Service call failed. Can't get detections."

        return "Failed to get detection"


class GetDistanceToObjectsTool(GroundingDinoBaseTool):
    name: str = "GetDistanceToObjectsTool"
    description: str = "A tool for calculating distance to specified objects using a ros2 action. The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result."

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
        logger = self.connector.node.get_logger()

        try:
            threshold = self.connector.node.get_parameter(
                "outlier_sigma_threshold"
            ).value
            if not isinstance(threshold, float):
                logger.error(
                    f"Parameter outlier_sigma_threshold was set badly: {type(threshold)}: {threshold} expected float. Using default value 1.0"
                )
                threshold = 1.0
        except (ParameterUninitializedException, ParameterNotDeclaredException):
            logger.warning(
                "Parameter outlier_sigma_threshold not found in node, using default value: 1.0"
            )
            threshold = 1.0

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
        resolved = get_future_result(future)
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
