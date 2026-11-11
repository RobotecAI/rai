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

"""Detection tools for object detection and distance measurement.

Note: Tools are coupled to GroundingDINO interface (RAIGroundingDino service type,
box_threshold/text_threshold parameters).

Model support: Currently model-specific (GroundingDINO). For future model-agnostic
support, a generic RAIDetection service interface would be needed. Alternatively,
model-specific tools can be created for each model (e.g., YoloDetectionTool).
"""

from typing import Any, List, NamedTuple, Type

import numpy as np
import sensor_msgs.msg
from pydantic import BaseModel, Field
from rai.communication.ros2.api import convert_ros_img_to_ndarray

from rai_perception.tools.base import PerceptionTool


class GetDetectionToolInput(BaseModel):
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


class GroundingDinoBaseTool(PerceptionTool):
    target_args = {"camera_topic": "topic"}

    async def _detections(
        self, camera_topic: str, object_names: list[str]
    ) -> list[DetectionData]:
        image = await self.receive(camera_topic)
        response = await self.detect(image, " , ".join(object_names))
        return self._parse_detection_array(response)

    def _parse_detection_array(self, detection_response: Any) -> list[DetectionData]:
        detected = []
        for detection in detection_response.detections.detections:
            hypothesis = detection.results[0].hypothesis
            bbox = BoundingBox(
                x_center=detection.bbox.center.position.x,
                y_center=detection.bbox.center.position.y,
                width=detection.bbox.size_x,
                height=detection.bbox.size_y,
            )
            detected.append(
                DetectionData(
                    class_name=hypothesis.class_id,
                    confidence=hypothesis.score,
                    bbox=bbox,
                )
            )
        return detected


class GetDetectionTool(GroundingDinoBaseTool):
    """Detects objects in a camera image with the detection service
    (detection_service_name, default "/detection")."""

    name: str = "GetDetectionTool"
    description: str = (
        "A tool for detecting specified objects using a ROS2 detection service. "
        "Requires a detection service (e.g., DetectionService) to be running. "
        "The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result."
    )
    args_schema: Type[GetDetectionToolInput] = GetDetectionToolInput

    async def _acall(self, camera_topic: str, object_names: list[str]) -> str:
        detected = await self._detections(camera_topic, object_names)
        names = ", ".join([det.class_name for det in detected])
        return f"I have detected the following items in the picture {names or 'None'}"


class GetDistanceToObjectsTool(GroundingDinoBaseTool):
    """Distances to objects found by the detection service, from a depth image:
    (1) object detection, (2) the mean depth in each box, outliers removed."""

    name: str = "GetDistanceToObjectsTool"
    description: str = (
        "A tool for calculating distance to specified objects using a ROS2 detection service and depth camera. "
        "Executes a 2-stage pipeline: (1) Object Detection, (2) Distance Calculation from depth data. "
        "Requires a detection service (e.g., DetectionService) to be running. "
        "The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result."
    )
    target_args = {"camera_topic": "topic", "depth_topic": "topic"}
    args_schema: Type[GetDistanceToObjectsInput] = GetDistanceToObjectsInput

    outlier_sigma_threshold: float = Field(default=1.0, exclude=True)
    conversion_ratio: float = Field(default=0.001, exclude=True)

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

    async def _acall(
        self, camera_topic: str, depth_topic: str, object_names: list[str]
    ) -> str:
        depth_img_msg = await self.receive(depth_topic)
        detected = await self._detections(camera_topic, object_names)
        measurements = self._get_distance_from_detections(
            depth_img_msg,
            detected,
            self.outlier_sigma_threshold,
            self.conversion_ratio,
        )
        measurement_string = ", ".join(
            f"{name}: {distance:.2f}m away" for name, distance in measurements
        )
        return f"I have detected the following items in the picture {measurement_string or 'no objects'}"
