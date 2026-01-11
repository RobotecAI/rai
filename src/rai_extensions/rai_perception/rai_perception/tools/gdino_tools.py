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

from typing import Any, Dict, List, NamedTuple, Type

import numpy as np
import sensor_msgs.msg
from langchain_core.tools import BaseTool
from pydantic import BaseModel, ConfigDict, Field
from rai.communication.ros2 import ROS2Connector
from rai.communication.ros2.api import convert_ros_img_to_ndarray
from rai.communication.ros2.ros_async import get_future_result
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)
from rclpy.task import Future

from rai_interfaces.srv import RAIGroundingDino

# Parameter prefix for ROS2 configuration
DISTANCE_TOOL_PARAM_PREFIX = "perception.distance_to_objects"


# --------------------- Inputs ---------------------
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

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def _run(self, *args, **kwargs):
        """Abstract method - must be implemented by subclasses."""
        raise NotImplementedError("Subclasses must implement _run method")

    def _get_detection_service_name(self) -> str:
        """Get detection service name from ROS2 parameter or use default."""
        from rai_perception.components.service_utils import get_detection_service_name

        return get_detection_service_name(self.connector)

    @property
    def service_name(self) -> str:
        """Get the detection service name used by this tool."""
        return self._get_detection_service_name()

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about required ROS2 services and their status.

        Returns:
            Dictionary with service names, types, descriptions, and current status.
            Useful for understanding service dependencies and debugging service connection issues.
        """
        from rai_perception.components.service_utils import (
            check_service_available,
            get_detection_service_name,
        )

        detection_service = get_detection_service_name(self.connector)

        return {
            "required_services": self.required_services,
            "current_service_name": detection_service,
            "service_status": {
                "detection": {
                    "name": detection_service,
                    "available": check_service_available(
                        self.connector, detection_service
                    ),
                },
            },
        }

    def check_service_dependencies(self) -> Dict[str, bool]:
        """Check if all required services are available.

        Returns:
            Dictionary mapping service names to availability status (True/False).
            Raises no exceptions - returns status for all services.
        """
        from rai_perception.components.service_utils import (
            check_service_available,
            get_detection_service_name,
        )

        detection_service = get_detection_service_name(self.connector)

        return {
            detection_service: check_service_available(
                self.connector, detection_service
            ),
        }

    def _call_gdino_node(
        self, camera_img_message: sensor_msgs.msg.Image, object_names: list[str]
    ) -> Future:
        from rai_perception.components.service_utils import create_service_client

        service_name = self._get_detection_service_name()
        cli = create_service_client(self.connector, RAIGroundingDino, service_name)
        req = RAIGroundingDino.Request()
        req.source_img = camera_img_message
        req.classes = " , ".join(object_names)
        req.box_threshold = self.box_threshold
        req.text_threshold = self.text_threshold

        return cli.call_async(req)

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
    """Tool for detecting objects in camera images using a detection service.

    **Service Dependencies:**
    - Detection Service: Required for object detection (default: "/detection")
      Configure via ROS2 parameter: /detection_tool/service_name

    **Pipeline:**
    - Single-stage: Calls detection service with image and object names
    - Returns detected objects with bounding boxes and confidence scores
    """

    name: str = "GetDetectionTool"
    description: str = (
        "A tool for detecting specified objects using a ROS2 detection service. "
        "Requires a detection service (e.g., DetectionService) to be running. "
        "The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result."
    )

    # Service dependencies for role expressiveness
    required_services: list[dict[str, str]] = [
        {
            "service_name": "/detection",
            "service_type": "DetectionService",
            "description": "Provides object detection capabilities (e.g., GroundingDINO)",
            "parameter": "/detection_tool/service_name",
        },
    ]

    args_schema: Type[GetDetectionToolInput] = GetDetectionToolInput

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


class GetDistanceToObjectsTool(GroundingDinoBaseTool):
    """Tool for calculating distances to detected objects using detection service and depth data.

    **Service Dependencies:**
    - Detection Service: Required for object detection (default: "/detection")
      Configure via ROS2 parameter: /detection_tool/service_name

    **Pipeline:**
    - Stage 1: Object Detection - Calls detection service to find objects
    - Stage 2: Distance Calculation - Uses depth image to compute distances to detected objects
    """

    name: str = "GetDistanceToObjectsTool"
    description: str = (
        "A tool for calculating distance to specified objects using a ROS2 detection service and depth camera. "
        "Executes a 2-stage pipeline: (1) Object Detection, (2) Distance Calculation from depth data. "
        "Requires a detection service (e.g., DetectionService) to be running. "
        "The tool call might take some time to execute and is blocking - you will not be able to check their feedback, only will be informed about the result."
    )

    # Service dependencies for role expressiveness
    required_services: list[dict[str, str]] = [
        {
            "service_name": "/detection",
            "service_type": "DetectionService",
            "description": "Provides object detection capabilities (e.g., GroundingDINO)",
            "parameter": "/detection_tool/service_name",
        },
    ]

    # Pipeline stages for role expressiveness
    pipeline_stages: list[dict[str, str]] = [
        {
            "stage": "Object Detection",
            "description": "Calls detection service to find objects in the image",
        },
        {
            "stage": "Distance Calculation",
            "description": "Uses depth image to compute distances to detected object bounding boxes",
        },
    ]

    args_schema: Type[GetDistanceToObjectsInput] = GetDistanceToObjectsInput

    # ROS2 parameters loaded in model_post_init
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

    def model_post_init(self, __context: Any) -> None:
        """Initialize tool with ROS2 parameters."""
        self._load_parameters()

    def _load_parameters(self) -> None:
        """Load ROS2 parameters with defaults."""
        node = self.connector.node
        node_logger = node.get_logger()

        def get_param(name: str, default: Any, param_type: type) -> Any:
            """Helper to get parameter with type checking and default fallback."""
            try:
                value = node.get_parameter(f"{DISTANCE_TOOL_PARAM_PREFIX}.{name}").value
                if not isinstance(value, param_type):
                    node_logger.error(
                        f"Parameter {name} has wrong type: {type(value)}, expected {param_type}. Using default: {default}"
                    )
                    return default
                return value
            except (ParameterUninitializedException, ParameterNotDeclaredException):
                # Auto-declare parameter with default value
                node.declare_parameter(f"{DISTANCE_TOOL_PARAM_PREFIX}.{name}", default)
                node_logger.debug(
                    f"Parameter {name} not found, using default: {default}"
                )
                return default

        self.outlier_sigma_threshold = get_param("outlier_sigma_threshold", 1.0, float)
        self.conversion_ratio = get_param("conversion_ratio", 0.001, float)

    def _run(
        self,
        camera_topic: str,
        depth_topic: str,
        object_names: list[str],
    ):
        camera_img_msg = self._get_image_message(camera_topic)
        depth_img_msg = self._get_image_message(depth_topic)
        future = self._call_gdino_node(camera_img_msg, object_names)

        resolved = get_future_result(future)
        if resolved is not None:
            detected = self._parse_detection_array(resolved)
            measurements = self._get_distance_from_detections(
                depth_img_msg,
                detected,
                self.outlier_sigma_threshold,
                self.conversion_ratio,
            )
            measurement_string = ", ".join(
                [
                    f"{measurement[0]}: {measurement[1]:.2f}m away"
                    for measurement in measurements
                ]
            )
            return f"I have detected the following items in the picture {measurement_string or 'no objects'}"
        return "Failed"

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the tool's pipeline stages.

        Returns:
            Dictionary with pipeline stages and descriptions.
            Useful for understanding tool behavior and debugging pipeline issues.
        """
        return {
            "pipeline_stages": self.pipeline_stages,
        }

    def get_service_info(self) -> Dict[str, Any]:
        """Get information about required ROS2 services and their status.

        Returns:
            Dictionary with service names, types, descriptions, and current status.
            Useful for understanding service dependencies and debugging service connection issues.
        """
        from rai_perception.components.service_utils import (
            check_service_available,
            get_detection_service_name,
        )

        detection_service = get_detection_service_name(self.connector)

        return {
            "required_services": self.required_services,
            "current_service_name": detection_service,
            "service_status": {
                "detection": {
                    "name": detection_service,
                    "available": check_service_available(
                        self.connector, detection_service
                    ),
                },
            },
        }

    def check_service_dependencies(self) -> Dict[str, bool]:
        """Check if all required services are available.

        Returns:
            Dictionary mapping service names to availability status (True/False).
            Raises no exceptions - returns status for all services.
        """
        from rai_perception.components.service_utils import (
            check_service_available,
            get_detection_service_name,
        )

        detection_service = get_detection_service_name(self.connector)

        return {
            detection_service: check_service_available(
                self.connector, detection_service
            ),
        }
