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

"""ROS2 node component that subscribes to camera images, calls detection services, and publishes detections.

Architectural Note:
This belongs in `components/` (not `services/`) because it's a client component:
- Services expose ROS2 services (server-side, e.g., DetectionService)
- Components are client-side: subscribe/publish topics, call services
This node subscribes to camera topics, calls detection services, and publishes results - it does not expose a service.
"""

import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from rai.communication.ros2 import QoS, ROS2Connector
from rai.communication.ros2.runtime import ROS2Runtime
from sensor_msgs.msg import CameraInfo, Image

from rai_interfaces.msg import RAIDetectionArray
from rai_perception.components.perception_utils import enhance_detection_with_3d_pose
from rai_perception.components.service_utils import detect, service_available

CONFIG_DIR = Path(__file__).parent.parent / "configs"

# Parameters, their defaults when the YAML files do not set them, and the YAML
# file (section) they are read from.
DETECTION_PUBLISHER_DEFAULTS: Dict[str, Any] = {
    "camera_topic": "/camera/camera/color/image_raw",
    "detection_topic": "/detection_array",
    "dino_service": "/detection",
    # "class1:threshold1, class2, class3:threshold3"; classes without a
    # threshold use default_class_threshold.
    "detection_classes": "person, cup, bottle, box, bag, chair, table, shelf, door, window, couch, sofa, bed",
    "default_class_threshold": 0.3,
    "detection_interval": 2.0,  # minimum seconds between detections
    "box_threshold": 0.3,  # the DINO call uses min(this, the lowest class threshold)
    "text_threshold": 0.25,
}
PERCEPTION_UTILS_DEFAULTS: Dict[str, Any] = {
    "depth_topic": "",  # optional, for 3D poses
    "camera_info_topic": "",  # optional, for 3D poses
    "depth_fallback_region_size": 5,
}

logger = logging.getLogger(__name__)


def _yaml_parameters(path: Path, section: str) -> Dict[str, Any]:
    with open(path, "r") as f:
        return (yaml.safe_load(f) or {}).get(section, {}).get("ros__parameters", {})


class DetectionPublisher:
    """Subscribes to camera images, calls the detection (DINO) service and
    publishes RAIDetectionArray messages, with 3D poses when depth and camera
    info topics are configured.

    Parameters come from the YAML files in configs/ (or the files named by the
    detection_publisher_config and perception_utils_config parameters); ROS
    parameters of the connector's node override them. Everything goes through
    the connector, so egress hooks see the published detections.
    """

    def __init__(self, connector: ROS2Connector):
        self.connector = connector
        self.params = self._load_parameters()
        self.last_image: Optional[Image] = None
        self.last_depth_image: Optional[Image] = None
        self.last_camera_info: Optional[CameraInfo] = None
        self.last_detection_time = 0.0
        self.last_log_time = 0.0
        self.log_interval = 5.0  # Log summary every 5 seconds

        dino_service = self.params["dino_service"]
        if not service_available(connector, dino_service):
            logger.warning(f"DINO service not available yet: {dino_service}")
        connector.subscribe(
            self.params["camera_topic"], self.image_callback, msg_type=Image, maxsize=1
        )
        for topic, callback, msg_type in (
            (self.params["depth_topic"], self.depth_callback, Image),
            (self.params["camera_info_topic"], self.camera_info_callback, CameraInfo),
        ):
            if topic:
                connector.subscribe(topic, callback, msg_type=msg_type, maxsize=1)
            else:
                logger.info("No depth or camera info topic: 3D poses not computed")
        logger.info(
            f"Detecting on {self.params['camera_topic']} with {dino_service}, "
            f"publishing to {self.params['detection_topic']}"
        )

    def _load_parameters(self) -> Dict[str, Any]:
        files = {
            "detection_publisher": self.connector.parameter(
                "detection_publisher_config", ""
            )
            or CONFIG_DIR / "detection_publisher.yaml",
            "perception_utils": self.connector.parameter("perception_utils_config", "")
            or CONFIG_DIR / "perception_utils.yaml",
        }
        params: Dict[str, Any] = {}
        for section, defaults in (
            ("detection_publisher", DETECTION_PUBLISHER_DEFAULTS),
            ("perception_utils", PERCEPTION_UTILS_DEFAULTS),
        ):
            from_yaml = _yaml_parameters(Path(files[section]), section)
            for name, default in defaults.items():
                params[name] = self.connector.parameter(
                    name, from_yaml.get(name, default)
                )
        return params

    def _parse_detection_classes(
        self, detection_classes_str: str
    ) -> Tuple[List[str], Dict[str, float]]:
        """Parse detection_classes string to extract class names and per-class thresholds.

        Format: "class1:threshold1, class2, class3:threshold3"
        Classes without explicit thresholds use default_class_threshold.

        Returns:
            Tuple of (class_names_list, class_thresholds_dict)
        """
        default_threshold = self.params["default_class_threshold"]
        class_names = []
        class_thresholds = {}
        for item in detection_classes_str.split(","):
            class_name, _, threshold_str = item.strip().partition(":")
            class_name = class_name.strip()
            if not class_name:
                continue
            threshold = default_threshold
            if threshold_str:
                try:
                    threshold = float(threshold_str.strip())
                except ValueError:
                    logger.warning(
                        f"Invalid threshold value in '{item}', using default"
                    )
            class_names.append(class_name)
            class_thresholds[class_name] = threshold
        return class_names, class_thresholds

    def depth_callback(self, msg: Image):
        """Store latest depth image."""
        self.last_depth_image = msg

    def camera_info_callback(self, msg: CameraInfo):
        """Store latest camera info."""
        self.last_camera_info = msg

    async def image_callback(self, msg: Image):
        """Detect objects in an image, at most once per detection_interval."""
        if time.time() - self.last_detection_time < self.params["detection_interval"]:
            return
        self.last_image = msg
        class_names, class_thresholds = self._parse_detection_classes(
            self.params["detection_classes"]
        )
        # The lowest threshold gets every relevant detection from DINO; the
        # per-class thresholds filter them afterwards.
        min_threshold = min(
            class_thresholds.values(), default=self.params["default_class_threshold"]
        )
        try:
            response = await detect(
                self.connector,
                msg,
                ", ".join(class_names),
                service=self.params["dino_service"],
                box_threshold=min(self.params["box_threshold"], min_threshold),
                text_threshold=self.params["text_threshold"],
            )
        except Exception as e:
            logger.warning(f"Detection on {self.params['dino_service']} failed: {e}")
            return
        detection_array = self._detection_array(response, msg.header, class_thresholds)
        await self.connector.publish(
            self.params["detection_topic"], detection_array, qos=QoS.SENSOR
        )
        self.last_detection_time = time.time()

    def _detection_array(
        self, response: Any, image_header: Any, class_thresholds: Dict[str, float]
    ) -> RAIDetectionArray:
        """The detections above their class thresholds, in the image's frame,
        with 3D poses where depth and camera info allow."""
        default_threshold = self.params["default_class_threshold"]
        detections = []
        for det in response.detections.detections:
            if det.results:
                hypothesis = det.results[0].hypothesis
                threshold = class_thresholds.get(hypothesis.class_id, default_threshold)
                if hypothesis.score < threshold:
                    logger.debug(
                        f"Filtered out {hypothesis.class_id} detection with score "
                        f"{hypothesis.score:.3f} (threshold: {threshold:.3f})"
                    )
                    continue
            if not det.header.frame_id:
                det.header.frame_id = image_header.frame_id
            det.header.stamp = image_header.stamp
            enhance_detection_with_3d_pose(
                det,
                self.last_depth_image,
                self.last_camera_info,
                self.params["depth_fallback_region_size"],
            )
            detections.append(det)

        now = time.time()
        if now - self.last_log_time >= self.log_interval:
            classes = {
                d.results[0].hypothesis.class_id for d in detections if d.results
            }
            logger.info(
                f"Publishing {len(detections)} detections: {', '.join(classes)}"
            )
            self.last_log_time = now
        return RAIDetectionArray(
            header=image_header,
            detections=detections,
            detection_classes=response.detections.detection_classes,
        )


def main(args: Optional[List[str]] = None):
    """Run the detection publisher node until Ctrl-C. --ros-args parameters
    (e.g. -p detection_publisher_config:=path) apply to its node."""
    runtime = ROS2Runtime("detection_publisher", cli_args=args or sys.argv)
    with ROS2Connector(runtime=runtime) as connector:
        DetectionPublisher(connector=connector)
        try:
            threading.Event().wait()
        except KeyboardInterrupt:
            logger.info("Shutting down detection publisher...")


if __name__ == "__main__":
    main()
