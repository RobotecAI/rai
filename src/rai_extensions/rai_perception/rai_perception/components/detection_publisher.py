# Copyright (C) 2025 Julia Jia
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

import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import rclpy
import yaml
from cv_bridge import CvBridge
from rai.communication.ros2 import ROS2Connector
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import CameraInfo, Image

from rai_interfaces.msg import RAIDetectionArray
from rai_interfaces.srv import RAIGroundingDino
from rai_perception.components.perception_utils import enhance_detection_with_3d_pose


class DetectionPublisher:
    """ROS2 node that subscribes to camera images, calls DINO service, and publishes detections.

    Design Notes:
    - Uses ROS2Connector for node lifecycle and infrastructure
    - Uses connector.node.* directly for raw ROS2 messages and QoS control
    - Uses connector.node.create_client() for async service calls (non-blocking)
    """

    def __init__(self, connector: ROS2Connector):
        """Initialize DetectionPublisher.

        Args:
            connector: ROS2Connector instance for ROS2 communication.
        """
        self.connector = connector
        self._initialize_parameters()
        self.bridge = CvBridge()
        self._initialize_clients()
        self._initialize_subscriptions()
        self._initialize_publishers()
        self.last_image: Optional[Image] = None
        self.last_depth_image: Optional[Image] = None
        self.last_camera_info: Optional[CameraInfo] = None
        self.last_detection_time = 0.0
        self.last_log_time = 0.0
        self.log_interval = 5.0  # Log summary every 5 seconds

    def _initialize_parameters(self):
        """Initialize ROS2 parameters from YAML files."""
        # Get config directory (configs folder at package root)
        package_root = Path(__file__).parent.parent
        config_dir = package_root / "configs"

        # Declare config file path parameters first
        config_params = [
            (
                "detection_publisher_config",
                "",
                ParameterType.PARAMETER_STRING,
                "Path to detection_publisher YAML config file (empty = use default in config/)",
            ),
            (
                "perception_utils_config",
                "",
                ParameterType.PARAMETER_STRING,
                "Path to perception_utils YAML config file (empty = use default in config/)",
            ),
        ]
        for name, default, param_type, description in config_params:
            self.connector.node.declare_parameter(
                name,
                default,
                descriptor=ParameterDescriptor(
                    type=param_type, description=description
                ),
            )

        # Get config file paths
        detection_pub_config_path = (
            self.connector.node.get_parameter("detection_publisher_config")
            .get_parameter_value()
            .string_value
        )
        perception_utils_config_path = (
            self.connector.node.get_parameter("perception_utils_config")
            .get_parameter_value()
            .string_value
        )

        # Load detection_publisher parameters
        if detection_pub_config_path:
            detection_pub_yaml = Path(detection_pub_config_path)
        else:
            detection_pub_yaml = config_dir / "detection_publisher.yaml"

        with open(detection_pub_yaml, "r") as f:
            detection_pub_config = yaml.safe_load(f)
        detection_pub_params = detection_pub_config.get("detection_publisher", {}).get(
            "ros__parameters", {}
        )

        # Load perception_utils parameters
        if perception_utils_config_path:
            perception_utils_yaml = Path(perception_utils_config_path)
        else:
            perception_utils_yaml = config_dir / "perception_utils.yaml"

        with open(perception_utils_yaml, "r") as f:
            perception_utils_config = yaml.safe_load(f)
        perception_utils_params = perception_utils_config.get(
            "perception_utils", {}
        ).get("ros__parameters", {})

        # Declare detection_publisher parameters
        parameters = [
            (
                "camera_topic",
                detection_pub_params.get(
                    "camera_topic", "/camera/camera/color/image_raw"
                ),
                ParameterType.PARAMETER_STRING,
                "Camera image topic to subscribe to",
            ),
            (
                "detection_topic",
                detection_pub_params.get("detection_topic", "/detection_array"),
                ParameterType.PARAMETER_STRING,
                "Topic to publish RAIDetectionArray messages",
            ),
            (
                "dino_service",
                detection_pub_params.get("dino_service", "/detection"),
                ParameterType.PARAMETER_STRING,
                "GroundingDINO service name",
            ),
            (
                "detection_classes",
                detection_pub_params.get(
                    "detection_classes",
                    "person, cup, bottle, box, bag, chair, table, shelf, door, window, couch, sofa, bed",
                ),
                ParameterType.PARAMETER_STRING,
                "Comma-separated list of object classes to detect. Format: 'class1:threshold1, class2, class3:threshold3' where classes without thresholds use default_class_threshold",
            ),
            (
                "default_class_threshold",
                detection_pub_params.get("default_class_threshold", 0.3),
                ParameterType.PARAMETER_DOUBLE,
                "Default box threshold for classes without explicit threshold in detection_classes",
            ),
            (
                "detection_interval",
                detection_pub_params.get("detection_interval", 2.0),
                ParameterType.PARAMETER_DOUBLE,
                "Minimum time between detections (seconds)",
            ),
            (
                "box_threshold",
                detection_pub_params.get("box_threshold", 0.3),
                ParameterType.PARAMETER_DOUBLE,
                "DINO box threshold (used as minimum for DINO call to get all detections)",
            ),
            (
                "text_threshold",
                detection_pub_params.get("text_threshold", 0.25),
                ParameterType.PARAMETER_DOUBLE,
                "DINO text threshold",
            ),
        ]

        for name, default, param_type, description in parameters:
            self.connector.node.declare_parameter(
                name,
                default,
                descriptor=ParameterDescriptor(
                    type=param_type,
                    description=description,
                ),
            )

        # Declare perception_utils parameters
        perception_params = [
            (
                "depth_topic",
                perception_utils_params.get("depth_topic", ""),
                ParameterType.PARAMETER_STRING,
                "Depth image topic (optional, for 3D pose computation)",
            ),
            (
                "camera_info_topic",
                perception_utils_params.get("camera_info_topic", ""),
                ParameterType.PARAMETER_STRING,
                "Camera info topic (optional, for 3D pose computation)",
            ),
            (
                "depth_fallback_region_size",
                perception_utils_params.get("depth_fallback_region_size", 5),
                ParameterType.PARAMETER_INTEGER,
                "Region size for depth fallback when center pixel has no depth",
            ),
        ]

        for name, default, param_type, description in perception_params:
            self.connector.node.declare_parameter(
                name,
                default,
                descriptor=ParameterDescriptor(
                    type=param_type,
                    description=description,
                ),
            )

    def _get_string_parameter(self, name: str) -> str:
        """Get string parameter value."""
        return (
            self.connector.node.get_parameter(name).get_parameter_value().string_value
        )

    def _get_double_parameter(self, name: str) -> float:
        return (
            self.connector.node.get_parameter(name).get_parameter_value().double_value
        )

    def _get_integer_parameter(self, name: str) -> int:
        return (
            self.connector.node.get_parameter(name).get_parameter_value().integer_value
        )

    def _parse_detection_classes(
        self, detection_classes_str: str
    ) -> Tuple[List[str], Dict[str, float]]:
        """Parse detection_classes string to extract class names and per-class thresholds.

        Format: "class1:threshold1, class2, class3:threshold3"
        Classes without explicit thresholds use default_class_threshold.

        Returns:
            Tuple of (class_names_list, class_thresholds_dict)
        """
        default_threshold = self._get_double_parameter("default_class_threshold")
        class_names = []
        class_thresholds = {}

        for item in detection_classes_str.split(","):
            item = item.strip()
            if not item:
                continue

            if ":" in item:
                class_name, threshold_str = item.split(":", 1)
                class_name = class_name.strip()
                try:
                    threshold = float(threshold_str.strip())
                    class_names.append(class_name)
                    class_thresholds[class_name] = threshold
                except ValueError:
                    self.connector.node.get_logger().warning(
                        f"Invalid threshold value in '{item}', using default"
                    )
                    class_names.append(class_name)
                    class_thresholds[class_name] = default_threshold
            else:
                class_name = item.strip()
                class_names.append(class_name)
                class_thresholds[class_name] = default_threshold

        return class_names, class_thresholds

    def _initialize_clients(self):
        """Initialize service clients.

        Note: We use self.connector.node.create_client() directly instead of ROS2Connector's
        service_call() because we need async service calls (call_async) for non-blocking
        detection processing. The connector's service_call() is synchronous and designed
        for the connector's message wrapper system.
        """
        dino_service = self._get_string_parameter("dino_service")
        self.dino_client = self.connector.node.create_client(
            RAIGroundingDino, dino_service
        )
        self.connector.node.get_logger().info(
            f"Waiting for DINO service: {dino_service}"
        )
        # Use short timeout - _process_image() will check again before actual use
        if not self.dino_client.wait_for_service(timeout_sec=0.1):
            self.connector.node.get_logger().warning(
                f"DINO service not available: {dino_service}"
            )
        else:
            self.connector.node.get_logger().info(f"DINO service ready: {dino_service}")

    def _initialize_subscriptions(self):
        """Initialize ROS2 subscriptions."""
        camera_topic = self._get_string_parameter("camera_topic")
        self.image_subscription = self.connector.node.create_subscription(
            Image, camera_topic, self.image_callback, qos_profile_sensor_data
        )
        self.connector.node.get_logger().info(
            f"Subscribed to camera topic: {camera_topic} "
            f"(QoS: {qos_profile_sensor_data.reliability.name})"
        )

        # Optional depth and camera info subscriptions for 3D pose computation
        depth_topic = self._get_string_parameter("depth_topic")
        camera_info_topic = self._get_string_parameter("camera_info_topic")

        if depth_topic:
            self.depth_subscription = self.connector.node.create_subscription(
                Image, depth_topic, self.depth_callback, qos_profile_sensor_data
            )
            self.connector.node.get_logger().info(
                f"Subscribed to depth topic: {depth_topic}"
            )
        else:
            self.depth_subscription = None
            self.connector.node.get_logger().info(
                "No depth topic provided, 3D poses will not be computed"
            )

        if camera_info_topic:
            self.camera_info_subscription = self.connector.node.create_subscription(
                CameraInfo,
                camera_info_topic,
                self.camera_info_callback,
                qos_profile_sensor_data,
            )
            self.connector.node.get_logger().info(
                f"Subscribed to camera info topic: {camera_info_topic}"
            )
        else:
            self.camera_info_subscription = None
            self.connector.node.get_logger().info(
                "No camera info topic provided, 3D poses will not be computed"
            )

    def _initialize_publishers(self):
        """Initialize ROS2 publishers."""
        detection_topic = self._get_string_parameter("detection_topic")
        self.detection_publisher = self.connector.node.create_publisher(
            RAIDetectionArray, detection_topic, qos_profile_sensor_data
        )
        self.connector.node.get_logger().info(
            f"Publishing to detection topic: {detection_topic} "
            f"(QoS: reliability={qos_profile_sensor_data.reliability.name})"
        )

    def depth_callback(self, msg: Image):
        """Store latest depth image."""
        self.last_depth_image = msg

    def camera_info_callback(self, msg: CameraInfo):
        """Store latest camera info."""
        self.last_camera_info = msg

    def image_callback(self, msg: Image):
        """Process incoming camera image."""
        self.connector.node.get_logger().debug(
            f"Received camera image (stamp: {msg.header.stamp.sec}.{msg.header.stamp.nanosec:09d}, "
            f"frame_id: {msg.header.frame_id})"
        )
        current_time = time.time()
        detection_interval = self._get_double_parameter("detection_interval")

        # Throttle detections
        if current_time - self.last_detection_time < detection_interval:
            time_since_last = current_time - self.last_detection_time
            self.connector.node.get_logger().debug(
                f"Throttling: {time_since_last:.2f}s since last detection (interval: {detection_interval}s)"
            )
            return

        self.last_image = msg
        self.connector.node.get_logger().debug("Processing camera image...")
        self._process_image(msg)

    def _process_image(self, image_msg: Image):
        """Call DINO service and publish detections."""
        if not self.dino_client.wait_for_service(timeout_sec=0.1):
            self.connector.node.get_logger().warning(
                "DINO service not ready, skipping detection"
            )
            return

        detection_classes_str = self._get_string_parameter("detection_classes")
        class_names, class_thresholds = self._parse_detection_classes(
            detection_classes_str
        )

        # Use minimum threshold for DINO call to ensure we get all relevant detections
        # Results will be filtered by per-class thresholds in _handle_dino_response
        min_threshold = (
            min(class_thresholds.values())
            if class_thresholds
            else self._get_double_parameter("default_class_threshold")
        )
        box_threshold = min(self._get_double_parameter("box_threshold"), min_threshold)
        text_threshold = self._get_double_parameter("text_threshold")

        # Store class_thresholds for filtering in response handler
        self._current_class_thresholds = class_thresholds

        request = RAIGroundingDino.Request()
        request.source_img = image_msg
        request.classes = ", ".join(class_names)
        request.box_threshold = box_threshold
        request.text_threshold = text_threshold

        self.connector.node.get_logger().debug(
            f"Calling DINO service with {len(class_names)} classes (box_threshold={box_threshold:.3f})"
        )

        future = self.dino_client.call_async(request)
        future.add_done_callback(
            lambda f: self._handle_dino_response(f, image_msg.header)
        )

    def _handle_dino_response(self, future, image_header):
        """Handle DINO service response."""
        try:
            response = future.result()
            if response is None:
                self.connector.node.get_logger().warning("DINO service returned None")
                return

            # Get class thresholds for filtering (set in _process_image)
            class_thresholds = getattr(self, "_current_class_thresholds", {})

            # Filter detections by per-class thresholds
            filtered_detections = []
            for det in response.detections.detections:
                if det.results and len(det.results) > 0:
                    result = det.results[0]
                    class_id = result.hypothesis.class_id
                    score = result.hypothesis.score

                    # Get threshold for this class (use default if not found)
                    threshold = class_thresholds.get(
                        class_id, self._get_double_parameter("default_class_threshold")
                    )

                    if score >= threshold:
                        filtered_detections.append(det)
                    else:
                        self.connector.node.get_logger().debug(
                            f"Filtered out {class_id} detection with score {score:.3f} "
                            f"(threshold: {threshold:.3f})"
                        )
                else:
                    # Keep detections without results (shouldn't happen, but be safe)
                    filtered_detections.append(det)

            # Create RAIDetectionArray message
            detection_array = RAIDetectionArray()
            detection_array.header = image_header
            detection_array.header.frame_id = image_header.frame_id
            detection_array.detections = filtered_detections
            detection_array.detection_classes = response.detections.detection_classes

            # Ensure each detection has the correct frame_id and enhance with 3D poses
            for det in detection_array.detections:
                if not det.header.frame_id:
                    det.header.frame_id = image_header.frame_id
                det.header.stamp = image_header.stamp

                # Enhance detection with 3D pose if pose is empty
                region_size = self._get_integer_parameter("depth_fallback_region_size")
                if enhance_detection_with_3d_pose(
                    det,
                    self.last_depth_image,
                    self.last_camera_info,
                    self.bridge,
                    region_size,
                ):
                    if det.results and len(det.results) > 0:
                        result = det.results[0]
                        computed_pose = result.pose.pose
                        self.connector.node.get_logger().debug(
                            f"Computed 3D pose for {result.hypothesis.class_id}: "
                            f"({computed_pose.position.x:.3f}, {computed_pose.position.y:.3f}, "
                            f"{computed_pose.position.z:.3f})"
                        )
                elif det.results and len(det.results) > 0:
                    result = det.results[0]
                    pose = result.pose.pose
                    if (
                        pose.position.x == 0.0
                        and pose.position.y == 0.0
                        and pose.position.z == 0.0
                    ):
                        self.connector.node.get_logger().debug(
                            f"Could not compute 3D pose for {result.hypothesis.class_id} "
                            f"(depth or camera info not available)"
                        )

            # Log detection details for debugging
            detection_count = len(detection_array.detections)
            current_time = time.time()
            should_log = current_time - self.last_log_time >= self.log_interval

            if detection_count > 0:
                # Log individual detections only at DEBUG level
                for i, det in enumerate(detection_array.detections):
                    results_count = len(det.results) if det.results else 0
                    if results_count > 0:
                        result = det.results[0]
                        self.connector.node.get_logger().debug(
                            f"Detection {i}: class={result.hypothesis.class_id}, "
                            f"score={result.hypothesis.score:.3f}"
                        )
                    else:
                        self.connector.node.get_logger().warning(
                            f"Detection {i} has no results! frame_id={det.header.frame_id}"
                        )

                # Throttled summary log
                if should_log:
                    classes_found = [
                        det.results[0].hypothesis.class_id
                        for det in detection_array.detections
                        if det.results and len(det.results) > 0
                    ]
                    self.connector.node.get_logger().info(
                        f"Published {detection_count} detections: {', '.join(set(classes_found))}"
                    )
                    self.last_log_time = current_time
            else:
                if should_log:
                    self.connector.node.get_logger().debug(
                        "No detections found in image"
                    )
                    self.last_log_time = current_time

            # Publish detections
            self.detection_publisher.publish(detection_array)
            self.last_detection_time = time.time()

        except Exception as e:
            self.connector.node.get_logger().error(
                f"Error processing DINO response: {e}"
            )


def main(args=None):
    """Main entry point for the detection publisher node."""
    rclpy.init(args=args)
    connector = ROS2Connector(
        node_name="detection_publisher", executor_type="multi_threaded"
    )
    detection_publisher = DetectionPublisher(connector=connector)
    detection_publisher.connector.node.get_logger().info("=" * 60)
    detection_publisher.connector.node.get_logger().info(
        "Detection Publisher Node Started"
    )
    detection_publisher.connector.node.get_logger().info("=" * 60)
    try:
        rclpy.spin(detection_publisher.connector.node)
    except KeyboardInterrupt:
        detection_publisher.connector.node.get_logger().info(
            "Shutting down detection publisher..."
        )
    finally:
        detection_publisher.connector.shutdown()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
