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

import logging
import time
from typing import Any, Callable, Dict, Optional, Type

import numpy as np
from geometry_msgs.msg import Point32
from pydantic import BaseModel, Field
from rai.tools.ros2.base import BaseROS2Tool
from rai.tools.timeout import RaiTimeoutError, timeout
from rclpy.exceptions import (
    ParameterNotDeclaredException,
    ParameterUninitializedException,
)
from sensor_msgs.msg import PointCloud

from rai_perception.components.gripping_points import (
    GrippingPointEstimator,
    GrippingPointEstimatorConfig,
    PointCloudFilter,
    PointCloudFilterConfig,
    PointCloudFromSegmentation,
    PointCloudFromSegmentationConfig,
    _publish_gripping_point_debug_data,
)
from rai_perception.components.perception_presets import apply_preset
from rai_perception.components.topic_utils import discover_camera_topics

# Parameter prefix for ROS2 configuration
GRIPPING_POINTS_TOOL_PARAM_PREFIX = "perception.gripping_points"

# Debug publishing constants
DEBUG_PUBLISH_DURATION_SEC = 5.0
DEBUG_PUBLISH_RATE_HZ = 10.0

logger = logging.getLogger(__name__)


class GetObjectGrippingPointsToolInput(BaseModel):
    object_name: str = Field(
        ...,
        description="The name of the object to get the gripping point of e.g. 'box', 'apple', 'screwdriver'",
    )
    debug: bool = Field(
        default=False,
        description="If True, publish intermediate pipeline results to ROS2 topics for visualization in RVIZ. Adds computational overhead - not suitable for production.",
    )


class GetObjectGrippingPointsTool(BaseROS2Tool):
    """Tool for extracting 3D gripping points from objects using a multi-stage pipeline.

    This tool orchestrates a 3-stage pipeline:
    1. Point Cloud Extraction: Detects and segments objects, extracts point clouds from depth data
    2. Point Cloud Filtering: Removes outliers and noise from point clouds
    3. Gripping Point Estimation: Estimates optimal gripping points from filtered point clouds

    **Service Dependencies:**
    - Detection Service: Required for object detection (default: "/detection")
    - Segmentation Service: Required for object segmentation (default: "/segmentation")

    **Pipeline Components:**
    - PointCloudFromSegmentation: Extracts segmented point clouds from camera/depth data
    - PointCloudFilter: Filters outliers using configurable strategies
    - GrippingPointEstimator: Estimates gripping points using configurable strategies

    Use debug=True to publish intermediate pipeline results to ROS2 topics for visualization.
    """

    name: str = "get_object_gripping_points"
    description: str = (
        "Get gripping points for specified object/objects. Returns 3D coordinates where a robot gripper can grasp the object. "
        "Executes a 3-stage pipeline: (1) Point Cloud Extraction from detection/segmentation, "
        "(2) Point Cloud Filtering to remove outliers, (3) Gripping Point Estimation. "
        "Requires detection and segmentation services to be running. "
        "Set debug=True to publish intermediate pipeline results to ROS2 topics for visualization in RVIZ (adds overhead, not for production)."
    )

    # Pipeline stages for role expressiveness
    pipeline_stages: list[dict[str, str]] = [
        {
            "stage": "Point Cloud Extraction",
            "component": "PointCloudFromSegmentation",
            "description": "Detects objects, segments them, and extracts point clouds from depth data",
        },
        {
            "stage": "Point Cloud Filtering",
            "component": "PointCloudFilter",
            "description": "Removes outliers and noise from point clouds using configurable filtering strategies",
        },
        {
            "stage": "Gripping Point Estimation",
            "component": "GrippingPointEstimator",
            "description": "Estimates optimal gripping points from filtered point clouds",
        },
    ]

    # Service dependencies for role expressiveness
    required_services: list[dict[str, str]] = [
        {
            "service_name": "/detection",
            "service_type": "DetectionService",
            "description": "Provides object detection capabilities (e.g., GroundingDINO)",
            "parameter": "/detection_tool/service_name",
        },
        {
            "service_name": "/segmentation",
            "service_type": "SegmentationService",
            "description": "Provides object segmentation capabilities (e.g., Grounded SAM)",
            "parameter": "/segmentation_tool/service_name",
        },
    ]

    # Components initialized in model_post_init
    # Using Field(exclude=True) to exclude from serialization, but still get Pydantic validation
    # These won't appear in tool schema because LangChain only uses args_schema for that
    gripping_point_estimator: Optional[GrippingPointEstimator] = Field(
        default=None, exclude=True
    )
    point_cloud_filter: Optional[PointCloudFilter] = Field(default=None, exclude=True)
    point_cloud_from_segmentation: Optional[PointCloudFromSegmentation] = Field(
        default=None, exclude=True
    )

    # Configs initialized in model_post_init
    segmentation_config: Optional[PointCloudFromSegmentationConfig] = Field(
        default=None, exclude=True
    )
    estimator_config: Optional[GrippingPointEstimatorConfig] = Field(
        default=None, exclude=True
    )
    filter_config: Optional[PointCloudFilterConfig] = Field(default=None, exclude=True)

    # ROS2 parameters loaded in _load_parameters()
    camera_topic: str = Field(default="/camera/rgb/image_raw", exclude=True)
    depth_topic: str = Field(default="/camera/depth/image_raw", exclude=True)
    camera_info_topic: str = Field(default="/camera/rgb/camera_info", exclude=True)
    target_frame: str = Field(default="base_link", exclude=True)
    source_frame: str = Field(default="camera_link", exclude=True)
    timeout_sec: float = Field(default=10.0, exclude=True)
    conversion_ratio: float = Field(default=0.001, exclude=True)

    args_schema: Type[GetObjectGrippingPointsToolInput] = (
        GetObjectGrippingPointsToolInput
    )

    def _run(self, object_name: str, debug: bool = False) -> str:
        @timeout(
            self.timeout_sec,
            f"Gripping point detection for object '{object_name}' exceeded {self.timeout_sec} seconds",
        )
        def _run_with_timeout():
            # Stage 1: Point cloud extraction
            pcl, stage_time = self._run_stage(
                stage_num=1,
                stage_name="Point Cloud Extraction",
                stage_func=lambda: self.point_cloud_from_segmentation.run(object_name),
                empty_result_msg=f"No {object_name}s detected.",
                debug_config={
                    "topic": "/debug/gripping_points/raw_point_clouds",
                    "point_clouds": None,  # Will be set from result
                }
                if debug
                else None,
            )
            if isinstance(pcl, str):  # Empty result message
                return pcl

            # Stage 2: Point cloud filtering
            total_points_before = sum(len(pts) for pts in pcl)
            pcl_filtered, stage_time = self._run_stage(
                stage_num=2,
                stage_name="Point Cloud Filtering",
                stage_func=lambda: self.point_cloud_filter.run(pcl),
                empty_result_msg=f"No {object_name}s detected after applying filtering",
                debug_config={
                    "topic": "/debug/gripping_points/filtered_point_clouds",
                    "point_clouds": None,  # Will be set from result
                    "total_points_before": total_points_before,
                }
                if debug
                else None,
            )
            if isinstance(pcl_filtered, str):  # Empty result message
                return pcl_filtered

            # Stage 3: Gripping point estimation
            gripping_points, stage_time = self._run_stage(
                stage_num=3,
                stage_name="Gripping Point Estimation",
                stage_func=lambda: self.gripping_point_estimator.run(pcl_filtered),
                empty_result_msg=None,  # No early exit for stage 3
                debug_config={
                    "topic": None,
                    "filtered_point_clouds": pcl_filtered,
                }
                if debug
                else None,
            )

            return self._format_result_message(object_name, gripping_points)

        try:
            return _run_with_timeout()
        except RaiTimeoutError as e:
            self.connector.node.get_logger().warning(f"Timeout: {e}")
            return f"Timeout: Gripping point detection for object '{object_name}' exceeded {self.timeout_sec} seconds"
        except Exception:
            raise

    @property
    def detection_service_name(self) -> str:
        """Get the detection service name used by this tool."""
        return self.point_cloud_from_segmentation._get_detection_service_name()

    @property
    def segmentation_service_name(self) -> str:
        """Get the segmentation service name used by this tool."""
        return self.point_cloud_from_segmentation._get_segmentation_service_name()

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the tool's pipeline stages and components.

        Returns:
            Dictionary with pipeline stages, component names, and descriptions.
            Useful for understanding tool behavior and debugging pipeline issues.
        """
        return {
            "pipeline_stages": self.pipeline_stages,
            "component_classes": {
                "PointCloudFromSegmentation": "Extracts segmented point clouds from camera/depth data",
                "PointCloudFilter": "Filters outliers from point clouds using configurable strategies",
                "GrippingPointEstimator": "Estimates gripping points from filtered point clouds",
            },
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
            get_segmentation_service_name,
        )

        detection_service = get_detection_service_name(self.connector)
        segmentation_service = get_segmentation_service_name(self.connector)

        return {
            "required_services": self.required_services,
            "current_service_names": {
                "detection": detection_service,
                "segmentation": segmentation_service,
            },
            "service_status": {
                "detection": {
                    "name": detection_service,
                    "available": check_service_available(
                        self.connector, detection_service
                    ),
                },
                "segmentation": {
                    "name": segmentation_service,
                    "available": check_service_available(
                        self.connector, segmentation_service
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
            get_segmentation_service_name,
        )

        detection_service = get_detection_service_name(self.connector)
        segmentation_service = get_segmentation_service_name(self.connector)

        return {
            detection_service: check_service_available(
                self.connector, detection_service
            ),
            segmentation_service: check_service_available(
                self.connector, segmentation_service
            ),
        }

    def get_config(self) -> Dict[str, Any]:
        """Get current ROS2 parameter configuration for observability.

        Returns:
            Dictionary mapping parameter names to their current values.
            Includes all deployment-specific parameters and service names.
        """
        return {
            "target_frame": self.target_frame,
            "source_frame": self.source_frame,
            "camera_topic": self.camera_topic,
            "depth_topic": self.depth_topic,
            "camera_info_topic": self.camera_info_topic,
            "timeout_sec": self.timeout_sec,
            "conversion_ratio": self.conversion_ratio,
            "detection_service_name": self.detection_service_name,
            "segmentation_service_name": self.segmentation_service_name,
        }

    # --------------------- Initialization ---------------------

    def model_post_init(self, __context: Any) -> None:
        """Initialize tool with ROS2 parameters and components."""
        # Initialize configs if not provided (they can be passed as constructor kwargs and will be validated)
        # These are excluded from tool schema because LangChain only uses args_schema for that
        if self.segmentation_config is None:
            self.segmentation_config = PointCloudFromSegmentationConfig()
        if self.estimator_config is None or self.filter_config is None:
            # Use default_grasp preset if configs not provided
            filter_config, estimator_config = apply_preset("default_grasp")
            if self.filter_config is None:
                self.filter_config = filter_config
            if self.estimator_config is None:
                self.estimator_config = estimator_config

        self._load_parameters()
        self._validate_topics_early()
        self._initialize_components()

        logger.info("GetObjectGrippingPointsTool initialized")

    def _load_parameters(self) -> None:
        """Load ROS2 parameters with defaults."""
        node = self.connector.node
        node_logger = node.get_logger()
        auto_declared = []
        overridden = []

        def get_param(name: str, default: Any, param_type: type) -> Any:
            """Helper to get parameter with type checking and default fallback."""
            try:
                value = node.get_parameter(
                    f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.{name}"
                ).value
                if not isinstance(value, param_type):
                    node_logger.error(
                        f"Parameter {name} has wrong type: {type(value)}, expected {param_type}. Using default: {default}"
                    )
                    return default
                # Parameter was found and is valid - it's an override (even if same as default)
                overridden.append((name, value))
                return value
            except (ParameterUninitializedException, ParameterNotDeclaredException):
                # Auto-declare parameter with default value
                node.declare_parameter(
                    f"{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.{name}", default
                )
                auto_declared.append((name, default))
                node_logger.debug(
                    f"Parameter {name} not found, using default: {default}"
                )
                return default

        self.camera_topic = get_param("camera_topic", "/camera/rgb/image_raw", str)
        self.depth_topic = get_param("depth_topic", "/camera/depth/image_raw", str)
        self.camera_info_topic = get_param(
            "camera_info_topic", "/camera/rgb/camera_info", str
        )
        self.target_frame = get_param("target_frame", "base_link", str)
        self.source_frame = get_param("source_frame", "camera_link", str)
        self.timeout_sec = get_param("timeout_sec", 10.0, float)
        self.conversion_ratio = get_param("conversion_ratio", 0.001, float)

        # Log auto-declared parameters
        if auto_declared:
            for name, default in auto_declared:
                logger.info(
                    f"Auto-declared parameter {GRIPPING_POINTS_TOOL_PARAM_PREFIX}.{name} = {default}"
                )

        # Log overridden parameters
        if overridden:
            for name, value in overridden:
                logger.info(
                    f"Overridden parameter {GRIPPING_POINTS_TOOL_PARAM_PREFIX}.{name} = {value}"
                )

    def _validate_topics_early(self) -> None:
        """Validate that required topics exist and provide suggestions if missing."""
        try:
            all_topics = [
                topic[0] for topic in self.connector.get_topics_names_and_types()
            ]
        except Exception:
            # If we can't query topics (e.g., ROS2 not fully initialized), skip validation
            logger.debug("Could not query topics for early validation, skipping")
            return

        required_topics = [
            self.camera_topic,
            self.depth_topic,
            self.camera_info_topic,
        ]
        missing = [t for t in required_topics if t not in all_topics]

        if missing:
            discovered = discover_camera_topics(self.connector)
            warning_msg = self._format_topic_suggestions(missing, discovered)
            logger.warning(warning_msg)
        else:
            logger.debug("All required topics are available")

    def _initialize_components(self) -> None:
        """Initialize PCL components with loaded parameters."""
        self.point_cloud_from_segmentation = PointCloudFromSegmentation(
            connector=self.connector,
            camera_topic=self.camera_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            source_frame=self.source_frame,
            target_frame=self.target_frame,
            conversion_ratio=self.conversion_ratio,
            config=self.segmentation_config,
        )
        self.gripping_point_estimator = GrippingPointEstimator(
            config=self.estimator_config
        )
        self.point_cloud_filter = PointCloudFilter(config=self.filter_config)

    # --------------------- Helper Methods ---------------------

    def _format_topic_suggestions(self, missing: list[str], discovered: dict) -> str:
        """Format topic validation warning message with suggestions."""
        lines = [
            "GetObjectGrippingPointsTool: Some required topics are not currently available:",
            f"  Missing topics: {missing}",
            "  Note: Topics may not be available yet (this is an early check).",
            "  If topics remain unavailable after waiting, check for topic name mismatches.",
            "  To remap to your robot/simulation topics, set ROS2 parameters before tool initialization:",
            f"    node.declare_parameter('{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_topic', '/your/robot/camera/topic')",
            f"    node.declare_parameter('{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.depth_topic', '/your/robot/depth/topic')",
            f"    node.declare_parameter('{GRIPPING_POINTS_TOOL_PARAM_PREFIX}.camera_info_topic', '/your/robot/camera_info/topic')",
        ]

        available_info = []
        if discovered.get("image_topics"):
            available_info.append(
                f"  Available image topics: {discovered['image_topics'][:3]}"
            )
        if discovered.get("depth_topics"):
            available_info.append(
                f"  Available depth topics: {discovered['depth_topics'][:3]}"
            )
        if discovered.get("camera_info_topics"):
            available_info.append(
                f"  Available camera_info topics: {discovered['camera_info_topics'][:3]}"
            )

        if available_info:
            lines.append("  Available topics on your system:")
            lines.extend(available_info)

        return "\n".join(lines)

    def _run_stage(
        self,
        stage_num: int,
        stage_name: str,
        stage_func: Callable,
        empty_result_msg: Optional[str],
        debug_config: Optional[dict],
    ) -> tuple[Any, float]:
        """Execute a pipeline stage with timing and optional debug logging.

        Returns:
            Tuple of (stage_result, stage_time). If empty_result_msg is provided and
            result is empty, returns (empty_result_msg, stage_time) instead.
        """
        stage_start = time.time()
        result = stage_func()
        stage_time = time.time() - stage_start

        if debug_config:
            # Extract debug config values
            topic = debug_config.get("topic")
            point_clouds = debug_config.get("point_clouds")
            total_points_before = debug_config.get("total_points_before")
            gripping_points = debug_config.get("gripping_points")
            filtered_point_clouds = debug_config.get("filtered_point_clouds")

            # Auto-detect point_clouds or gripping_points from result if not provided
            if (
                point_clouds is None
                and gripping_points is None
                and isinstance(result, list)
                and len(result) > 0
            ):
                first_item = result[0]
                # Check if result contains point clouds (Nx3 arrays) or gripping points (3-element arrays/lists)
                if isinstance(first_item, np.ndarray):
                    if (
                        len(first_item.shape) == 2 and first_item.shape[1] == 3
                    ):  # Nx3 array (point cloud)
                        point_clouds = result
                    elif (
                        len(first_item.shape) == 1 and len(first_item) == 3
                    ):  # 3-element array (gripping point)
                        gripping_points = result
                elif isinstance(first_item, (list, tuple)) and len(first_item) == 3:
                    # 3-element list/tuple (gripping point)
                    gripping_points = result

            # Calculate additional info for stage 2 (filtering)
            additional_info = None
            if (
                total_points_before is not None
                and isinstance(result, list)
                and len(result) > 0
            ):
                total_points_after = sum(len(pts) for pts in result)
                points_removed = total_points_before - total_points_after
                additional_info = (
                    f"{total_points_after} points remaining ({points_removed} removed)"
                )

            # Calculate total_points for logging
            total_points = None
            if point_clouds:
                total_points = sum(len(pts) for pts in point_clouds)

            self._log_and_publish_stage(
                stage=stage_num,
                stage_name=stage_name,
                instances=len(result) if isinstance(result, list) else 1,
                point_clouds=point_clouds,
                stage_time=stage_time,
                topic=topic,
                additional_info=additional_info,
                total_points=total_points,
                gripping_points=gripping_points,
                filtered_point_clouds=filtered_point_clouds,
            )

        # Check for empty results
        if empty_result_msg and isinstance(result, list) and len(result) == 0:
            return empty_result_msg, stage_time

        return result, stage_time

    def _format_result_message(self, object_name: str, gripping_points: list) -> str:
        """Format the final result message from gripping points."""
        if len(gripping_points) == 0:
            return f"No gripping point found for the object {object_name}\n"
        elif len(gripping_points) == 1:
            return f"The gripping point of the object {object_name} is {gripping_points[0]}\n"
        else:
            message = f"Multiple gripping points found for the object {object_name}\n"
            for i, gp in enumerate(gripping_points):
                message += (
                    f"The gripping point of the object {i + 1} {object_name} is {gp}\n"
                )
            return message

    # --------------------- Debug Helpers ---------------------

    def _log_and_publish_stage(
        self,
        stage: int,
        stage_name: str,
        instances: int,
        point_clouds: list | None,
        stage_time: float,
        topic: str | None,
        additional_info: str | None = None,
        total_points: int | None = None,
        gripping_points: list | None = None,
        filtered_point_clouds: list | None = None,
    ) -> None:
        """Log stage information and publish debug data if applicable.

        Args:
            stage: Stage number (1, 2, or 3)
            stage_name: Human-readable stage name
            instances: Number of instances found/processed
            point_clouds: Point cloud data to publish (if applicable)
            stage_time: Time taken for this stage in seconds
            topic: ROS2 topic to publish to (None to skip publishing)
            additional_info: Optional additional information to include in log
            total_points: Total number of points (if not provided, will calculate from point_clouds)
            gripping_points: Gripping points for stage 3
            filtered_point_clouds: Filtered point clouds for stage 3
        """
        if total_points is None:
            total_points = sum(len(pts) for pts in point_clouds) if point_clouds else 0

        log_msg = (
            f"[Stage {stage}: {stage_name}] Found {instances} instance(s)"
            + (f", {total_points} total points" if total_points > 0 else "")
            + (f", {additional_info}" if additional_info else "")
            + f", took {stage_time:.3f}s"
        )
        logger.info(log_msg)

        # Publish point cloud data for stages 1 and 2
        if topic and point_clouds:
            self._publish_point_cloud_debug(point_clouds, topic)

        # Publish gripping points for stage 3
        if stage == 3 and gripping_points and filtered_point_clouds:
            self._publish_gripping_points_debug(filtered_point_clouds, gripping_points)

    def _publish_point_cloud_debug(
        self,
        point_clouds: list,
        topic_name: str,
        publish_duration: float = DEBUG_PUBLISH_DURATION_SEC,
    ) -> None:
        """Publish point cloud debug data to ROS2 topic for visualization in RVIZ.

        Args:
            point_clouds: List of point cloud arrays (Nx3) per instance
            topic_name: ROS2 topic name to publish to
            publish_duration: Duration in seconds to publish the data
        """
        if not point_clouds:
            return

        logger.warning(
            f"Debug mode: Publishing point cloud data to {topic_name} for {publish_duration}s "
            "(adds computational overhead - not suitable for production)"
        )

        all_points = np.concatenate(point_clouds, axis=0)
        msg = PointCloud()
        self._set_point_cloud_header(msg)
        msg.points = [
            Point32(x=float(p[0]), y=float(p[1]), z=float(p[2])) for p in all_points
        ]

        pub = self.connector.node.create_publisher(PointCloud, topic_name, 10)
        sleep_duration = 1.0 / DEBUG_PUBLISH_RATE_HZ
        start_time = time.time()

        while time.time() - start_time < publish_duration:
            self._update_point_cloud_header(msg)
            pub.publish(msg)
            time.sleep(sleep_duration)

    def _publish_gripping_points_debug(
        self,
        point_clouds: list,
        gripping_points: list,
        publish_duration: float = DEBUG_PUBLISH_DURATION_SEC,
    ) -> None:
        """Publish gripping points debug data to ROS2 topics for visualization in RVIZ.

        Args:
            point_clouds: List of filtered point cloud arrays (Nx3) per instance
            gripping_points: List of gripping point arrays (3,) per instance
            publish_duration: Duration in seconds to publish the data
        """
        obj_points_xyz = [np.array(pc, dtype=np.float32) for pc in point_clouds]
        gripping_points_xyz = [np.array(gp, dtype=np.float32) for gp in gripping_points]

        _publish_gripping_point_debug_data(
            self.connector,
            obj_points_xyz,
            gripping_points_xyz,
            base_frame_id=self.target_frame,
            publish_duration=publish_duration,
        )

    def _set_point_cloud_header(self, msg: PointCloud) -> None:
        """Set initial header for PointCloud message."""
        msg.header.frame_id = self.target_frame
        msg.header.stamp = self.connector.node.get_clock().now().to_msg()

    def _update_point_cloud_header(self, msg: PointCloud) -> None:
        """Update timestamp in PointCloud message header."""
        msg.header.stamp = self.connector.node.get_clock().now().to_msg()
