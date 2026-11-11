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

"""Tools for extracting 3D gripping points from objects.

Note: Tools depend on PointCloudFromSegmentation component which is coupled to
GroundingDINO and Grounded SAM interfaces (RAIGroundingDino/RAIGroundedSam service types).

Model support: Currently model-specific (GroundingDINO + Grounded SAM). For future
model-agnostic support, generic RAIDetection/RAISegmentation service interfaces would
be needed. Alternatively, model-specific tools/services can be created for each model.
"""

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, Optional, Type

import numpy as np
from geometry_msgs.msg import Point, Pose, Vector3
from pydantic import BaseModel, Field, PrivateAttr
from rai.communication.ros2 import QoS
from rai.tools import Effect
from rai.tools.ros2.base import BaseROS2Tool
from sensor_msgs_py import point_cloud2
from std_msgs.msg import Header
from visualization_msgs.msg import Marker, MarkerArray

from rai_perception.components.gripping_points import (
    GrippingPointEstimator,
    GrippingPointEstimatorConfig,
    PointCloudFilter,
    PointCloudFilterConfig,
    PointCloudFromSegmentation,
    PointCloudFromSegmentationConfig,
)
from rai_perception.components.perception_presets import apply_preset
from rai_perception.components.service_utils import (
    DETECTION_SERVICE,
    SEGMENTATION_SERVICE,
    missing,
    service_available,
)
from rai_perception.components.topic_utils import discover_camera_topics

DEBUG_RAW_TOPIC = "/debug/gripping_points/raw_point_clouds"
DEBUG_FILTERED_TOPIC = "/debug/gripping_points/filtered_point_clouds"
DEBUG_POINTS_TOPIC = "/debug_gripping_points_pointcloud"
DEBUG_MARKERS_TOPIC = "/debug_gripping_points_markerarray"

# Output label and description suffix of each estimator strategy.
STRATEGY_OUTPUT = {
    "centroid": ("Centroid", ""),
    "top_plane": ("Centroid", " (center of top plane)"),
    "biggest_plane": ("Centroid", " (center of largest plane)"),
}

logger = logging.getLogger(__name__)

# CPU-bound stages run here, not in asyncio's default executor, which
# asyncio.run joins: a timed-out call returns at once, the stage finishes later.
_cpu = ThreadPoolExecutor(max_workers=4, thread_name_prefix="rai_perception")


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

    1. Point Cloud Extraction (PointCloudFromSegmentation): detects and segments
       objects with the detection and segmentation services, then extracts
       point clouds from depth data
    2. Point Cloud Filtering (PointCloudFilter): removes outliers and noise
    3. Gripping Point Estimation (GrippingPointEstimator): estimates gripping
       points from the filtered point clouds

    Topics, frames and service names are tool fields. The whole call, ROS waits
    included, is bounded by timeout_s. Use debug=True to publish intermediate
    pipeline results (latched) for visualization in RViz.
    """

    default_effect = Effect.READ
    target_fields = {
        "camera_topic": "topic",
        "depth_topic": "topic",
        "camera_info_topic": "topic",
        "detection_service_name": "service",
        "segmentation_service_name": "service",
    }
    name: str = "get_object_gripping_points"
    description: str = (
        "Get gripping points for all detected objects of a specified type in the target frame using advanced strategies. "
        "This tool supports multiple gripping strategies (top_grasp, precise_grasp) optimized for specific grasping scenarios. "
        "For general object position queries, use get_object_positions instead. "
        "This tool provides accurate positional data but does not distinguish between different colors of the same object type. "
        "While position detection is reliable, please note that object classification may occasionally be inaccurate."
    )
    args_schema: Type[BaseModel] = GetObjectGrippingPointsToolInput
    timeout_s: Optional[float] = 60.0

    camera_topic: str = Field(default="/camera/rgb/image_raw", exclude=True)
    depth_topic: str = Field(default="/camera/depth/image_raw", exclude=True)
    camera_info_topic: str = Field(default="/camera/rgb/camera_info", exclude=True)
    target_frame: str = Field(default="base_link", exclude=True)
    source_frame: str = Field(default="camera_link", exclude=True)
    detection_service_name: str = Field(default=DETECTION_SERVICE, exclude=True)
    segmentation_service_name: str = Field(default=SEGMENTATION_SERVICE, exclude=True)
    conversion_ratio: float = Field(
        default=1.0,
        exclude=True,
        description="Optional scaling factor applied after encoding-aware depth conversion. "
        "Encoding-aware conversion automatically handles 16UC1/mono16 (mm) and 32FC1 (m) encodings. "
        "Default 1.0 means no additional scaling (recommended).",
    )

    # Pipeline configs; filter and estimator default to the default_grasp preset.
    segmentation_config: PointCloudFromSegmentationConfig = Field(
        default_factory=PointCloudFromSegmentationConfig, exclude=True
    )
    filter_config: Optional[PointCloudFilterConfig] = Field(default=None, exclude=True)
    estimator_config: Optional[GrippingPointEstimatorConfig] = Field(
        default=None, exclude=True
    )

    # Output format customization; by default determined by the estimator strategy.
    output_point_label: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Custom label for points in output (e.g., 'Centroid', 'GrippingPoint').",
    )
    output_description_suffix: Optional[str] = Field(
        default=None,
        exclude=True,
        description="Custom description suffix for output message (e.g., ' (center of top plane)').",
    )

    _extraction: PointCloudFromSegmentation = PrivateAttr()
    _filter: PointCloudFilter = PrivateAttr()
    _estimator: GrippingPointEstimator = PrivateAttr()

    def model_post_init(self, context: Any) -> None:
        super().model_post_init(context)
        if self.filter_config is None or self.estimator_config is None:
            filter_config, estimator_config = apply_preset("default_grasp")
            self.filter_config = self.filter_config or filter_config
            self.estimator_config = self.estimator_config or estimator_config
        self._extraction = PointCloudFromSegmentation(
            connector=self.connector,
            camera_topic=self.camera_topic,
            depth_topic=self.depth_topic,
            camera_info_topic=self.camera_info_topic,
            source_frame=self.source_frame,
            target_frame=self.target_frame,
            conversion_ratio=self.conversion_ratio,
            config=self.segmentation_config,
            detection_service=self.detection_service_name,
            segmentation_service=self.segmentation_service_name,
        )
        self._filter = PointCloudFilter(config=self.filter_config)
        self._estimator = GrippingPointEstimator(config=self.estimator_config)
        logger.info(f"{type(self).__name__} configuration: {self.get_config()}")
        self._warn_missing_topics()

    @property
    def point_cloud_from_segmentation(self) -> PointCloudFromSegmentation:
        return self._extraction

    @property
    def point_cloud_filter(self) -> PointCloudFilter:
        return self._filter

    @property
    def gripping_point_estimator(self) -> GrippingPointEstimator:
        return self._estimator

    def get_config(self) -> Dict[str, Any]:
        """The deployment configuration: topics, frames, services, timeout."""
        return {
            "target_frame": self.target_frame,
            "source_frame": self.source_frame,
            "camera_topic": self.camera_topic,
            "depth_topic": self.depth_topic,
            "camera_info_topic": self.camera_info_topic,
            "timeout_s": self.timeout_s,
            "conversion_ratio": self.conversion_ratio,
            "detection_service_name": self.detection_service_name,
            "segmentation_service_name": self.segmentation_service_name,
        }

    def check_service_dependencies(self) -> Dict[str, bool]:
        """Each required service and whether it is in the ROS graph now."""
        return {
            s: service_available(self.connector, s)
            for s in (self.detection_service_name, self.segmentation_service_name)
        }

    def _warn_missing_topics(self) -> None:
        """An early hint when the configured topics are not (yet) in the graph."""
        try:
            topics = [self.camera_topic, self.depth_topic, self.camera_info_topic]
            absent = missing(self.connector, topics, "topic")
        except Exception:  # the graph may not be queryable yet
            return
        if not absent:
            return
        found = discover_camera_topics(self.connector)
        logger.warning(
            f"{type(self).__name__}: topics {absent} are not available (yet). "
            "Set camera_topic, depth_topic and camera_info_topic to your robot's topics. "
            f"Image topics: {found['image_topics'][:3]}, depth topics: "
            f"{found['depth_topics'][:3]}, camera_info topics: {found['camera_info_topics'][:3]}"
        )

    async def _acall(self, object_name: str, debug: bool = False) -> str:
        start = time.monotonic()
        clouds = await self._extraction.run(object_name)
        start = _log_stage(1, "Point Cloud Extraction", clouds, start)
        if not clouds:
            return f"No {object_name}s detected."
        if debug:
            await self._publish_cloud(DEBUG_RAW_TOPIC, clouds)

        filtered = await _in_thread(self._filter.run, clouds)
        start = _log_stage(2, "Point Cloud Filtering", filtered, start)
        if not filtered:
            return f"No {object_name}s detected after applying filtering"
        if debug:
            await self._publish_cloud(DEBUG_FILTERED_TOPIC, filtered)

        points = await _in_thread(self._estimator.run, filtered)
        _log_stage(3, "Gripping Point Estimation", points, start)
        if debug and points:
            await self._publish_cloud(DEBUG_POINTS_TOPIC, filtered)
            await self._publish_markers(points)
        return self._format_result_message(object_name, points)

    def _format_result_message(self, object_name: str, gripping_points: list) -> str:
        if not gripping_points:
            return f"No {object_name}s detected."
        strategy = self.estimator_config.strategy if self.estimator_config else ""
        label, suffix = STRATEGY_OUTPUT.get(strategy, ("GrippingPoint", ""))
        if self.output_point_label is not None:
            label = self.output_point_label
            plural = label if label.endswith("s") else f"{label}s"
        else:
            plural = "Centroids" if label == "Centroid" else "Gripping points"
        if self.output_description_suffix is not None:
            suffix = self.output_description_suffix

        def fmt(gp: Any) -> str:
            if isinstance(gp, (np.ndarray, list, tuple)) and len(gp) >= 3:
                return f"{label}(x={gp[0]:.6f}, y={gp[1]:.6f}, z={gp[2]:.6f})"
            logger.warning(f"Gripping point has unexpected format: {gp}")
            return str(gp)

        frame = f"in {self.target_frame} frame"
        if len(gripping_points) == 1:
            return f"{plural} of detected {object_name}{suffix} {frame}: {fmt(gripping_points[0])}."
        points = ", ".join(fmt(gp) for gp in gripping_points)
        return f"{plural} of detected {object_name}s{suffix} {frame}: [{points}]. Sizes of the detected objects are unknown."

    def _header(self) -> Header:
        return Header(frame_id=self.target_frame, stamp=self.connector.now().to_msg())

    async def _publish_cloud(self, topic: str, clouds: list) -> None:
        """Publish point clouds (latched, so RViz gets them late) for debugging."""
        logger.warning(f"Debug mode: publishing point clouds to {topic}")
        points = np.concatenate([np.asarray(c, dtype=np.float32) for c in clouds])
        msg = point_cloud2.create_cloud_xyz32(self._header(), points)
        await self.connector.publish(topic, msg, qos=QoS.LATCHED)

    async def _publish_markers(self, gripping_points: list) -> None:
        markers = MarkerArray()
        header = self._header()
        for i, p in enumerate(gripping_points):
            marker = Marker(header=header, id=i, type=Marker.SPHERE, action=Marker.ADD)
            marker.pose = Pose(
                position=Point(x=float(p[0]), y=float(p[1]), z=float(p[2]))
            )
            marker.scale = Vector3(x=0.04, y=0.04, z=0.04)
            marker.color.r, marker.color.a = 1.0, 1.0
            markers.markers.append(marker)
        await self.connector.publish(DEBUG_MARKERS_TOPIC, markers, qos=QoS.LATCHED)


async def _in_thread(fn: Any, arg: Any) -> Any:
    return await asyncio.get_running_loop().run_in_executor(_cpu, fn, arg)


def _log_stage(stage: int, name: str, result: list, start: float) -> float:
    """Log a pipeline stage and return the time it ended."""
    now = time.monotonic()
    logger.info(
        f"[Stage {stage}: {name}] Found {len(result)} instance(s), took {now - start:.3f}s"
    )
    return now


class GetObjectPositionsToolInput(BaseModel):
    """Input schema for GetObjectPositionsTool."""

    object_name: str = Field(
        ...,
        description="The name of the object type to get the positions of (e.g., 'apple', 'cube', 'carrot')",
    )


class GetObjectPositionsTool(GetObjectGrippingPointsTool):
    """get_object_positions: GetObjectGrippingPointsTool with the default_grasp
    preset (centroid strategy), the v1 interface on the new pipeline.

    The name "get_object_positions" reads as a spatial query, so agents use it
    for spatial reasoning more readily than "get_object_gripping_points".
    """

    name: str = "get_object_positions"
    description: str = (
        "Retrieve the positions of all objects of a specified type in the target frame. "
        "This tool provides accurate positional data but does not distinguish between different colors of the same object type. "
        "While position detection is reliable, please note that object classification may occasionally be inaccurate."
    )
    args_schema: Type[BaseModel] = GetObjectPositionsToolInput

    def model_post_init(self, context: Any) -> None:
        self.filter_config, self.estimator_config = apply_preset("default_grasp")
        super().model_post_init(context)

    async def _acall(self, object_name: str) -> str:  # type: ignore[override]
        return await super()._acall(object_name)
