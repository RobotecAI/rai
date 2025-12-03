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

import logging
from pathlib import Path
from typing import Dict

import rclpy
import yaml
from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Point
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray

from rai_semap.core.backend.sqlite_backend import SQLiteBackend
from rai_semap.core.semantic_map_memory import SemanticMapMemory
from rai_semap.utils.ros2_log import ROS2LogHandler


class SemanticMapVisualizer(Node):
    """ROS2 node for visualizing semantic map annotations in RViz2."""

    def __init__(self):
        super().__init__("semantic_map_visualizer")

        handler = ROS2LogHandler(self)
        handler.setLevel(logging.DEBUG)
        python_logger = logging.getLogger("rai_semap")
        python_logger.setLevel(logging.DEBUG)
        python_logger.handlers.clear()
        python_logger.addHandler(handler)
        python_logger.propagate = False

        self._initialize_parameters()
        self._initialize_semap_memory()
        self._initialize_publisher()
        self._initialize_timer()

        self.class_colors, self.default_color = self._generate_class_colors()

    def _initialize_parameters(self):
        """Initialize ROS2 parameters."""
        parameters = [
            (
                "database_path",
                "semantic_map.db",
                ParameterType.PARAMETER_STRING,
                "Path to SQLite database file",
            ),
            (
                "location_id",
                "rosbot_xl_demo",
                ParameterType.PARAMETER_STRING,
                "Location identifier to query",
            ),
            (
                "map_frame_id",
                "map",
                ParameterType.PARAMETER_STRING,
                "Frame ID of the SLAM map",
            ),
            (
                "map_resolution",
                0.05,
                ParameterType.PARAMETER_DOUBLE,
                "OccupancyGrid resolution (meters/pixel)",
            ),
            (
                "marker_topic",
                "/semantic_map_markers",
                ParameterType.PARAMETER_STRING,
                "Topic for publishing MarkerArray messages",
            ),
            (
                "update_rate",
                1.0,
                ParameterType.PARAMETER_DOUBLE,
                "Rate (Hz) for updating markers",
            ),
            (
                "marker_scale",
                0.3,
                ParameterType.PARAMETER_DOUBLE,
                "Scale factor for marker size",
            ),
            (
                "show_text_labels",
                True,
                ParameterType.PARAMETER_BOOL,
                "Whether to show text labels with object class names",
            ),
            (
                "marker_lifetime",
                0.0,
                ParameterType.PARAMETER_DOUBLE,
                "Marker lifetime in seconds (0 = never expire)",
            ),
            (
                "class_colors_config",
                "",
                ParameterType.PARAMETER_STRING,
                "Path to YAML file with class color definitions (empty = use default in config/)",
            ),
        ]

        for name, default, param_type, description in parameters:
            self.declare_parameter(
                name,
                default,
                descriptor=ParameterDescriptor(
                    type=param_type, description=description
                ),
            )

    def _get_string_parameter(self, name: str) -> str:
        """Get string parameter value."""
        return self.get_parameter(name).get_parameter_value().string_value

    def _get_double_parameter(self, name: str) -> float:
        """Get double parameter value."""
        return self.get_parameter(name).get_parameter_value().double_value

    def _get_bool_parameter(self, name: str) -> bool:
        """Get bool parameter value."""
        return self.get_parameter(name).get_parameter_value().bool_value

    def _initialize_semap_memory(self):
        """Initialize semantic map memory backend."""
        database_path = self._get_string_parameter("database_path")
        location_id = self._get_string_parameter("location_id")
        map_frame_id = self._get_string_parameter("map_frame_id")
        map_resolution = self._get_double_parameter("map_resolution")

        backend = SQLiteBackend(database_path)
        self.memory = SemanticMapMemory(
            backend=backend,
            location_id=location_id,
            map_frame_id=map_frame_id,
            resolution=map_resolution,
        )
        self.get_logger().info(
            f"Initialized semantic map memory: location_id={location_id}, "
            f"map_frame_id={map_frame_id}, database_path={database_path}"
        )

    def _initialize_publisher(self):
        """Initialize marker publisher."""
        marker_topic = self._get_string_parameter("marker_topic")
        self.marker_publisher = self.create_publisher(MarkerArray, marker_topic, 10)
        self.get_logger().info(f"Publishing markers to: {marker_topic}")

    def _initialize_timer(self):
        """Initialize update timer."""
        update_rate = self._get_double_parameter("update_rate")
        timer_period = 1.0 / update_rate if update_rate > 0 else 1.0
        self.timer = self.create_timer(timer_period, self._update_markers)
        self.get_logger().info(f"Update rate: {update_rate} Hz")

    def _generate_class_colors(self) -> tuple[Dict[str, ColorRGBA], ColorRGBA]:
        """Load color map for object classes from YAML config."""
        config_path = self._get_string_parameter("class_colors_config")

        if config_path:
            yaml_path = Path(config_path)
        else:
            current_dir = Path(__file__).parent
            yaml_path = current_dir / "config" / "visualizer.yaml"

        default_color = ColorRGBA(r=0.5, g=0.5, b=0.5, a=0.8)
        colors = {}
        if yaml_path.exists():
            try:
                with open(yaml_path, "r") as f:
                    config = yaml.safe_load(f)

                default_color_list = config.get("default_color", [0.5, 0.5, 0.5, 0.8])
                default_color = ColorRGBA(
                    r=default_color_list[0],
                    g=default_color_list[1],
                    b=default_color_list[2],
                    a=default_color_list[3] if len(default_color_list) > 3 else 0.8,
                )

                class_colors_config = config.get("class_colors", {})
                for class_name, color_value in class_colors_config.items():
                    if isinstance(color_value, list):
                        colors[class_name] = ColorRGBA(
                            r=color_value[0],
                            g=color_value[1],
                            b=color_value[2],
                            a=color_value[3] if len(color_value) > 3 else 0.8,
                        )
                    else:
                        colors[class_name] = ColorRGBA(
                            r=color_value.get("r", 0.5),
                            g=color_value.get("g", 0.5),
                            b=color_value.get("b", 0.5),
                            a=color_value.get("a", 0.8),
                        )
                self.get_logger().info(
                    f"Loaded {len(colors)} class colors from {yaml_path}"
                )
            except Exception as e:
                self.get_logger().warning(
                    f"Failed to load class colors from {yaml_path}: {e}"
                )
        else:
            self.get_logger().warning(
                f"Class colors config file not found: {yaml_path}"
            )

        return colors, default_color

    def _get_class_color(self, object_class: str) -> ColorRGBA:
        """Get color for object class, with fallback for unknown classes."""
        if object_class in self.class_colors:
            return self.class_colors[object_class]
        return self.default_color

    def _create_sphere_marker(self, annotation, marker_id: int, scale: float) -> Marker:
        """Create a sphere marker for an annotation."""
        marker = Marker()
        marker.header.frame_id = self._get_string_parameter("map_frame_id")
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = annotation.object_class
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD

        marker.pose = annotation.pose
        marker.scale.x = scale
        marker.scale.y = scale
        marker.scale.z = scale

        color = self._get_class_color(annotation.object_class)
        color.a = 0.6 + 0.4 * annotation.confidence
        marker.color = color

        marker_lifetime = self._get_double_parameter("marker_lifetime")
        if marker_lifetime > 0:
            lifetime_duration = Duration()
            lifetime_duration.sec = int(marker_lifetime)
            lifetime_duration.nanosec = int(
                (marker_lifetime - int(marker_lifetime)) * 1e9
            )
            marker.lifetime = lifetime_duration

        return marker

    def _create_text_marker(self, annotation, marker_id: int, scale: float) -> Marker:
        """Create a text marker for an annotation label."""
        marker = Marker()
        marker.header.frame_id = self._get_string_parameter("map_frame_id")
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = f"{annotation.object_class}_text"
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD

        marker.pose = annotation.pose
        marker.pose.position.z += scale * 0.5
        marker.scale.z = scale * 0.3

        marker.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
        marker.text = f"{annotation.object_class}\n{annotation.confidence:.2f}"

        marker_lifetime = self._get_double_parameter("marker_lifetime")
        if marker_lifetime > 0:
            lifetime_duration = Duration()
            lifetime_duration.sec = int(marker_lifetime)
            lifetime_duration.nanosec = int(
                (marker_lifetime - int(marker_lifetime)) * 1e9
            )
            marker.lifetime = lifetime_duration

        return marker

    def _update_markers(self):
        """Query database and publish markers."""
        location_id = self._get_string_parameter("location_id")
        center = Point(x=0.0, y=0.0, z=0.0)

        try:
            annotations = self.memory.query_by_location(
                center, radius=1e10, location_id=location_id
            )
        except Exception as e:
            self.get_logger().error(f"Failed to query annotations: {e}")
            return

        if not annotations:
            self.get_logger().debug("No annotations found")
            marker_array = MarkerArray()
            marker_array.markers = []
            self.marker_publisher.publish(marker_array)
            return

        marker_array = MarkerArray()
        marker_scale = self._get_double_parameter("marker_scale")
        show_text = self._get_bool_parameter("show_text_labels")

        marker_id = 0
        for annotation in annotations:
            sphere_marker = self._create_sphere_marker(
                annotation, marker_id, marker_scale
            )
            marker_array.markers.append(sphere_marker)
            marker_id += 1

            if show_text:
                text_marker = self._create_text_marker(
                    annotation, marker_id, marker_scale
                )
                marker_array.markers.append(text_marker)
                marker_id += 1

        self.marker_publisher.publish(marker_array)
        self.get_logger().debug(
            f"Published {len(annotations)} annotations as {len(marker_array.markers)} markers"
        )


def main(args=None):
    """Main entry point for the semantic map visualizer."""
    rclpy.init(args=args)
    node = SemanticMapVisualizer()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
