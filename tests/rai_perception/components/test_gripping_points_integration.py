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

"""
Manual test for GetGrippingPointTool with various demo scenarios. Each test:
- Finds gripping points of specified object in the target frame.
- Publishes debug data for visualization.
- Saves annotated image of the gripping points.

The manipulation demo app and rviz2 need to be started before running the test. The test will fail if the gripping points are not found.

Usage:
pytest tests/rai_perception/components/test_gripping_points_integration.py::test_gripping_points_manipulation_demo -m "manual" -s -v --grasp default_grasp
"""

import re
import traceback

import numpy as np
import pytest
import rclpy
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector
from rai_perception import GetObjectGrippingPointsTool
from rai_perception.components.gripping_points import (
    GrippingPointEstimatorConfig,
    PointCloudFilterConfig,
    PointCloudFromSegmentationConfig,
)
from rai_perception.components.perception_presets import apply_preset
from rai_perception.components.visualization_utils import (
    save_gripping_points_annotated_image,
)
from rai_perception.tools.gripping_points_tools import GRIPPING_POINTS_TOOL_PARAM_PREFIX

# Default configurations
MANIPULATION_DEMO_TOPICS = {
    "camera": "/color_image5",
    "depth": "/depth_image5",
    "camera_info": "/color_camera_info5",
}

MANIPULATION_DEMO_FRAMES = {"target": "panda_link0", "source": "RGBDCamera5"}

REQUIRED_SERVICES = ["/segmentation", "/detection"]


def extract_gripping_points_from_result(result: str) -> list[np.ndarray]:
    """Extract gripping points from tool result string.

    Args:
        result: Formatted result string from GetObjectGrippingPointsTool

    Returns:
        List of gripping points as numpy arrays (Nx3)
    """
    pattern = r"\[([0-9.eE\-\+\s]+)\]"
    matches = re.findall(pattern, result)
    gripping_points = []
    for match in matches:
        coords = [float(x) for x in match.split()]
        if len(coords) == 3:
            gripping_points.append(np.array(coords, dtype=np.float32))
    return gripping_points


def _setup_ros2_parameters(
    node,
    topics: dict[str, str],
    frames: dict[str, str],
    timeout_sec: float = 30.0,
    conversion_ratio: float = 1.0,
) -> None:
    """Set up ROS2 parameters for the gripping points tool."""
    param_prefix = GRIPPING_POINTS_TOOL_PARAM_PREFIX
    parameters = {
        "target_frame": frames["target"],
        "source_frame": frames["source"],
        "camera_topic": topics["camera"],
        "depth_topic": topics["depth"],
        "camera_info_topic": topics["camera_info"],
        "timeout_sec": timeout_sec,
        "conversion_ratio": conversion_ratio,
    }
    for param_name, param_value in parameters.items():
        node.declare_parameter(f"{param_prefix}.{param_name}", param_value)


def _create_tool(
    connector: ROS2Connector,
    filter_config: PointCloudFilterConfig,
    estimator_config: GrippingPointEstimatorConfig,
) -> GetObjectGrippingPointsTool:
    """Create and configure GetObjectGrippingPointsTool."""
    return GetObjectGrippingPointsTool(
        connector=connector,
        segmentation_config=PointCloudFromSegmentationConfig(),
        estimator_config=estimator_config,
        filter_config=filter_config,
    )


def run_gripping_points_test(
    test_object: str,
    grasp: str,
    topics: dict[str, str],
    frames: dict[str, str],
    debug_enabled: bool = False,
) -> None:
    """Run gripping points test with given grasp preset configuration."""
    rclpy.init()
    connector = ROS2Connector(executor_type="single_threaded")

    try:
        print("Waiting for ROS2 services and topics...")
        wait_for_ros2_services(connector, REQUIRED_SERVICES)
        wait_for_ros2_topics(connector, list(topics.values()))
        print("All services and topics available")

        _setup_ros2_parameters(connector.node, topics, frames)

        print(
            f"\nTesting GetGrippingPointTool with object '{test_object}', grasp '{grasp}'"
        )

        filter_config, estimator_config = apply_preset(grasp)
        tool = _create_tool(connector, filter_config, estimator_config)
        result = tool._run(object_name=test_object, debug=debug_enabled)
        print(f"\nTool result:\n{result}")

        gripping_points = extract_gripping_points_from_result(result)
        assert len(gripping_points) > 0, "No gripping points found"

        print(f"\nFound {len(gripping_points)} gripping point(s) in target frame:")
        for i, gp in enumerate(gripping_points, 1):
            print(f"  GP{i}: [{gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f}]")

        if debug_enabled:
            annotated_image_path = f"{test_object}_{grasp}_gripping_points.jpg"
            save_gripping_points_annotated_image(
                connector,
                gripping_points,
                topics["camera"],
                topics["camera_info"],
                frames["source"],
                frames["target"],
                annotated_image_path,
            )
            print(f"Saved annotated image as '{annotated_image_path}'")

    except Exception as e:
        print(f"Test failed: {e}")
        traceback.print_exc()
        raise

    finally:
        if hasattr(connector, "executor"):
            connector.executor.shutdown()
        connector.shutdown()


@pytest.mark.manual
def test_gripping_points_manipulation_demo(grasp) -> None:
    """Manual test requiring manipulation-demo app to be started."""
    run_gripping_points_test(
        test_object="cube",
        grasp=grasp,
        topics=MANIPULATION_DEMO_TOPICS,
        frames=MANIPULATION_DEMO_FRAMES,
        debug_enabled=True,
    )
