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

import asyncio
import re

import numpy as np
import pytest
from rai.communication.ros2 import ROS2Connector
from rai_perception import GetObjectGrippingPointsTool, wait_for_perception_dependencies
from rai_perception.components.perception_presets import apply_preset
from rai_perception.components.visualization_utils import (
    save_gripping_points_annotated_image,
)

# Default configurations
MANIPULATION_DEMO_TOPICS = {
    "camera": "/color_image5",
    "depth": "/depth_image5",
    "camera_info": "/color_camera_info5",
}

MANIPULATION_DEMO_FRAMES = {"target": "panda_link0", "source": "RGBDCamera5"}


def extract_gripping_points_from_result(result: str) -> list[np.ndarray]:
    """The (x, y, z) points in a GetObjectGrippingPointsTool result."""
    pattern = r"x=([-0-9.eE+]+), y=([-0-9.eE+]+), z=([-0-9.eE+]+)"
    return [np.array(m, dtype=np.float32) for m in re.findall(pattern, result)]


def run_gripping_points_test(
    test_object: str,
    grasp: str,
    topics: dict[str, str],
    frames: dict[str, str],
    debug_enabled: bool = False,
) -> None:
    """Run gripping points test with given grasp preset configuration."""
    with ROS2Connector() as connector:
        filter_config, estimator_config = apply_preset(grasp)
        tool = GetObjectGrippingPointsTool(
            connector=connector,
            target_frame=frames["target"],
            source_frame=frames["source"],
            camera_topic=topics["camera"],
            depth_topic=topics["depth"],
            camera_info_topic=topics["camera_info"],
            timeout_s=30.0,
            filter_config=filter_config,
            estimator_config=estimator_config,
        )
        print("Waiting for ROS2 services and topics...")
        wait_for_perception_dependencies(connector, [tool])

        print(f"\nTesting {tool.name} with object '{test_object}', grasp '{grasp}'")
        result = tool.run({"object_name": test_object, "debug": debug_enabled})
        print(f"\nTool result:\n{result}")

        gripping_points = extract_gripping_points_from_result(result)
        assert len(gripping_points) > 0, "No gripping points found"
        for i, gp in enumerate(gripping_points, 1):
            print(f"  GP{i}: [{gp[0]:.3f}, {gp[1]:.3f}, {gp[2]:.3f}]")

        if debug_enabled:
            annotated_image_path = f"{test_object}_{grasp}_gripping_points.jpg"
            asyncio.run(
                save_gripping_points_annotated_image(
                    connector,
                    gripping_points,
                    topics["camera"],
                    topics["camera_info"],
                    frames["source"],
                    frames["target"],
                    annotated_image_path,
                )
            )
            print(f"Saved annotated image as '{annotated_image_path}'")


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
