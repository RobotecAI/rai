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

import sys
import tempfile
from pathlib import Path
from typing import Optional

import pytest
import rclpy
from geometry_msgs.msg import Pose

# Add src/rai_semap to Python path
rai_semap_path = Path(__file__).parent.parent.parent / "src" / "rai_semap"
sys.path.insert(0, str(rai_semap_path))

from rai_semap.core.semantic_map_memory import SemanticAnnotation  # noqa: E402

# Common test constants
TEST_LOCATION_ID = "test_location"
TEST_DETECTION_SOURCE = "GroundingDINO"
TEST_SOURCE_FRAME = "camera_frame"
TEST_BASE_TIMESTAMP = 1234567890


@pytest.fixture(scope="module")
def ros2_context():
    """Initialize ROS2 context for testing."""
    rclpy.init()
    yield
    rclpy.shutdown()


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield db_path
    Path(db_path).unlink(missing_ok=True)


def make_pose(x: float, y: float, z: float = 0.0) -> Pose:
    """Create a Pose with specified position."""
    pose = Pose()
    pose.position.x = x
    pose.position.y = y
    pose.position.z = z
    return pose


def make_annotation(
    annotation_id: str,
    object_class: str,
    x: float,
    y: float,
    z: float = 0.0,
    confidence: float = 0.9,
    timestamp: int = TEST_BASE_TIMESTAMP,
    detection_source: str = TEST_DETECTION_SOURCE,
    source_frame: str = TEST_SOURCE_FRAME,
    location_id: str = TEST_LOCATION_ID,
    vision_detection_id: Optional[str] = None,
    metadata: Optional[dict] = None,
) -> SemanticAnnotation:
    """Create a SemanticAnnotation with common defaults."""
    pose = make_pose(x, y, z)
    return SemanticAnnotation(
        id=annotation_id,
        object_class=object_class,
        pose=pose,
        confidence=confidence,
        timestamp=timestamp,
        detection_source=detection_source,
        source_frame=source_frame,
        location_id=location_id,
        vision_detection_id=vision_detection_id,
        metadata=metadata,
    )
