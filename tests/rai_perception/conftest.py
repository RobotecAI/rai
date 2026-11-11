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
# See the specific language governing permissions and
# limitations under the License.

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from rai.communication.ros2 import ROS2Connector
from rclpy.time import Time


def pytest_addoption(parser):
    """Add custom command line options."""
    parser.addoption(
        "--grasp",
        action="store",
        default="default_grasp",
        help="Grasp preset configuration: default_grasp, precise_grasp, or top_grasp",
    )


def pytest_configure(config):
    for warning in (
        "ignore:Importing from timm.models.layers is deprecated:FutureWarning",
        "ignore:rai_perception.vision_markup.boxer.GDBoxer is deprecated:DeprecationWarning",
        "ignore:rai_perception.vision_markup.segmenter.GDSegmenter is deprecated:DeprecationWarning",
        "ignore:rai_perception.vision_markup is deprecated:DeprecationWarning",
    ):
        config.addinivalue_line("filterwarnings", warning)


@pytest.fixture
def grasp(request):
    """Fixture to get grasp preset from command line."""
    return request.config.getoption("--grasp")


def make_mock_connector() -> MagicMock:
    """A stand-in ROS2Connector (it passes isinstance checks).

    ROS parameters live in connector.params (connector.declare_parameter declares a
    default there); receive, call, publish and transform are AsyncMocks; the
    ROS graph is empty; names resolve to absolute names.
    """
    connector = MagicMock(spec=ROS2Connector)
    connector.params = {}
    connector.declare_parameter.side_effect = connector.params.setdefault
    connector.now.return_value = Time()
    connector.resolve.side_effect = lambda name, kind="topic": "/" + name.lstrip("/")
    for name in ("receive", "call", "publish", "transform"):
        setattr(connector, name, AsyncMock())
    connector.get_services_names_and_types.return_value = []
    connector.get_topics_names_and_types.return_value = []
    return connector


@pytest.fixture
def mock_connector() -> MagicMock:
    return make_mock_connector()


def patch_ros2_for_service_tests(mock_connector):
    """Services that create their own connector get mock_connector."""
    return patch(
        "rai_perception.services.base_vision_service.ROS2Connector",
        return_value=mock_connector,
    )


def create_valid_weights_file(weights_path: Path, size_mb: int = 2) -> None:
    """Helper to create a valid weights file for testing.

    Args:
        weights_path: Path where the weights file should be created
        size_mb: Size of the file in megabytes (default: 2MB)
    """
    weights_path.parent.mkdir(parents=True, exist_ok=True)
    weights_path.write_bytes(b"0" * (size_mb * 1024 * 1024))


def create_valid_model_dir(model_dir: Path) -> None:
    """Helper to create a stand-in Hugging Face model directory for testing.

    Args:
        model_dir: Directory where the model files should be created
    """
    model_dir.mkdir(parents=True, exist_ok=True)
    (model_dir / "config.json").write_text("{}")
    (model_dir / "model.safetensors").write_bytes(b"0" * 1024)


def get_weights_path(tmp_path: Path) -> Path:
    """Helper to get the standard weights path for testing.

    Args:
        tmp_path: Temporary directory path

    Returns:
        Path to the weights file
    """
    return tmp_path / "vision" / "weights" / "test_weights.pth"
