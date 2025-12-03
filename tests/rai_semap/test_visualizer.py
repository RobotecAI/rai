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

import tempfile
from pathlib import Path

import pytest
import rclpy
import yaml
from std_msgs.msg import ColorRGBA

from rai_semap.ros2.visualizer import SemanticMapVisualizer


def set_parameter(node, name: str, value, param_type):
    """Helper to set a single parameter on a node."""
    node.set_parameters([rclpy.parameter.Parameter(name, param_type, value)])


@pytest.fixture
def temp_config_file():
    """Create a temporary config file for testing."""
    config_data = {
        "default_color": [0.5, 0.5, 0.5, 0.8],
        "class_colors": {
            "chair": [0.2, 0.8, 0.4, 0.8],
            "table": [0.6, 0.2, 0.8, 0.8],
        },
    }
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_data, f)
        config_path = f.name
    yield config_path
    Path(config_path).unlink(missing_ok=True)


@pytest.fixture
def visualizer(ros2_context, temp_db_path):
    """Create a SemanticMapVisualizer instance for testing."""
    node = SemanticMapVisualizer()
    set_parameter(
        node, "database_path", temp_db_path, rclpy.parameter.Parameter.Type.STRING
    )
    yield node
    node.destroy_node()


def test_visualizer_initialization(visualizer):
    """Test that SemanticMapVisualizer initializes correctly."""
    assert visualizer is not None
    assert visualizer.get_name() == "semantic_map_visualizer"
    assert visualizer.class_colors is not None
    assert visualizer.default_color is not None
    assert visualizer.marker_publisher is not None


def test_generate_class_colors(visualizer):
    """Test generating class colors from config."""
    colors, default_color = visualizer._generate_class_colors()

    assert isinstance(colors, dict)
    assert isinstance(default_color, ColorRGBA)
    assert default_color.r == 0.5
    assert default_color.g == 0.5
    assert default_color.b == 0.5
    assert default_color.a == 0.8


def test_generate_class_colors_from_custom_config(
    ros2_context, temp_db_path, temp_config_file
):
    """Test generating class colors from custom config file."""
    node = SemanticMapVisualizer()
    set_parameter(
        node, "database_path", temp_db_path, rclpy.parameter.Parameter.Type.STRING
    )
    set_parameter(
        node,
        "class_colors_config",
        temp_config_file,
        rclpy.parameter.Parameter.Type.STRING,
    )

    colors, default_color = node._generate_class_colors()

    assert "chair" in colors
    assert "table" in colors
    assert colors["chair"].r == 0.2
    assert colors["chair"].g == 0.8
    assert colors["chair"].b == 0.4
    assert default_color.r == 0.5

    node.destroy_node()


def test_get_class_color(visualizer):
    """Test getting color for object class."""
    visualizer.class_colors["test_class"] = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

    color = visualizer._get_class_color("test_class")
    assert color.r == 1.0
    assert color.g == 0.0
    assert color.b == 0.0

    unknown_color = visualizer._get_class_color("unknown_class")
    assert unknown_color == visualizer.default_color


def test_get_string_parameter(visualizer):
    """Test getting string parameter."""
    set_parameter(
        visualizer,
        "database_path",
        "/test/path.db",
        rclpy.parameter.Parameter.Type.STRING,
    )
    assert visualizer._get_string_parameter("database_path") == "/test/path.db"


def test_get_double_parameter(visualizer):
    """Test getting double parameter."""
    set_parameter(
        visualizer, "map_resolution", 0.1, rclpy.parameter.Parameter.Type.DOUBLE
    )
    assert visualizer._get_double_parameter("map_resolution") == 0.1


def test_get_bool_parameter(visualizer):
    """Test getting bool parameter."""
    set_parameter(
        visualizer, "show_text_labels", False, rclpy.parameter.Parameter.Type.BOOL
    )
    assert visualizer._get_bool_parameter("show_text_labels") is False
