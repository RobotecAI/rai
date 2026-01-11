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

"""Shared test fixtures and helpers for config tests."""

from pathlib import Path
from unittest.mock import MagicMock

import yaml


def create_yaml_config_file(config_path: Path, config_data: dict) -> None:
    """Create a YAML config file with given data.

    Args:
        config_path: Path where the YAML file should be created
        config_data: Dictionary to write as YAML
    """
    with open(config_path, "w") as f:
        yaml.dump(config_data, f)


def create_python_config_file(config_path: Path, config_content: str) -> None:
    """Create a Python config file with given content.

    Args:
        config_path: Path where the Python file should be created
        config_content: String content to write to the file
    """
    with open(config_path, "w") as f:
        f.write(config_content)


def create_mock_node_with_parameter(param_name: str, param_value: str) -> MagicMock:
    """Create a mock ROS2 node with a parameter set.

    Args:
        param_name: Name of the parameter
        param_value: Value of the parameter

    Returns:
        Mock node with parameter configured
    """
    mock_node = MagicMock()
    mock_param = MagicMock()
    mock_param.value = param_value
    mock_node.has_parameter.return_value = True
    mock_node.get_parameter.return_value = mock_param
    return mock_node


def create_mock_node_without_parameter() -> MagicMock:
    """Create a mock ROS2 node without the parameter.

    Returns:
        Mock node without parameter
    """
    mock_node = MagicMock()
    mock_node.has_parameter.return_value = False
    return mock_node
