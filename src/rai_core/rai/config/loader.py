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

import importlib.util
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_yaml_config(
    config_path: Path, node_name: Optional[str] = None
) -> Dict[str, Any]:
    """Load YAML config file, optionally extracting ROS2 parameters.

    Args:
        config_path: Path to YAML config file
        node_name: Optional node name to extract ros__parameters section.
                   If provided, extracts config[node_name][ros__parameters].
                   If None, returns entire config dict.

    Returns:
        Dictionary containing config values

    Raises:
        FileNotFoundError: If config file doesn't exist
        yaml.YAMLError: If YAML parsing fails
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if node_name:
        node_config = config.get(node_name, {})
        return node_config.get("ros__parameters", {})
    return config


def load_python_config(config_path: Path) -> Any:
    """Load Python config file as a module.

    Args:
        config_path: Path to Python config file

    Returns:
        Config module object

    Raises:
        FileNotFoundError: If config file doesn't exist
        ImportError: If module cannot be imported
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("config", config_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load config module from {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_config_path(
    param_name: str,
    node,
    default_dir: Path,
    default_filename: str,
) -> Path:
    """Get config file path from ROS2 parameter or default.

    Args:
        param_name: ROS2 parameter name for config path
        node: ROS2 node instance
        default_dir: Default directory if parameter not set
        default_filename: Default filename if parameter not set

    Returns:
        Path to config file
    """
    if node.has_parameter(param_name):
        config_path_str = node.get_parameter(param_name).value
        if config_path_str:
            return Path(config_path_str)
    return default_dir / default_filename
