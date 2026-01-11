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

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, OpaqueFunction


def generate_detection_publisher_cmd(context):
    """Generate detection publisher command with conditional parameters."""
    detection_publisher_config = context.launch_configurations.get(
        "detection_publisher_config", ""
    )
    perception_utils_config = context.launch_configurations.get(
        "perception_utils_config", ""
    )

    cmd = [
        "python",
        "-m",
        "rai_perception.components.detection_publisher",
        "--ros-args",
    ]

    # Add config file parameters only if provided
    if detection_publisher_config:
        cmd.extend(["-p", f"detection_publisher_config:={detection_publisher_config}"])
    if perception_utils_config:
        cmd.extend(["-p", f"perception_utils_config:={perception_utils_config}"])

    return [ExecuteProcess(cmd=cmd, output="screen")]


def generate_semap_cmd(context):
    """Generate semantic map node command with conditional parameters."""
    node_config = context.launch_configurations.get("node_config", "")

    cmd = [
        "python",
        "-m",
        "rai_semap.ros2.node",
        "--ros-args",
    ]

    # Add config file parameter only if provided
    if node_config:
        cmd.extend(["-p", f"node_config:={node_config}"])

    return [ExecuteProcess(cmd=cmd, output="screen")]


def generate_launch_description():
    # Declare launch arguments
    node_config_arg = DeclareLaunchArgument(
        "node_config",
        default_value="",
        description="Path to node YAML config file (empty = use default in config/)",
    )

    detection_publisher_config_arg = DeclareLaunchArgument(
        "detection_publisher_config",
        default_value="",
        description="Path to detection_publisher YAML config file (empty = use default in config/)",
    )

    perception_utils_config_arg = DeclareLaunchArgument(
        "perception_utils_config",
        default_value="",
        description="Path to perception_utils YAML config file (empty = use default in config/)",
    )

    return LaunchDescription(
        [
            node_config_arg,
            detection_publisher_config_arg,
            perception_utils_config_arg,
            OpaqueFunction(function=generate_detection_publisher_cmd),
            OpaqueFunction(function=generate_semap_cmd),
        ]
    )
