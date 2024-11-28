# Copyright (C) 2024 Robotec.AI
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
#

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "max_lines",
                default_value="512",
                description="Maximum number of lines to store in rai_state_logs (integer)",
            ),
            DeclareLaunchArgument(
                "include_meta",
                default_value="true",
                description="Include metadata in rai_state_logs",
                choices=["true", "false"],
            ),
            DeclareLaunchArgument(
                "clear_on_retrieval",
                default_value="true",
                description="Clear logs on retrieval",
                choices=["true", "false"],
            ),
            Node(
                package="rai_state_logs",
                executable="rai_state_logs_node",
                name="rai_state_logs_node",
                output="screen",
                parameters=[
                    {"max_lines": LaunchConfiguration("max_lines")},
                    {"include_meta": LaunchConfiguration("include_meta")},
                    {"clear_on_retrieval": LaunchConfiguration("clear_on_retrieval")},
                ],
            ),
        ]
    )
