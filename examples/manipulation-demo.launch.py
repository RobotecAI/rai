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

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node, RosTimer
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    # Declare the game_launcher argument
    game_launcher_arg = DeclareLaunchArgument(
        "game_launcher",
        default_value="",
        description="Path to the game launcher executable",
    )

    return LaunchDescription(
        [
            # Include the game_launcher argument
            game_launcher_arg,
            # Launch the openset nodes
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        FindPackageShare("rai_bringup"),
                        "/launch/openset.launch.py",
                    ]
                ),
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        "src/examples/rai-manipulation-demo/Project/Examples/panda_moveit_config_demo.launch.py",
                    ]
                ),
            ),
            # Launch the robotic_manipulation node after 3 seconds, to ensure the moveit node is ready
            RosTimer(
                period=3.0,
                actions=[
                    Node(
                        package="robotic_manipulation",
                        executable="robotic_manipulation",
                        name="robotic_manipulation_node",
                        output="screen",
                    ),
                ],
            ),
            # Launch the game launcher
            ExecuteProcess(
                cmd=[
                    LaunchConfiguration("game_launcher"),
                    "-bg_ConnectToAssetProcessor=0",
                ],
                output="screen",
            ),
        ]
    )
