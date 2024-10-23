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
from launch.actions import DeclareLaunchArgument, ExecuteProcess
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


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
            # Launch the manipulation demo Python script
            ExecuteProcess(
                cmd=["python3", "examples/manipulation-demo.py"], output="screen"
            ),
            # Launch the robotic_manipulation node
            Node(
                package="robotic_manipulation",
                executable="robotic_manipulation",
                name="robotic_manipulation_node",
                output="screen",
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
