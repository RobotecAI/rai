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

import shlex

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    robot_description_package = LaunchConfiguration("robot_description_package")
    game_launcher = LaunchConfiguration("game_launcher")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "game_launcher",
                description="Path to the O3DE game launcher executable",
            ),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [FindPackageShare("rai_whoami"), "/launch/rai_whoami.launch.py"]
                )
            ),
            ExecuteProcess(
                cmd=[
                    "python",
                    "examples/rosbot-xl-demo.py",
                    "--allowlist",
                    "examples/rosbotxl_allowlist.txt",
                ],
                output="screen",
            ),
            ExecuteProcess(
                cmd=shlex.split("streamlit run src/rai_hmi/rai_hmi/text_hmi.py")
                + [robot_description_package, "examples/rosbotxl_allowlist.txt"],
                output="screen",
            ),
            ExecuteProcess(
                cmd=[game_launcher, "-bg_ConnectToAssetProcessor=0"],
                output="screen",
            ),
            ExecuteProcess(
                cmd=["bash", "run-nav.bash"],
                cwd="src/examples/rai-rosbot-xl-demo",
                output="screen",
            ),
        ]
    )
