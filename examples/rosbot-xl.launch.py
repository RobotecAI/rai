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
from launch.actions import ExecuteProcess, IncludeLaunchDescription
from launch.launch_description_sources import (
    AnyLaunchDescriptionSource,
    PythonLaunchDescriptionSource,
)
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    game_launcher = LaunchConfiguration("game_launcher")
    return LaunchDescription(
        [
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource(
                    [
                        FindPackageShare("rai_bringup"),
                        "/launch/sim_whoami_demo.launch.py",
                    ]
                ),
                launch_arguments={
                    "allowlist": "examples/rosbot-xl_allowlist.txt",
                    "demo_script": "examples/rosbot-xl-demo.py",
                    "robot_description_package": "rosbot_xl_whoami",
                    "game_launcher": game_launcher,
                }.items(),
            ),
            ExecuteProcess(
                cmd=["bash", "run-nav.bash"],
                cwd="src/examples/rai-rosbot-xl-demo",
                output="screen",
            ),
            IncludeLaunchDescription(
                AnyLaunchDescriptionSource(
                    PathJoinSubstitution(
                        [
                            FindPackageShare("rai_bringup"),
                            "launch",
                            "openset.launch.py",
                        ]
                    )
                )
            ),
        ]
    )
