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
# See the License for the specific language governing permissions and
# limitations under the License.

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


# TODO (mkotynia) think about separation of launches
def generate_launch_description():
    launch_moveit = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                "src/examples/rai-manipulation-demo/Project/Examples/panda_moveit_config_demo.launch.py",
            ]
        )
    )

    launch_robotic_manipulation = Node(
        package="robotic_manipulation",
        executable="robotic_manipulation",
        output="screen",
        parameters=[
            {"use_sim_time": True},
        ],
    )

    launch_openset = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [
                FindPackageShare("rai_bringup"),
                "/launch/openset.launch.py",
            ]
        ),
    )

    return LaunchDescription(
        [
            launch_openset,
            launch_moveit,
            launch_robotic_manipulation,
        ]
    )
