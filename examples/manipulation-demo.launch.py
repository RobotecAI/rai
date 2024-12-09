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

import rclpy
from launch import LaunchContext, LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
    OpaqueFunction,
    RegisterEventHandler,
)
from launch.event_handlers import OnExecutionComplete, OnProcessStart
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from rclpy.qos import QoSProfile, ReliabilityPolicy
from rosgraph_msgs.msg import Clock


def generate_launch_description():
    # Declare the game_launcher argument
    game_launcher_arg = DeclareLaunchArgument(
        "game_launcher",
        default_value="",
        description="Path to the game launcher executable",
    )

    launch_game_launcher = ExecuteProcess(
        cmd=[
            LaunchConfiguration("game_launcher"),
            "-bg_ConnectToAssetProcessor=0",
        ],
        output="screen",
    )

    def wait_for_clock_message(context: LaunchContext, *args, **kwargs):
        rclpy.init()
        node = rclpy.create_node("wait_for_game_launcher")
        node.create_subscription(
            Clock,
            "/clock",
            lambda msg: rclpy.shutdown(),
            QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT),
        )
        rclpy.spin(node)
        return None

    # Game launcher will start publishing the clock message after loading the simulation
    wait_for_game_launcher = OpaqueFunction(function=wait_for_clock_message)

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
        name="robotic_manipulation_node",
        output="screen",
        parameters=[
            {"use_sim_time": True},
        ],
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
            # Launch the game launcher and wait for it to load
            launch_game_launcher,
            RegisterEventHandler(
                event_handler=OnProcessStart(
                    target_action=launch_game_launcher,
                    on_start=[
                        wait_for_game_launcher,
                    ],
                )
            ),
            # Launch the MoveIt node after loading the simulation
            RegisterEventHandler(
                event_handler=OnExecutionComplete(
                    target_action=wait_for_game_launcher,
                    on_completion=[
                        launch_moveit,
                        launch_robotic_manipulation,
                    ],
                )
            ),
        ]
    )
