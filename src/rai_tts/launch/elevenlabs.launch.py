import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    config = os.path.join(
        get_package_share_directory("rai_tts"), "config", "elevenlabs.yaml"
    )
    print(config)
    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=config,
                description="Path to the config file",
            ),
            Node(
                package="rai_tts",
                executable="tts_node",
                name="tts_node",
                parameters=[LaunchConfiguration("config_file")],
            ),
        ]
    )
