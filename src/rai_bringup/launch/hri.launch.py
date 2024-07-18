from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    asr_node = Node(
        package="rai_asr", executable="asr_node", name="asr_node", output="screen"
    )

    hmi_node = Node(
        package="rai_hmi", executable="hmi_node", name="hmi_node", output="screen"
    )

    tts_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            [FindPackageShare("rai_tts"), "/launch/elevenlabs.launch.py"]
        )
    )

    return LaunchDescription([asr_node, hmi_node, tts_launch])
