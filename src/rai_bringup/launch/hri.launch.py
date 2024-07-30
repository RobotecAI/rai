from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    IncludeLaunchDescription,
    OpaqueFunction,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def include_tts_launch(context, *args, **kwargs):
    tts_vendor = LaunchConfiguration("tts_vendor").perform(context)
    if tts_vendor == "opentts":
        launch_file = "opentts.launch.py"
    else:
        launch_file = "elevenlabs.launch.py"

    return [
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                [FindPackageShare("rai_tts"), f"/launch/{launch_file}"]
            )
        )
    ]


def generate_launch_description():
    tts_vendor_arg = DeclareLaunchArgument(
        "tts_vendor",
        default_value="elevenlabs",
        description="TTS vendor to use (opentts or elevenlabs)",
    )

    robot_description_package_arg = DeclareLaunchArgument(
        "robot_description_package",
        default_value="husarion_whoami",
        description="Robot description package to use",
    )

    asr_node = Node(
        package="rai_asr", executable="asr_node", name="asr_node", output="screen"
    )

    hmi_node = Node(
        package="rai_hmi",
        executable="hmi_node",
        name="hmi_node",
        parameters=[
            {
                "robot_description_package": LaunchConfiguration(
                    "robot_description_package"
                )
            }
        ],
        output="screen",
    )

    tts_launch_inclusion = OpaqueFunction(function=include_tts_launch)

    whoami_node = Node(
        package="rai_whoami",
        executable="rai_whoami_node",
        name="rai_whoami_node",
        parameters=[
            {
                "robot_description_package": LaunchConfiguration(
                    "robot_description_package"
                )
            }
        ],
        output="screen",
    )

    return LaunchDescription(
        [
            tts_vendor_arg,
            robot_description_package_arg,
            asr_node,
            hmi_node,
            tts_launch_inclusion,
            whoami_node,
        ]
    )
