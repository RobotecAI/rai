from typing import List

import rclpy
from langchain_core.tools import BaseTool
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from rai import get_llm_model
from rai.agents.langchain.core import create_react_runnable
from rai.communication.ros2 import wait_for_ros2_services, wait_for_ros2_topics
from rai.communication.ros2.connectors import ROS2Connector
from rai.tools.ros2.manipulation import (
    GetObjectPositionsTool,
    MoveObjectFromToTool,
)
from rai.tools.ros2.simple import GetROS2ImageConfiguredTool
from rai_perception.tools import GetGrabbingPointTool

from rai_whoami.models import EmbodimentInfo


def create_agent():
    connector = ROS2Connector(executor_type="single_threaded")

    required_services = ["/grounded_sam_segment", "/grounding_dino_classify"]
    required_topics = ["/color_image5", "/depth_image5", "/color_camera_info5"]
    wait_for_ros2_services(connector, required_services)
    wait_for_ros2_topics(connector, required_topics)

    node = connector.node
    node.declare_parameter("conversion_ratio", 1.0)

    camera_tool = GetROS2ImageConfiguredTool(connector=connector, topic="/color_image5")
    tools: List[BaseTool] = [
        GetObjectPositionsTool(
            connector=connector,
            target_frame="panda_link0",
            source_frame="RGBDCamera5",
            camera_topic="/color_image5",
            depth_topic="/depth_image5",
            camera_info_topic="/color_camera_info5",
            get_grabbing_point_tool=GetGrabbingPointTool(connector=connector),
        ),
        MoveObjectFromToTool(connector=connector, manipulator_frame="panda_link0"),
        camera_tool,
    ]

    llm = get_llm_model(model_type="complex_model", streaming=True)
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/manipulation_embodiment.json"
    )

    agent = create_react_runnable(
        llm=llm,
        tools=tools,
        system_prompt=embodiment_info.to_langchain(),
    )
    return agent, camera_tool


def get_manipulation_launch_description():
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
            game_launcher_arg,
            launch_game_launcher,
            launch_openset,
            launch_moveit,
            launch_robotic_manipulation,
        ]
    )


def get_manipulation_launch_description_no_binary():
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
