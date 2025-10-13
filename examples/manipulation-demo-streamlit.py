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
# See the License for the specific language goveself.rning permissions and
# limitations under the License.

import importlib
import sys
from pathlib import Path

import rclpy
import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.communication.ros2.connectors.ros2_connector import ROS2Connector
from rai.messages import HumanMultimodalMessage

from rai_bench.manipulation_o3de import get_scenarios
from rai_bench.manipulation_o3de.benchmark import Scenario
from rai_sim.o3de.o3de_bridge import (
    O3DEngineArmManipulationBridge,
    O3DExROS2SimulationConfig,
)
from rai_sim.simulation_bridge import SceneConfig

manipulation_demo = importlib.import_module("manipulation-demo")


def launch_description():
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


@st.cache_resource
def init_ros():
    rclpy.init()
    return "ros"


@st.cache_resource
def initialize_graph():
    return manipulation_demo.create_agent()


@st.cache_resource
def initialize_o3de(scenario_path: str, o3de_config_path: str):
    simulation_config = O3DExROS2SimulationConfig.load_config(
        config_path=Path(o3de_config_path)
    )
    scene_config = SceneConfig.load_base_config(Path(scenario_path))
    scenario = Scenario(
        task=None,
        scene_config=scene_config,
        scene_config_path=scenario_path,
    )
    o3de = O3DEngineArmManipulationBridge(ROS2Connector())
    o3de.init_simulation(simulation_config=simulation_config)
    o3de.launch_robotic_stack(
        required_robotic_ros2_interfaces=simulation_config.required_robotic_ros2_interfaces,
        launch_description=launch_description(),
    )
    o3de.setup_scene(scenario.scene_config)


def main(scenario: Scenario, simulation_config: O3DExROS2SimulationConfig):
    st.set_page_config(
        page_title="RAI Manipulation Demo",
        page_icon=":robot:",
    )
    st.title("RAI Manipulation Demo")
    st.markdown("---")
    st.sidebar.header("Tool Calls History")

    if "ros" not in st.session_state:
        ros = init_ros()
        st.session_state["ros"] = ros

    if "o3de" not in st.session_state:
        o3de = initialize_o3de(scenario, simulation_config)
        st.session_state["o3de"] = o3de

    if "graph" not in st.session_state:
        graph = initialize_graph()
        st.session_state["graph"] = graph

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            AIMessage(content="Hi! I am a robotic arm. What can I do for you?")
        ]

    prompt = st.chat_input()
    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                st.chat_message("assistant").write(msg.content)
        elif isinstance(msg, HumanMultimodalMessage):
            continue
        elif isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, ToolMessage):
            with st.sidebar.expander(f"Tool: {msg.name}", expanded=False):
                st.code(msg.content, language="json")

    if prompt:
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)
        with st.chat_message("assistant"):
            st_callback = get_streamlit_cb(st.container())
            streamlit_invoke(
                st.session_state["graph"], st.session_state.messages, [st_callback]
            )


if __name__ == "__main__":
    levels = [
        "medium",
        "hard",
        "very_hard",
    ]
    scenarios: list[Scenario] = get_scenarios(levels=levels)
    scenario_names = [Path(s.scene_config_path).stem for s in scenarios]
    print(scenario_names)

    if len(sys.argv) > 1:
        layout = sys.argv[1]
        if layout not in scenario_names:
            raise ValueError(f"Invalid layout: {layout}. Select from {scenario_names}")
    else:
        layout = "3carrots_1a_1t_2bc_2yc"
    o3de_config_path = (
        "src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml"
    )

    scenario_idx = scenario_names.index(layout)
    scenario = str(scenarios[scenario_idx].scene_config_path)

    main(scenario, o3de_config_path)
