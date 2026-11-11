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

import logging
import os
from pathlib import Path

import streamlit as st
from langchain_core.messages import HumanMessage
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from manipulation_common import create_agent
from rai.communication.ros2 import ROS2Connector, encode_image
from rai.messages import human
from rai.ui.streamlit import chat, reset

from rai_bench.manipulation_o3de import get_scenarios
from rai_bench.manipulation_o3de.benchmark import Scenario
from rai_sim.o3de.o3de_bridge import (
    O3DEngineArmManipulationBridge,
    O3DExROS2SimulationConfig,
)
from rai_sim.simulation_bridge import SceneConfig

logger = logging.getLogger(__name__)


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
        launch_arguments={
            "enable_legacy_service_names": "true",  # Enable both legacy and new service names for v1/v2 compatibility
        }.items(),
    )

    return LaunchDescription(
        [
            launch_openset,
            launch_moveit,
            launch_robotic_manipulation,
        ]
    )


@st.cache_resource
def initialize_agent(version: str):
    return create_agent(version=version)


@st.cache_resource
def initialize_o3de(
    scenario_path: str, o3de_config_path: str, agent_version: str = "v2"
):
    simulation_config = O3DExROS2SimulationConfig.load_config(
        config_path=Path(o3de_config_path)
    )

    # Adjust required services based on agent version
    # v1 uses legacy service names, v2 uses new service names
    if agent_version == "v2":
        # Replace legacy service names with new ones for v2
        services = simulation_config.required_robotic_ros2_interfaces["services"]
        services = [
            "/detection" if s == "/grounding_dino_classify" else s for s in services
        ]
        services = [
            "/segmentation" if s == "/grounded_sam_segment" else s for s in services
        ]
        simulation_config.required_robotic_ros2_interfaces["services"] = services

    scene_config = SceneConfig.load_base_config(Path(scenario_path))
    scenario = Scenario(
        task=None,
        scene_config=scene_config,
        scene_config_path=scenario_path,
    )
    o3de = O3DEngineArmManipulationBridge(ROS2Connector())

    # Clear scene at the beginning
    o3de.init_simulation(simulation_config=simulation_config)
    o3de.launch_robotic_stack(
        required_robotic_ros2_interfaces=simulation_config.required_robotic_ros2_interfaces,
        launch_description=launch_description(),
    )
    o3de.setup_scene(scenario.scene_config)
    return o3de, scenario


def setup_new_scene(o3de, scenario_path: str):
    """Setup a new scene with the given scenario path"""
    scene_config = SceneConfig.load_base_config(Path(scenario_path))
    scenario = Scenario(
        task=None,
        scene_config=scene_config,
        scene_config_path=scenario_path,
    )
    o3de.setup_scene(scenario.scene_config)
    return scenario


SCENARIO_NAMES = {
    "3rc": "3 Red Cubes",
    "4carrots": "4 Carrots",
    "2rc_2a": "2 Red Cubes, 2 Apples",
    "3rc_2a_1carrot": "3 Red Cubes, 2 Apples, 1 Carrot",
    "3carrots_3a_2rc": "3 Carrots, 3 Apples, 2 Red Cubes",
}


def get_scenario_path(scenarios, selected_layout: str):
    selected_scenario_path = None
    for s in scenarios:
        if Path(s.scene_config_path).stem == selected_layout:
            selected_scenario_path = s.scene_config_path
            break
    return selected_scenario_path


def main(o3de_config_path: str):
    st.set_page_config(
        page_title="RAI Manipulation Demo",
        page_icon=":robot:",
    )
    st.sidebar.header("Configuration")

    agent_version = os.getenv("AGENT_VERSION", "v2")

    levels = ["medium", "hard", "very_hard"]
    scenarios: list[Scenario] = get_scenarios(levels=levels)
    scenarios = [
        s for s in scenarios if Path(s.scene_config_path).stem in SCENARIO_NAMES
    ]

    selected_layout = st.sidebar.selectbox(
        "Select Layout:",
        options=list(SCENARIO_NAMES),
        format_func=lambda x: str(SCENARIO_NAMES.get(x, x)),
        help="Choose a scene layout for the manipulation demo",
    )
    st.sidebar.info(f"Selected: {SCENARIO_NAMES.get(selected_layout, selected_layout)}")

    if "o3de" not in st.session_state:
        initial_layout = list(SCENARIO_NAMES)[0]
        scenario_path = get_scenario_path(scenarios, initial_layout)
        if scenario_path is None:
            st.error(f"Could not find scenario: {initial_layout}")
            st.stop()
        o3de, initial_scenario = initialize_o3de(
            scenario_path, o3de_config_path, agent_version=agent_version
        )
        st.session_state["o3de"] = o3de
        st.session_state["current_scenario"] = initial_scenario
        st.session_state["current_layout"] = initial_layout

    # After O3DE: the agent waits for the perception services it launches.
    logger.info(f"Initializing agent version: {agent_version}")
    agent, camera_tool = initialize_agent(version=agent_version)
    o3de = st.session_state["o3de"]
    changed = st.session_state["current_layout"] != selected_layout
    st.sidebar.markdown("---")
    reload = st.sidebar.button("Reload Layout", help="Reload the current scene layout")
    if changed or reload:
        scenario_path = get_scenario_path(scenarios, selected_layout)
        if scenario_path is None:
            st.sidebar.error("Could not find scenario path for the layout")
        else:
            try:
                try:
                    o3de.clear_scene()
                except Exception as clear_error:
                    st.sidebar.warning(f"Could not clear scene: {clear_error}")
                st.session_state["current_scenario"] = setup_new_scene(
                    o3de, scenario_path
                )
                st.session_state["current_layout"] = selected_layout
            except Exception as e:
                st.sidebar.error(f"Failed to set up the scene: {e}")
            else:
                reset(agent)  # a new scene makes earlier object positions stale
                st.rerun()

    def with_camera_image(prompt: str) -> HumanMessage:
        image = camera_tool.connector.sync.receive(camera_tool.topic, timeout=5.0)
        return human(prompt, [encode_image(image)])

    chat(
        agent,
        title="RAI Manipulation Demo",
        greeting="Hi! I am a robotic arm. What can I do for you?",
        prepare=with_camera_image,
    )


if __name__ == "__main__":
    levels = [
        "medium",
        "hard",
        "very_hard",
    ]
    scenarios: list[Scenario] = get_scenarios(levels=levels)
    o3de_config_path = (
        "src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml"
    )
    main(o3de_config_path)
