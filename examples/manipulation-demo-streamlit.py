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

from pathlib import Path

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage
from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
)
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare
from manipulation_common import create_agent
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
def initialize_graph():
    agent, camera_tool = create_agent()
    return agent, camera_tool


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
    o3de = O3DEngineArmManipulationBridge(ROS2Connector(executor_type="multi_threaded"))

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
    st.title("RAI Manipulation Demo")
    st.markdown("---")

    # Layout selection in sidebar
    st.sidebar.header("Configuration")

    # Get available scenarios for layout selection
    levels = ["medium", "hard", "very_hard"]
    scenarios: list[Scenario] = get_scenarios(levels=levels)
    scenarios = [
        s for s in scenarios if Path(s.scene_config_path).stem in SCENARIO_NAMES
    ]

    # Create layout selection widget
    layout_options = list(SCENARIO_NAMES.keys())
    scenario = "3rc"
    # Determine the current selection index
    current_index = (
        layout_options.index(Path(scenario).stem)
        if Path(scenario).stem in layout_options
        else 0
    )

    selected_layout_option = st.sidebar.selectbox(
        "Select Layout:",
        options=layout_options,
        format_func=lambda x: str(SCENARIO_NAMES.get(x, x)),
        index=current_index,
        help="Choose a scene layout for the manipulation demo",
    )

    # Convert selection back to layout name
    selected_layout = selected_layout_option

    # Display selected layout info
    st.sidebar.info(f"Selected: {SCENARIO_NAMES.get(selected_layout, selected_layout)}")

    # Display current scene info if available (removed to reduce log noise)

    # Check if layout has changed
    if "current_layout" not in st.session_state:
        st.session_state["current_layout"] = selected_layout
    elif st.session_state["current_layout"] != selected_layout:
        # User selected a predefined layout
        st.sidebar.success("Layout changed! Setting up new scene...")
        # Find the scenario path for the selected layout
        selected_scenario_path = None
        selected_scenario_path = get_scenario_path(scenarios, selected_layout)

        if selected_scenario_path and "o3de" in st.session_state:
            # Setup new scene with the selected layout
            try:
                # Clear the current scene first
                try:
                    # Actually clear the scene by despawning all entities
                    while st.session_state["o3de"].spawned_entities:
                        st.session_state["o3de"]._despawn_entity(
                            st.session_state["o3de"].spawned_entities[0]
                        )
                except Exception as clear_error:
                    st.sidebar.warning(f"Could not clear scene: {str(clear_error)}")

                # Setup new scene
                new_scenario = setup_new_scene(
                    st.session_state["o3de"], selected_scenario_path
                )
                st.session_state["current_scenario"] = new_scenario
                st.session_state["current_layout"] = selected_layout

                # Reset agent history for new layout
                st.session_state["messages"] = [
                    AIMessage(content="Hi! I am a robotic arm. What can I do for you?")
                ]
                st.rerun()
            except Exception as e:
                st.sidebar.error(f"Failed to setup new scene: {str(e)}")
        else:
            st.session_state["current_layout"] = selected_layout

    # Add reload button in configuration section
    st.sidebar.markdown("---")
    if st.sidebar.button("Reload Layout", help="Reload the current scene layout"):
        if "o3de" in st.session_state and "current_layout" in st.session_state:
            # Find the scenario path for the current layout
            current_scenario_path = None
            for s in scenarios:
                if Path(s.scene_config_path).stem == st.session_state["current_layout"]:
                    current_scenario_path = s.scene_config_path
                    break

            if current_scenario_path:
                try:
                    # Clear the current scene first
                    st.sidebar.info("Clearing current scene...")
                    try:
                        st.session_state["o3de"].clear_scene()
                        st.sidebar.success("Scene cleared")
                    except Exception as clear_error:
                        st.sidebar.warning(f"Could not clear scene: {str(clear_error)}")

                    # Reload the scene
                    st.sidebar.info(
                        f"Reloading scene: {st.session_state['current_layout']}"
                    )
                    new_scenario = setup_new_scene(
                        st.session_state["o3de"], current_scenario_path
                    )
                    st.session_state["current_scenario"] = new_scenario

                    # Reset agent history for reloaded scene
                    st.session_state["messages"] = [
                        AIMessage(
                            content="Hi! I am a robotic arm. What can I do for you?"
                        )
                    ]
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Failed to reload scene: {str(e)}")
            else:
                st.sidebar.error("Could not find scenario path for current layout")
        else:
            st.sidebar.warning("Demo not initialized yet")

    if "o3de" not in st.session_state:
        selected_scenario_path = get_scenario_path(scenarios, scenario)
        if selected_scenario_path is None:
            st.error(f"Could not find scenario: {scenario}")
            st.stop()
        o3de, initial_scenario = initialize_o3de(
            selected_scenario_path, o3de_config_path
        )
        st.session_state["o3de"] = o3de
        st.session_state["current_scenario"] = initial_scenario

    if "graph" not in st.session_state:
        graph, camera_tool = initialize_graph()
        st.session_state["graph"] = graph
        st.session_state["camera_tool"] = camera_tool

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

    if prompt:
        if "camera_tool" in st.session_state:
            _, artifact = st.session_state["camera_tool"]._run()
        else:
            artifact = None
        message = HumanMultimodalMessage(
            content=prompt, images=artifact.get("images", [])
        )
        st.session_state.messages.append(message)

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
    o3de_config_path = (
        "src/rai_bench/rai_bench/manipulation_o3de/predefined/configs/o3de_config.yaml"
    )
    main(o3de_config_path)
