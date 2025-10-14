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
import random
import tempfile
from pathlib import Path

import rclpy
import streamlit as st
import yaml
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

manipulation_demo = importlib.import_module("manipulation-demo-v2")


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
    if not rclpy.ok():
        rclpy.init()
    return ""


@st.cache_resource
def initialize_graph():
    agent, camera_tool = manipulation_demo.create_agent()
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


def generate_random_scenario(object_counts: dict) -> str:
    """Generate a random scenario configuration with specified object counts.

    Args:
        object_counts: Dictionary with object types as keys and counts as values
        e.g., {"apple": 2, "tomato": 1, "blue_cube": 3}

    Returns:
        Path to the generated scenario file
    """
    # Available object types and their prefab names
    object_types = {
        "apple": "apple",
        "tomato": "tomato",
        "carrot": "carrot",
        "blue_cube": "blue_cube",
        "red_cube": "red_cube",
        "yellow_cube": "yellow_cube",
        "corn": "corn",
    }

    entities = []
    entity_counter = 1

    # Generate entities for each object type
    for obj_type, count in object_counts.items():
        if obj_type not in object_types:
            continue

        prefab_name = object_types[obj_type]

        for i in range(count):
            # Generate random position within reasonable bounds
            x = random.uniform(0.1, 0.4)  # Front to back
            y = random.uniform(-0.5, 0.5)  # Left to right
            z = 0.05  # Height on table

            entity = {
                "name": f"{obj_type}{entity_counter}",
                "prefab_name": prefab_name,
                "pose": {
                    "translation": {"x": x, "y": y, "z": z},
                    "rotation": {"x": 0.0, "y": 0.0, "z": 0.0, "w": 1.0},
                },
            }
            entities.append(entity)
            entity_counter += 1

    # Create scenario configuration
    scenario_config = {"entities": entities}

    # Create temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(scenario_config, f, default_flow_style=False)
        return f.name


SCENARIO_NAMES = [
    "1bc_1rc_1yc",
    "1rc_2bc_3yc",
    "1carrot_1a_2t_1bc_1rc_3yc_stacked",
    "4carrots",
    "3a_4t_2bc",
    "2a_1c_2rc",
    "1carrot_1a_1t_1bc_1corn",
]


def get_scenario_path(scenarios, selected_layout: str):
    selected_scenario_path = None
    for s in scenarios:
        if Path(s.scene_config_path).stem == selected_layout:
            selected_scenario_path = s.scene_config_path
            break
    return selected_scenario_path


def main(simulation_config: O3DExROS2SimulationConfig):
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

    # Create layout selection widget with random scenario option
    layout_options = ["🎲 Random Scenario"] + SCENARIO_NAMES
    scenario = "1bc_1rc_1yc"
    # Determine the current selection index
    if (
        "current_layout" in st.session_state
        and st.session_state["current_layout"] == "random_scenario"
    ):
        current_index = 0  # Random scenario is first option
    else:
        current_index = (
            SCENARIO_NAMES.index(Path(scenario).stem) + 1
            if Path(scenario).stem in SCENARIO_NAMES
            else 1
        )

    selected_layout_option = st.sidebar.selectbox(
        "Select Layout:",
        options=layout_options,
        index=current_index,
        help="Choose a scene layout for the manipulation demo",
    )

    # Convert selection back to layout name
    if selected_layout_option == "🎲 Random Scenario":
        selected_layout = "random_scenario"
    else:
        selected_layout = selected_layout_option

    # Display selected layout info
    if selected_layout == "random_scenario":
        st.sidebar.info("Selected: 🎲 Random Scenario")
    else:
        st.sidebar.info(f"Selected: {selected_layout}")

    # Random scenario generator
    st.sidebar.markdown("---")
    st.sidebar.subheader("🎲 Random Scenario Generator")

    # Object count inputs
    object_counts = {}
    available_objects = [
        "apple",
        "tomato",
        "carrot",
        "blue_cube",
        "red_cube",
        "yellow_cube",
        "corn",
    ]

    for obj in available_objects:
        count = st.sidebar.number_input(
            f"Number of {obj.replace('_', ' ').title()}s:",
            min_value=0,
            max_value=10,
            value=0,
            key=f"count_{obj}",
        )
        if count > 0:
            object_counts[obj] = count

    # Generate random scenario button
    if st.sidebar.button(
        "🎲 Generate Random Scenario", disabled=not any(object_counts.values())
    ):
        if any(object_counts.values()):
            try:
                # Generate the scenario file
                scenario_path = generate_random_scenario(object_counts)

                # Load the scenario into the scene
                if "o3de" in st.session_state:
                    # Clear the current scene first
                    st.sidebar.info("🧹 Clearing current scene...")
                    try:
                        st.sidebar.success("✅ Scene cleared")
                    except Exception as clear_error:
                        st.sidebar.warning(
                            f"⚠️ Could not clear scene: {str(clear_error)}"
                        )

                    # Setup new scene with random scenario
                    st.sidebar.info("🎲 Setting up random scenario...")
                    new_scenario = setup_new_scene(
                        st.session_state["o3de"], scenario_path
                    )
                    st.session_state["current_scenario"] = new_scenario
                    st.session_state["current_layout"] = "random_scenario"

                    # Reset agent history for new random scenario
                    st.session_state["messages"] = [
                        AIMessage(
                            content="Hi! I am a robotic arm. What can I do for you?"
                        )
                    ]

                    st.sidebar.success(
                        f"✅ Random scenario generated with {sum(object_counts.values())} objects!"
                    )
                else:
                    st.sidebar.warning("⚠️ Demo not initialized yet")
            except Exception as e:
                st.sidebar.error(f"❌ Failed to generate random scenario: {str(e)}")
        else:
            st.sidebar.warning("⚠️ Please specify at least one object count")

    # Display current scene info if available
    if "current_scenario" in st.session_state:
        current_scene_name = Path(
            st.session_state["current_scenario"].scene_config_path
        ).stem
        if current_scene_name == "random_scenario":
            st.sidebar.success("Active Scene: Random Scenario")
        else:
            st.sidebar.success(f"Active Scene: {current_scene_name}")

    # Check if layout has changed
    if "current_layout" not in st.session_state:
        st.session_state["current_layout"] = selected_layout
    elif st.session_state["current_layout"] != selected_layout:
        if selected_layout == "random_scenario":
            # User selected random scenario from dropdown - show info
            st.sidebar.info(
                "🎲 Random scenario selected. Use the controls below to generate a scenario."
            )
            st.session_state["current_layout"] = selected_layout
        else:
            # User selected a predefined layout
            st.sidebar.success("🔄 Layout changed! Setting up new scene...")
            # Find the scenario path for the selected layout
            selected_scenario_path = None
            selected_scenario_path = get_scenario_path(scenarios, selected_layout)

            if selected_scenario_path and "o3de" in st.session_state:
                # Setup new scene with the selected layout
                try:
                    # Clear the current scene first
                    st.sidebar.info("🧹 Clearing current scene...")
                    try:
                        st.sidebar.success("✅ Scene cleared")
                    except Exception as clear_error:
                        st.sidebar.warning(
                            f"⚠️ Could not clear scene: {str(clear_error)}"
                        )

                    # Setup new scene
                    st.sidebar.info(f"🎯 Setting up scene: {selected_layout}")
                    new_scenario = setup_new_scene(
                        st.session_state["o3de"], selected_scenario_path
                    )
                    st.session_state["current_scenario"] = new_scenario
                    st.session_state["current_layout"] = selected_layout

                    # Reset agent history for new layout
                    st.session_state["messages"] = [
                        AIMessage(
                            content="Hi! I am a robotic arm. What can I do for you?"
                        )
                    ]

                    st.sidebar.success(f"✅ Scene updated to: {selected_layout}")
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to setup new scene: {str(e)}")
            else:
                st.session_state["current_layout"] = selected_layout

    # Add reload and restart buttons
    col1, col2 = st.sidebar.columns(2)

    with col1:
        if st.button("🔄 Reload Layout", help="Reload the current scene layout"):
            if "o3de" in st.session_state and "current_layout" in st.session_state:
                # Find the scenario path for the current layout
                current_scenario_path = None
                for s in scenarios:
                    if (
                        Path(s.scene_config_path).stem
                        == st.session_state["current_layout"]
                    ):
                        current_scenario_path = s.scene_config_path
                        break

                if current_scenario_path:
                    try:
                        # Clear the current scene first
                        st.sidebar.info("🧹 Clearing current scene...")
                        try:
                            st.session_state["o3de"].clear_scene()
                            st.sidebar.success("✅ Scene cleared")
                        except Exception as clear_error:
                            st.sidebar.warning(
                                f"⚠️ Could not clear scene: {str(clear_error)}"
                            )

                        # Reload the scene
                        st.sidebar.info(
                            f"🔄 Reloading scene: {st.session_state['current_layout']}"
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

                        st.sidebar.success(
                            f"✅ Scene reloaded: {st.session_state['current_layout']}"
                        )
                    except Exception as e:
                        st.sidebar.error(f"❌ Failed to reload scene: {str(e)}")
                else:
                    st.sidebar.error(
                        "❌ Could not find scenario path for current layout"
                    )
            else:
                st.sidebar.warning("⚠️ Demo not initialized yet")

    with col2:
        if st.button(
            "🔄 Restart Demo", help="Restart the demo with the selected layout"
        ):
            # Reset layout and call o3de.reset_arm()
            if "o3de" in st.session_state:
                try:
                    st.session_state["o3de"].reset_arm()
                    st.sidebar.success("✅ Arm has been reset.")
                except Exception as e:
                    st.sidebar.error(f"❌ Failed to reset arm: {e}")
            st.session_state["current_layout"] = selected_layout
            st.rerun()

    st.sidebar.markdown("---")
    st.sidebar.header("Tool Calls History")

    if "ros" not in st.session_state:
        ros = init_ros()
        st.session_state["ros"] = ros

    if "o3de" not in st.session_state:
        selected_scenario_path = get_scenario_path(scenarios, scenario)
        o3de, initial_scenario = initialize_o3de(
            selected_scenario_path, simulation_config
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
        elif isinstance(msg, ToolMessage):
            with st.sidebar.expander(f"Tool: {msg.name}", expanded=False):
                st.code(msg.content, language="json")

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
