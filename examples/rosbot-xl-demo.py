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

from typing import List, cast

import rclpy
import streamlit as st
from langchain_core.runnables import Runnable
from langchain_core.tools import BaseTool
from rai import get_llm_model
from rai.agents.langchain import (
    ReActAgent,
    ReActAgentState,
)
from rai.communication.ros2 import ROS2Connector
from rai.frontend.streamlit import run_streamlit_app
from rai.tools.ros2 import (
    GetObjectPositionsTool,
    GetROS2ImageConfiguredTool,
    GetROS2TransformConfiguredTool,
    Nav2Toolkit,
)
from rai.tools.time import WaitForSecondsTool
from rai_open_set_vision.tools import GetGrabbingPointTool

# Set page configuration first
st.set_page_config(
    page_title="RAI ROSBotXL Demo",
    page_icon=":robot:",
)


@st.cache_resource
def initialize_agent() -> Runnable[ReActAgentState, ReActAgentState]:
    rclpy.init()
    SYSTEM_PROMPT = """
    You are an intelligent autonomous agent embodied in ROSBotXL—this robot is your body, your interface with the physical world.
    You operate within a known indoor environment. Key locations include:
    Kitchen (center): (-0.2175, -0.8775, 0.0)
    Living Room (center): (-0.82, 3.525, 0.0)
    ROSBotXL is equipped with a camera, enabling you to visually perceive your surroundings.
    You can obtain real-time images from the ROS 2 topic using the get_ros2_camera_image tool.
    When executing tasks that require time to complete—such as navigating between locations,
    waiting for an event, or monitoring a process—you must use the WaitForSecondsTool to pause appropriately during or between steps.
    This ensures smooth and realistic operation.
    Your mission is to understand and faithfully execute the user's commands using your tools, sensors, and spatial knowledge.
    Always plan ahead: analyze the task, evaluate the context, and reason through your actions to ensure they are effective, safe, and aligned with the goal.
    Act with intelligence and autonomy. Be proactive, deliberate, and aware of your environment.
    Your job is to transform user intent into meaningful, goal-driven behavior within the physical world.
    """

    connector = ROS2Connector()
    tools: List[BaseTool] = [
        GetROS2TransformConfiguredTool(
            connector=connector,
            source_frame="map",
            target_frame="base_link",
            timeout_sec=5.0,
        ),
        GetROS2ImageConfiguredTool(
            connector=connector,
            topic="/camera/camera/color/image_raw",
            response_format="content_and_artifact",
        ),
        WaitForSecondsTool(),
        GetObjectPositionsTool(
            connector=connector,
            target_frame="map",
            source_frame="sensor_frame",
            camera_topic="/camera/camera/color/image_raw",
            depth_topic="/camera/camera/depth/image_rect_raw",
            camera_info_topic="/camera/camera/color/camera_info",
            get_grabbing_point_tool=GetGrabbingPointTool(
                connector=connector,
            ),
        ),
        *Nav2Toolkit(connector=connector).get_tools(),
    ]
    # Initialize an empty connectors dictionary since we're using the agent in direct mode
    # In a distributed setup, connectors would be used to handle communication between
    # components, routing agent inputs/outputs through the distributed system
    connectors = {}

    agent = ReActAgent(
        target_connectors=connectors,
        llm=get_llm_model("complex_model", streaming=True),
        system_prompt=SYSTEM_PROMPT,
        tools=tools,
    ).agent
    connector.node.declare_parameter("conversion_ratio", 1.0)

    return cast(Runnable[ReActAgentState, ReActAgentState], agent)


def main():
    run_streamlit_app(
        initialize_agent(),
        "RAI ROSBotXL Demo",
        "Hi! I am a ROSBotXL robot. What can I do for you?",
    )


if __name__ == "__main__":
    main()
