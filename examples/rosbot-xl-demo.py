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

from typing import List

import rclpy
import streamlit as st
from langchain_core.tools import BaseTool
from rai import get_llm
from rai.agents import ReActAgent
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2 import (
    GetObjectPositionsTool,
    GetROS2ImageConfiguredTool,
    GetROS2TransformConfiguredTool,
)
from rai.tools.ros2.navigation.nav2_blocking import (
    GetCurrentPoseTool,
    NavigateToPoseBlockingTool,
)
from rai.tools.time import WaitForSecondsTool
from rai.ui.streamlit import chat
from rai_perception.tools import GetGrabbingPointTool

from rai_whoami import EmbodimentInfo

# Set page configuration first
st.set_page_config(
    page_title="RAI ROSBotXL Demo",
    page_icon=":robot:",
)


@st.cache_resource
def initialize_agent() -> ReActAgent:
    rclpy.init()
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/rosbotxl_embodiment.json"
    )

    connector = ROS2Connector(use_sim_time=True)
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
                connector=connector, conversion_ratio=1.0
            ),
        ),
        NavigateToPoseBlockingTool(
            connector=connector, frame_id="map", action_name="navigate_to_pose"
        ),
        GetCurrentPoseTool(
            connector=connector, frame_id="map", robot_frame_id="base_link"
        ),
    ]

    agent = ReActAgent(
        get_llm("complex", streaming=True),
        tools,
        system_prompt=embodiment_info.to_langchain(),
    )
    return agent


chat(
    initialize_agent(),
    title="RAI ROSBotXL Demo",
    greeting="Hi! I am a ROSBotXL robot. What can I do for you?",
)
