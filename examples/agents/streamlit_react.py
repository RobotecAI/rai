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


import rclpy
import streamlit as st
from rai.agents import ReActAgent
from rai.communication.ros2 import ROS2Connector
from rai.tools.ros2 import ROS2Toolkit
from rai.ui.streamlit import chat


@st.cache_resource
def make_agent() -> ReActAgent:
    rclpy.init()
    return ReActAgent(tools=ROS2Toolkit(connector=ROS2Connector()).get_tools())


st.set_page_config(page_title="RAI ReAct agent", page_icon=":robot:")
chat(make_agent(), title="RAI ReAct agent", greeting="Hi! What can I do for you?")
