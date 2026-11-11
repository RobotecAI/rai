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

import streamlit as st
from rai import get_llm
from rai.agents import ReActAgent
from rai.tools.ros2 import ROS2CLIToolkit
from rai.ui.streamlit import chat

from rai_whoami import EmbodimentInfo


@st.cache_resource
def initialize_agent():
    llm = get_llm("complex", streaming=True)
    embodiment_info = EmbodimentInfo.from_file(
        "examples/embodiments/ros2_debugging_assistant_embodiment.json"
    )
    return ReActAgent(
        llm,
        ROS2CLIToolkit().get_tools(),
        system_prompt=embodiment_info.to_langchain(),
    )


st.set_page_config(
    page_title="ROS 2 Debugging Assistant",
    page_icon=":robot:",
)

chat(
    initialize_agent(),
    title="ROS 2 Debugging Assistant",
    greeting="Hi! I am a ROS 2 assistant. How can I help you?",
)
