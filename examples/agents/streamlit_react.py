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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.agents.langchain import ReActAgent
from rai.communication.ros2 import ROS2Connector
from rai.messages import HumanMultimodalMessage
from rai.tools.ros2 import ROS2Toolkit


@st.cache_resource
def initialize_graph():
    rclpy.init()
    ros2_connector = ROS2Connector()
    tools = ROS2Toolkit(connector=ros2_connector).get_tools()
    agent = ReActAgent(target_connectors={}, tools=tools).agent
    return agent


def main():
    st.set_page_config(
        page_title="RAI Manipulation Demo",
        page_icon=":robot:",
    )
    st.title("RAI Manipulation Demo")
    st.markdown("---")

    st.sidebar.header("Tool Calls History")

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
    main()
