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
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from rai.agents.conversational_agent import create_conversational_agent
from rai.agents.integrations.streamlit import get_streamlit_cb, streamlit_invoke
from rai.tools.ros2.cli import (
    ros2_action,
    ros2_interface,
    ros2_node,
    ros2_param,
    ros2_service,
    ros2_topic,
)
from rai.utils.model_initialization import get_llm_model


@st.cache_resource
def initialize_graph():
    llm = get_llm_model(model_type="complex_model", streaming=True)
    agent = create_conversational_agent(
        llm,
        [ros2_topic, ros2_interface, ros2_node, ros2_service, ros2_action, ros2_param],
        system_prompt="""You are a ROS 2 expert helping a user with their ROS 2 questions. You have access to various tools that allow you to query the ROS 2 system.
                Be proactive and use the tools to answer questions. Retrieve as much information from the ROS 2 system as possible.
                """,
    )
    return agent


def main():
    st.set_page_config(
        page_title="ROS 2 Debugging Assistant",
        page_icon=":robot:",
    )
    st.title("ROS 2 Debugging Assistant")
    st.markdown("---")

    st.sidebar.header("Tool Calls History")

    if "graph" not in st.session_state:
        graph = initialize_graph()
        st.session_state["graph"] = graph

    if "messages" not in st.session_state:
        st.session_state["messages"] = [
            AIMessage(content="Hi! I am a ROS 2 assistant. How can I help you?")
        ]

    prompt = st.chat_input()
    for msg in st.session_state.messages:
        if isinstance(msg, AIMessage):
            if msg.content:
                st.chat_message("assistant").write(msg.content)
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
