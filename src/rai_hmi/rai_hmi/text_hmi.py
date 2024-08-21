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
#

import logging
import sys
from typing import cast

import rclpy
import streamlit as st
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    ToolCall,
    ToolMessage,
)
from langchain_openai.chat_models import ChatOpenAI

from rai_hmi.agent import create_conversational_agent
from rai_hmi.base import BaseHMINode

logger = logging.getLogger(__name__)


st.set_page_config(page_title="LangChain Chat App", page_icon="🦜")
robot_description_package = sys.argv[1] if len(sys.argv) > 1 else None

if robot_description_package:
    st.title(f"{robot_description_package.replace('_whoami', '')} chat app")
else:
    st.title("ROS 2 Chat App")
    logging.warning(
        "No robot_description_package provided. Some functionalities may not work."
    )


@st.cache_resource
def initialize_ros_node(robot_description_package: str):
    rclpy.init()

    node = BaseHMINode(
        f"{robot_description_package}_hmi_node",
        robot_description_package=robot_description_package,
    )

    return node


@st.cache_resource
def initialize_agent(_node):
    llm = ChatOpenAI(
        temperature=0.5,
        model="gpt-4o-mini",
        streaming=True,
    )
    llm = create_conversational_agent(
        llm, _node.tools, _node.system_prompt, logger=_node.get_logger()
    )
    return llm


def initialize_session_memory():
    if "memory" not in st.session_state:
        st.session_state.memory = []
    if "tool_calls" not in st.session_state:
        st.session_state.tool_calls = {}


def convert_langchain_message_to_streamlit_message(message):
    if isinstance(message, HumanMessage):
        return {"type": "user", "avatar": "🧑‍💻", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"type": "bot", "avatar": "🤖", "content": message.content}
    elif isinstance(message, ToolMessage):
        return {"type": "bot", "avatar": "🛠️", "content": message.content}
    else:
        return {"type": "unknown", "content": message.content}


def handle_history_message(message: BaseMessage):
    if isinstance(message, HumanMessage):
        user_chat_obj = st.chat_message("user", avatar="🧑‍💻")
        user_chat_obj.markdown(message.content)
    elif isinstance(message, AIMessage):
        if message.content == "":
            return
        bot_chat_obj = st.chat_message("bot", avatar="🤖")
        bot_chat_obj.markdown(message.content)
    elif isinstance(message, ToolMessage):
        tool_call = st.session_state.tool_calls[message.tool_call_id]
        label = tool_call.name + " status: "
        status = "✅" if message.status == "success" else "❌"
        tool_chat_obj = st.expander(label=label + status).chat_message(
            "bot", avatar="🛠️"
        )
        tool_chat_obj.markdown(message.content)
    else:
        st.write("Unknown message type")


if __name__ == "__main__":
    with st.spinner("Initializing ROS 2 node..."):
        node = initialize_ros_node(robot_description_package)
    agent = initialize_agent(_node=node)
    initialize_session_memory()

    status = {
        "robot_database": node.faiss_index is not None,
        "system_prompt": node.system_prompt == "",
    }
    with st.expander("System status", expanded=False):
        st.json(status)

    for message in st.session_state.memory:
        handle_history_message(message)

    if prompt := st.chat_input("What is your question?"):
        user_chat_obj = st.chat_message("user", avatar="🧑‍💻")
        user_chat_obj.markdown(prompt)
        st.session_state.memory.append(HumanMessage(content=prompt))

        message_placeholder = st.container()
        with message_placeholder:
            with st.spinner("Thinking..."):
                tool_placeholders = {}
                for state in agent.stream({"messages": st.session_state.memory}):
                    node_name = list(state.keys())[0]
                    if node_name == "thinker":
                        last_message = state[node_name]["messages"][-1]
                        if last_message.content:
                            st_message = convert_langchain_message_to_streamlit_message(
                                last_message
                            )
                            st.chat_message(
                                st_message["type"], avatar=st_message["avatar"]
                            ).markdown(st_message["content"])
                            continue

                        called_tools = last_message.tool_calls
                        for tool_call in called_tools:
                            tool_call = cast(ToolCall, tool_call)
                            st.session_state.tool_calls[tool_call["id"]] = tool_call
                            with st.expander(f"Tool call: {tool_call['name']}"):
                                st.markdown(f"Arguments: {tool_call['args']}")
                                tool_placeholders[tool_call["id"]] = st.empty()

                    elif node_name == "tools":
                        tool_messages = []
                        for message in state[node_name]["messages"][::-1]:
                            if message.type == "tool":
                                st.session_state.tool_calls[message.tool_call_id] = (
                                    message
                                )
                                if message.tool_call_id in tool_placeholders:
                                    st_message = (
                                        convert_langchain_message_to_streamlit_message(
                                            message
                                        )
                                    )
                                    with tool_placeholders[message.tool_call_id]:
                                        st.chat_message(
                                            st_message["type"],
                                            avatar=st_message["avatar"],
                                        ).markdown(st_message["content"])
                            else:
                                break
