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
import base64
import io
import logging
import sys
from queue import Queue
from typing import Dict, Optional, cast

import rclpy
import std_msgs.msg
import streamlit as st
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_core.tools import render_text_description_and_args
from langchain_openai.chat_models import ChatOpenAI
from PIL import Image
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, HistoryPolicy, QoSProfile, ReliabilityPolicy

from rai.agents.conversational_agent import create_conversational_agent
from rai.messages import HumanMultimodalMessage
from rai.node import RaiBaseNode
from rai.tools.ros.native_actions import GetCameraImage
from rai.utils.artifacts import get_stored_artifacts
from rai.utils.model_initialization import get_llm_model
from rai_hmi.base import BaseHMINode
from rai_hmi.basic_nav import (
    backup,
    drive_on_heading,
    get_location,
    get_ros2_interfaces,
    get_ros2_interfaces_tool,
    go_to_pose,
    led_strip,
    led_strip_array,
    spin,
    wait_n_sec_tool,
)

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
def initialize_ros_node(robot_description_package: Optional[str]):
    node = BaseHMINode(
        f"{robot_description_package}_hmi_node",
        robot_description_package=robot_description_package,
        queue=Queue(),
    )

    return node


def decode_base64_into_image(base64_image: str):
    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    return image


@st.cache_resource
def initialize_agent(_node: BaseHMINode):
    llm = ChatOpenAI(
        temperature=0.5,
        model="gpt-4o",
        streaming=True,
    )
    rai_node = RaiBaseNode(node_name="rai_tool_node")
    rclpy.spin_once(rai_node, timeout_sec=1.0)
    tools = [
        get_ros2_interfaces_tool,
        drive_on_heading,
        backup,
        spin,
        go_to_pose,
        get_location,
        led_strip,
        led_strip_array,
        wait_n_sec_tool,
        GetCameraImage(node=rai_node),
    ]
    tools_desc = (
        f"\nThese are available tools:\n{render_text_description_and_args(tools)}"
    )
    interfaces_desc = f"\nThese are available interfaces:\n{get_ros2_interfaces()}"
    rules = """
        <rule> Before every navigation action check you location, to verify if nav2 action took expected result. Actions might fail, but make navigation progress. You need to check that and adapt </rule>
        While the navigation task is running you can do other things. You can also check the task status.
        When you are asked to perform a task carefully, you can split navigation tasks into a couple of smaller ones.
        < rule> don't decide to respond to the user until you are sure that the navigation task is completed </rule>
    """
    system_prompt = _node.system_prompt + tools_desc + interfaces_desc + rules
    _node.system_prompt = system_prompt
    print(f"Agents system prompt: {system_prompt=}")
    agent = create_conversational_agent(
        llm,
        _node.tools + tools,
        system_prompt,
        logger=_node.get_logger(),
    )
    return agent


def initialize_session_memory(system_prompt: str = ""):
    if "memory" not in st.session_state:
        st.session_state.memory = [SystemMessage(content=system_prompt)]
    if "tool_calls" not in st.session_state:
        st.session_state.tool_calls = {}


def convert_langchain_message_to_streamlit_message(
    message: BaseMessage,
) -> Dict[str, str]:
    message.content = cast(str, message.content)  # type: ignore
    if isinstance(message, HumanMessage):
        return {"type": "user", "avatar": "🧑‍💻", "content": message.content}
    elif isinstance(message, AIMessage):
        return {"type": "bot", "avatar": "🤖", "content": message.content}
    elif isinstance(message, ToolMessage):
        return {"type": "bot", "avatar": "🛠️", "content": message.content}
    else:
        return {"type": "unknown", "avatar": "❓", "content": message.content}


def handle_history_message(message: BaseMessage):
    message.content = cast(str, message.content)  # type: ignore
    if isinstance(message, HumanMessage):
        if isinstance(
            message, HumanMultimodalMessage
        ):  # we do not handle user's images
            return
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
        with tool_chat_obj:
            st.markdown(message.content)
            artifacts = get_stored_artifacts(message.tool_call_id)
            for artifact in artifacts:
                if "images" in artifact:
                    base_64_image = artifact["images"][0]
                    image = decode_base64_into_image(base_64_image)
                    st.image(image)
    elif isinstance(message, SystemMessage):
        return  # we do not handle system messages
    else:
        raise ValueError("Unknown message type")


def publish_response_to_ros(node: Node, response: str):
    llm = get_llm_model("simple_model")
    response = llm.invoke(
        [
            SystemMessage(
                content=(
                    "You are an experienced reporter deployed on a autonomous robot. "  # type: ignore
                    "Your task is to summarize the message in a way that is very easy to understand"
                    "for the user. Also the message will be played using tts, so please make it concise "
                    "and don't include any special characters."
                    "Do not use markdown formatting. Keep it short and concise. If the message is empty, please return empty string."
                    "The message have to be in first person!"
                )
            ),
            HumanMessage(content=response),
        ]
    ).content
    node.get_logger().info(f"LLM shortened the response to:\n{response}")

    reliable_qos = QoSProfile(
        reliability=ReliabilityPolicy.RELIABLE,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_ALL,
    )
    publisher = node.create_publisher(std_msgs.msg.String, "/to_human", reliable_qos)
    msg = std_msgs.msg.String()
    msg.data = response
    publisher.publish(msg)
    publisher.destroy()


if __name__ == "__main__":
    with st.spinner("Initializing ROS 2 node..."):
        node = initialize_ros_node(robot_description_package)
    agent = initialize_agent(_node=node)
    initialize_session_memory(system_prompt=node.system_prompt)

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
                for state in agent.stream(
                    {"messages": st.session_state.memory},
                    config={"recursion_limit": 500},
                ):
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
                            publish_response_to_ros(node, st_message["content"])

                        called_tools = last_message.tool_calls
                        for tool_call in called_tools:
                            tool_call = cast(ToolCall, tool_call)
                            st.session_state.tool_calls[tool_call["id"]] = tool_call
                            with st.expander(f"Tool call: {tool_call['name']}"):
                                st.markdown(f"Arguments: {tool_call['args']}")
                                tool_placeholders[tool_call["id"]] = st.empty()

                    elif node_name == "tools":
                        tool_messages = []
                        last_ai_msg_idx = 0
                        for message in state[node_name]["messages"]:
                            if isinstance(message, AIMessage):
                                last_ai_msg_idx = state[node_name]["messages"].index(
                                    message
                                )

                        for message in state[node_name]["messages"][
                            last_ai_msg_idx + 1 :  # noqa: E203
                        ]:
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
                                        artifacts = get_stored_artifacts(
                                            message.tool_call_id
                                        )
                                        for artifact in artifacts:
                                            if "images" in artifact:
                                                base_64_image = artifact["images"][0]
                                                image = decode_base64_into_image(
                                                    base_64_image
                                                )
                                                st.image(image)
                            else:
                                break
