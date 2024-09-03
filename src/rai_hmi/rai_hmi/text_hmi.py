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
from pprint import pformat
from typing import Dict, Optional, cast

import rclpy
import streamlit as st
from langchain.tools import tool
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolCall,
    ToolMessage,
)
from langchain_openai.chat_models import ChatOpenAI
from PIL import Image
from pydantic import BaseModel

from rai.agents.conversational_agent import create_conversational_agent
from rai.agents.state_based import get_stored_artifacts
from rai.messages import HumanMultimodalMessage
from rai_hmi.base import BaseHMINode
from rai_hmi.task import Task

logger = logging.getLogger(__name__)

st.set_page_config(page_title="LangChain Chat App", page_icon="🦜")


# ---------- Cached Resources ----------
@st.cache_resource
def parse_args():
    robot_description_package = sys.argv[1] if len(sys.argv) > 1 else None
    return robot_description_package


@st.cache_resource
def initialize_agent(_node: BaseHMINode):
    llm = ChatOpenAI(
        temperature=0.5,
        model="gpt-4o-mini",
        streaming=True,
    )

    @tool
    def add_task_to_queue(task: Task):
        """Use this tool to add a task to the queue. The task will be handled by the executor part of your system."""
        _node.add_task_to_queue(task)
        return f"Task added to the queue: {task.json()}"

    tools = [add_task_to_queue]
    agent = create_conversational_agent(
        llm, _node.tools + tools, _node.system_prompt, logger=_node.get_logger()
    )
    return agent


@st.cache_resource
def initialize_ros_node(robot_description_package: Optional[str]):
    rclpy.init()

    node = BaseHMINode(
        f"{robot_description_package}_hmi_node",
        robot_description_package=robot_description_package,
    )

    return node


class EMOJIS:
    human = "🧑‍💻"
    bot = "🤖"
    tool = "🛠️"
    unknown = "❓"
    success = "✅"
    failure = "❌"


# ---------- Helpers ----------
def display_agent_message(message: BaseMessage):
    message.content = cast(str, message.content)  # type: ignore
    if isinstance(message, HumanMessage):
        if isinstance(
            message, HumanMultimodalMessage
        ):  # we do not handle user's images
            return
        st.chat_message("user", avatar=EMOJIS.human).markdown(message.content)
    elif isinstance(message, AIMessage):
        if message.content == "":
            return
        st.chat_message("bot", avatar=EMOJIS.bot).markdown(message.content)
    elif isinstance(message, ToolMessage):
        tool_call = st.session_state.tool_calls[message.tool_call_id]
        label = tool_call.name + " status: "
        status = EMOJIS.success if message.status == "success" else EMOJIS.failure
        tool_chat_obj = st.expander(label=label + status).chat_message(
            "bot", avatar=EMOJIS.tool
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


# ---------- Streamlit Application ----------
class SystemStatus(BaseModel):
    robot_database: bool
    system_prompt: bool


class StreamlitAppLayout:
    """App has two columns: chat + robot state"""

    def __init__(self, robot_description_package) -> None:
        if robot_description_package:
            st.title(f"{robot_description_package.replace('_whoami', '')} chat app")
        else:
            st.title("ROS 2 Chat App")
            logging.warning(
                "No robot_description_package provided. Some functionalities may not work."
            )
        self.n_columns = 2

    def draw(self, system_status: SystemStatus):
        self.draw_app_status_expander(system_status)
        self.draw_columns()

    def draw_columns(self):
        self.chat_column, self.mission_column = st.columns(self.n_columns)

    def draw_app_status_expander(self, system_status: SystemStatus):
        with st.expander("System status", expanded=False):
            st.json(system_status.model_dump())

    def write_chat_msg(self, msg: BaseMessage):
        with self.chat_column:
            display_agent_message(msg)

    def write_mission_msg(self, msg: BaseMessage):
        with self.mission_column:
            display_agent_message(msg)


class Memory:
    def __init__(self) -> None:
        if "mission_memory" not in st.session_state:
            st.session_state.mission_memory = []

        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = []

        if "tool_calls" not in st.session_state:
            st.session_state.tool_calls = {}

    @property
    def mission_memory(self):
        return st.session_state.mission_memory

    @property
    def chat_memory(self):
        return st.session_state.chat_memory

    @property
    def tool_calls(self):
        return st.session_state.tool_calls

    def __repr__(self) -> str:
        return f"===> Chat <===\n{pformat(self.chat_memory)}\n\n===> Mission <===\n{pformat(self.mission_memory)}\n\n===> Tool calls <===\n{pformat(self.tool_calls)}"


class Chat:
    def __init__(self, memory: Memory, layout: StreamlitAppLayout) -> None:
        self.memory = memory
        self.layout = layout

    def user(self, txt):
        logger.info(f'User said: "{txt}"')
        msg = HumanMessage(content=txt)
        self.memory.chat_memory.append(msg)
        self.layout.write_chat_msg(msg)

    def bot(self, msg):
        logger.info(f'Bot said: "{msg}"')
        self.memory.chat_memory.append(msg)
        self.layout.write_chat_msg(msg)


class Agent:
    def __init__(self, node, memory) -> None:
        self.agent = initialize_agent(node)
        self.memory = memory

    def stream(self):
        # Copy, because agent's memory != streamlit app memory. App memory is used to
        # recreate the page, agent might manipulate it's history to perform the task.
        # In simplest case it adds thinker message to the state.
        messages = self.memory.chat_memory.copy()

        return self.agent.stream({"messages": messages})


class StreamlitApp:
    def __init__(self, robot_description_package) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.robot_description_package = robot_description_package

        self.layout = StreamlitAppLayout(self.robot_description_package)
        self.memory = Memory()
        self.chat = Chat(self.memory, self.layout)

        self.node = self.initialize_node(robot_description_package)
        self.agent = Agent(self.node, self.memory)

    def run(self):
        system_status = self.get_system_status()
        self.layout.draw(system_status)

        self.recreate_from_memory()
        self.interact()

    def get_system_status(self) -> SystemStatus:
        return SystemStatus(
            robot_database=self.node.faiss_index is not None,
            system_prompt=self.node.system_prompt == "",
        )

    def initialize_node(self, robot_description_package):
        with st.spinner("Initializing ROS 2 node..."):
            node = initialize_ros_node(robot_description_package)
            self.logger.info("ROS 2 node initialized")
        return node

    def recreate_from_memory(self):
        with self.layout.chat_column:
            for message in self.memory.chat_memory:
                display_agent_message(message)

        with self.layout.mission_column:
            for message in self.memory.mission_memory:
                display_agent_message(message)

    def interact(self):
        if prompt := st.chat_input("What is your question?"):
            self.chat.user(prompt)

            message_placeholder = st.container()
            with message_placeholder:
                with st.spinner("Thinking..."):
                    tool_placeholders = {}
                    for state in self.agent.stream():
                        node_name = list(state.keys())[0]
                        if node_name == "thinker":
                            last_message = state[node_name]["messages"][-1]
                            if last_message.content:
                                self.chat.bot(last_message)

                            called_tools = last_message.tool_calls
                            for tool_call in called_tools:
                                tool_call = cast(ToolCall, tool_call)
                                st.session_state.tool_calls[tool_call["id"]] = tool_call
                                with st.expander(f"Tool call: {tool_call['name']}"):
                                    st.markdown(f"Arguments: {tool_call['args']}")
                                    tool_placeholders[tool_call["id"]] = st.empty()

                        elif node_name == "tools":
                            last_ai_msg_idx = 0
                            for message in state[node_name]["messages"]:
                                if isinstance(message, AIMessage):
                                    last_ai_msg_idx = state[node_name][
                                        "messages"
                                    ].index(message)

                            for message in state[node_name]["messages"][
                                last_ai_msg_idx + 1 :  # noqa: E203
                            ]:
                                if message.type == "tool":
                                    st.session_state.tool_calls[
                                        message.tool_call_id
                                    ] = message
                                    if message.tool_call_id in tool_placeholders:
                                        st_message = convert_langchain_message_to_streamlit_message(
                                            message
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
                                                    base_64_image = artifact["images"][
                                                        0
                                                    ]
                                                    image = decode_base64_into_image(
                                                        base_64_image
                                                    )
                                                    st.image(image)
                                else:
                                    break


def decode_base64_into_image(base64_image: str):
    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    return image


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


if __name__ == "__main__":
    robot_description_package = parse_args()
    app = StreamlitApp(robot_description_package)
    app.run()
