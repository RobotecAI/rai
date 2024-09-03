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
#
import base64
import io
import logging
import sys
import threading
import time
import uuid
from pprint import pformat
from queue import Queue
from typing import List, Optional, cast

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
from streamlit.delta_generator import DeltaGenerator

from rai.agents.conversational_agent import create_conversational_agent
from rai.agents.state_based import get_stored_artifacts
from rai.messages import HumanMultimodalMessage
from rai_hmi.base import BaseHMINode
from rai_hmi.chat_msgs import EMOJIS, MissionMessage
from rai_hmi.task import Task, TaskInput

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="LangChain Chat App", page_icon="🦜")

MODEL = "gpt-4o"
MAX_DISPLAY = 5


class Memory:
    def __init__(self) -> None:
        if "mission_memory" not in st.session_state:
            st.session_state.mission_memory = []

        if "chat_memory" not in st.session_state:
            st.session_state.chat_memory = []

        if "tool_calls" not in st.session_state:
            st.session_state.tool_calls = {}

        if "missions_uids" not in st.session_state:
            st.session_state.missions_uids = set()

    @property
    def mission_memory(self):
        return st.session_state.mission_memory

    @property
    def missions_uids(self):
        return st.session_state.missions_uids

    @property
    def chat_memory(self):
        return st.session_state.chat_memory

    @property
    def tool_calls(self):
        return st.session_state.tool_calls

    def register_tool_calls(self, tool_calls: List[ToolCall]):
        for tool_call in tool_calls:
            tool_call = cast(ToolCall, tool_call)
            tool_id = tool_call["id"]
            self.tool_calls[tool_id] = tool_call

    def add_mission(self, msg: MissionMessage):
        self.mission_memory.append(msg)
        self.missions_uids.add(msg.uid)

    def get_mission_memory(self, uid: Optional[str] = None) -> List[MissionMessage]:
        logger.info(f"Mission memory: {self.mission_memory}")
        if not uid:
            return self.mission_memory

        if uid not in self.missions_uids:
            raise AssertionError(f"Mission with {uid} not found")

        return [m for m in self.mission_memory if m["uid"] == uid]

    def __repr__(self) -> str:
        return f"===> Chat <===\n{pformat(self.chat_memory)}\n\n===> Mission <===\n{pformat(self.mission_memory)}\n\n===> Tool calls <===\n{pformat(self.tool_calls)}"


# ---------- Cached Resources ----------
@st.cache_resource
def parse_args():
    robot_description_package = sys.argv[1] if len(sys.argv) > 1 else None
    return robot_description_package


@st.cache_resource
def initialize_agent(_node: BaseHMINode, _memory: Memory):
    llm = ChatOpenAI(
        temperature=0.5,
        model=MODEL,
        streaming=True,
    )

    @tool
    def get_mission_memory(uid: Optional[str] = None) -> List[MissionMessage]:
        """List mission memory. Pass uid if memory for specific mission is required."""
        return _memory.get_mission_memory(uid)

    @tool
    def add_task_to_queue(task: TaskInput):
        """Use this tool to add a task to the queue. The task will be handled by the executor part of your system."""

        uid = uuid.uuid4()
        _node.add_task_to_queue(
            Task(
                name=task.name,
                description=task.description,
                priority=task.priority,
                uid=uid,
            )
        )
        return f"Task added to the queue: {task.json()}"

    tools = [add_task_to_queue, get_mission_memory]
    agent = create_conversational_agent(
        llm, _node.tools + tools, _node.system_prompt, logger=_node.get_logger()
    )
    return agent


@st.cache_resource
def initialize_mission_queue() -> Queue:
    return Queue()


@st.cache_resource
def initialize_ros_node(
    _feedbacks_queue: Queue, robot_description_package: Optional[str]
):
    rclpy.init()

    node = BaseHMINode(
        f"{robot_description_package}_hmi_node",
        queue=_feedbacks_queue,
        robot_description_package=robot_description_package,
    )
    threading.Thread(target=rclpy.spin, args=(node,), daemon=True).start()

    return node


def display_agent_message(
    message,  # TODO(boczekbartek): add typhint
    tool_chat_obj: Optional[DeltaGenerator] = None,
    no_expand: bool = False,
):
    """
    Display LLM message in streamlit UI.

    Args:
        message: The message to display
        tool_chat_obj: Pre-existing Streamlit object for the tool message.
        no_expand: Skip expanders due - Streamlit does not support nested expanders.

    """
    message.content = cast(str, message.content)  # type: ignore
    if not message.content:
        return  # Tool messages might not have any content, skip displying them
    if isinstance(message, HumanMessage):
        if isinstance(
            message, HumanMultimodalMessage
        ):  # we do not handle user's images
            return
        st.chat_message("user", avatar=EMOJIS.human).markdown(message.content)
    elif isinstance(message, AIMessage):
        st.chat_message("bot", avatar=EMOJIS.bot).markdown(message.content)
    elif isinstance(message, ToolMessage):
        tool_call = st.session_state.tool_calls[message.tool_call_id]
        label = tool_call.name + " status: "
        status = EMOJIS.success if message.status == "success" else EMOJIS.failure
        if not tool_chat_obj:
            if not no_expand:
                tool_chat_obj = st.expander(label=label + status).chat_message(
                    "bot", avatar=EMOJIS.tool
                )
            else:
                tool_chat_obj = st.chat_message("bot", avatar=EMOJIS.tool)
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
    elif isinstance(message, MissionMessage):
        avatar, content = message.render_steamlit()
        st.chat_message("bot", avatar=avatar).markdown(content)
    else:
        raise ValueError("Unknown message type")


# ---------- Streamlit Application ----------
class SystemStatus(BaseModel):
    robot_database: bool
    system_prompt: bool


class Layout:
    """App has two columns: chat + robot state"""

    def __init__(
        self, robot_description_package, max_display: int = MAX_DISPLAY
    ) -> None:
        self.max_display = max_display
        if robot_description_package:
            st.title(f"{robot_description_package.replace('_whoami', '')} chat app")
        else:
            st.title("ROS 2 Chat App")
            logging.warning(
                "No robot_description_package provided. Some functionalities may not work."
            )
        self.n_columns = 2
        self.tool_placeholders = dict()

    def draw(self, system_status: SystemStatus):
        self.draw_app_status_expander(system_status)
        self.draw_columns()

    def draw_app_status_expander(self, system_status: SystemStatus):
        with st.expander("System status", expanded=False):
            st.json(system_status.model_dump())

    def draw_columns(self):
        self.chat_column, self.mission_column = st.columns(self.n_columns)

    def create_tool_expanders(self, tool_calls: List[ToolCall]):
        with self.chat_column:
            for tool_call in tool_calls:
                tool_call = cast(ToolCall, tool_call)
                with st.expander(f"Tool call: {tool_call['name']}"):
                    st.markdown(f"Arguments: {tool_call['args']}")
                    self.tool_placeholders[tool_call["id"]] = st.empty()

    def write_chat_msg(self, msg: BaseMessage):
        with self.chat_column:
            if isinstance(msg, ToolMessage):
                display_agent_message(msg, self.tool_placeholders[msg.tool_call_id])
            else:
                display_agent_message(msg)

    def write_mission_msg(self, msg: MissionMessage):
        with self.mission_column:
            logger.info(f'Mission said: "{msg}"')
            display_agent_message(msg)

    def show_chat(self, history):
        with self.chat_column:
            self.__show_history(history)

    def show_mission(self, history):
        with self.mission_column:
            self.__show_history(history)

    def __show_history(self, history):
        show, hide = self.__split_history(history, self.max_display)
        with st.expander("Untoggle to see full chat history"):
            for message in hide:
                display_agent_message(message, no_expand=True)

        for message in show:
            display_agent_message(message)

    @staticmethod
    def __split_history(history, max_display):
        n_messages = len(history)
        if n_messages > max_display:
            n_hide = n_messages - max_display
            return history[:max_display], history[-n_hide:]
        else:
            return history, []


class Chat:
    def __init__(self, memory: Memory, layout: Layout) -> None:
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

    def tool(self, msg):
        logger.info(f'Tool said: "{msg}"')
        self.memory.chat_memory.append(msg)
        self.layout.write_chat_msg(msg)

    def mission(self, msg: MissionMessage):
        logger.info(f'Mission said: "{msg}"')
        self.memory.add_mission(msg)
        self.layout.write_mission_msg(msg)


class Agent:
    def __init__(self, node, memory) -> None:
        self.memory = memory
        self.agent = initialize_agent(node, self.memory)

    def stream(self):
        # Copy, because agent's memory != streamlit app memory. App memory is used to
        # recreate the page, agent might manipulate it's history to perform the task.
        # In simplest case it adds thinker message to the state.
        messages = self.memory.chat_memory.copy()
        logger.info(f"Sending messages:\n{pformat(messages)}")

        return self.agent.stream({"messages": messages})


class StreamlitApp:
    def __init__(self, robot_description_package) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)

        self.robot_description_package = robot_description_package

        self.layout = Layout(self.robot_description_package)
        self.memory = Memory()
        self.chat = Chat(self.memory, self.layout)

        self.mission_queue = initialize_mission_queue()
        self.ros_node = self.initialize_node(
            self.mission_queue, robot_description_package
        )
        self.agent = Agent(self.ros_node, self.memory)

    def update_mission(self):
        while True:
            if self.mission_queue.empty():
                time.sleep(0.5)
            msg = self.mission_queue.get()
            self.chat.mission(msg)

    def run(self):
        system_status = self.get_system_status()
        self.layout.draw(system_status)

        self.recreate_from_memory()

        st.chat_input(
            "What is your question?", on_submit=self.prompt_callback, key="prompt"
        )

        self.update_mission()

    def get_system_status(self) -> SystemStatus:
        return SystemStatus(
            robot_database=self.ros_node.faiss_index is not None,
            system_prompt=self.ros_node.system_prompt == "",
        )

    def initialize_node(self, feedbacks_queue, robot_description_package):
        self.logger.info("Initializing ROS 2 node...")
        with st.spinner("Initializing ROS 2 node..."):
            node = initialize_ros_node(feedbacks_queue, robot_description_package)
            self.logger.info("ROS 2 node initialized")
        return node

    def recreate_from_memory(self):
        """
        Recreates the page from the memory. It is because Streamlit reloads entire page every widget action.
        See: https://docs.streamlit.io/get-started/fundamentals/main-concepts

        Args:
            max_display (int, optional): Max number of messages to display. Rest will be hidden in a toggle. Defaults to 5.

        """
        self.layout.show_chat(self.memory.chat_memory)
        self.layout.show_mission(self.memory.mission_memory)

    def prompt_callback(self):
        prompt = st.session_state.prompt
        self.chat.user(prompt)

        message_placeholder = st.container()
        with self.layout.chat_column:
            with message_placeholder:
                with st.spinner("Thinking..."):
                    for state in self.agent.stream():
                        # logger.info(f"State:\n{pformat(state)}")
                        node_name = list(state.keys())[0]
                        if node_name == "thinker":
                            last_message = state[node_name]["messages"][-1]
                            self.chat.bot(last_message)
                            self.memory.register_tool_calls(last_message.tool_calls)
                            self.layout.create_tool_expanders(last_message.tool_calls)

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
                                    self.memory.tool_calls[message.tool_call_id] = (
                                        message
                                    )
                                    self.chat.tool(message)
                                else:
                                    break


def decode_base64_into_image(base64_image: str):
    image = Image.open(io.BytesIO(base64.b64decode(base64_image)))
    return image


if __name__ == "__main__":
    robot_description_package = parse_args()
    app = StreamlitApp(robot_description_package)
    app.run()
